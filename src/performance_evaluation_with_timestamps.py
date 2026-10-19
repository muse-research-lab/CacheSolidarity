import asyncio
import time
import os
import random
from datetime import datetime
import argparse
import pandas as pd
import gc
import torch
import logging
from typing import Optional
import json
import numpy as np

from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.user_metadata import UserMetadata
from vllm import RequestOutput as vllmRequestOutput
from vllm.user_monitor import UserMonitor
from llmperf.postprocessing.output import RequestOutput, ExperimentOutput

import glob
import shutil


# === CONFIG ===
GPU_UTIL = 0.9
SWAP_SPACE = 0
user_monitor = UserMonitor()


# === HELPERS ===
def load_user_prompts(csv_path):
    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if not {'user_id', 'sentence'}.issubset(df.columns):
        raise ValueError(f"'user_id' and 'sentence' columns required in {csv_path}")
    return list(zip(df['user_id'].astype(str), df['sentence']))


def sanitize_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def setup_logger(log_path):
    print(f"[INFO] Setting up logger at: {log_path}")
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s'
    )

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.handlers = [file_handler]
    return logger


# === ASYNC REQUEST HANDLER ===
async def send_request(
    llm,
    prompt,
    user_id: str,
    slo: float,
    request_number: int,
    user_monitor: UserMonitor
) -> tuple[str, RequestOutput]:
    """Send a single request to the model and return (request_id, RequestOutput)."""
    start_ts = time.time()
    request_id = f"{user_id}_{int(start_ts * 1000)}_{random.randint(0, 9999)}"

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        top_p=1.0,
        ignore_eos=True
    )

    final_output: Optional[vllmRequestOutput] = None
    user_metadata = UserMetadata(uid=user_id, slo=slo)
    final_prompt = {"prompt": prompt}

    async for output in llm.generate(
        final_prompt, sampling_params, request_id=request_id, user_md=user_metadata
    ):
        final_output = output

    request_output = RequestOutput.from_vllm_output(
        request_id,
        final_output,
        modality_token_index=-1
    )
    return request_id, request_output


# === MAIN DATASET RUNNER ===
async def run_dataset(csv_path: str, log_output_dir: str, sequential: bool = False,
                      request_rate: Optional[float] = None, time_quantum: int = 100,
                      duration: Optional[int] = None, seed: int = 42,
                      model_path: str = "./models/Llama-2-13b-chat-hf",
                      sys_mode: str = "APC",
                      experiment_name: str = "default",
                      tensor_parallel_size: int = 1):
    print(f"[RUN] Starting dataset: {csv_path}")
    dataset_name = sanitize_filename(csv_path)
    log_path = os.path.join(log_output_dir, f"{dataset_name}.log")
    logger = setup_logger(log_path)

    rate_label = request_rate if request_rate is not None else "NA"
    output_dir = os.path.join(
        "./temp_results",
        experiment_name,
        os.path.basename(model_path),
        dataset_name,
        f"Rate_{rate_label}",
        sys_mode,
    )
    os.makedirs(output_dir, exist_ok=True)
    os.environ["VLLM_BLOCK_LOG_DIR"] = output_dir
    logger.info(f"Using result directory: {output_dir}")

    # Initialize ALL modes as disabled
    APC_enabled = False
    Isolation_enabled = False
    CacheSolidarity_enabled = False
    SafeKV_enabled = False

    if sys_mode not in ["APC", "Isolation", "CacheSolidarity", "SafeKV"]:
        logger.error(f"Invalid sys_mode: {sys_mode}. Must be one of ['APC', 'Isolation', 'CacheSolidarity', 'SafeKV']")
        return
    
    if sys_mode == "APC":
        logger.info("APC mode selected")
        APC_enabled = True
    elif sys_mode == "Isolation":
        logger.info("Isolation mode selected")
        Isolation_enabled = True
    elif sys_mode == "CacheSolidarity":
        logger.info("CacheSolidarity mode selected")
        CacheSolidarity_enabled = True
    elif sys_mode == "SafeKV":
        logger.info("SafeKV mode selected")
        SafeKV_enabled = True

    try:
        user_prompts = load_user_prompts(csv_path)
    except Exception as e:
        logger.error(f"Failed to load prompts from {csv_path}: {e}")
        print(f"[ERROR] Failed to load CSV: {e}")
        return

    logger.info(f"Running dataset: {csv_path} with {len(user_prompts)} total requests")
    print(f"[RUN] Number of requests: {len(user_prompts)}")

    print("[RUN] Initializing vLLM engine...")
    engine_args = AsyncEngineArgs(
        model=model_path,
        gpu_memory_utilization=GPU_UTIL,
        max_model_len=4096,
        swap_space=SWAP_SPACE,
        disable_log_stats=False,
        enable_prefix_caching=True,
        enable_safekv=SafeKV_enabled,
        enable_CacheSolidarity=CacheSolidarity_enabled,
        enable_APC=APC_enabled,
        enable_Isolation=Isolation_enabled,
        tensor_parallel_size=tensor_parallel_size,
    )
    llm = AsyncLLM.from_engine_args(engine_args)
    print("[RUN] Engine initialized")

    results = []
    user_request_count = {}
    start_time = time.time()

    try:
        if sequential and request_rate is None:
            print("[RUN] Running in sequential mode")
            for i, (user_id, prompt) in enumerate(user_prompts, start=1):
                request_number = user_request_count.get(user_id, 0) + 1
                user_request_count[user_id] = request_number
                slo = 0.0

                request_id, request_output = await send_request(
                    llm, prompt, user_id, slo, request_number, user_monitor
                )
                results.append(request_output)
                print(f"[SEQ] Completed {i}/{len(user_prompts)} requests: {request_id}")

        elif request_rate is None:
            print("[RUN] Running concurrently (no rate limiting)")
            all_tasks = []
            for user_id, prompt in user_prompts:
                request_number = user_request_count.get(user_id, 0) + 1
                user_request_count[user_id] = request_number
                slo = 0.0
                all_tasks.append(
                    send_request(llm, prompt, user_id, slo, request_number, user_monitor)
                )

            completed = await asyncio.gather(*all_tasks)
            for i, (request_id, request_output) in enumerate(completed, start=1):
                results.append(request_output)
                print(f"[CONC] Completed {i}/{len(all_tasks)} requests: {request_id}")

        else:
            print(f"[RUN] Running rate-limited mode: {request_rate} req/s")
            if duration is None:
                duration = len(user_prompts) / request_rate
            np.random.seed(seed)
            lam = request_rate * (time_quantum / 1000)
            quantums_per_sec = 1000 / time_quantum
            arrival_times = np.random.poisson(lam=lam, size=int(duration * quantums_per_sec))
            timestamps = []
            for i, n in enumerate(arrival_times):
                timestamps += [i * (time_quantum / 1000)] * n

            all_tasks = []
            start_time = time.time()
            for idx, (user_id, prompt) in enumerate(user_prompts):
                if idx >= len(timestamps):
                    break
                scheduled_time = timestamps[idx]
                now = time.time() - start_time
                delay = scheduled_time - now
                if delay > 0:
                    await asyncio.sleep(delay)
                request_number = user_request_count.get(user_id, 0) + 1
                user_request_count[user_id] = request_number
                slo = 0.0
                task = asyncio.create_task(
                    send_request(llm, prompt, user_id, slo, request_number, user_monitor)
                )
                all_tasks.append(task)

            completed = await asyncio.gather(*all_tasks)
            for i, (request_id, request_output) in enumerate(completed, start=1):
                results.append(request_output)

    finally:
        elapsed_time = time.time() - start_time
        print(f"[RUN] Dataset completed in {elapsed_time:.2f}s")

        experiment_output = ExperimentOutput(
            id=f"{dataset_name}",
            elapsed_time=elapsed_time,
            request_outputs=results
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = os.path.basename(model_path)
        config_name = f"file_{sanitize_filename(csv_path)}_rate{request_rate}_{timestamp}"

        output_dir = (
            experiment_output.output_path
            or os.path.join(
                "./temp_results",
                experiment_name,
                model_name,
                sanitize_filename(csv_path),
                f"Rate_{request_rate if request_rate is not None else 'NA'}",
                sys_mode,
            )
        )
        os.makedirs(output_dir, exist_ok=True)
        os.environ["VLLM_BLOCK_LOG_DIR"] = output_dir

        output_path = os.path.join(output_dir, f"{experiment_output.id}.jsonl")

        with open(output_path, "w", encoding="utf-8") as file:
            for req in experiment_output.request_outputs:
                req_dict = req.__dict__.copy()
                req_dict.update({
                    "processor_time": req.processor_time,
                    "encoder_time": req.encoder_time,
                    "ttft": req.ttft,
                    "tbt": req.tbt,
                    "e2e": req.e2e,
                    "queued_time": req.queued_time,
                    "prefill_time": req.prefill_time,
                    "decode_time": req.decode_time,
                    "inference_time": req.inference_time,
                })
                file.write(json.dumps(req_dict) + "\n")

        print(f"[SAVE] Experiment output saved at {output_path}")

        # Cleanup
        if 'llm' in locals():
            del llm
        gc.collect()
        torch.cuda.empty_cache()
        print("[CLEANUP] GPU memory cleared")


# === ARG PARSER ===
def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM experiments from prompt CSVs.")
    parser.add_argument('--log_output_dir', type=str, default="./logs", help='Directory to save logs')
    parser.add_argument('--csv_paths', type=str, nargs='+', required=True, help='Paths to CSV files')
    parser.add_argument('--sequential', action='store_true', help='Run requests sequentially')
    parser.add_argument('--request_rate', type=float, default=None, help='Requests per second')
    parser.add_argument('--time_quantum', type=int, default=100, help='Time quantum in ms for Poisson')
    parser.add_argument('--duration', type=int, default=None, help='Duration in seconds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--tensor_parallel_size', '--tensot_parallel_size', type=int, default=1,
                        help='Tensor parallel size for the vLLM engine')
    parser.add_argument('--model_path', type=str, default="./models/Llama-2-13b-chat-hf",
                        help='Path to the LLM model')
    parser.add_argument('--sys_mode', type=str, default="APC", help='System mode (e.g., APC, Isolation, CacheSolidarity, SafeKV)')
    parser.add_argument('--experiment_name', type=str, default="default", help='Name of the experiment')
    return parser.parse_args()


# === MAIN EXECUTION ===
if __name__ == '__main__':
    args = parse_args()

    for csv_path in args.csv_paths:
        print(f"[MAIN] Running dataset: {csv_path}")

        asyncio.run(run_dataset(
            csv_path,
            args.log_output_dir,
            sequential=args.sequential,
            request_rate=args.request_rate,
            time_quantum=args.time_quantum,
            duration=args.duration,
            seed=args.seed,
            model_path=args.model_path,
            sys_mode=args.sys_mode,
            experiment_name=args.experiment_name,
            tensor_parallel_size=args.tensor_parallel_size,
        ))

