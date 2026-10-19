"""Benchmark the latency of processing a single batch of requests."""
import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.inputs import PromptStrictInputs
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

def main(args: argparse.Namespace):
    print(args)

    llm = LLM(
        model=args.model,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        ray_workers_use_nsight=args.ray_workers_use_nsight,
        use_v2_block_manager=args.use_v2_block_manager,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=False,
        download_dir=args.download_dir,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    # Generate dummy prompts
    # dummy_prompt_token_ids = np.random.randint(10000, size=(args.batch_size, args.input_len))
    # dummy_inputs: List[PromptStrictInputs] = [{"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()]

    fixed_prompt_token_ids = np.random.randint(10000, size=(args.input_len,))  # Create a fixed prompt
    dummy_inputs: List[PromptStrictInputs] = [{"prompt_token_ids": fixed_prompt_token_ids.tolist()} for _ in range(args.batch_size)]  # Use the same prompt for all


    def run_to_completion_tftt(profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir))
            ) as p:
                outputs = llm.generate(dummy_inputs, sampling_params=sampling_params, use_tqdm=False)
            print(p.key_averages())
            return outputs 

        else:
            start_time = time.perf_counter()
            first_token_time = None  

            outputs = []
            for output in llm.generate(dummy_inputs, sampling_params=sampling_params, use_tqdm=False):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                outputs.append(output)  
            if first_token_time:
                ttft = first_token_time - start_time
            else:
                raise ValueError("First token time not captured!")
            total_latency = time.perf_counter() - start_time

            return ttft, total_latency, outputs  


    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion_tftt(profile_dir=None)

    if args.profile:
        profile_dir = args.profile_result_dir
        if not profile_dir:
            profile_dir = Path(".") / "vllm_benchmark_result" / f"latency_result_{time.time()}"
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion_tftt(profile_dir=profile_dir)
        return

    # Benchmark
    ttfts = []  
    latencies = []  

    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        ttft, latency, outputs = run_to_completion_tftt(profile_dir=None)
        ttfts.append(ttft)
        latencies.append(latency)

    # Calculate and display percentiles 
    ttft_percentiles = np.percentile(ttfts, [10, 25, 50, 75, 90])
    print(f'Avg TTFT: {np.mean(ttfts)} seconds')
    for perc, value in zip([10, 25, 50, 75, 90], ttft_percentiles):
        print(f'{perc}% percentile TTFT: {value} seconds')


        # Calculate and display percentiles 
    latencies_percentiles = np.percentile(latencies, [10, 25, 50, 75, 90])
    print(f'Avg latencies: {np.mean(latencies)} seconds')
    for perc, value in zip([10, 25, 50, 75, 90], latencies_percentiles):
        print(f'{perc}% percentile latencies: {value} seconds')


    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
            "outputs": [output.tolist() for output in all_outputs], 
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.'
    )
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--speculative-model', type=str, default=None)
    parser.add_argument('--num-speculative-tokens', type=int, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization', '-q', choices=[*QUANTIZATION_METHODS, None], default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n', type=int, default=1, help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters-warmup', type=int, default=10, help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters', type=int, default=30, help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code', action='store_true', help='Trust remote code from Hugging Face')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
                        help='Data type for model weights and activations. The "auto" option will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.')
    parser.add_argument('--enforce-eager', action='store_true', help='Enforce eager mode and disable CUDA graph')
    parser.add_argument('--kv-cache-dtype', type=str, choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'], default="auto",
                        help='Data type for kv cache storage. If "auto", will use model data type.')
    parser.add_argument('--quantization-param-path', type=str, default=None,
                        help='Path to the JSON file containing the KV cache scaling factors.')
    parser.add_argument('--profile', action='store_true', help='Profile the generation process of a single batch')
    parser.add_argument('--profile-result-dir', type=str, default=None,
                        help='Path to save the PyTorch profiler output.')
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"],
                        help='Device type for vLLM execution, supporting CUDA and CPU.')
    parser.add_argument('--block-size', type=int, default=16, help='Block size of key/value cache')
    parser.add_argument('--enable-chunked-prefill', action='store_true', help='If True, the prefill requests can be chunked based on the max_num_batched_tokens')
    parser.add_argument('--use-v2-block-manager', action='store_true')
    parser.add_argument('--ray-workers-use-nsight', action='store_true', help="If specified, use nsight to profile ray workers")
    parser.add_argument('--download-dir', type=str, default=None,
                        help='Directory to download and load the weights, default to the default cache dir of Hugging Face')
    parser.add_argument('--output-json', type=str, default=None, help='Path to save the latency results in JSON format.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.')

    args = parser.parse_args()
    main(args)
