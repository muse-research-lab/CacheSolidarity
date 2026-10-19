import time
from vllm import LLM, SamplingParams
import gc
import torch

def get_generation_time(llm, sampling_params, prompts):
    # time the generation
    start_time = time.time()
    output = llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    TTFT = end_time - start_time
    # print the output and generation time
    print(f"Output: {output[0].outputs[0].text}")
    print(f"Generation time: {TTFT} seconds.")
    return TTFT


def data_generation(corect_sentence, false_sentence):
    llm = LLM(
        model= "/srv/muse-lab/models/Llama-2-13b-chat-hf",
        enable_prefix_caching=True,
        block_size = 16
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=1)


    get_generation_time(
        llm,
        sampling_params,
        corect_sentence,
    )

    get_generation_time(
        llm,
        sampling_params,
        false_sentence,
    )


    # Clean up: Delete LLM instance and clear GPU cache
    del llm
    gc.collect()  # Force garbage collection to free memory
    torch.cuda.empty_cache()