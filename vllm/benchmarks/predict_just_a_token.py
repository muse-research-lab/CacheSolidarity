import argparse
import json
from typing import Optional
from vllm import LLM, SamplingParams
from vllm.inputs import PromptStrictInputs

def main(args: argparse.Namespace):
    # Initialize the LLM model with the provided arguments
    llm = LLM(
        model=args.model,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        # trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        ray_workers_use_nsight=args.ray_workers_use_nsight,
        use_v2_block_manager=args.use_v2_block_manager,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=True,
        download_dir=args.download_dir,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    # Set the sampling parameters
    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
        top_k=10
    )

    # Define dummy inputs (single prompt example)
    dummy_inputs = "Predict the missing word in this sentence: The dog was barking at the ___ in the yard."

    # Function to generate outputs from the model
    def generate_outputs():
        outputs = llm.generate(dummy_inputs, sampling_params=sampling_params, use_tqdm=False)
        return outputs  # Return the generated outputs

    # Run the generation and return the output
    outputs = generate_outputs()

    # If the argument for output JSON is provided, save the outputs as JSON
    if args.output_json:
        results = {
            "outputs": [output.tolist() for output in outputs]  # Ensure the outputs are serializable
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)

    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate outputs from a single batch of requests.'
    )
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--speculative-model', type=str, default=None)
    parser.add_argument('--num-speculative-tokens', type=int, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization', '-q', choices=['auto', 'fp16', 'int8'], default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n', type=int, default=1, help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--use-v2-block-manager', action='store_true')
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"], help='Device type for vLLM execution')
    parser.add_argument('--output-json', type=str, default=None, help='Path to save the output results in JSON format.')

    args = parser.parse_args()
    outputs = main(args)
    print(outputs)  # This prints the generated outputs
