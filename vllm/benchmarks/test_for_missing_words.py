import time
import random
import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tokenizer_path = "/srv/muse-lab/models/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def hide_words_from_blocks(tokens, block_size):
    """Generates the whole sentence with missing words from blocks in reverse order.
    Previously hidden words in blocks will remain hidden in subsequent iterations.
    """
    blocks = [tokens[i:i + block_size] for i in range(0, len(tokens), block_size)]  # Divide tokens into blocks
    print(blocks)
    modified_sentences = []
    
    # Start with the full sentence
    full_sentence = tokenizer.convert_tokens_to_string(tokens)
    modified_sentences.append((full_sentence, tokens))  # Append original tokens for comparison

    # Keep track of the hidden tokens for each block
    hidden_indices = [-1] * len(blocks)  # Initially, no word is hidden (use -1 as a placeholder)

    # Hide progressively from each block
    for hide_count in range(1, len(blocks) + 1):
        temp_tokens = tokens.copy()  # Work on a copy of tokens

        # Hide one word from the last `hide_count` blocks
        for j in range(1, hide_count + 1):
            block_idx = -j  # Starting from the last block

            if blocks[block_idx]:
                # Check if a word has already been hidden in this block
                if hidden_indices[block_idx] == -1:
                    # No word hidden yet, so randomly choose one
                    random_index = random.randint(0, len(blocks[block_idx]) - 1)
                    hidden_indices[block_idx] = random_index  # Store the hidden word index
                else:
                    # A word was already hidden in this block, use the same index
                    random_index = hidden_indices[block_idx]

                try:
                    # print(temp_tokens[(block_idx * block_size) + random_index])
                    temp_tokens[(block_idx * block_size) + random_index] = "_"  # Replace the token with "[HIDDEN]"
                except:
                    print("Something went wrong!")

        # Convert the modified tokens back into a sentence
        modified_sentence = tokenizer.convert_tokens_to_string(temp_tokens)
        modified_sentences.append((modified_sentence, temp_tokens))

    return modified_sentences


# Function to count how many blocks are in common with the original
def count_common_blocks(original_tokens, modified_tokens, block_size):
    """Counts how many blocks a modified sentence has in common with the original."""
    original_blocks = [original_tokens[i:i + block_size] for i in range(0, len(original_tokens), block_size)]
    modified_blocks = [modified_tokens[i:i + block_size] for i in range(0, len(modified_tokens), block_size)]
    
    # Count the number of blocks that are identical
    common_blocks_count = sum(1 for orig_block, mod_block in zip(original_blocks, modified_blocks) if orig_block == mod_block)
    return common_blocks_count

# Function to calculate the generation time
def get_generation_time(llm, sampling_params, prompt):
    """Sends the prompt to the model and times the response."""
    start_time = time.time()
    output = llm.generate(prompt, sampling_params=sampling_params)
    end_time = time.time()
    
    generation_time = end_time - start_time
    return generation_time

def main(block_size, input_csv, output_csv):
    # Load the tokenizer
    tokenizer_path = "/srv/muse-lab/models/Llama-2-13b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Initialize the LLM model
    llm = LLM(
        model="/srv/muse-lab/models/Llama-2-13b-chat-hf",
        enable_prefix_caching=True
    )

    # Sampling parameters
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Load the input CSV file and select only the 'prompt' column
    input_data = pd.read_csv(input_csv)
    # input_data = input_data.iloc[0:3, :]
    # Ensure 'prompt' is a column in the DataFrame
    if 'prompt' not in input_data.columns:
        raise ValueError("The input CSV file must contain a 'prompt' column.")

    # Get the 'prompt' column as a pandas Series
    sentences = input_data['prompt']

    # Prepare the output DataFrame
    output_data = []

    # Loop through each sentence in the 'prompt' column
    for idx, sentence in sentences.items():
        # print(f"\nProcessing sentence {idx + 1}: {sentence}")

        # Tokenize the sentence
        tokens = tokenizer.tokenize(sentence)

        # Get the modified sentences (with progressively hidden words)
        modified_sentences = hide_words_from_blocks(tokens, block_size)

        # Process each modified sentence
        original_token_count = len(tokens)
        original_block_count = len(tokens) // block_size + (len(tokens) % block_size > 0)
        generation_time = get_generation_time(llm, sampling_params, sentence)
        print(generation_time)
        for mod_idx, (prompt, modified_tokens) in enumerate(modified_sentences):
            # Get the generation time for each modified sentence
            generation_time = get_generation_time(llm, sampling_params, prompt)

            # Count how many blocks are in common with the original
            common_blocks = count_common_blocks(tokens, modified_tokens, block_size)
            print(f"Generation time: {generation_time}")
            print(f"Common blocks: {common_blocks}")
            print(f"Original blocks count: {original_block_count}")
            # Append the result to the output_data
            output_data.append({
                "sentence_idx": idx + 1,
                "sentence": sentence,
                "modification_idx": mod_idx,
                "original_block_count": original_block_count,
                "common_blocks": common_blocks,
                "generation_time": generation_time
            })

    # Convert the output data to a DataFrame and save it as a CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False)

    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Process sentences and calculate block modifications.")
    parser.add_argument("--block_size", type=int, required=True, help="Block size to divide the tokens.")
    parser.add_argument("--input_csv", type=str, default="input_sentences.csv", help="Input CSV file with sentences.")
    parser.add_argument("--output_csv", type=str, default="output_results.csv", help="Output CSV file for results.")

    # Parse arguments
    args = parser.parse_args()

    # Run the main process
    main(args.block_size, args.input_csv, args.output_csv)
