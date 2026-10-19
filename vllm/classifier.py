import time
import csv
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
import torch

block_size = 32
num_tokens = 50
# Step 7: Run the test for 100 iterations and save results to CSV
csv_file_path = "/home/panagiotis.pennas/vllm/data_classifier_32_50.csv"


# Initialize the tokenizer
tokenizer_path = "/srv/muse-lab/models/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def generate_sentence_of_tokens(tokenizer, num_tokens):
    sentence = "This is a generated sentence with repeating words to ensure that we reach the required number of tokens. "
    tokens = tokenizer.tokenize(sentence)
    full_sentence = sentence

    # Keep appending the sentence until we reach the required number of tokens
    while len(tokens) < num_tokens:
        full_sentence += sentence
        tokens = tokenizer.tokenize(full_sentence)

    
    truncated_tokens = tokens[:num_tokens]
    full_sentence = tokenizer.convert_tokens_to_string(truncated_tokens)
    return full_sentence

# Step 2: Split sentence into blocks
def split_sentence_into_blocks(tokenizer, sentence, block_size):
    tokens = tokenizer.tokenize(sentence)
    blocks = [tokens[i:i + block_size] for i in range(0, len(tokens), block_size)]
    return blocks

# Step 3: Replace one token in each block
def replace_token_in_each_block(blocks, replace_index=0, new_token="_"):
    for block in blocks:
        if len(block) > replace_index:
            block[replace_index] = new_token  # Replace token at the given index in each block
    return blocks

# Step 4: Replace tokens in all blocks except the first one
def replace_token_in_all_but_first_block(blocks, replace_index=0, new_token="_"):
    for i in range(1, len(blocks)):  # Start from block 1, skip block 0
        if len(blocks[i]) > replace_index:
            blocks[i][replace_index] = new_token  # Replace token at the given index in other blocks
    return blocks

# Step 5: Convert token blocks back to sentence
def convert_blocks_to_sentence(tokenizer, blocks):
    all_tokens = [token for block in blocks for token in block]
    return tokenizer.convert_tokens_to_string(all_tokens)

def print_token_and_block_info(tokens, blocks):
    print(f"Number of tokens: {len(tokens)}")
    print(f"Number of blocks: {len(blocks)}")


original_sentence = generate_sentence_of_tokens(tokenizer, num_tokens)

# Define block size and split the sentence into blocks
blocks = split_sentence_into_blocks(tokenizer, original_sentence, block_size)

# Replace the first token in each block with '[HIDDEN]'
modified_blocks = replace_token_in_each_block(blocks, replace_index=0, new_token='_')
modified_sentence = convert_blocks_to_sentence(tokenizer, modified_blocks)

tokens = tokenizer.tokenize(original_sentence)
print("Original sentence token and block info:")
print_token_and_block_info(tokens, blocks)

modified_tokens = tokenizer.tokenize(modified_sentence)
print("\nModified sentence token and block info:")
print_token_and_block_info(modified_tokens, modified_blocks)

# Generate LONG_PROMPT_3 where only the blocks after the first have hidden tokens
blocks_for_long_prompt_3 = split_sentence_into_blocks(tokenizer, original_sentence, block_size)
modified_blocks_3 = replace_token_in_all_but_first_block(blocks_for_long_prompt_3, replace_index=0, new_token='_')
LONG_PROMPT_3 = convert_blocks_to_sentence(tokenizer, modified_blocks_3)

tokens_long_prompt_3 = tokenizer.tokenize(LONG_PROMPT_3)
print("\nLONG_PROMPT_3 token and block info:")
print_token_and_block_info(tokens_long_prompt_3, blocks_for_long_prompt_3)

# Use the three sentences as LONG_PROMPT_1, LONG_PROMPT_2, and LONG_PROMPT_3
LONG_PROMPT_1 = original_sentence
LONG_PROMPT_2 = modified_sentence

# Step 6: Use vLLM to send the prompt to the LLM and time the generation
def get_generation_time(llm, sampling_params, prompt):
    start_time = time.time()
    output = llm.generate(prompt, sampling_params=sampling_params)
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation time: {end_time - start_time} seconds.")
    return generation_time, output[0].outputs[0].text

# Sampling parameters (temperature 0, max_tokens 1)
sampling_params = SamplingParams(temperature=0, max_tokens=1)


# Open the CSV file and write the header
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Prompt Source", "Generation Time (s)"])

    # Perform 100 iterations
    for i in range(1, 101):
        print(f"Iteration {i}: Running all three prompts and cleaning cache...")

        # Initialize the LLM instance at the start of each iteration
        llm = LLM(
            model="/srv/muse-lab/models/Llama-2-13b-chat-hf",
            enable_prefix_caching=True
        )

        # Run for LONG_PROMPT_1
        generation_time_1, output_text_1 = get_generation_time(llm, sampling_params, LONG_PROMPT_1)
        writer.writerow([i, "LONG_PROMPT_1", generation_time_1])

        # Run for LONG_PROMPT_2
        generation_time_2, output_text_2 = get_generation_time(llm, sampling_params, LONG_PROMPT_2)
        writer.writerow([i, "LONG_PROMPT_2", generation_time_2])

        # Run for LONG_PROMPT_3
        generation_time_3, output_text_3 = get_generation_time(llm, sampling_params, LONG_PROMPT_3)
        writer.writerow([i, "LONG_PROMPT_3", generation_time_3])

        # Clean up: Delete LLM instance and clear GPU cache
        del llm
        gc.collect()  # Force garbage collection to free memory
        torch.cuda.empty_cache()  # Free up GPU memory

print(f"CSV file saved at: {csv_file_path}")
