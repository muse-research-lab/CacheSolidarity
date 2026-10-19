import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/srv/muse-lab/models/Llama-2-13b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("/srv/muse-lab/models/Llama-2-13b-chat-hf")

# Load the filtered CSV file
input_csv = "/home/panagiotis.pennas/vllm/benchmarks/filtered_output.csv"  # Replace with your filtered CSV file
output_csv = "/home/panagiotis.pennas/vllm/benchmarks/prediction_attempts_with_ranks.csv"  # Output file to record the results

# Read the CSV file into a DataFrame
data = pd.read_csv(input_csv)

# Limit to the first 1000 rows
data = data.head(1000)
# data = data.iloc[[10, 11, 12]]
# Function to predict tokens and check conditions
def predict_hidden_tokens(sentence, hidden_tokens):
    # Prepare the input by hiding the last two tokens
    tokens = tokenizer.tokenize(sentence)
    input_sentence = "".join(tokens[:-2])  # Exclude the last two tokens
    inputs = tokenizer(input_sentence, return_tensors="pt")

    # Get logits (probabilities) for the first predicted token
    outputs = model(**inputs)
    logits_first = outputs.logits[:, -1, :]
    probabilities_first = torch.nn.functional.softmax(logits_first, dim=-1)

    # Get the top 100 most probable first tokens
    top_k = 100
    top_k_probs_first, top_k_indices_first = torch.topk(probabilities_first, top_k)
    top_k_tokens_first = [tokenizer.decode([idx], skip_special_tokens=True).strip() for idx in top_k_indices_first[0]]

    # Check if the first hidden token is in the top 100
    first_hidden_token = hidden_tokens[0]
    first_without_first_character = first_hidden_token[1:] 
    first_token_rank = -1  # Default rank if not found
    for rank, token in enumerate(top_k_tokens_first):
        if (token == first_hidden_token) or (token == first_without_first_character):
            first_token_rank = rank + 1  # 1-based index
            break

    if first_token_rank == -1:
        print(f"First hidden token '{first_hidden_token}' not in top 100 most probable tokens. Skipping.")
        return -1, -1

    # Use the first hidden token to predict the second token
    extended_inputs = tokenizer(input_sentence + first_hidden_token, return_tensors="pt")
    extended_outputs = model(**extended_inputs)
    logits_second = extended_outputs.logits[:, -1, :]
    probabilities_second = torch.nn.functional.softmax(logits_second, dim=-1)

    # Get the top 100 most probable second tokens
    top_k_probs_second, top_k_indices_second = torch.topk(probabilities_second, top_k)
    top_k_tokens_second = [tokenizer.decode([idx], skip_special_tokens=True).strip() for idx in top_k_indices_second[0]]

    # Check if the second hidden token is in the top 100
    second_hidden_token = hidden_tokens[1]
    second_token_rank = -1  # Default rank if not found
    for rank, token in enumerate(top_k_tokens_second):
        if (token == second_hidden_token) or (token == second_hidden_token[1:]):
            second_token_rank = rank + 1  # 1-based index
            print("Match")
            break

    return first_token_rank, second_token_rank

# Open a CSV file to save the results
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Value", "hidden_token_1", "hidden_token_2", "hidden_token_1_rank", "hidden_token_2_rank"])

    # Process each sentence
    for index, row in data.iterrows():
        print(f"Processing index {index}")
        sentence = row['Value']
        tokens = tokenizer.tokenize(sentence)

        # Ensure there are at least two tokens to hide
        if len(tokens) < 2:
            print(f"Skipping sentence with insufficient tokens: {sentence}")
            continue

        # Hide the last two tokens
        hidden_tokens = tokens[-2:]
        print(f"Hidden tokens to find: {hidden_tokens}")

        # Predict tokens and handle conditions
        first_token_rank, second_token_rank = predict_hidden_tokens(sentence, hidden_tokens)

        if first_token_rank is None:
            continue  # Skip if the first hidden token is not in the top 100

        # Write the result to the CSV
        writer.writerow([sentence, hidden_tokens[0], hidden_tokens[1], first_token_rank, second_token_rank])

print(f"Prediction attempts recorded in {output_csv}")
