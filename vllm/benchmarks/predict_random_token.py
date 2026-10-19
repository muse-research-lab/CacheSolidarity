import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv
import random

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/srv/muse-lab/models/Llama-2-13b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("/srv/muse-lab/models/Llama-2-13b-chat-hf")

# Load the filtered CSV file
input_csv = "/home/panagiotis.pennas/vllm/benchmarks/filtered_output.csv"  # Replace with your filtered CSV file
output_csv = "/home/panagiotis.pennas/vllm/benchmarks/prediction_attempts.csv"  # Output file to record the results

# Read the CSV file into a DataFrame
data = pd.read_csv(input_csv)

# Limit to the first 1000 rows
# data = data.head(1000)
data = data.iloc[[10, 11, 12]]
# Function to predict the hidden token
def predict_hidden_token(sentence, hidden_token, token_index, max_attempts=1):
    for attempt in range(1, max_attempts + 1):
        print(f"Attempt:{attempt}")
        
        # Prepare the input by hiding the selected token
        tokens = tokenizer.tokenize(sentence)
        tokens[token_index] = "<mask>"  # Replace the hidden token with a placeholder
        input_sentence = "Replace '<mask>' with a proper word" + "".join(tokens)
        detok = tokenizer.convert_tokens_to_string(tokens)
        question = "Replace '<mask>' with the most probable missing token: " + detok
        print(f"Input sentence: {detok}")
        inputs = tokenizer(question, return_tensors="pt")

        # Generate predictions
        outputs = model.generate(**inputs, max_new_tokens=1)
        predicted_token_id = outputs[0][-1]
        predicted_token = tokenizer.decode([predicted_token_id], skip_special_tokens=True).strip()
        print(f"Predicted token:{predicted_token}")

        # Check if the prediction matches the hidden token
        if predicted_token == hidden_token:
            print("Predicted token matches hidden!")
            return predicted_token, attempt
    
    return "", max_attempts  # Return empty prediction and max attempts if not found

# Open a CSV file to save the results
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Value", "Hidden Token", "Predicted Token", "Token Index", "Attempts"])

    # Process each sentence
    for index, row in data.iterrows():
        print(index)
        sentence = row['Value']
        # question = "Replace '<mask>' in this sentence with the most probable token: " + sentence
        print(sentence)
        tokens = tokenizer.tokenize(sentence)

        # Randomly select a token to hide
        token_index = random.randint(0, len(tokens) - 1)
        hidden_token = tokens[token_index]
        print(f"Token to be found:{hidden_token}")

        # Predict the hidden token
        predicted_token, attempts = predict_hidden_token(sentence, hidden_token, token_index)

        # Write the result to the CSV
        writer.writerow([sentence, hidden_token, predicted_token, token_index, attempts])

print(f"Prediction attempts recorded in {output_csv}")
