import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/srv/muse-lab/models/Llama-2-13b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("/srv/muse-lab/models/Llama-2-13b-chat-hf")


# Load the filtered CSV file
input_csv = "/home/panagiotis.pennas/vllm/benchmarks/filtered_output.csv"  # Replace with your filtered CSV file
output_csv = "/home/panagiotis.pennas/vllm/benchmarks/prediction_attempts.csv"  # Output file to record the results

# Read the CSV file into a DataFrame
data = pd.read_csv(input_csv)

# Limit to the first 100 rows
data = data.head(1000)
# data = data.iloc[[10, 11, 12]]

# Function to predict the hidden token
def predict_hidden_token(sentence, hidden_token, max_attempts=1):
    for attempt in range(1, max_attempts + 1):
        print(f"Attempt:{attempt}")
        # Prepare the input by hiding the last token
        tokens = tokenizer.tokenize(sentence)
        input_sentence = "".join(tokens[:-1])  # Exclude the last token
        inputs = tokenizer(input_sentence, return_tensors="pt")

        # Generate predictions
        outputs = model.generate(**inputs, max_new_tokens=1)
        predicted_token_id = outputs[0][-1]
        predicted_token = tokenizer.decode([predicted_token_id], skip_special_tokens=True).strip()
        print(f"Predicted token:{predicted_token}")
        # Check if the prediction matches the hidden token
        if (predicted_token == hidden_token) or (predicted_token == hidden_token[1:]):
            print("Predicted token matches hidden!")
            return predicted_token, attempt
        else:
            return predicted_token, 0
    return max_attempts  # Return max attempts if not found

# Open a CSV file to save the results
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Value", "hidden_token", "predicted_token","attempts"])

    # Process each sentence
    for index, row in data.iterrows():
        print(index)
        sentence = row['Value']
        print(sentence)
        tokens = tokenizer.tokenize(sentence)

        # Hide the last token
        hidden_token = tokens[-1]
        print(f"Token to be found:{hidden_token}")
        sentence_without_last = "".join(tokens[:-1])
        print(f"Given sentence:{sentence_without_last}")

        # Predict the hidden token
        predicted_token, attempts = predict_hidden_token(sentence, hidden_token)

        # Write the result to the CSV
        writer.writerow([sentence, hidden_token, predicted_token ,attempts])

print(f"Prediction attempts recorded in {output_csv}")
