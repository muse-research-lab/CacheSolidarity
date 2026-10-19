import time
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, pipeline
import torch

# Load your model
model = '/srv/muse-lab/models/Llama-2-13b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

# Initialize the model pipeline
llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def get_top_k_predictions(prompt: str) -> pd.DataFrame:
    """
    Generate a dataset with the LONG_PROMPT as the first column and each generated sequence as a separate column.
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=10,
        eos_token_id=tokenizer.eos_token_id,
        max_length=400,
    )
    
    data = {"LONG_PROMPT": [prompt]}
    for i, seq in enumerate(sequences):
        generated_text = seq['generated_text']
        if "The correct sentence is " in generated_text:
            corrected_sentence = generated_text.split("The correct sentence is ", 1)[1]
            data[f"Generated_Sequence_{i+1}"] = [corrected_sentence]
        else:
            data[f"Generated_Sequence_{i+1}"] = ["(No valid output)"]
        
        # Print the generated text before storing it
        print(f"Generated Sequence {i+1}: {generated_text}")
    
    df = pd.DataFrame(data)
    return df

# Load the CSV file containing the missing word texts
csv_input_file = '/home/panagiotis.pennas/dataset.csv'  # Replace with your actual file name
df_input = pd.read_csv(csv_input_file)

# Initialize the list to collect the results
all_generated_results = []

# Loop through each row and generate predictions for each Missing_word_text
for index, row in df_input.iterrows():
    LONG_PROMPT = row['Missing_word_text']  # Use the Missing_word_text from the CSV
    prompt = LONG_PROMPT + "Question: Fill the gap of the given text and give me back the completed text. Your answer: The completed text is "
    
    # Get top k predictions for the current row's Missing_word_text
    dataset = get_top_k_predictions(prompt)
    
    # Append the results for this particular row to the list
    all_generated_results.append(dataset)

# Concatenate all the generated results into one DataFrame
final_dataset = pd.concat(all_generated_results, ignore_index=True)

# Save the final dataset to CSV
csv_filename = "generated_sentences.csv"
final_dataset.to_csv(csv_filename, index=False)
print(f"Dataset saved as {csv_filename}")
