import time
import csv
from vllm import LLM, SamplingParams
import os
import sys


csv.field_size_limit(sys.maxsize)

def process_csv(input_csv, output_csv, llm, sampling_params):
    """
    Reads prompts from a CSV file if it exists; otherwise, creates one with default prompts.
    Processes each prompt twice, and writes the results and generation times to a new CSV file.
    """
    # Check if the input file exists
    if not os.path.exists(input_csv):
        print(f"Input file {input_csv} not found. Creating a default dataset.")
        # Create a default dataset if input file is missing
        default_data = [{"Value": "Question: what is the age of John Doe? Your answer: The age of John Doe is "}]
        with open(input_csv, mode='w', newline='') as infile:
            writer = csv.DictWriter(infile, fieldnames=["Value"])
            writer.writeheader()
            writer.writerows(default_data)
    
    # Calculate the total number of rows in the input CSV
    with open(input_csv, mode='r') as infile:
        total_prompts = sum(1 for _ in csv.DictReader(infile))

    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["Execution Time 1", "Execution Time 2"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process rows
        for i, row in enumerate(reader):
            prompt = row['Value']  # Read the prompt from the "Values" column
            print(f"Processing prompt {i + 1}/{total_prompts}")

            # Run the prompt twice and record execution times
            _, time1 = get_generation_time(llm, sampling_params, prompt)
            _, time2 = get_generation_time(llm, sampling_params, prompt)

            # Add the execution times to the row
            row["Execution Time 1"] = time1
            row["Execution Time 2"] = time2

            # Write the updated row to the output CSV
            writer.writerow(row)

def get_generation_time(llm, sampling_params, prompt):
    """
    Run the LLM generation and return the output text and generation time.
    """
    start_time = time.time()
    output = llm.generate(prompt, sampling_params=sampling_params)[0]
    end_time = time.time()
    generation_time = end_time - start_time
    return output.outputs[0].text, generation_time

# Set up the LLM and sampling parameters
llm = LLM(
    model='/srv/muse-lab/models/Llama-2-13b-chat-hf',
    enable_prefix_caching=True
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

# Process the CSV file
input_csv = '/home/panagiotis.pennas/vllm/filtered_values_8_tokens.csv' # Replace with your input CSV file
output_csv = '/home/panagiotis.pennas/output_16.csv'  # Replace with your desired output CSV file
process_csv(input_csv, output_csv, llm, sampling_params)
