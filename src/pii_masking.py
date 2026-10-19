#!/usr/bin/env python3

import os
import sys
import pandas as pd
from tqdm import tqdm
from ai4privacy import protect
from transformers import AutoTokenizer
import re
import csv

# Enable tqdm progress bars for pandas
tqdm.pandas()


def main(input_file, output_file, tokenizer_path, score_threshold=0.5):
    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------
    print(f"Loading dataset: {input_file}")

    df = pd.read_csv(input_file,  sep='\t')
    if "user_prompt" not in df.columns:
        raise ValueError(
            f"Input CSV must contain a 'user_prompt' column. "
            f"Found columns: {df.columns.tolist()}"
        )

    print(f"Rows loaded: {len(df)}")

    # ---------------------------------------------------------
    # 2. Run AI4Privacy protection
    # ---------------------------------------------------------
    print("Running PII detection and masking...")

    df["protect_result"] = df["user_prompt"].progress_apply(
        lambda x: protect(
            str(x),
            verbose=True,
            score_threshold=score_threshold
        )
    )

    # ---------------------------------------------------------
    # 3. Extract masked text
    # ---------------------------------------------------------
    df["masked_prompt"] = df["protect_result"].progress_apply(
        lambda x: x["masked_text"]
    )

    # ---------------------------------------------------------
    # 4. Extract original PII values
    # ---------------------------------------------------------
    df["pii_values"] = df["protect_result"].progress_apply(
        lambda x: [
            item["value"]
            for item in x.get("replacements", [])
        ]
    )

    # ---------------------------------------------------------
    # 5. Determine whether prompt changed
    # ---------------------------------------------------------
    df["changed"] = (
        df["user_prompt"] != df["masked_prompt"]
    ).astype(int)

    # ---------------------------------------------------------
    # 6. Load tokenizer
    # ---------------------------------------------------------
    print(f"Loading tokenizer: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True
    )

    # ---------------------------------------------------------
    # 7. Tokenize masked prompts
    # ---------------------------------------------------------
    print("Tokenizing masked prompts...")

    df["tokens"] = df["masked_prompt"].progress_apply(
        lambda x: tokenizer.tokenize(str(x))
    )

    # ---------------------------------------------------------
    # 8. Count tokens
    # ---------------------------------------------------------
    df["token_count"] = df["tokens"].progress_apply(len)

    # ---------------------------------------------------------
    # 9. Find MASK token positions
    # ---------------------------------------------------------
    df["PI_position"] = df["tokens"].progress_apply(find_pi_positions)

    # ---------------------------------------------------------
    # 10. Remove temporary column
    # ---------------------------------------------------------
    df.drop(
        columns=["protect_result"],
        inplace=True
    )

    # ---------------------------------------------------------
    # 11. Save output
    # ---------------------------------------------------------
    print(f"Saving output: {output_file}")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(
        output_file,
        index=False
    )

    print("\nProcessing complete!")
    print(f"Final CSV saved to: {output_file}")
    print(f"Rows processed: {len(df)}")
    print(f"Rows changed: {df['changed'].sum()}")


def find_pi_positions(tokens):
    positions = []

    for i in range(len(tokens) - 4):
        # Look for: ▁[, PI, I, _, NUMBER
        if (
            tokens[i] == "▁["
            and tokens[i + 1] == "MA"
            and tokens[i + 2] == "SK"
            and tokens[i + 3] == "]"
        ):

            positions.append(i + 3)

    return positions


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Command-line arguments
    #
    # Usage:
    #
    # python script.py INPUT.csv OUTPUT.csv TOKENIZER_PATH [THRESHOLD]
    #
    # ---------------------------------------------------------

    if len(sys.argv) not in (4, 5):
        print(
            "\nUsage:\n"
            "  python script.py <input_csv> <output_csv> "
            "<tokenizer_path> [score_threshold]\n"
        )

        print("Example:")
        print(
            "  python script.py "
            "input.csv "
            "output.csv "
            "models/Llama-2-13b-chat-hf"
        )

        print("\nExample with threshold:")
        print(
            "  python script.py "
            "input.csv "
            "output.csv "
            "models/Llama-2-13b-chat-hf "
            "0.7"
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # Read command-line arguments
    # ---------------------------------------------------------

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    tokenizer_path = sys.argv[3]

    # Default threshold = 0.5
    score_threshold = (
        float(sys.argv[4])
        if len(sys.argv) == 5
        else 0.5
    )

    # ---------------------------------------------------------
    # Run processing
    # ---------------------------------------------------------

    print("\n========================================")
    print("PII Processing")
    print("========================================")
    print(f"Input CSV:      {input_file}")
    print(f"Output CSV:     {output_file}")
    print(f"Tokenizer:      {tokenizer_path}")
    print(f"Threshold:      {score_threshold}")
    print("========================================\n")

    main(
        input_file=input_file,
        output_file=output_file,
        tokenizer_path=tokenizer_path,
        score_threshold=score_threshold
    )