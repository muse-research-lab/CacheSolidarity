"""
BERT PII Masked Word Rank Predictor
------------------------------------
Usage:
    python bert_pii_rank.py <path/to/your_file.csv>

Output:
    A new CSV saved alongside the input file with '_match_ranks' appended to the name.
    E.g. Flan_0.5_processed.csv → Flan_0.5_processed_match_ranks.csv
"""

import sys
import ast
import string
from pathlib import Path

import pandas as pd
import torch
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm

tqdm.pandas()


# ---------------------------------------------------------------------------
# 0. CLI argument
# ---------------------------------------------------------------------------

def parse_args():
    if len(sys.argv) < 2:
        print("Usage: python bert_pii_rank.py <path/to/input.csv>")
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"Error: file not found → {csv_path}")
        sys.exit(1)
    return csv_path

# ---------------------------------------------------------------------------
# 1. Setup BERT + NLTK
# ---------------------------------------------------------------------------

def setup():
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.eval()

    return tokenizer, model, stop_words

# ---------------------------------------------------------------------------
# 2. Predict masked words
# ---------------------------------------------------------------------------

def predict_all_masked_words(text, tokenizer, model, stop_words, top_k=1000):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    mask_token_indices = (
        inputs.input_ids == tokenizer.mask_token_id
    ).nonzero(as_tuple=True)[1]

    if len(mask_token_indices) == 0:
        return {}

    results = {}
    for mask_idx in mask_token_indices:
        mask_logits = logits[0, mask_idx, :]
        top_k_indices = torch.topk(mask_logits, top_k).indices.tolist()
        top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

        valid_tokens = [
            token for token in top_k_tokens
            if token not in stop_words
            and token not in string.punctuation
            and token != "..."
            and token != "[UNK]"
        ]
        results[mask_idx.item()] = valid_tokens[:top_k]

    return results

# ---------------------------------------------------------------------------
# 3. Safe eval + normalisation helpers
# ---------------------------------------------------------------------------

def safe_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def norm(x):
    return str(x).strip().lower()

# ---------------------------------------------------------------------------
# 4. Rank computation
# ---------------------------------------------------------------------------

def get_match_ranks(real_list, pred_dict):
    pred_lists = list(pred_dict.values())
    ranks = []

    for i, (real, preds) in enumerate(zip(real_list, pred_lists), start=1):
        preds_norm = [norm(p) for p in preds]
        real_norm = norm(real)

        if real_norm in preds_norm:
            rank = preds_norm.index(real_norm) + 1  # 1-based
        else:
            rank = -1  # not found

        ranks.append(rank)

    return ranks

# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

def main():
    csv_path = parse_args()

    print(f"\n[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Total rows: {len(df)}")

    # Keep only changed rows and limit to first 10 (matching original notebook)
    df = df[df["changed"] == 1]
    print(f"[INFO] Rows after filter (changed==1, first 10): {len(df)}")
    df["masked_prompt"] = df["masked_prompt"].fillna("").astype(str)

    # --- Setup models ---
    print("[INFO] Loading BERT model …")
    tokenizer, model, stop_words = setup()

    # --- Predict masked words ---
    print("[INFO] Running BERT masked-word predictions …")
    df["Predicted_Words"] = df["masked_prompt"].progress_apply(
        lambda x: predict_all_masked_words(x, tokenizer, model, stop_words, top_k=1000)
    )

    # --- Safe-eval string columns ---
    df["pii_values"] = df["pii_values"].apply(safe_eval)
    df["Predicted_Words"] = df["Predicted_Words"].apply(safe_eval)

    # --- Compute match ranks ---
    print("[INFO] Computing match ranks …")
    df["Match_Rank"] = df.progress_apply(
        lambda row: get_match_ranks(row["pii_values"], row["Predicted_Words"]),
        axis=1,
    )

    # --- Save output ---
    out_path = csv_path.parent / (csv_path.stem + "_match_ranks.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[INFO] Results saved to: {out_path}")


if __name__ == "__main__":
    main()