#!/usr/bin/env python3

import os
import json
import re
import time
import argparse

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =========================================================
# ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser(description="Two-tier PII detector")

parser.add_argument("--input_csv",  type=str, required=True)
parser.add_argument("--output_dir", type=str,
    default="./data/privacy_evaluation/llm_pii_detection_results",)
parser.add_argument("--max_rows",        type=int, default=None)
parser.add_argument("--max_new_tokens",  type=int, default=512,
    help="Max tokens the LLM may generate (default: 512)")
parser.add_argument("--checkpoint_every", type=int, default=500,
    help="Save a checkpoint CSV every N rows (default: 500)")

args = parser.parse_args()

INPUT_FILE        = args.input_csv
OUTPUT_DIR        = args.output_dir
MAX_ROWS          = args.max_rows
MAX_NEW_TOKENS    = args.max_new_tokens
CHECKPOINT_EVERY  = args.checkpoint_every

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Checkpoint path: same name as output but with _checkpoint suffix
BASE_NAME       = os.path.splitext(os.path.basename(INPUT_FILE))[0]
OUTPUT_PATH     = os.path.join(OUTPUT_DIR, BASE_NAME + ".csv")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, BASE_NAME + "_checkpoint.csv")

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "./models/Llama-3.2-1B-Instruct"

PII_RULES = [
    ("email",           r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    ("phone",           r"\+?\d[\d\s\-\(\)]{7,}\d"),
    ("url",             r"https?://[^\s]+"),
    ("ip_address",      r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("ipv6",            r"\b(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}\b"),
    ("ssn",             r"\b\d{3}-\d{2}-\d{4}\b"),
    ("passport",        r"\b[A-Z0-9]{6,9}\b"),
    ("driver_license",  r"\b[A-Z0-9]{5,15}\b"),
    ("tax_id",          r"\b\d{2}-\d{7}\b"),
    ("national_id",     r"\b\d{8,14}\b"),
    ("date_of_birth",   r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b"),
    ("credit_card",     r"\b(?:\d[ -]?){13,16}\b"),
    ("iban",            r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    ("bank_account",    r"\b\d{8,20}\b"),
    ("routing_number",  r"\b\d{9}\b"),
    ("swift_bic",       r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b"),
    ("api_key",         r"\bsk-[a-zA-Z0-9]{10,}\b"),
    ("jwt_token",       r"eyJ[a-zA-Z0-9_\-]+=*\.[a-zA-Z0-9_\-]+=*\.?[a-zA-Z0-9_\-+/=]*"),
    ("aws_access_key",  r"\bAKIA[0-9A-Z]{16}\b"),
    ("github_token",    r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ("private_key",     r"-----BEGIN (RSA|DSA|EC|OPENSSH|PGP)? PRIVATE KEY-----"),
    ("bearer_token",    r"Bearer\s+[A-Za-z0-9\-._~+/]+=*"),
    ("medical_record_number", r"\bMRN[:\s]?\d{5,15}\b"),
    ("insurance_id",    r"\b[A-Z]{2,5}\d{6,12}\b"),
    ("zip_code",        r"\b\d{5}(?:-\d{4})?\b"),
    ("postal_code",     r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b"),
    ("street_address",  r"\b\d{1,5}\s+\w+(\s\w+)*\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b"),
    ("license_plate",   r"\b[A-Z0-9]{2,8}-?[A-Z0-9]{1,8}\b"),
    ("vin",             r"\b[A-HJ-NPR-Z0-9]{17}\b"),
    ("bitcoin_address", r"\b(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b"),
    ("ethereum_address",r"\b0x[a-fA-F0-9]{40}\b"),
    ("telegram_handle", r"@[A-Za-z0-9_]{5,32}"),
    ("discord_tag",     r".+#\d{4}"),
    ("mac_address",     r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
]

# =========================================================
# DEVICE + MODEL
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model.to(device)
model.eval()
print("Model loaded.")

# =========================================================
# REGEX DETECTOR
# =========================================================

def regex_pii_detector(text: str) -> dict:
    entities = []
    for pii_type, pattern in PII_RULES:
        for match in re.findall(pattern, text):
            entities.append({"type": pii_type, "value": match})
    return {"contains_pii": len(entities) > 0, "entities": entities}

# =========================================================
# SAFE LLM RESULT  (guarantees both keys are always present)
# =========================================================

def _safe_llm_result(contains_pii: bool, entities: list) -> dict:
    """Always returns a dict with both 'contains_pii' and 'entities'."""
    return {"contains_pii": contains_pii, "entities": entities}

# =========================================================
# LLM DETECTOR
# =========================================================

def llm_pii_detector(text: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a PII detection API. "
                "Return ONLY valid JSON. "
                "Do not explain anything."
            ),
        },
        {
            "role": "user",
            "content": f"""Detect PII entities in the following text.

Return ONLY JSON:

{{
  "contains_pii": true,
  "entities": [
    {{
      "type": "name|email|phone|address|id|medical|financial|other",
      "value": "text"
    }}
  ]
}}

Text:
{text}
""",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Relaxed parsing: regex for contains_pii, then try JSON for entities
    # ------------------------------------------------------------------
    contains_pii = bool(
        re.search(r'"contains_pii"\s*:\s*true', decoded, re.IGNORECASE)
    )

    # Try strict JSON parse
    try:
        start = decoded.find("{")
        end   = decoded.rfind("}") + 1
        if start != -1 and end > start:
            parsed   = json.loads(decoded[start:end])
            entities = parsed.get("entities", [])
            # Use contains_pii from the parsed JSON if present, fallback to regex
            contains_pii = parsed.get("contains_pii", contains_pii)
            return _safe_llm_result(contains_pii, entities)
    except json.JSONDecodeError:
        pass

    # Try to salvage entities from truncated output via regex
    entity_pattern = r'\{\s*"type"\s*:\s*"([^"]+)"\s*,\s*"value"\s*:\s*"([^"]+)"\s*\}'
    entities = [
        {"type": m[0], "value": m[1]}
        for m in re.findall(entity_pattern, decoded)
    ]
    if entities:
        pass

    return _safe_llm_result(contains_pii, entities)

# =========================================================
# MAIN DETECTOR
# =========================================================

def detect_pii(text: str) -> dict:
    start = time.perf_counter()

    try:
        regex_result = regex_pii_detector(text)
        if regex_result["contains_pii"]:
            return {
                "contains_pii":           True,
                "detection_tier":         "regex",
                "entities":               regex_result["entities"],
                "detection_time_seconds": time.perf_counter() - start,
            }

        llm_result = llm_pii_detector(text)
        return {
            "contains_pii":           llm_result["contains_pii"],
            "detection_tier":         "llm" if llm_result["contains_pii"] else "none",
            "entities":               llm_result["entities"],      # always present now
            "detection_time_seconds": time.perf_counter() - start,
        }

    except Exception as e:
        # Last-resort fallback: log and continue so the whole run doesn't crash
        print(f"[ERROR] detect_pii failed on row (skipping): {e}")
        return {
            "contains_pii":           False,
            "detection_tier":         "error",
            "entities":               [],
            "detection_time_seconds": time.perf_counter() - start,
        }

# =========================================================
# LOAD CSV + RESUME FROM CHECKPOINT
# =========================================================

print("=" * 80)
print(f"Processing: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
if MAX_ROWS is not None:
    df = df.iloc[:MAX_ROWS]

if "user_prompt" not in df.columns:
    raise ValueError("CSV does not contain 'user_prompt' column.")

# Resume: if a checkpoint exists, skip already-processed rows
start_idx   = 0
done_results = []

if os.path.exists(CHECKPOINT_PATH):
    ck = pd.read_csv(CHECKPOINT_PATH)
    start_idx    = len(ck)
    # Re-extract just the result columns from the checkpoint
    result_cols  = ["contains_pii", "detection_tier", "entities", "detection_time_seconds"]
    done_results = ck[result_cols].to_dict("records")
    print(f"[RESUME] Found checkpoint with {start_idx} rows — continuing from row {start_idx}")
else:
    print("[INFO] No checkpoint found — starting from the beginning")

# =========================================================
# MAIN LOOP
# =========================================================

results = list(done_results)

for idx, row in tqdm(
    df.iloc[start_idx:].iterrows(),
    total=len(df) - start_idx,
    desc="Running PII detection",
    initial=start_idx,
):
    results.append(detect_pii(str(row["user_prompt"])))

    # Periodic checkpoint save
    if len(results) % CHECKPOINT_EVERY == 0:
        ck_df = pd.concat(
            [df.iloc[: len(results)].reset_index(drop=True),
             pd.DataFrame(results)],
            axis=1,
        )
        ck_df["entities"] = ck_df["entities"].apply(json.dumps)
        ck_df.to_csv(CHECKPOINT_PATH, index=False)
        print(f"\n[CHECKPOINT] Saved {len(results)} rows → {CHECKPOINT_PATH}")

# =========================================================
# SAVE FINAL OUTPUT + CLEAN UP CHECKPOINT
# =========================================================

final_df             = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
final_df["entities"] = final_df["entities"].apply(json.dumps)
final_df.to_csv(OUTPUT_PATH, index=False)

# Remove checkpoint now that we have the complete file
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print(f"[INFO] Checkpoint removed.")

print("=" * 80)
print(f"Saved to: {OUTPUT_PATH}")
print("DONE.")