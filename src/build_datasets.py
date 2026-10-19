#!/usr/bin/env python3
"""
Build privacy-evaluation prompt CSVs from four sources: ShareGPT, Flan,
OASST1, and UltraChat.

Setup:
    pip install -U datasets huggingface_hub requests

Usage:
    python build_datasets.py                       # run everything
    python build_datasets.py --only sharegpt flan   # run a subset
    python build_datasets.py --data-dir ./data/privacy_evaluation
    python build_datasets.py --ultrachat-limit 5000 # change UltraChat cap (default 10000, 0 = no cap)

Each dataset function writes "<Name>.csv" (a single "user_prompt" column)
into --data-dir.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import requests

DATASETS = ["sharegpt", "flan", "oasst1", "ultrachat"]


def write_csv(output_path: Path, values):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_prompt"])
        writer.writerows([[v] for v in values])
    print(f"    Extracted {len(values)} values -> {output_path}")


def build_sharegpt(data_dir: Path):
    print("==> ShareGPT")
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    json_path = data_dir / "ShareGPT_V3_unfiltered_cleaned_split.json"

    if not json_path.exists():
        print(f"    Downloading {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(json_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    else:
        print(f"    Found existing {json_path}, skipping download")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    values = []
    for record in data:
        for conversation in record.get("conversations", []):
            values.append(conversation.get("value", ""))

    write_csv(data_dir / "ShareGPT.csv", values)


def build_flan(data_dir: Path):
    print("==> Flan")
    from datasets import load_dataset

    ds = load_dataset("DataProvenanceInitiative/Commercial-Flan-Collection-Dialog")
    train = ds["train"]
    write_csv(data_dir / "Flan.csv", train["inputs"])


def build_oasst1(data_dir: Path):
    print("==> OASST1")
    from datasets import load_dataset

    ds = load_dataset("OpenAssistant/oasst1")
    train = ds["train"]
    dataset_en = train.filter(lambda x: x["lang"] == "en")
    write_csv(data_dir / "OASST1.csv", dataset_en["text"])


def build_ultrachat(data_dir: Path, limit: int):
    print("==> UltraChat")
    from datasets import load_dataset

    ds = load_dataset("openbmb/UltraChat")
    train = ds["train"]

    rows = train["data"] if limit <= 0 else train["data"][:limit]

    # Each row in "data" is a list of conversation turns; flatten to one
    # prompt per row, mirroring how the ShareGPT conversations are flattened.
    values = []
    for turns in rows:
        if isinstance(turns, (list, tuple)):
            values.extend(turns)
        else:
            values.append(turns)

    write_csv(data_dir / "UltraChat.csv", values)


BUILDERS = {
    "sharegpt": build_sharegpt,
    "flan": build_flan,
    "oasst1": build_oasst1,
    "ultrachat": build_ultrachat,
}


def main():
    parser = argparse.ArgumentParser(description="Build evaluation prompt CSVs.")
    parser.add_argument(
        "--data-dir",
        default="./data/privacy_evaluation",
        help="Directory to download into and write CSVs to.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=DATASETS,
        default=None,
        help="Only build these datasets (default: all).",
    )
    parser.add_argument(
        "--ultrachat-limit",
        type=int,
        default=10000,
        help="Max UltraChat rows to process (0 = no cap). Default 10000.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    targets = args.only or DATASETS

    failures = []
    for name in targets:
        try:
            if name == "ultrachat":
                BUILDERS[name](data_dir, args.ultrachat_limit)
            else:
                BUILDERS[name](data_dir)
        except Exception as e:
            print(f"    FAILED: {name} -> {e}\n")
            failures.append(name)

    if failures:
        print("\nThe following datasets failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll requested datasets built successfully.")


if __name__ == "__main__":
    main()