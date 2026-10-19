#!/usr/bin/env python3
"""
download_models.py

Downloads all the pretrained models used in this paper's experiments from the
Hugging Face Hub, so reviewers can reproduce the code without hunting for
model identifiers themselves.

Usage
-----
    pip install -U huggingface_hub

    # Download everything into ./models (default)
    python download_models.py

    # Download into a custom directory
    python download_models.py --output-dir /path/to/models

    # Download only specific models (see --list for valid keys)
    python download_models.py --only gemma-3-4b-it Llama-2-7b-chat-hf

    # See which models are configured
    python download_models.py --list

Notes
-----
* Several of these models are "gated" on the Hugging Face Hub (the Meta
  Llama family and Google's Gemma family). Reviewers must:
    1. Create a Hugging Face account.
    2. Request/accept access on each gated model's page.
    3. Authenticate locally, e.g. `huggingface-cli login`, or set the
       HF_TOKEN environment variable before running this script.
* Some of these are large (the 70B model in particular is >100GB). Make
  sure there is enough disk space, or use --only to fetch a subset.
* If a model was fine-tuned/renamed locally and does not correspond
  exactly to the public Hub identifier below, double check the MODELS
  dict and adjust the repo_id accordingly (see the comment on llama3_70b).
"""

import argparse
import glob
import os
import shutil
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print(
        "The 'huggingface_hub' package is required.\n"
        "Install it with: pip install -U huggingface_hub",
        file=sys.stderr,
    )
    sys.exit(1)


# Local name (as used in the paper/code) -> Hugging Face Hub repo id
MODELS = {
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "llava-onevision-qwen2-0.5b-ov-hf": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "llava-onevision-qwen2-7b-ov-chat-hf": "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    # NOTE: "llama3_70b" is ambiguous between the base and instruct-tuned
    # checkpoints. Adjust to "meta-llama/Meta-Llama-3-70B" (base) if that is
    # what was actually used in the paper.
    "llama3_70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "Qwen2-VL-2B-Instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen2-VL-7B-Instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
}

# Repos that require the HF license to be accepted + an authenticated token.
# Downloading these without access silently returns only public files
# (e.g. README.md) with no error, so we check for these explicitly.
GATED_MODELS = {
    "gemma-3-4b-it",
    "Llama-3.2-1B-Instruct",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "llama3_70b",
}


WEIGHT_EXTENSIONS = ("*.safetensors", "*.bin", "*.pt", "*.pth")


def _has_weights(target_dir: str) -> bool:
    for pattern in WEIGHT_EXTENSIONS:
        # weights sometimes live in subfolders (e.g. Llama-3.2's "original/")
        if glob.glob(os.path.join(target_dir, "**", pattern), recursive=True):
            return True
    return False


def validate_download(target_dir: str) -> tuple[bool, str]:
    """Returns (ok, reason). A real model dir needs a config + weight files,
    not just a README, which is what a skipped gated download leaves behind."""
    if not os.path.isdir(target_dir):
        return False, "directory does not exist"
    if not os.path.isfile(os.path.join(target_dir, "config.json")):
        return False, "missing config.json"
    if not _has_weights(target_dir):
        return False, "no weight files (*.safetensors/*.bin/*.pt) found"
    return True, "ok"


def download_model(
    local_name: str, repo_id: str, output_dir: str, token: str | None, force: bool
) -> bool:
    target_dir = os.path.join(output_dir, local_name)

    if local_name in GATED_MODELS and not token:
        print(
            f"WARNING: '{local_name}' ({repo_id}) is a gated model but no token was "
            "found (--token / HF_TOKEN). The download may silently skip the model "
            "weights and only fetch public files like README.md.",
            file=sys.stderr,
        )

    if force and os.path.isdir(target_dir):
        print(f"--force: removing existing {target_dir}")
        shutil.rmtree(target_dir)

    print(f"\n=== Downloading {local_name}  (repo: {repo_id}) ===")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            token=token,
            # Skip files we typically don't need for inference/fine-tuning code.
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.tflite"],
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAILED to download {local_name} ({repo_id}): {e}", file=sys.stderr)
        print(
            "  -> If this is a gated model, make sure you have requested access\n"
            "     on the model's Hugging Face page and are authenticated\n"
            "     (huggingface-cli login or HF_TOKEN env var).",
            file=sys.stderr,
        )
        return False

    ok, reason = validate_download(target_dir)
    if not ok:
        print(
            f"FAILED (post-download check) for {local_name}: {reason}.\n"
            f"  -> The call to snapshot_download did not raise an error, but "
            f"{target_dir} does not look like a complete model.\n"
            f"  -> This usually means the repo is gated and access/token was "
            f"missing or not yet approved. Re-run with --force after fixing "
            f"access.",
            file=sys.stderr,
        )
        return False

    print(f"Done: {target_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download models used in this paper.")
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Directory to download models into (default: ./models)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="MODEL_NAME",
        help="Download only these models (by local name). See --list for valid names.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured models and exit.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token. Defaults to HF_TOKEN env var / cached login.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and re-download any existing model directory (use after "
        "fixing gated-model access for a model that previously failed).",
    )
    args = parser.parse_args()

    if args.list:
        print("Configured models (local name -> HF repo id):")
        for name, repo_id in MODELS.items():
            print(f"  {name:40s} -> {repo_id}")
        return

    token = args.token or os.environ.get("HF_TOKEN")

    if args.only:
        unknown = [m for m in args.only if m not in MODELS]
        if unknown:
            print(f"Unknown model name(s): {unknown}", file=sys.stderr)
            print("Run with --list to see valid names.", file=sys.stderr)
            sys.exit(1)
        to_download = {k: MODELS[k] for k in args.only}
    else:
        to_download = MODELS

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for local_name, repo_id in to_download.items():
        results[local_name] = download_model(
            local_name, repo_id, args.output_dir, token, args.force
        )

    print("\n=== Summary ===")
    for local_name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAILED'}] {local_name}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()