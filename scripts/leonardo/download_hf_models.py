#!/usr/bin/env python3
"""Pre-download HuggingFace models for offline use on LEONARDO compute nodes.

Run on a login node (has internet). Models are cached to $HF_HOME so that
compute nodes (no internet) can load them with TRANSFORMERS_OFFLINE=1.

Usage:
    export HF_HOME=/leonardo_scratch/fast/<account>/project/data/huggingface_cache
    python3 scripts/leonardo/download_hf_models.py
"""

import os
import sys


def main():
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print("ERROR: HF_HOME is not set.")
        print("  export HF_HOME=/path/to/huggingface_cache")
        sys.exit(1)

    print(f"HF_HOME = {hf_home}")
    os.makedirs(hf_home, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoProcessor

    models = [
        "microsoft/Florence-2-base",
        "microsoft/Florence-2-large",
    ]

    for model_name in models:
        print(f"\n--- Downloading {model_name} ---")

        print(f"  Downloading model weights...")
        AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        print(f"  Model cached.")

        print(f"  Downloading processor/tokenizer...")
        AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"  Processor cached.")

    print("\nAll models downloaded and cached.")
    print(f"Cache location: {hf_home}")
    print("Set TRANSFORMERS_OFFLINE=1 on compute nodes to use cached models.")


if __name__ == "__main__":
    main()
