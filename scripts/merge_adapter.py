#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                trust_remote_code=True, device_map="cpu")
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(args.output_dir)
    print(f"Saved merged model to {args.output_dir}")


if __name__ == "__main__":
    main()
