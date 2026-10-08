from __future__ import annotations

import argparse
from pathlib import Path

from .inference import InferenceConfig, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the two-stage DARES evidence selector")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decider-model", required=True)
    parser.add_argument("--selector-model", required=True)
    parser.add_argument("--decider-prompt", type=Path)
    parser.add_argument("--selector-prompt", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--candidate-chars", type=int, default=800)
    parser.add_argument("--decider-max-tokens", type=int, default=384)
    parser.add_argument("--selector-max-tokens", type=int, default=192)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = len(run(InferenceConfig(
        input_path=args.input, output_dir=args.output_dir,
        decider_model=args.decider_model, selector_model=args.selector_model,
        decider_prompt=args.decider_prompt, selector_prompt=args.selector_prompt,
        tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
        candidate_chars=args.candidate_chars, decider_max_tokens=args.decider_max_tokens,
        selector_max_tokens=args.selector_max_tokens,
    )))
    print(f"Wrote {count} selections to {args.output_dir}")


if __name__ == "__main__":
    main()
