#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dares.io import normalize_task, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    tasks = [normalize_task(row, index) for index, row in enumerate(rows)]
    blocks = sum(len(task["candidate_blocks"]) for task in tasks)
    print(f"valid tasks={len(tasks)} candidate_blocks={blocks}")


if __name__ == "__main__":
    main()
