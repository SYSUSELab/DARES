#!/usr/bin/env python3
"""Replace each task's candidate pool with a retriever/reranker Top-k pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict) -> str:
    return str(row.get("task_id", row.get("id")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    tasks = read(args.tasks)
    selections = {key(row): row for row in read(args.selection)}
    if len(tasks) != args.expected:
        raise ValueError(f"Expected {args.expected} tasks, found {len(tasks)}")

    output = []
    for task in tasks:
        selection = selections.get(key(task))
        if selection is None:
            raise KeyError(f"Missing selection for task {key(task)}")
        if isinstance(selection.get("extracted_blocks"), list):
            blocks = [str(block) for block in selection["extracted_blocks"]]
        else:
            original = task.get("All_contexts", [])
            blocks = [original[int(index)] for index in selection.get("selected_blocks", [])]
        row = dict(task)
        row["All_contexts"] = blocks
        row["candidate_pool_source"] = selection.get("selection_strategy", args.selection.stem)
        row["candidate_pool_size"] = len(blocks)
        output.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
