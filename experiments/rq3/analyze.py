#!/usr/bin/env python3
"""Measure evidence compactness and overlap with retrieval baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict) -> str:
    return str(row.get("task_id", row.get("id")))


def selected(row: dict) -> set[int]:
    if isinstance(row.get("selected_blocks"), list):
        return {int(value) for value in row["selected_blocks"]}
    if isinstance(row.get("selected_indices"), list):
        return {int(value) for value in row["selected_indices"]}
    return {int(value[1:]) for value in row.get("selected_block_ids", []) if str(value).startswith("B")}


def compare(benchmark: str, ours_path: Path, baseline_name: str, baseline_path: Path) -> dict:
    ours = {key(row): selected(row) for row in read(ours_path)}
    baseline = {key(row): selected(row) for row in read(baseline_path)}
    if ours.keys() != baseline.keys():
        raise ValueError(f"Task IDs are not aligned for {benchmark}/{baseline_name}")
    ours_total = baseline_total = shared = 0
    for task_id, ours_ids in ours.items():
        baseline_ids = baseline[task_id]
        ours_total += len(ours_ids)
        baseline_total += len(baseline_ids)
        shared += len(ours_ids & baseline_ids)
    tasks = len(ours)
    return {
        "benchmark": benchmark,
        "baseline": baseline_name,
        "tasks": tasks,
        "dares_blocks_per_task": ours_total / tasks,
        "baseline_blocks_per_task": baseline_total / tasks,
        "shared_dares_evidence": shared / ours_total if ours_total else 1.0,
        "baseline_context_removed": 1 - shared / baseline_total if baseline_total else 0.0,
        "block_reduction": 1 - ours_total / baseline_total if baseline_total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codereval-ours", type=Path, required=True)
    parser.add_argument("--codereval-rlcoder", type=Path, required=True)
    parser.add_argument("--codereval-reranker", type=Path, required=True)
    parser.add_argument("--repoeval-ours", type=Path, required=True)
    parser.add_argument("--repoeval-rlcoder", type=Path, required=True)
    parser.add_argument("--repoeval-reranker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for benchmark in ("codereval", "repoeval"):
        ours = getattr(args, f"{benchmark}_ours")
        rows.append(compare(benchmark, ours, "rlcoder", getattr(args, f"{benchmark}_rlcoder")))
        rows.append(compare(benchmark, ours, "reranker", getattr(args, f"{benchmark}_reranker")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
