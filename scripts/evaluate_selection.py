#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dares.io import read_jsonl
from dares.schema import extract_json_object


def ids(row: dict[str, Any]) -> set[str]:
    for key in ("selected_block_ids", "gold_selected_block_ids"):
        if isinstance(row.get(key), list):
            return set(row[key])
    messages = row.get("messages", [])
    if messages:
        return set(extract_json_object(messages[-1]["content"]).get("selected_block_ids", []))
    return set()


def key(row: dict[str, Any]) -> str:
    return str(row.get("task_id", row.get("id", ""))).split("::", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    predictions = {key(row): ids(row) for row in read_jsonl(args.predictions)}
    references = {key(row): ids(row) for row in read_jsonl(args.references)}
    tp = fp = fn = exact = 0
    macro_f1 = []
    for task_id, gold in references.items():
        predicted = predictions.get(task_id, set())
        tp_i, fp_i, fn_i = len(predicted & gold), len(predicted - gold), len(gold - predicted)
        tp += tp_i; fp += fp_i; fn += fn_i
        exact += predicted == gold
        macro_f1.append(2 * tp_i / (2 * tp_i + fp_i + fn_i) if predicted or gold else 1.0)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    result = {"tasks": len(references), "exact_match": exact / len(references),
              "micro_precision": precision, "micro_recall": recall,
              "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
              "macro_f1": sum(macro_f1) / len(macro_f1)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
