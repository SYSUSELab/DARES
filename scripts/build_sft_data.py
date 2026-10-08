#!/usr/bin/env python3
"""Materialize Decider and Selector chat data from task-level annotations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dares.io import candidate_text, normalize_task, read_jsonl, write_jsonl
from dares.prompts import load_template, render_decider, render_selector
from dares.schema import validate_decision, validate_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--decider-output", type=Path, required=True)
    parser.add_argument("--selector-output", type=Path, required=True)
    parser.add_argument("--candidate-chars", type=int, default=800)
    parser.add_argument("--decider-prompt", type=Path)
    parser.add_argument("--selector-prompt", type=Path)
    args = parser.parse_args()
    decider_template = load_template(args.decider_prompt, "decider.txt")
    selector_template = load_template(args.selector_prompt, "selector.txt")
    decider_rows, selector_rows = [], []
    for index, source in enumerate(read_jsonl(args.input)):
        task = normalize_task(source, index)
        decision = validate_decision(source["decision"])
        selected = validate_selection(
            {"selected_block_ids": source["selected_block_ids"]}, len(task["candidate_blocks"])
        )
        decision_text = json.dumps(decision, ensure_ascii=False, separators=(",", ":"))
        selection_text = json.dumps({"selected_block_ids": selected}, ensure_ascii=False,
                                    separators=(",", ":"))
        common = {key: source[key] for key in ("split",) if key in source}
        decider_rows.append({
            "id": task["id"], **common,
            "messages": [
                {"role": "user", "content": render_decider(decider_template, task)},
                {"role": "assistant", "content": decision_text},
            ],
        })
        texts = [candidate_text(item)[:args.candidate_chars] for item in task["candidate_blocks"]]
        selector_rows.append({
            "id": task["id"], **common, "gold_selected_block_ids": selected,
            "messages": [
                {"role": "user", "content": render_selector(
                    selector_template, task, decision_text, texts
                )},
                {"role": "assistant", "content": selection_text},
            ],
        })
    write_jsonl(args.decider_output, decider_rows)
    write_jsonl(args.selector_output, selector_rows)
    print(f"wrote decider={len(decider_rows)} selector={len(selector_rows)}")


if __name__ == "__main__":
    main()
