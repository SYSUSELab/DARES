#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dares.io import read_jsonl
from dares.schema import extract_json_object, validate_decision, validate_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--stage", choices=("decider", "selector"), required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    for index, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"row {index}: final message must be assistant")
        value = extract_json_object(messages[-1]["content"])
        if args.stage == "decider":
            validate_decision(value)
        else:
            candidate_ids = set()
            for message in messages:
                if message.get("role") == "user":
                    import re
                    candidate_ids.update(re.findall(r'<candidate_block id="(B\d+)">', message["content"]))
            selected = value.get("selected_block_ids")
            if not isinstance(selected, list) or any(item not in candidate_ids for item in selected):
                raise ValueError(f"row {index}: invalid selected candidate ID")
    print(f"valid stage={args.stage} rows={len(rows)}")


if __name__ == "__main__":
    main()
