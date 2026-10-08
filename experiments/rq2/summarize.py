#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("*/Selected_Contexts.jsonl")):
        data = read(path)
        counts = [len(row.get("selected_block_ids", [])) for row in data]
        rows.append({"variant": path.parent.name, "tasks": len(data),
                     "nonempty": sum(bool(value) for value in counts),
                     "average_blocks": sum(counts) / len(counts)})
    output = args.root / "selection_summary.json"
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(output.read_text())


if __name__ == "__main__":
    main()
