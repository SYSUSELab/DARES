#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to Qwen3-4B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/models}"

for stage in decider selector; do
  "$PYTHON_BIN" "$ROOT/scripts/merge_adapter.py" \
    --base-model "$BASE_MODEL" \
    --adapter "$ROOT/weights/$stage" \
    --output-dir "$OUTPUT_ROOT/$stage"
done

echo "Decider: $OUTPUT_ROOT/decider"
echo "Selector: $OUTPUT_ROOT/selector"
