#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -f "$ROOT/configs/paths.env" ]] && source "$ROOT/configs/paths.env"
OUT="${OUTPUT_ROOT:-$ROOT/outputs/rq2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINED_D="${DECIDER_MODEL:-$ROOT/models/decider}"
TRAINED_S="${SELECTOR_MODEL:-$ROOT/models/selector}"
BASE="${BASE_MODEL:?Set BASE_MODEL}"

declare -A DECIDERS=(
  [trained_decider__trained_selector]="$TRAINED_D"
  [untuned_decider__trained_selector]="$BASE"
  [trained_decider__untuned_selector]="$TRAINED_D"
  [untuned_decider__untuned_selector]="$BASE"
)
declare -A SELECTORS=(
  [trained_decider__trained_selector]="$TRAINED_S"
  [untuned_decider__trained_selector]="$TRAINED_S"
  [trained_decider__untuned_selector]="$BASE"
  [untuned_decider__untuned_selector]="$BASE"
)

for variant in trained_decider__trained_selector untuned_decider__trained_selector trained_decider__untuned_selector untuned_decider__untuned_selector; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" "$PYTHON_BIN" "$ROOT/scripts/infer_benchmark.py" \
    --tasks "${CODEREVAL_TASKS:?Set CODEREVAL_TASKS}" --expected-tasks 230 \
    --stage1-model "${DECIDERS[$variant]}" --stage2-model "${SELECTORS[$variant]}" \
    --output-dir "$OUT/$variant" --tp "${TP_SIZE:-4}" --max-model-len 8192 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.72}"
done

"$PYTHON_BIN" "$ROOT/experiments/rq2/summarize.py" --root "$OUT"

if [[ "${RUN_GENERATORS:-0}" == "1" ]]; then
  for variant in "${!DECIDERS[@]}"; do
    CONTEXTS_FILE="$OUT/$variant/Selected_Contexts.jsonl" OUTPUT_ROOT="$OUT/generation/$variant" \
      bash "${CODEREVAL_GENERATOR_RUNNER:?Set CODEREVAL_GENERATOR_RUNNER}"
  done
fi
