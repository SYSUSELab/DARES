#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -f "$ROOT/configs/paths.env" ]] && source "$ROOT/configs/paths.env"
OUT="${OUTPUT_ROOT:-$ROOT/outputs/rq1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
COMMON=(--stage1-model "${DECIDER_MODEL:-$ROOT/models/decider}" --stage2-model "${SELECTOR_MODEL:-$ROOT/models/selector}" --tp "${TP_SIZE:-4}" --max-model-len 8192 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.72}")

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" "$PYTHON_BIN" "$ROOT/scripts/infer_benchmark.py" \
  --tasks "${CODEREVAL_TASKS:?Set CODEREVAL_TASKS}" --expected-tasks 230 \
  --output-dir "$OUT/codereval" "${COMMON[@]}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" "$PYTHON_BIN" "$ROOT/scripts/infer_benchmark.py" \
  --tasks "${REPOEVAL_TASKS:?Set REPOEVAL_TASKS}" --expected-tasks 1600 \
  --path-metadata "${REPOEVAL_PATH_METADATA:?Set REPOEVAL_PATH_METADATA}" \
  --output-dir "$OUT/repoeval" "${COMMON[@]}"

if [[ "${RUN_GENERATORS:-0}" == "1" ]]; then
  CONTEXTS_FILE="$OUT/codereval/Selected_Contexts.jsonl" OUTPUT_ROOT="$OUT/codereval_generation" bash "${CODEREVAL_GENERATOR_RUNNER:?Set CODEREVAL_GENERATOR_RUNNER}"
  SELECTION_FILE="$OUT/repoeval/Selected_Contexts.jsonl" OUTPUT_ROOT="$OUT/repoeval_generation" bash "${REPOEVAL_GENERATOR_RUNNER:?Set REPOEVAL_GENERATOR_RUNNER}"
fi
