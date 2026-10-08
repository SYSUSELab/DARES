#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -f "$ROOT/configs/paths.env" ]] && source "$ROOT/configs/paths.env"
OUT="${OUTPUT_ROOT:-$ROOT/outputs/rq4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DECIDER="${DECIDER_MODEL:-$ROOT/models/decider}"
SELECTOR="${SELECTOR_MODEL:-$ROOT/models/selector}"
mkdir -p "$OUT/tasks"

run_one() {
  local benchmark="$1" source="$2" tasks="$3" selection="$4" expected="$5"
  local restricted="$OUT/tasks/${benchmark}_${source}.jsonl"
  "$PYTHON_BIN" "$ROOT/scripts/build_retriever_pool.py" \
    --tasks "$tasks" --selection "$selection" --output "$restricted" --expected "$expected"
  extra=()
  [[ "$benchmark" == "repoeval" ]] && extra=(--path-metadata "${REPOEVAL_PATH_METADATA:?Set REPOEVAL_PATH_METADATA}")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" "$PYTHON_BIN" "$ROOT/scripts/infer_benchmark.py" \
    --tasks "$restricted" --expected-tasks "$expected" \
    --stage1-model "$DECIDER" --stage2-model "$SELECTOR" \
    --output-dir "$OUT/$benchmark/$source" --tp "${TP_SIZE:-4}" \
    --max-model-len 8192 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.72}" \
    "${extra[@]}"
}

run_one codereval rlcoder "${CODEREVAL_TASKS:?Set CODEREVAL_TASKS}" "${CODEREVAL_RLCODER:?Set CODEREVAL_RLCODER}" 230
run_one codereval reranker "$CODEREVAL_TASKS" "${CODEREVAL_RERANKER:?Set CODEREVAL_RERANKER}" 230
run_one repoeval rlcoder "${REPOEVAL_TASKS:?Set REPOEVAL_TASKS}" "${REPOEVAL_RLCODER:?Set REPOEVAL_RLCODER}" 1600
run_one repoeval reranker "$REPOEVAL_TASKS" "${REPOEVAL_RERANKER:?Set REPOEVAL_RERANKER}" 1600

if [[ "${RUN_GENERATORS:-0}" == "1" ]]; then
  for source in rlcoder reranker; do
    CONTEXTS_FILE="$OUT/codereval/$source/Selected_Contexts.jsonl" OUTPUT_ROOT="$OUT/generation/codereval/$source" bash "${CODEREVAL_GENERATOR_RUNNER:?Set CODEREVAL_GENERATOR_RUNNER}"
    SELECTION_FILE="$OUT/repoeval/$source/Selected_Contexts.jsonl" OUTPUT_ROOT="$OUT/generation/repoeval/$source" bash "${REPOEVAL_GENERATOR_RUNNER:?Set REPOEVAL_GENERATOR_RUNNER}"
  done
fi
