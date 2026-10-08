#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -f "$ROOT/configs/paths.env" ]] && source "$ROOT/configs/paths.env"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${OUTPUT_ROOT:-$ROOT/outputs/rq3}"

"$PYTHON_BIN" "$ROOT/experiments/rq3/analyze.py" \
  --codereval-ours "${CODEREVAL_DARES:-$ROOT/outputs/rq1/codereval/Selected_Contexts.jsonl}" \
  --codereval-rlcoder "${CODEREVAL_RLCODER:?Set CODEREVAL_RLCODER}" \
  --codereval-reranker "${CODEREVAL_RERANKER:?Set CODEREVAL_RERANKER}" \
  --repoeval-ours "${REPOEVAL_DARES:-$ROOT/outputs/rq1/repoeval/Selected_Contexts.jsonl}" \
  --repoeval-rlcoder "${REPOEVAL_RLCODER:?Set REPOEVAL_RLCODER}" \
  --repoeval-reranker "${REPOEVAL_RERANKER:?Set REPOEVAL_RERANKER}" \
  --output "$OUT/evidence_quality.json"
