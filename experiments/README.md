# Experiment entry points

The four subdirectories correspond to the paper's research questions. All
scripts read local benchmark paths from `configs/paths.env`, use greedy
decoding, and write outputs below `outputs/` unless `OUTPUT_ROOT` is set.

- `rq1/run.sh`: construct CoderEval and RepoEval evidence and optionally invoke
  downstream generation/evaluation runners.
- `rq2/run.sh`: run the trained/untuned Decider and Selector ablations and
  optionally evaluate their downstream generations.
- `rq3/run.sh`: calculate evidence compactness and overlap with RLCoder and the
  relevance reranker.
- `rq4/run.sh`: restrict the initial pool to RLCoder or reranker Top-10,
  reapply DARES, and optionally evaluate downstream generation.

The downstream runner variables are deliberately configurable because the
official CoderEval execution environment is Docker-based, whereas RepoEval
uses its own generation and exact-match evaluator. A runner receives the
selected-context path and output root through environment variables documented
in `configs/paths.env.example`.
