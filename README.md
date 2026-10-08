# DARES

This repository contains the implementation and review artifact for **DARES: Decision-Guided Adaptive Repository Evidence Selection**.

DARES is a retriever-independent context-selection layer for repository-level code generation. Given the code visible at the completion position and a pool of retrieved repository blocks, DARES first identifies unresolved generation decisions and then selects an ordered, task-dependent evidence set.

## Overview

DARES contains two independently fine-tuned components:

- **Decider** identifies the target, objective, implementation anchor, and repository-specific decisions that cannot be resolved from the visible code alone.
- **Selector** selects a directly imitable anchor and complementary evidence from the retrieved candidate pool.

The canonical prompts used by the released models are available in [`prompts/`](prompts). Candidate retrieval is intentionally separated from evidence selection; DARES can consume candidate pools produced by CodeRAG or another repository retriever.

## Environment

We recommend Python 3.10 and four CUDA GPUs for the default inference configuration.

```bash
conda create -n dares python=3.10 -y
conda activate dares
pip install -e ".[inference,train]"
```

## Released weights

The repository includes the trained LoRA adapters under `weights/decider/` and `weights/selector/`. They are based on `Qwen/Qwen3-4B-Instruct-2507` and stored with Git LFS:

```bash
git lfs install
git lfs pull
BASE_MODEL=/path/to/Qwen3-4B-Instruct bash scripts/prepare_models.sh
```

The merged checkpoints are written to `models/decider` and `models/selector`.

## Data format

DARES accepts one JSON object per task:

```json
{"id":"task-id","repository":"owner/repository","current_file":"package/module.py","cursor_prefix":"def target(...):\n    ...","candidate_blocks":[{"path":"package/helper.py","code":"def helper(...): ..."}]}
```

For compatibility with the evaluation artifacts, `task_id`, `project`, `path`, `input`, and `All_contexts` are also supported. The SFT data used for the released adapters is under [`data/sft/`](data/sft).

## Inference

```bash
python scripts/select_context.py \
  --input examples/tasks.jsonl \
  --output-dir outputs/example \
  --decider-model models/decider \
  --selector-model models/selector \
  --tensor-parallel-size 4 \
  --max-model-len 8192
```

The command produces structured Decider outputs and an ordered set of selected candidate IDs and block contents. Decoding is greedy.

## Training

The Decider and Selector are trained independently from the same base model using LoRA SFT:

```bash
accelerate launch --multi_gpu --num_processes 4 scripts/train_lora.py \
  --model-path /path/to/Qwen3-4B-Instruct \
  --train-file data/sft/stage1_train.jsonl \
  --validation-file data/sft/stage1_validation.jsonl \
  --output-dir outputs/decider_lora
```

Repeat with `stage2_train.jsonl` and `stage2_validation.jsonl` for the Selector. The default arguments reproduce the training configuration used for the released adapters. Use [`scripts/merge_adapter.py`](scripts/merge_adapter.py) to merge a trained adapter.

## Evaluation

Copy the path template and edit it for the local benchmark installation:

```bash
cp configs/paths.env.example configs/paths.env
```

### RQ1: generation performance and inference cost

```bash
bash experiments/rq1/run.sh
```

This builds DARES contexts for CoderEval and RepoEval. Set `RUN_GENERATORS=1` to invoke the configured downstream generation and official evaluation runners.

### RQ2: component ablation

```bash
bash experiments/rq2/run.sh
```

This evaluates trained and untuned Decider/Selector combinations. Set `RUN_GENERATORS=1` to run downstream generation for every variant.

### RQ3: selected-evidence quality

```bash
bash experiments/rq3/run.sh
```

This reports evidence-set size, overlap with retrieval baselines, retained DARES evidence, and removed baseline context.

### RQ4: compatibility with upstream retrievers

```bash
bash experiments/rq4/run.sh
```

This applies DARES to candidate pools supplied by RLCoder and the relevance reranker. Set `RUN_GENERATORS=1` to run the configured downstream generators and evaluators.

## Repository structure

```text
configs/       Local path configuration template
dares/         Core prompt rendering, schemas, I/O, and inference
data/sft/      Decider and Selector SFT data
experiments/   RQ1--RQ4 reproduction scripts
prompts/       Canonical paper prompt templates
scripts/       Training, inference, conversion, and evaluation utilities
weights/       Released Decider and Selector LoRA adapters
```

## Citation

Citation information will be added after publication.
