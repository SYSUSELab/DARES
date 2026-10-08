#!/usr/bin/env python3
"""Run the released two-stage DARES selector on a benchmark JSONL file.

Stage 1: visible cursor prefix -> compact retrieval hint.
Stage 2: hint + original CodeRAG candidate pool -> selected block IDs.
GT and right_context are never used.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECIDER_TEMPLATE = (ROOT / 'prompts' / 'decider.txt').read_text(encoding='utf-8')
SELECTOR_TEMPLATE = (ROOT / 'prompts' / 'selector.txt').read_text(encoding='utf-8')
S2_DIRECT = """Select useful evidence from the original CodeRAG repository candidates
for completing the visible target code. Select one directly imitable implementation
anchor when available, followed by only non-redundant evidence or dependencies needed
by the target. Return exactly {"selected_block_ids":["B..."]}, ordered as anchor then
evidence/dependencies, with as many IDs as the task warrants. Do not target a fixed count."""


def read(path: Path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def complete(path: Path, expected: int):
    try:
        return sum(bool(line.strip()) for line in path.open(encoding='utf-8')) == expected
    except FileNotFoundError:
        return False


def parse_object(raw: str):
    text = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.I)
    text = re.sub(r'\s*```$', '', text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {}


def render(tokenizer, messages):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def infer(model_path: Path, messages, tp: int, max_model_len: int,
          max_tokens: int, utilization: float):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path), trust_remote_code=True,
        tensor_parallel_size=tp, max_model_len=max_model_len,
        dtype='bfloat16', gpu_memory_utilization=utilization,
        enforce_eager=True, disable_custom_all_reduce=True,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [render(tokenizer, value) for value in messages]
    lengths = [len(tokenizer.encode(value, add_special_tokens=False)) for value in prompts]
    oversized = [(index, length) for index, length in enumerate(lengths)
                 if length > max_model_len - max_tokens]
    if oversized:
        raise ValueError(f'Prompt exceeds model window: {oversized[:5]}')
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=max_tokens),
        use_tqdm=True,
    )
    texts = [item.outputs[0].text for item in outputs]
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return texts


def task_key(task):
    return task.get('task_id', task.get('id'))


def repository_name(task):
    if task.get('project') or task.get('repository'):
        return str(task.get('project') or task.get('repository'))
    # RepoEval candidate headers contain .../line_and_api_level/<repository>/...
    for block in task.get('All_contexts', []):
        match = re.search(r'/line_and_api_level/([^/]+)/', str(block).splitlines()[0])
        if match:
            return match.group(1)
    return 'unknown_repository'


def current_path(task):
    return str(task.get('path') or task.get('current_file') or task.get('_inferred_path') or 'unknown_current_file')


def head(task):
    return (
        f"<repository>{repository_name(task)}</repository>\n"
        f"<current_file>{current_path(task)}</current_file>\n"
        f"<cursor_prefix>\n{str(task['input'])[:6000]}\n</cursor_prefix>"
    )


def decider_prompt(task):
    return (DECIDER_TEMPLATE
            .replace('{REPOSITORY_NAME}', repository_name(task))
            .replace('{CURRENT_FILE_PATH}', current_path(task))
            .replace('{CODE_BEFORE_COMPLETION_POSITION}', str(task['input'])[:6000]))


def selector_prompt(task, hint, rendered):
    return (SELECTOR_TEMPLATE
            .replace('{REPOSITORY_NAME}', repository_name(task))
            .replace('{CURRENT_FILE_PATH}', current_path(task))
            .replace('{CODE_BEFORE_COMPLETION_POSITION}', str(task['input'])[:6000])
            .replace('{STAGE_1_OUTPUT}', hint)
            .replace('{CANDIDATE_BLOCKS}', '\n'.join(rendered)))


def clean_hint(value, task):
    if not isinstance(value, dict):
        value = {}
    anchor = value.get('anchor') if isinstance(value.get('anchor'), dict) else {}
    decisions = value.get('decisions') if isinstance(value.get('decisions'), list) else []
    clean_decisions = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        clean_decisions.append({
            'type': str(decision.get('type', 'repository_contract')),
            'requirement': str(decision.get('requirement', '')),
            'entities': [str(x) for x in decision.get('entities', [])]
                        if isinstance(decision.get('entities'), list) else [],
        })
    return {
        'target': str(value.get('target') or task_key(task)),
        'objective': str(value.get('objective', '')),
        'anchor': {
            'pattern': str(anchor.get('pattern', '')),
            'entities': [str(x) for x in anchor.get('entities', [])]
                        if isinstance(anchor.get('entities'), list) else [],
        },
        'decisions': clean_decisions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--stage1-model', type=Path, required=True)
    parser.add_argument('--stage2-model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tp', type=int, default=4)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.72)
    parser.add_argument('--expected-tasks', type=int, default=230)
    parser.add_argument('--path-metadata', type=Path)
    parser.add_argument('--force', action='store_true')
    parser.add_argument(
        '--ablation', choices=('full', 'direct', 'no_anchor', 'no_decisions'),
        default='full',
        help=('full: complete two-stage method; direct: Stage 2 selects without a '
              'Stage-1 hint; no_anchor/no_decisions: remove that field from the hint.'),
    )
    args = parser.parse_args()

    tasks = read(args.tasks)
    if args.path_metadata:
        metadata = read(args.path_metadata)
        if len(metadata) != len(tasks):
            raise ValueError('Path metadata is not aligned with tasks')
        for task, row in zip(tasks, metadata):
            path = (row.get('identity_score') or {}).get('current_path')
            if path:
                task['_inferred_path'] = path
    if len(tasks) != args.expected_tasks:
        raise ValueError(f'Expected {args.expected_tasks} tasks, got {len(tasks)}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1_file = args.output_dir/'stage1_guidance.jsonl'
    selection_file = args.output_dir/'Selected_Contexts.jsonl'

    if args.ablation == 'direct':
        guidance = [{
            'id': task['id'], 'task_id': task_key(task),
            'guidance': None, 'raw_output': None,
        } for task in tasks]
        write(stage1_file, guidance)
    elif not args.force and complete(stage1_file, args.expected_tasks):
        guidance = read(stage1_file)
        print(f'[reuse] {stage1_file}')
    else:
        messages = [[{'role': 'user', 'content': decider_prompt(task)}] for task in tasks]
        raw = infer(args.stage1_model, messages, args.tp, args.max_model_len,
                    384, args.gpu_memory_utilization)
        guidance = []
        for task, output in zip(tasks, raw):
            guidance.append({
                'id': task['id'], 'task_id': task_key(task),
                'guidance': clean_hint(parse_object(output), task),
                'raw_output': output,
            })
        write(stage1_file, guidance)

    messages, metadata = [], []
    for task, hint_row in zip(tasks, guidance):
        candidates = task['All_contexts']
        candidate_indices = list(range(len(candidates)))
        rendered = []
        for index in candidate_indices:
            rendered.append(
                f'<candidate_block id="B{index}">\n'
                f'{candidates[index][:800]}\n</candidate_block>'
            )
        guidance_value = hint_row.get('guidance')
        if args.ablation == 'direct':
            user = (head(task) + '\n<candidates>\n' + '\n\n'.join(rendered)
                    + '\n</candidates>')
            messages.append([{'role': 'system', 'content': S2_DIRECT},
                             {'role': 'user', 'content': user}])
        else:
            hint_value = json.loads(json.dumps(guidance_value))
            if args.ablation == 'no_anchor':
                hint_value['anchor'] = {'pattern': '', 'entities': []}
            elif args.ablation == 'no_decisions':
                hint_value['decisions'] = []
            hint = json.dumps(hint_value, ensure_ascii=False, separators=(',', ':'))
            user = selector_prompt(task, hint, rendered)
            messages.append([{'role': 'user', 'content': user}])
        metadata.append((task, candidate_indices))

    raw = infer(args.stage2_model, messages, args.tp, args.max_model_len,
                192, args.gpu_memory_utilization)
    rows = []
    for (task, candidate_indices), output in zip(metadata, raw):
        value = parse_object(output)
        proposed = value.get('selected_block_ids', []) if isinstance(value, dict) else []
        valid = {f'B{index}' for index in candidate_indices}
        selected_ids = []
        if isinstance(proposed, list):
            for block_id in proposed:
                if isinstance(block_id, str) and block_id in valid and block_id not in selected_ids:
                    selected_ids.append(block_id)
        indices = [int(block_id[1:]) for block_id in selected_ids]
        rows.append({
            'id': task['id'], 'task_id': task_key(task),
            'selected_blocks': indices,
            'selected_block_ids': selected_ids,
            'extracted_blocks': [task['All_contexts'][index] for index in indices],
            'candidate_pool': candidate_indices,
            'raw_output': output,
            'selection_strategy': f'dares_{args.ablation}',
            'gold_used': False,
        })
    write(selection_file, rows)
    histogram = {}
    for row in rows:
        count = len(row['selected_blocks'])
        histogram[count] = histogram.get(count, 0) + 1
    summary = {
        'tasks': len(rows),
        'nonempty': sum(bool(row['selected_blocks']) for row in rows),
        'total_blocks': sum(len(row['selected_blocks']) for row in rows),
        'average_blocks': sum(len(row['selected_blocks']) for row in rows)/len(rows),
        'selection_histogram': dict(sorted(histogram.items())),
        'ablation': args.ablation,
        'output': str(selection_file),
    }
    (args.output_dir/'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
