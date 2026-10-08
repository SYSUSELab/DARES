from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import candidate_text, normalize_task, read_jsonl, write_jsonl
from .prompts import load_template, render_decider, render_selector
from .schema import extract_json_object, validate_decision, validate_selection


@dataclass
class InferenceConfig:
    input_path: Path
    output_dir: Path
    decider_model: str
    selector_model: str
    decider_prompt: Path | None = None
    selector_prompt: Path | None = None
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 8
    candidate_chars: int = 800
    decider_max_tokens: int = 384
    selector_max_tokens: int = 192
    trust_remote_code: bool = True


def _chat_prompt(tokenizer: Any, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _load_engine(model: str, config: InferenceConfig) -> tuple[Any, Any]:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM
    except ImportError as error:
        raise RuntimeError('Install inference dependencies with pip install -e ".[inference]"') from error
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=config.trust_remote_code)
    engine = LLM(
        model=model,
        tokenizer=model,
        trust_remote_code=config.trust_remote_code,
        tensor_parallel_size=config.tensor_parallel_size,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_seqs=config.max_num_seqs,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )
    return engine, tokenizer


def _generate(engine: Any, prompts: list[str], max_tokens: int) -> list[str]:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = engine.generate(prompts, params, use_tqdm=True)
    return [output.outputs[0].text.strip() for output in outputs]


def _release_engine(engine: Any) -> None:
    del engine
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def run(config: InferenceConfig) -> list[dict[str, Any]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [normalize_task(row, index) for index, row in enumerate(read_jsonl(config.input_path))]
    decider_template = load_template(config.decider_prompt, "decider.txt")
    selector_template = load_template(config.selector_prompt, "selector.txt")

    engine, tokenizer = _load_engine(config.decider_model, config)
    decider_prompts = [_chat_prompt(tokenizer, render_decider(decider_template, task)) for task in tasks]
    raw_decisions = _generate(engine, decider_prompts, config.decider_max_tokens)
    decision_rows = []
    for task, raw in zip(tasks, raw_decisions):
        try:
            decision = validate_decision(extract_json_object(raw))
            error = None
        except ValueError as exception:
            decision, error = {}, str(exception)
        decision_rows.append({"id": task["id"], "decision": decision,
                              "raw_decider_output": raw, "error": error})
    write_jsonl(config.output_dir / "decisions.jsonl", decision_rows)
    _release_engine(engine)

    engine, tokenizer = _load_engine(config.selector_model, config)
    selector_prompts = []
    all_candidate_texts = []
    for task, decision_row in zip(tasks, decision_rows):
        texts = [candidate_text(item)[:config.candidate_chars] for item in task["candidate_blocks"]]
        all_candidate_texts.append(texts)
        hint = json.dumps(decision_row["decision"], ensure_ascii=False) if decision_row["decision"] else decision_row["raw_decider_output"]
        selector_prompts.append(_chat_prompt(
            tokenizer, render_selector(selector_template, task, hint, texts)
        ))
    raw_selections = _generate(engine, selector_prompts, config.selector_max_tokens)
    results = []
    for task, decision_row, texts, raw in zip(tasks, decision_rows, all_candidate_texts, raw_selections):
        try:
            ids = validate_selection(extract_json_object(raw), len(texts))
            error = None
        except ValueError as exception:
            ids, error = [], str(exception)
        indices = [int(block_id[1:]) for block_id in ids]
        selected = [task["candidate_blocks"][index] for index in indices]
        results.append({
            "id": task["id"], "repository": task["repository"],
            "current_file": task["current_file"], "decision": decision_row["decision"],
            "selected_block_ids": ids, "selected_indices": indices,
            "selected_blocks": indices,
            "extracted_blocks": selected,
            "selected_contexts": selected, "Selected_Contexts": selected,
            "raw_selector_output": raw, "error": error,
        })
    write_jsonl(config.output_dir / "selections.jsonl", results)
    write_jsonl(config.output_dir / "Selected_Contexts.jsonl", results)
    _release_engine(engine)
    return results
