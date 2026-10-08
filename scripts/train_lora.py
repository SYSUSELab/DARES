#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class ChatDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.examples = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row.get("messages")
                if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
                    raise ValueError(f"{path}:{line_number}: final message must be assistant")
                prefix = apply_template(tokenizer, messages[:-1], True)
                complete = apply_template(tokenizer, messages, False)
                input_ids = tokenizer(complete, add_special_tokens=False, truncation=True,
                                      max_length=max_length)["input_ids"]
                prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=True,
                                       max_length=max_length)["input_ids"]
                labels = input_ids.copy()
                labels[:min(len(prefix_ids), len(labels))] = [-100] * min(len(prefix_ids), len(labels))
                self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def apply_template(tokenizer, messages, generation_prompt: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        width = max(len(example["input_ids"]) for example in examples)
        input_ids, labels, masks = [], [], []
        for example in examples:
            padding = width - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * padding)
            labels.append(example["labels"] + [-100] * padding)
            masks.append([1] * len(example["input_ids"]) + [0] * padding)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(masks)}


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT for a DARES stage")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    train_data = ChatDataset(args.train_file, tokenizer, args.max_length)
    validation_data = ChatDataset(args.validation_file, tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, logging_steps=10,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        report_to="none", seed=args.seed, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data,
                      eval_dataset=validation_data, data_collator=Collator(tokenizer.pad_token_id))
    trainer.train()
    final_dir = args.output_dir / "final_adapter"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    if trainer.is_world_process_zero():
        print(f"Saved adapter to {final_dir}")


if __name__ == "__main__":
    main()
