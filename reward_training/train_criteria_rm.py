#!/usr/bin/env python3
"""Train a five-attribute reward regressor from explicitly supplied local JSON files."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

CRITERIA = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")


def format_prompt_response(prompt: str, response: str) -> str:
    """Original plain-text regression format; this flow does not use a chat template."""
    return "Prompt:\n" + prompt + "\n\nResponse:\n" + response


def criterion_labels(example: dict) -> list[float]:
    values = [float(example[name]) for name in CRITERIA]
    if any(not np.isfinite(value) or value < 0 or value > 4 for value in values):
        raise ValueError("Criterion labels must be finite scores in [0, 4]")
    return [value / 4.0 for value in values]


def compute_eval_metrics(eval_prediction):
    predictions, labels = eval_prediction
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    mse = ((predictions - labels) ** 2).mean(axis=0)
    mae = np.abs(predictions - labels).mean(axis=0)
    result = {"mse_avg": float(mse.mean()), "mae_avg": float(mae.mean())}
    for index, name in enumerate(CRITERIA):
        result[f"mse_{name}"] = float(mse[index])
        result[f"mae_{name}"] = float(mae[index])
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_file", required=True, type=Path)
    parser.add_argument("--validation_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--dataset_cache_dir")
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=float, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=300)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--pad_token")
    args = parser.parse_args(argv)
    if args.load_in_4bit and not args.use_lora:
        parser.error("--load_in_4bit requires --use_lora")
    if args.load_in_4bit and (args.use_cpu or not torch.cuda.is_available()):
        parser.error("4-bit loading requires CUDA")
    if min(args.seq_length, args.batch_size, args.eval_batch_size,
           args.gradient_accumulation_steps, args.eval_steps, args.save_steps,
           args.logging_steps) <= 0:
        parser.error("Sequence length, batch sizes and step intervals must be positive")
    if args.train_file.resolve() == args.validation_file.resolve():
        parser.error("Training and validation must use separate files")
    return args


def main(argv=None):
    args = parse_args(argv)
    for file in (args.train_file, args.validation_file):
        if not file.is_file():
            raise FileNotFoundError(file)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if args.pad_token:
        token_id = tokenizer.convert_tokens_to_ids(args.pad_token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError("Requested padding token is absent from the tokenizer")
        tokenizer.pad_token = args.pad_token
    elif tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires a padding token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_kwargs = dict(num_labels=len(CRITERIA), problem_type="regression",
                        low_cpu_mem_usage=True, ignore_mismatched_sizes=True,
                        torch_dtype=torch.float32)
    if args.load_in_4bit:
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16),
            device_map={"": torch.cuda.current_device()})
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not args.gradient_checkpointing
    if args.use_lora:
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        elif args.gradient_checkpointing:
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type=TaskType.SEQ_CLS,
            target_modules=args.lora_target_modules, modules_to_save=["score"]),
            autocast_adapter_dtype=False)
    dataset = load_dataset("json", data_files={"train": str(args.train_file),
                                               "validation": str(args.validation_file)},
                           cache_dir=args.dataset_cache_dir)
    required = {"prompt", "response", *CRITERIA}
    def preprocess(example):
        encoded = tokenizer(format_prompt_response(example["prompt"], example["response"]),
                            truncation=True, max_length=args.seq_length)
        encoded["labels"] = criterion_labels(example)
        return encoded
    processed = {}
    for split in ("train", "validation"):
        if not len(dataset[split]) or not required.issubset(dataset[split].column_names):
            raise ValueError(f"{split} must contain nonempty prompt/response and all five criterion columns")
        processed[split] = dataset[split].map(preprocess, remove_columns=dataset[split].column_names)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.num_epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, lr_scheduler_type="cosine",
        eval_strategy="steps", save_strategy="steps", eval_steps=args.eval_steps,
        save_steps=args.save_steps, logging_steps=args.logging_steps,
        logging_first_step=True, save_total_limit=args.save_limit,
        bf16=args.bf16, fp16=args.fp16, use_cpu=args.use_cpu,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], seed=args.seed, data_seed=args.seed,
        remove_unused_columns=False, save_safetensors=True,
        dataloader_pin_memory=torch.cuda.is_available())
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=processed["train"], eval_dataset=processed["validation"],
                      processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_eval_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_metrics("validation", metrics)
    (args.output_dir / "criteria_config.json").write_text(json.dumps(
        {"criteria": CRITERIA, "label_divisor": 4.0, "loss": "mean_squared_error",
         "text_format": "Prompt:\\n{prompt}\\n\\nResponse:\\n{response}"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
