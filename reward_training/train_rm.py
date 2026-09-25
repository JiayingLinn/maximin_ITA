#!/usr/bin/env python3
"""Train or validate one independent scalar RM on one local preference domain."""

import argparse
import gc
import json
import logging
import platform
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Sequence

import datasets
import numpy as np
import peft
import torch
import transformers
from transformers import set_seed

from shp_rm.constants import (
    DEFAULT_LORA_TARGETS,
)
from shp_rm.chat_format import tokenize_pair_text
from shp_rm.data import PairwiseCollator, load_and_tokenize_splits
from shp_rm.modeling import (
    compute_pair_rewards,
    load_model_and_tokenizer,
    save_adapter_manifest,
)
from shp_rm.trainer import (
    GradientCheckCallback,
    PairwiseRewardTrainer,
    build_training_arguments,
    forward_shape_check,
    pairwise_metrics,
)


LOGGER = logging.getLogger("fair_bon")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    LOGGER.addHandler(console)

    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def runtime_info() -> dict[str, Any]:
    cuda_capability = (
        list(torch.cuda.get_device_capability(torch.cuda.current_device()))
        if torch.cuda.is_available()
        else None
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "cuda_capability": cuda_capability,
        "native_bf16_supported": bool(
            cuda_capability is not None
            and cuda_capability[0] >= 8
            and torch.cuda.is_bf16_supported()
        ),
        "tf32_supported": bool(
            cuda_capability is not None and cuda_capability[0] >= 8
        ),
    }


def add_sample_counts(
    metrics: dict[str, Any], dataset_info: dict[str, Any]
) -> dict[str, Any]:
    metrics = dict(metrics)
    for split in ("train", "validation"):
        split_info = dataset_info.get("splits", {}).get(split)
        if split_info:
            metrics[f"{split}_raw_examples"] = split_info["raw_examples"]
            metrics[f"{split}_effective_examples"] = split_info["effective_examples"]
            metrics[f"{split}_examples_used"] = split_info["examples_used"]
    return metrics


def truncation_smoke_check(tokenizer: Any, max_length: int) -> dict[str, Any]:
    prompt = "history-token " * (max_length * 4)
    response = "UNIQUE_COMPLETE_RESPONSE_7391"
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )
    kept_ids = tokenize_pair_text(tokenizer, prompt, response, max_length)["input_ids"]
    decoded = tokenizer.decode(kept_ids, skip_special_tokens=False)
    result = {
        "full_token_count": len(full_ids),
        "kept_token_count": len(kept_ids),
        "response_preserved": response in decoded,
        "suffix_preserved": kept_ids == list(full_ids)[-max_length:],
    }
    result["passed"] = bool(
        result["full_token_count"] > max_length
        and result["kept_token_count"] == max_length
        and result["response_preserved"]
        and result["suffix_preserved"]
    )
    if not result["passed"]:
        raise AssertionError(f"Response-preserving truncation failed: {result}")
    return result


def save_log_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in history:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def reload_reference(
    args: argparse.Namespace,
    model: torch.nn.Module,
    dataset: Any,
    collator: PairwiseCollator,
) -> dict[str, Any]:
    features = [dataset[index] for index in range(min(2, len(dataset)))]
    batch = collator(features)
    batch.pop("labels")
    model.eval()
    model.config.use_cache = False
    with reward_inference_context(args):
        chosen, rejected = compute_pair_rewards(model, batch)
    return {
        "features": features,
        "chosen_rewards_before_save": chosen.detach().float().cpu().tolist(),
        "rejected_rewards_before_save": rejected.detach().float().cpu().tolist(),
        "atol": 1e-4,
        "rtol": 1e-4,
        "forward_policy": "explicit_training_precision_autocast_v1",
        "autocast_dtype": reward_autocast_dtype_name(args),
    }


def reward_autocast_dtype_name(args: argparse.Namespace) -> Optional[str]:
    if args.fp16:
        return "float16"
    if args.bf16:
        return "bfloat16"
    return None


def reward_inference_context(args: argparse.Namespace) -> Any:
    dtype_name = reward_autocast_dtype_name(args)
    if not torch.cuda.is_available() or dtype_name is None:
        return nullcontext()
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def verify_saved_checkpoint(
    args: argparse.Namespace,
    checkpoint: Path,
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Freshly load base + adapter and compare saved pre-reload rewards."""
    model, tokenizer, _ = load_model_and_tokenizer(
        args,
        adapter_path=str(checkpoint),
        training=False,
    )
    model.config.use_cache = False
    collator = PairwiseCollator(tokenizer)
    batch = collator(reference["features"])
    batch.pop("labels")
    with reward_inference_context(args):
        chosen, rejected = compute_pair_rewards(model, batch)

    before_chosen = np.asarray(reference["chosen_rewards_before_save"])
    before_rejected = np.asarray(reference["rejected_rewards_before_save"])
    after_chosen = chosen.detach().float().cpu().numpy()
    after_rejected = rejected.detach().float().cpu().numpy()
    atol = float(reference["atol"])
    rtol = float(reference["rtol"])
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "chosen_rewards_before_save": before_chosen.tolist(),
        "chosen_rewards_after_reload": after_chosen.tolist(),
        "rejected_rewards_before_save": before_rejected.tolist(),
        "rejected_rewards_after_reload": after_rejected.tolist(),
        "chosen_max_abs_difference": float(
            np.max(np.abs(before_chosen - after_chosen))
        ),
        "rejected_max_abs_difference": float(
            np.max(np.abs(before_rejected - after_rejected))
        ),
        "atol": atol,
        "rtol": rtol,
        "forward_policy": "explicit_training_precision_autocast_v1",
        "autocast_dtype": reward_autocast_dtype_name(args),
        "passed": bool(
            np.allclose(before_chosen, after_chosen, atol=atol, rtol=rtol)
            and np.allclose(before_rejected, after_rejected, atol=atol, rtol=rtol)
        ),
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not result["passed"]:
        raise AssertionError(f"Reloaded rewards do not match: {result}")
    return result


def run_training(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    set_seed(args.seed)

    model, tokenizer, structure = load_model_and_tokenizer(args, training=True)
    tokenized, dataset_info = load_and_tokenize_splits(
        args,
        tokenizer,
        requested_splits=("train", "validation"),
    )
    write_json(output_dir / "dataset_info.json", dataset_info)
    write_json(output_dir / "model_structure.json", structure)

    collator = PairwiseCollator(tokenizer)
    shape_result = forward_shape_check(
        model,
        tokenized["train"],
        collator,
        requested_batch_size=args.batch_size,
    )
    write_json(output_dir / "forward_shape_check.json", shape_result)
    truncation_result = None
    if args.smoke_test:
        truncation_result = truncation_smoke_check(tokenizer, args.seq_length)
        write_json(output_dir / "truncation_check.json", truncation_result)

    gradient_callback = GradientCheckCallback()
    trainer = PairwiseRewardTrainer(
        model=model,
        args=build_training_arguments(args, do_train=True),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=pairwise_metrics,
        processing_class=tokenizer,
        callbacks=[gradient_callback],
    )

    initial_validation = None
    if args.smoke_test:
        initial_validation = trainer.evaluate(metric_key_prefix="initial_validation")
        write_json(
            output_dir / "initial_validation_metrics.json",
            initial_validation,
        )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    gradient_result = gradient_callback.result()
    write_json(output_dir / "gradient_check.json", gradient_result)
    if not gradient_result["passed"]:
        raise AssertionError(f"LoRA or score-head gradient was missing: {gradient_result}")

    train_metrics = add_sample_counts(train_result.metrics, dataset_info)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    validation_metrics = add_sample_counts(validation_metrics, dataset_info)
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)
    trainer.save_state()
    write_json(output_dir / "validation_metrics.json", validation_metrics)

    # PEFT save_pretrained is used by Trainer here. modules_to_save=[score]
    # places the trained scalar head in the standard adapter file.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.model.config.to_json_file(output_dir / "reward_model_config.json")
    save_log_history(output_dir / "training_log.jsonl", trainer.state.log_history)
    artifact_manifest = save_adapter_manifest(trainer.model, output_dir)
    write_json(output_dir / "artifact_manifest.json", artifact_manifest)

    reference = reload_reference(
        args,
        trainer.model,
        tokenized["validation"],
        collator,
    )
    write_json(output_dir / "reload_reference.json", reference)

    logged_losses = [
        float(entry["loss"])
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    smoke_result = None
    if args.smoke_test:
        loss_decreased = (
            len(logged_losses) >= 2 and min(logged_losses[1:]) < logged_losses[0]
        )
        smoke_result = {
            "mapping_checks_passed": all(
                check["chosen_matches_source"] and check["rejected_matches_source"]
                for split_info in dataset_info["splits"].values()
                for check in split_info["mapping_checks"]
            ),
            "forward_shape": shape_result,
            "truncation_check": truncation_result,
            "gradient_check": gradient_result,
            "logged_training_losses": logged_losses,
            "training_loss_decreased": loss_decreased,
            "initial_validation_metrics": initial_validation,
            "final_validation_metrics": validation_metrics,
        }
        if not loss_decreased:
            raise AssertionError(f"Smoke training loss did not decrease: {logged_losses}")

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.verify_reload:
        reload_result = verify_saved_checkpoint(args, output_dir, reference)
        write_json(output_dir / "reload_verification.json", reload_result)
        if smoke_result is not None:
            smoke_result["reload_verification"] = reload_result

    if smoke_result is not None:
        smoke_result["passed"] = bool(
            smoke_result["mapping_checks_passed"]
            and smoke_result["forward_shape"]["passed"]
            and smoke_result["truncation_check"]["passed"]
            and smoke_result["gradient_check"]["passed"]
            and smoke_result["training_loss_decreased"]
            and (
                not args.verify_reload
                or smoke_result["reload_verification"]["passed"]
            )
        )
        write_json(output_dir / "smoke_test_results.json", smoke_result)

    LOGGER.info("Completed %s; artifacts are in %s", args.domain, output_dir.resolve())


def run_validation(args: argparse.Namespace) -> None:
    checkpoint = Path(args.model_path if args.full_model else args.adapter_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    set_seed(args.seed)
    if args.full_model:
        from shp_rm.fsdp import load_full_model_and_tokenizer
        model, tokenizer, _ = load_full_model_and_tokenizer(args, training=False)
    else:
        model, tokenizer, _ = load_model_and_tokenizer(
            args, adapter_path=str(checkpoint), training=False
        )
    tokenized, current_info = load_and_tokenize_splits(
        args,
        tokenizer,
        requested_splits=("validation",),
    )
    saved_info_path = checkpoint / "dataset_info.json"
    if saved_info_path.exists():
        dataset_info = json.loads(saved_info_path.read_text(encoding="utf-8"))
        dataset_info["splits"]["validation"] = current_info["splits"]["validation"]
    else:
        dataset_info = current_info

    collator = PairwiseCollator(tokenizer)
    forward_shape_check(model, tokenized["validation"], collator)
    trainer = PairwiseRewardTrainer(
        model=model,
        args=build_training_arguments(args, do_train=False),
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=pairwise_metrics,
        processing_class=tokenizer,
    )
    metrics = trainer.evaluate(metric_key_prefix="validation")
    metrics = add_sample_counts(metrics, dataset_info)
    write_json(Path(args.output_dir) / "validation_metrics.json", metrics)
    LOGGER.info("Validation metrics: %s", json.dumps(metrics, sort_keys=True))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model_path", required=True)
    # Llama 3.x has no pad token and would fall back to <|eot_id|>, the very
    # token its chat template ends with. Pass <|finetune_right_pad_id|> there.
    parser.add_argument("--pad_token", default=None)
    parser.add_argument("--dataset_path", required=True, help="Local dataset root containing DOMAIN/train.json and validation.json")
    parser.add_argument("--dataset_cache_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--full_model", action="store_true", help="Validation only: load a full checkpoint from --model_path")

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--eval_strategy", choices=("epoch", "steps"), default="steps")
    parser.add_argument("--eval_steps", type=int, default=300)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=list(DEFAULT_LORA_TARGETS),
    )
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_cpu", action="store_true")

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_validation_samples", type=int, default=None)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument(
        "--overwrite_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--verify_reload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--report_to", choices=("none",), default="none")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--config_check_only", action="store_true")

    args = parser.parse_args(argv)
    if args.full_model and args.mode != "eval":
        parser.error("--full_model is a validation option; use train_rm_fsdp.py for full training")
    if args.mode == "eval" and not args.full_model and not args.adapter_path:
        parser.error("Adapter validation requires --adapter_path")
    if args.seq_length <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("Sequence length and batch sizes must be positive")
    if args.eval_steps <= 0:
        parser.error("--eval_steps must be greater than zero")
    if args.save_steps <= 0:
        parser.error("--save_steps must be greater than zero")
    if args.fp16 and args.bf16:
        parser.error("--fp16 and --bf16 cannot both be enabled")
    if args.smoke_test:
        if args.max_train_samples is None:
            args.max_train_samples = 32
        if args.max_validation_samples is None:
            args.max_validation_samples = 16
        if args.max_steps < 0:
            args.max_steps = 12
        args.eval_strategy = "steps"
        args.eval_steps = min(args.eval_steps, max(1, args.max_steps // 2))
        args.logging_steps = 1
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.config_check_only:
        checked = vars(args).copy()
        if args.use_cpu or torch.cuda.is_available() or not (args.fp16 or args.bf16):
            resolved = build_training_arguments(
                args, do_train=args.mode == "train"
            )
            checked["resolved_training_arguments"] = {
                "device": str(resolved.device),
                "fp16": resolved.fp16,
                "bf16": resolved.bf16,
                "tf32": resolved.tf32,
                "eval_strategy": str(resolved.eval_strategy),
                "save_strategy": str(resolved.save_strategy),
                "eval_steps": resolved.eval_steps,
                "save_steps": resolved.save_steps,
                "load_best_model_at_end": resolved.load_best_model_at_end,
            }
        else:
            # This branch only supports CPU-side source checks. The production
            # worker has already required CUDA and therefore always executes
            # the real TrainingArguments hardware validation above.
            checked["resolved_training_arguments"] = {
                "status": "skipped_without_cuda"
            }
        checked["config_check"] = "passed"
        print(json.dumps(checked, indent=2, sort_keys=True, default=str))
        return

    output_dir = Path(args.output_dir)
    configure_logging(output_dir)
    saved_arguments = vars(args).copy()
    saved_arguments["runtime"] = runtime_info()
    saved_arguments["effective_train_batch_size_per_process"] = (
        args.batch_size * args.gradient_accumulation_steps
    )
    arguments_filename = (
        "training_arguments.json" if args.mode == "train" else "validation_arguments.json"
    )
    write_json(output_dir / arguments_filename, saved_arguments)
    LOGGER.info("Arguments: %s", json.dumps(saved_arguments, default=str, sort_keys=True))

    if (args.fp16 or args.bf16) and not torch.cuda.is_available():
        raise RuntimeError("FP16/BF16 training requires CUDA")
    if args.mode == "train":
        run_training(args)
    else:
        run_validation(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Fatal reward-model pipeline error")
        raise
