"""Bradley-Terry Trainer, validation metrics, and smoke-test checks."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch import nn
from transformers import Trainer, TrainerCallback, TrainingArguments

from .data import PairwiseCollator
from .modeling import compute_pair_rewards, extract_scalar_rewards


def bradley_terry_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """Mean -log sigmoid(r_chosen - r_rejected)."""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


class PairwiseRewardTrainer(Trainer):
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Any:
        del num_items_in_batch
        inputs = dict(inputs)
        inputs.pop("labels", None)

        chosen_rewards = extract_scalar_rewards(
            model(
                input_ids=inputs["input_ids_chosen"],
                attention_mask=inputs["attention_mask_chosen"],
            )
        )
        rejected_rewards = extract_scalar_rewards(
            model(
                input_ids=inputs["input_ids_rejected"],
                attention_mask=inputs["attention_mask_rejected"],
            )
        )
        loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
        if return_outputs:
            return loss, {
                "logits": torch.stack((chosen_rewards, rejected_rewards), dim=-1)
            }
        return loss


def pairwise_metrics(eval_prediction: Any) -> dict[str, float]:
    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.asarray(predictions).reshape(-1, 2)
    margins = predictions[:, 0] - predictions[:, 1]
    return {
        "pairwise_accuracy": float(np.mean(margins > 0.0)),
        "bradley_terry_loss": float(np.logaddexp(0.0, -margins).mean()),
        "mean_reward_margin": float(margins.mean()),
    }


class GradientCheckCallback(TrainerCallback):
    """Record a nonzero LoRA gradient and score-head gradient before stepping."""

    def __init__(self) -> None:
        self.lora_gradient_seen = False
        self.reward_head_gradient_seen = False
        self.first_lora_grad_norm: Optional[float] = None
        self.first_reward_head_grad_norm: Optional[float] = None
        self.first_checked_step: Optional[int] = None

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        model: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        if model is None:
            return control
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            norm = float(parameter.grad.detach().float().norm().cpu())
            if not np.isfinite(norm) or norm <= 0.0:
                continue
            if "lora_" in name and not self.lora_gradient_seen:
                self.lora_gradient_seen = True
                self.first_lora_grad_norm = norm
                self.first_checked_step = int(state.global_step)
            if "score" in name and not self.reward_head_gradient_seen:
                self.reward_head_gradient_seen = True
                self.first_reward_head_grad_norm = norm
                self.first_checked_step = int(state.global_step)
        return control

    def result(self) -> dict[str, Any]:
        return {
            "lora_gradient_seen": self.lora_gradient_seen,
            "reward_head_gradient_seen": self.reward_head_gradient_seen,
            "first_lora_grad_norm": self.first_lora_grad_norm,
            "first_reward_head_grad_norm": self.first_reward_head_grad_norm,
            "first_checked_step": self.first_checked_step,
            "passed": self.lora_gradient_seen and self.reward_head_gradient_seen,
        }


def forward_shape_check(
    model: nn.Module,
    dataset: Dataset,
    collator: PairwiseCollator,
    requested_batch_size: int = 2,
) -> dict[str, Any]:
    if len(dataset) == 0:
        raise ValueError("Cannot check forward shape with an empty dataset")
    batch_size = min(requested_batch_size, len(dataset))
    batch = collator([dataset[index] for index in range(batch_size)])
    batch.pop("labels")
    model.eval()
    chosen, rejected = compute_pair_rewards(model, batch)
    expected = (batch_size,)
    if tuple(chosen.shape) != expected or tuple(rejected.shape) != expected:
        raise AssertionError(
            f"Expected reward shape {expected}, got {tuple(chosen.shape)} and "
            f"{tuple(rejected.shape)}"
        )
    return {
        "batch_size": batch_size,
        "chosen_reward_shape": list(chosen.shape),
        "rejected_reward_shape": list(rejected.shape),
        "passed": True,
    }


def build_training_arguments(args: Any, do_train: bool) -> TrainingArguments:
    strategy = args.eval_strategy if do_train else "no"
    report_to = [] if args.report_to == "none" else [args.report_to]
    # Transformers only allows load_best_model_at_end when save_steps is a
    # multiple of eval_steps. Keep independently requested save/eval schedules
    # and save the actual end-of-training model at the output root when they
    # are incompatible. Periodic evaluations still update the best metric and
    # best checkpoint recorded in TrainerState.
    load_best_model_at_end = bool(
        do_train
        and (
            strategy == "epoch"
            or (
                strategy == "steps"
                and args.save_steps % args.eval_steps == 0
            )
        )
    )
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "do_train": do_train,
        "do_eval": True,
        "num_train_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optim": args.optim,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "eval_strategy": strategy,
        "save_strategy": strategy,
        "logging_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_total_limit": args.save_limit,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": "pairwise_accuracy",
        "greater_is_better": True,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "tf32": args.tf32,
        "use_cpu": args.use_cpu,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_drop_last": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "report_to": report_to,
        "run_name": args.run_name or f"rm-{args.domain}",
        "logging_dir": str(Path(args.output_dir) / "logs"),
        "save_safetensors": True,
    }
    if args.gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return TrainingArguments(**kwargs)
