"""Scalar sequence-classification model, LoRA, and PEFT persistence."""

import gc
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from safetensors.torch import save_file
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .constants import DEFAULT_LORA_TARGETS


LOGGER = logging.getLogger("fair_bon")


def load_tokenizer(
    path: str,
    pad_token: Optional[str] = None,
    **from_pretrained_kwargs: Any,
) -> Any:
    """Load the tokenizer and settle its pad token.

    Sequence classification pools the rightmost token that is not the pad token,
    so a pad token equal to the one a chat template ends with silently moves the
    pooled position one step back. Llama 3.x has no pad token of its own and
    would fall back to `<|eot_id|>`, which is exactly that final token; it ships
    `<|finetune_right_pad_id|>` for this. Pass it explicitly rather than letting
    the fallback decide.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        use_fast=True,
        **from_pretrained_kwargs,
    )
    if pad_token:
        pad_id = tokenizer.convert_tokens_to_ids(pad_token)
        if pad_id is None or pad_id == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer at {path} has no token {pad_token!r}")
        tokenizer.pad_token = pad_token
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            raise ValueError(
                f"pad_token {pad_token!r} is the eos token; pooling would move "
                "one position back. Choose a dedicated pad token."
            )
    elif tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"Tokenizer at {path} has no chat template")
    return tokenizer


def _base_model_kwargs(
    args: Any,
    training: bool,
    legacy_kbit_training_reload: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_labels": 1,
        "problem_type": "regression",
        "low_cpu_mem_usage": True,
        "ignore_mismatched_sizes": True,
    }
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires CUDA and bitsandbytes")
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        if legacy_kbit_training_reload:
            # Transformers 4.55.2/bitsandbytes forced this dtype when the old
            # training code omitted torch_dtype. Keep it explicit here so the
            # legacy base quantization path is reproducible.
            kwargs["torch_dtype"] = torch.float16
        else:
            # Keep embeddings, norms, and the scalar score head in the same FP32
            # dtype before and after PEFT reload. The quantized matrix kernels
            # still use compute_dtype.
            kwargs["torch_dtype"] = torch.float32
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    elif args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float32
    return kwargs


def _legacy_reload_compatibility(args: Any, adapter_path: Optional[str]) -> bool:
    """Return whether an old adapter needs the original k-bit prepare path."""
    requested = bool(getattr(args, "legacy_kbit_training_reload", False))
    if not adapter_path:
        return requested
    compatibility_path = Path(adapter_path) / "loader_compatibility.json"
    if not compatibility_path.is_file():
        return requested
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    if compatibility.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported loader compatibility schema: {compatibility_path}"
        )
    if compatibility.get("load_strategy") != "legacy_kbit_training_path":
        raise ValueError(
            f"Unsupported loader compatibility strategy: {compatibility_path}"
        )
    return True


def _validate_target_modules(model: nn.Module, targets: Sequence[str]) -> dict[str, int]:
    counts = {
        target: sum(
            name.rsplit(".", 1)[-1] == target for name, _ in model.named_modules()
        )
        for target in targets
    }
    missing = sorted(name for name, count in counts.items() if count == 0)
    if missing:
        raise ValueError(f"LoRA targets do not match the model: missing {missing}")
    score_modules = [
        name for name, _ in model.named_modules()
        if name == "score" or name.endswith(".score")
    ]
    if not score_modules:
        raise ValueError("No scalar sequence-classification head named 'score' found")
    return counts


def trainable_parameter_report(model: nn.Module) -> dict[str, Any]:
    trainable_lora = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    trainable_head = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and "score" in name
    ]
    if not trainable_lora:
        raise AssertionError("No trainable LoRA parameters")
    if not trainable_head:
        raise AssertionError("The scalar reward head is not trainable")
    return {
        "trainable_lora_parameter_names": trainable_lora,
        "trainable_reward_head_parameter_names": trainable_head,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_model_and_tokenizer(
    args: Any,
    adapter_path: Optional[str] = None,
    training: bool = True,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Build a fresh scalar base RM, then create or load its LoRA adapter."""
    legacy_kbit_training_reload = _legacy_reload_compatibility(args, adapter_path)
    if legacy_kbit_training_reload and not args.load_in_4bit:
        raise ValueError("Legacy k-bit reload compatibility requires 4-bit loading")
    if adapter_path:
        base_path = args.model_path
        tokenizer = load_tokenizer(adapter_path, getattr(args, 'pad_token', None))
    else:
        base_path = args.model_path
        tokenizer = load_tokenizer(base_path, getattr(args, 'pad_token', None))

    LOGGER.info("Loading scalar base model: %s", base_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_path,
        **_base_model_kwargs(
            args,
            training=training,
            legacy_kbit_training_reload=legacy_kbit_training_reload,
        ),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.num_labels = 1
    model.config.problem_type = "regression"
    structure: dict[str, Any] = {
        "base_model": base_path,
        "num_labels": 1,
        "legacy_kbit_training_reload": legacy_kbit_training_reload,
    }

    if adapter_path:
        if legacy_kbit_training_reload:
            # Adapters produced before the FP32 base-load fix were trained after
            # prepare_model_for_kbit_training cast the non-quantized modules to
            # FP32. Reproduce that exact order for validation and inference.
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=False,
            )
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=training,
            autocast_adapter_dtype=False,
        )
        if training:
            structure.update(trainable_parameter_report(model))
    else:
        structure["lora_target_module_counts"] = _validate_target_modules(
            model, args.lora_target_modules
        )
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        elif training and args.gradient_checkpointing:
            model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            inference_mode=not training,
            target_modules=list(args.lora_target_modules or DEFAULT_LORA_TARGETS),
            modules_to_save=["score"],
        )
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
        structure.update(trainable_parameter_report(model))

    model.config.use_cache = not (training and args.gradient_checkpointing)
    if training:
        model.print_trainable_parameters()
    else:
        model.eval()
    return model, tokenizer, structure


def extract_scalar_rewards(output: Any) -> torch.Tensor:
    logits = output.logits
    if logits.ndim != 2 or logits.shape[-1] != 1:
        raise ValueError(
            f"Expected scalar logits [batch, 1], received {tuple(logits.shape)}"
        )
    return logits[:, 0]


def input_device(model: nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def move_batch(batch: dict[str, torch.Tensor], model: nn.Module) -> dict[str, torch.Tensor]:
    device = input_device(model)
    return {name: tensor.to(device) for name, tensor in batch.items()}


def compute_pair_rewards(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = move_batch(batch, model)
    with torch.no_grad():
        chosen = extract_scalar_rewards(
            model(
                input_ids=batch["input_ids_chosen"],
                attention_mask=batch["attention_mask_chosen"],
            )
        )
        rejected = extract_scalar_rewards(
            model(
                input_ids=batch["input_ids_rejected"],
                attention_mask=batch["attention_mask_rejected"],
            )
        )
    return chosen, rejected


def save_adapter_manifest(model: nn.Module, output_dir: Path) -> dict[str, Any]:
    """Prove the standard adapter includes LoRA tensors and the score head."""
    state = get_peft_model_state_dict(model)
    lora_keys = sorted(key for key in state if "lora_" in key)
    head_state = {
        key: value.detach().contiguous().cpu()
        for key, value in state.items()
        if "score" in key
    }
    if not lora_keys:
        raise AssertionError("PEFT state contains no LoRA tensors")
    if not head_state:
        raise AssertionError("PEFT state contains no score head tensors")

    # This is an explicit inspection copy. Standard reload uses the adapter file,
    # which already contains these same modules_to_save tensors.
    save_file(head_state, str(output_dir / "reward_head.safetensors"))
    adapter_files = sorted(path.name for path in output_dir.glob("adapter_model.*"))
    return {
        "adapter_files": adapter_files,
        "lora_tensor_count": len(lora_keys),
        "lora_tensor_keys": lora_keys,
        "reward_head_file": "reward_head.safetensors",
        "reward_head_tensor_count": len(head_state),
        "reward_head_tensor_keys": sorted(head_state),
        "standard_reload_uses": adapter_files,
        "reward_head_is_embedded_in_adapter": True,
    }


def release_model(model: nn.Module) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
