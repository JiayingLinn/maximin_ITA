"""Full-parameter loading, FSDP validation, gradient checks and checkpoint saving."""

import logging
import math
from typing import Any, Optional
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, TrainerCallback
from .modeling import load_tokenizer
LOGGER = logging.getLogger('fair_bon')
OPTIMIZER_BYTES_PER_PARAMETER = {'adamw_torch': 16, 'adamw_torch_fused': 16, 'adamw_bnb_8bit': 10, 'adafactor': 9, 'sgd': 8}

def decoder_layer_class_name(model: Any) -> str:
    candidates = getattr(model, '_no_split_modules', None) or []
    named = [name for name in candidates if isinstance(name, str) and name]
    if not named:
        raise ValueError(f'{type(model).__name__} declares no _no_split_modules, so there is no decoder layer class to shard at. Pass one explicitly.')
    if len(named) > 1:
        raise ValueError(f'{type(model).__name__} declares several splittable modules {named}; name the one to wrap explicitly.')
    return named[0]

def estimate_memory(parameter_count: int, world_size: int, optim: str) -> dict[str, Any]:
    if world_size <= 0:
        raise ValueError('World size must be positive')
    per_parameter = OPTIMIZER_BYTES_PER_PARAMETER.get(optim)
    if per_parameter is None:
        raise ValueError(f'No memory model for optimizer {optim!r}; known: {sorted(OPTIMIZER_BYTES_PER_PARAMETER)}')
    total = parameter_count * per_parameter
    return {'parameters': parameter_count, 'optimizer': optim, 'bytes_per_parameter': per_parameter, 'state_bytes_total': total, 'state_bytes_per_gpu': total / world_size, 'state_gib_total': total / 2 ** 30, 'state_gib_per_gpu': total / world_size / 2 ** 30, 'world_size': world_size}

def load_full_model_and_tokenizer(args: Any, training: bool=True) -> tuple[nn.Module, Any, dict[str, Any]]:
    tokenizer = load_tokenizer(args.model_path, pad_token=getattr(args, 'pad_token', None))
    config = AutoConfig.from_pretrained(args.model_path)
    config.num_labels = 1
    config.problem_type = 'regression'
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, config=config, torch_dtype=torch.float32, low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.num_labels = 1
    model.config.problem_type = 'regression'
    model.config.use_cache = not (training and args.gradient_checkpointing)
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    total = sum((p.numel() for p in model.parameters()))
    if training and trainable != total:
        raise ValueError(f'Full fine-tuning expects every parameter trainable, got {trainable} of {total}')
    structure = {'base_model': args.model_path, 'num_labels': 1, 'full_finetune': True, 'parameter_count': total, 'trainable_parameter_count': trainable, 'decoder_layer_class': decoder_layer_class_name(model), 'load_dtype': 'float32'}
    return (model, tokenizer, structure)

def assert_sharding_is_real(model: nn.Module, world_size: int) -> dict[str, Any]:
    from torch.distributed.fsdp import FullyShardedDataParallel
    units = [module for module in model.modules() if isinstance(module, FullyShardedDataParallel)]
    if world_size > 1 and (not units):
        raise RuntimeError('World size is >1 but the model has no FSDP units: the launcher did not wrap anything. Launch with `accelerate launch --config_file` rather than plain python.')
    flattened = [unit for unit in units if not getattr(unit, '_use_orig_params', False)]
    if flattened:
        raise RuntimeError(f"""{len(flattened)} of {len(units)} FSDP units have use_orig_params=False, which violates this trainer's supported FSDP configuration. Set fsdp_use_orig_params: true in the accelerate CONFIG FILE; as a command-line flag it is silently ignored.""")
    if world_size > 1 and len(units) < 2:
        raise RuntimeError(f'Only {len(units)} FSDP unit(s): the auto-wrap policy matched nothing, so every rank holds the whole model and nothing is sharded. Name the decoder layer class explicitly with fsdp_transformer_layer_cls_to_wrap.')
    report = {'fsdp_units': len(units), 'use_orig_params': True, 'world_size': world_size}
    LOGGER.info('FSDP: %d units, use_orig_params=True, world_size=%d', len(units), world_size)
    return report

def require_fp16_capable_hardware(args: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {'cuda': False}
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    supports_bf16 = major >= 8
    if getattr(args, 'bf16', False) and (not supports_bf16):
        raise RuntimeError(f'{name} is compute capability {major}.{minor} and has no bf16. Use --fp16 on this hardware.')
    return {'cuda': True, 'device_name': name, 'compute_capability': f'{major}.{minor}', 'supports_bf16': supports_bf16}

class FsdpSanityCallback(TrainerCallback):

    def __init__(self, world_size: int, dataset: Any, collator: Any) -> None:
        self.trainer = None
        self.world_size = world_size
        self.dataset = dataset
        self.collator = collator
        self.sharding: dict[str, Any] = {}
        self.forward_check: dict[str, Any] = {}

    def on_train_begin(self, args: Any, state: Any, control: Any, model: Optional[nn.Module]=None, **kwargs: Any) -> Any:
        del args, state
        if self.trainer is not None:
            model = self.trainer.model_wrapped
        if model is None:
            return control
        self.sharding = assert_sharding_is_real(model, self.world_size)
        self.forward_check = sharded_forward_check(model, self.dataset, self.collator, self.trainer.accelerator.device if self.trainer is not None else kwargs.get('accelerator') or getattr(model, 'device', None))
        return control

def sharded_forward_check(model: nn.Module, dataset: Any, collator: Any, device: Any=None) -> dict[str, Any]:
    if len(dataset) == 0:
        raise ValueError('Cannot check forward shape with an empty dataset')
    batch_size = min(2, len(dataset))
    batch = collator([dataset[index] for index in range(batch_size)])
    batch.pop('labels', None)
    if device is None:
        device = next(model.parameters()).device
    elif hasattr(device, 'device'):
        device = device.device
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            chosen = model(input_ids=batch['input_ids_chosen'], attention_mask=batch['attention_mask_chosen']).logits
            rejected = model(input_ids=batch['input_ids_rejected'], attention_mask=batch['attention_mask_rejected']).logits
    finally:
        model.train(was_training)
    expected = (batch_size, 1)
    if tuple(chosen.shape) != expected or tuple(rejected.shape) != expected:
        raise AssertionError(f'Expected scalar logits {expected}, got {tuple(chosen.shape)} and {tuple(rejected.shape)}')
    return {'batch_size': batch_size, 'device': str(device), 'chosen_logits_shape': list(chosen.shape), 'rejected_logits_shape': list(rejected.shape), 'passed': True}

class FullGradientCheckCallback(TrainerCallback):

    def __init__(self) -> None:
        self.body_gradient_seen = False
        self.score_gradient_seen = False
        self.first_body_grad_norm: Optional[float] = None
        self.first_score_grad_norm: Optional[float] = None
        self.first_checked_step: Optional[int] = None

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, model: Optional[nn.Module]=None, **kwargs: Any) -> Any:
        del args, kwargs
        if model is None or (self.body_gradient_seen and self.score_gradient_seen):
            return control
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            norm = float(parameter.grad.detach().float().norm().cpu())
            if not math.isfinite(norm) or norm <= 0.0:
                continue
            is_score = 'score' in name
            if is_score and (not self.score_gradient_seen):
                self.score_gradient_seen = True
                self.first_score_grad_norm = norm
                self.first_checked_step = int(state.global_step)
            elif not is_score and (not self.body_gradient_seen):
                self.body_gradient_seen = True
                self.first_body_grad_norm = norm
                self.first_checked_step = int(state.global_step)
        return control

    def result(self) -> dict[str, Any]:
        local = [self.body_gradient_seen, self.score_gradient_seen]
        seen = list(local)
        ranks_reporting = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            flags = torch.tensor([float(local[0]), float(local[1]), 1.0], dtype=torch.float32, device=torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu'))
            torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.SUM)
            seen = [flags[0].item() > 0, flags[1].item() > 0]
            ranks_reporting = int(flags[2].item())
        return {'body_gradient_seen': bool(seen[0]), 'score_gradient_seen': bool(seen[1]), 'body_gradient_seen_on_this_rank': bool(local[0]), 'score_gradient_seen_on_this_rank': bool(local[1]), 'ranks_reporting': ranks_reporting, 'first_body_grad_norm': self.first_body_grad_norm, 'first_score_grad_norm': self.first_score_grad_norm, 'first_checked_step': self.first_checked_step, 'passed': bool(seen[0] and seen[1])}

def unwrap_for_saving(trainer: Any, output_dir: Any, state_dict: Optional[dict]=None) -> None:
    accelerator = trainer.accelerator
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(trainer.model_wrapped)
    if state_dict is None:
        state_dict = accelerator.get_state_dict(trainer.model_wrapped)
    unwrapped.save_pretrained(output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict, safe_serialization=True)
    accelerator.wait_for_everyone()
