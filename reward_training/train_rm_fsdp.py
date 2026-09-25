#!/usr/bin/env python3
"""Train a full-parameter scalar reward model using FSDP and local preference pairs."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence
from transformers import AutoConfig, set_seed
from shp_rm.data import PairwiseCollator, load_and_tokenize_splits
from shp_rm.fsdp import OPTIMIZER_BYTES_PER_PARAMETER, FsdpSanityCallback, FullGradientCheckCallback, decoder_layer_class_name, estimate_memory, load_full_model_and_tokenizer, require_fp16_capable_hardware, unwrap_for_saving
from shp_rm.trainer import PairwiseRewardTrainer, build_training_arguments, pairwise_metrics
from train_rm import configure_logging
LOGGER = logging.getLogger('fair_bon')

def parse_args(argv: Optional[Sequence[str]]=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--dataset_cache_dir', default=None)
    parser.add_argument('--pad_token', default=None)
    parser.add_argument('--seq_length', type=int, default=1024)
    parser.add_argument('--num_epochs', type=float, default=2.0)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-06)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--optim', default='adamw_torch', choices=sorted(OPTIMIZER_BYTES_PER_PARAMETER))
    parser.add_argument('--lr_scheduler_type', default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--eval_strategy', choices=('epoch', 'steps'), default='steps')
    parser.add_argument('--eval_steps', type=int, default=300)
    parser.add_argument('--save_steps', type=int, default=300)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_limit', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument('--fp16', action='store_true', default=True)
    precision.add_argument('--bf16', dest='bf16', action='store_true', default=False)
    parser.add_argument('--tf32', action='store_true', default=False)
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    parser.add_argument('--no-gradient_checkpointing', dest='gradient_checkpointing', action='store_false')
    parser.add_argument('--save_only_model', action='store_true', default=False, help='Write model weights at each save and no optimizer state. Required by adamw_bnb_8bit, whose int8 moments FSDP cannot gather, and only accepted alongside FULL_STATE_DICT -- transformers rejects it under SHARDED_STATE_DICT. Costs the ability to resume: a crashed run restarts from weights rather than continuing.')
    parser.add_argument('--overwrite_output_dir', action='store_true', default=False)
    parser.add_argument('--dataloader_num_workers', type=int, default=2)
    parser.add_argument('--preprocessing_num_workers', type=int, default=4)
    parser.add_argument('--overwrite_cache', action='store_true', default=False)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_validation_samples', type=int, default=None)
    parser.add_argument('--report_to', choices=('none',), default='none')
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--estimate_only', action='store_true', help='Print the per-GPU memory arithmetic for this model, optimizer and world size, then exit. Reads the config only, so it costs nothing and answers how many nodes to request before queueing for them.')
    parser.add_argument('--world_size', type=int, default=None, help='Only for --estimate_only; a real run reads it from the launcher.')
    args = parser.parse_args(argv)
    if min(args.eval_steps, args.save_steps, args.batch_size, args.eval_batch_size, args.seq_length) <= 0:
        parser.error('Step intervals, batch sizes and sequence length must be positive')
    if args.bf16:
        args.fp16 = False
    return args

def launcher_world_size() -> int:
    return int(os.environ.get('WORLD_SIZE', '1'))

def report_memory_estimate(args: argparse.Namespace) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(args.model_path)
    import torch
    from transformers import AutoModelForSequenceClassification
    config.num_labels = 1
    with torch.device('meta'):
        model = AutoModelForSequenceClassification.from_config(config)
    parameter_count = sum((p.numel() for p in model.parameters()))
    world_size = args.world_size or launcher_world_size()
    estimate = estimate_memory(parameter_count, world_size, args.optim)
    estimate['model_path'] = args.model_path
    estimate['decoder_layer_class'] = decoder_layer_class_name(model)
    return estimate

def main(argv: Optional[Sequence[str]]=None) -> None:
    args = parse_args(argv)
    if args.estimate_only:
        print(json.dumps(report_memory_estimate(args), indent=2, sort_keys=True))
        return
    output_dir = Path(args.output_dir)
    configure_logging(output_dir)
    world_size = launcher_world_size()
    if world_size < 2:
        raise RuntimeError('FSDP requires a distributed launcher with at least two processes')
    if args.optim in ('adamw_bnb_8bit', 'adafactor') and not args.save_only_model:
        raise ValueError('This optimizer requires --save_only_model and FULL_STATE_DICT')
    hardware = require_fp16_capable_hardware(args)
    LOGGER.info('hardware: %s', hardware)
    set_seed(args.seed)
    model, tokenizer, structure = load_full_model_and_tokenizer(args, training=True)
    LOGGER.info('%s: %.2fB parameters, sharding at %s over %d process(es)', args.model_path, structure['parameter_count'] / 1000000000.0, structure['decoder_layer_class'], world_size)
    estimate = estimate_memory(structure['parameter_count'], world_size, args.optim)
    LOGGER.info('training state: %.1f GiB total, %.1f GiB per GPU before activations', estimate['state_gib_total'], estimate['state_gib_per_gpu'])
    tokenized, dataset_info = load_and_tokenize_splits(args, tokenizer, ('train', 'validation'))
    collator = PairwiseCollator(tokenizer)
    training_arguments = build_training_arguments(args, do_train=True)
    training_arguments.load_best_model_at_end = False
    training_arguments.save_only_model = args.save_only_model
    gradient_callback = FullGradientCheckCallback()
    sanity_callback = FsdpSanityCallback(world_size, tokenized['validation'], collator)
    trainer = PairwiseRewardTrainer(model=model, args=training_arguments, train_dataset=tokenized['train'], eval_dataset=tokenized['validation'], data_collator=collator, compute_metrics=pairwise_metrics, callbacks=[sanity_callback, gradient_callback])
    sanity_callback.trainer = trainer
    result = trainer.train()
    metrics = trainer.evaluate()
    gradient_result = gradient_callback.result()
    if not gradient_result['passed']:
        raise AssertionError(f'Missing body or score-head gradient: {gradient_result}')
    unwrap_for_saving(trainer, output_dir)
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        (output_dir / 'fsdp_run.json').write_text(json.dumps({'structure': structure, 'sharding': sanity_callback.sharding, 'hardware': hardware, 'memory_estimate': estimate, 'forward_shape_check': sanity_callback.forward_check, 'gradient_check': gradient_result, 'dataset_info': dataset_info, 'train_metrics': result.metrics, 'validation_metrics': metrics, 'training_arguments': {'model_path': args.model_path, 'domain': args.domain, 'learning_rate': args.learning_rate, 'num_epochs': args.num_epochs, 'optim': args.optim, 'batch_size': args.batch_size, 'gradient_accumulation_steps': args.gradient_accumulation_steps, 'effective_train_batch_size': args.batch_size * args.gradient_accumulation_steps * world_size, 'seq_length': args.seq_length, 'fp16': args.fp16, 'bf16': args.bf16, 'full_finetune': True, 'world_size': world_size}}, indent=2, sort_keys=True), encoding='utf-8')
        (output_dir / '.training_complete').write_text(f'model={args.model_path}\ndomain={args.domain}\nlearning_rate={args.learning_rate}\nworld_size={world_size}\n', encoding='utf-8')
        LOGGER.info('wrote %s', output_dir)
if __name__ == '__main__':
    sys.exit(main())
