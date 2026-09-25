#!/usr/bin/env python3
"""Create a generic launch configuration and start local multi-GPU FSDP training."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_processes", type=int, required=True)
    parser.add_argument("--wrap_class", required=True,
                        help="Decoder layer class exposed by your model's _no_split_modules")
    parser.add_argument("--mixed_precision", choices=("fp16", "bf16"), default="fp16")
    args, training_args = parser.parse_known_args()
    if training_args and training_args[0] == "--":
        training_args = training_args[1:]
    if args.num_processes < 2:
        parser.error("FSDP requires at least two GPU processes")
    config = {
        "compute_environment": "LOCAL_MACHINE", "distributed_type": "FSDP",
        "num_machines": 1, "machine_rank": 0, "main_training_function": "main",
        "mixed_precision": args.mixed_precision, "use_cpu": False,
        "num_processes": args.num_processes,
        "fsdp_config": {
            "fsdp_version": 1, "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": args.wrap_class,
            "fsdp_use_orig_params": True, "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_cpu_ram_efficient_loading": True, "fsdp_sync_module_states": True,
            "fsdp_offload_params": False, "fsdp_forward_prefetch": False,
            "fsdp_backward_prefetch": "BACKWARD_PRE"}}
    entrypoint = Path(__file__).resolve().parents[1] / "train_rm_fsdp.py"
    with tempfile.TemporaryDirectory(prefix="fair-bon-fsdp-") as temporary:
        config_path = Path(temporary) / "accelerate.yaml"
        config_path.write_text(json.dumps(config))
        command = [sys.executable, "-m", "accelerate.commands.launch", "--config_file",
                   str(config_path), str(entrypoint), "--save_only_model",
                   f"--{args.mixed_precision}", *training_args]
        return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
