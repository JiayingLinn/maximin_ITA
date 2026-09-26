# Pairwise reward-model training

Train and validate reward models with LoRA/QLoRA or full-parameter FSDP.
See the [main README](../README.md) for the complete pipeline.
Run commands from the repository root and replace `<...>` placeholders.

```bash
# Install dependencies.
python -m pip install -r reward_training/requirements.txt
# Install additional dependencies for CUDA QLoRA.
python -m pip install -r reward_training/requirements-qlora.txt
export PYTHON_BIN=python
```

```bash
# Train with LoRA; replace --no-load_in_4bit with --load_in_4bit for CUDA QLoRA.
bash reward_training/scripts/train_rm.sh \
  --model_path '<BASE_MODEL_DIR>' --dataset_path '<PAIRWISE_DATA_ROOT>' \
  --domain '<DOMAIN>' --output_dir '<OUTPUT_DIR>' \
  --no-load_in_4bit --bf16

# Train with full-parameter FSDP on two GPUs.
bash reward_training/scripts/train_rm_fsdp.sh \
  --num_processes 2 --wrap_class '<DECODER_LAYER_CLASS>' --mixed_precision bf16 -- \
  --model_path '<BASE_MODEL_DIR>' --dataset_path '<PAIRWISE_DATA_ROOT>' \
  --domain '<DOMAIN>' --output_dir '<OUTPUT_DIR>'

# Validate an adapter using the same precision and quantization as training.
bash reward_training/scripts/validate_rm.sh \
  --model_path '<BASE_MODEL_DIR>' --adapter_path '<ADAPTER_DIR>' \
  --dataset_path '<PAIRWISE_DATA_ROOT>' --domain '<DOMAIN>' \
  --output_dir '<VALIDATION_OUTPUT_DIR>' --no-load_in_4bit --bf16

# Validate a full-parameter checkpoint.
bash reward_training/scripts/validate_rm.sh --full_model \
  --model_path '<CHECKPOINT_DIR>' --dataset_path '<PAIRWISE_DATA_ROOT>' \
  --domain '<DOMAIN>' --output_dir '<VALIDATION_OUTPUT_DIR>' \
  --no-load_in_4bit --bf16

# Show available options.
bash reward_training/scripts/train_rm.sh --help
bash reward_training/scripts/train_rm_fsdp.sh --help
bash reward_training/scripts/validate_rm.sh --help

# Run training and validation tests.
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false PYTHONPATH=reward_training \
  python -m unittest discover -s reward_training/tests -v
```
