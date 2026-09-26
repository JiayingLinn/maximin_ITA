# Pairwise reward-model training

Train scalar reward models with the Bradley-Terry preference loss using
LoRA/QLoRA or full-parameter FSDP. Model and data locations are runtime inputs.
The [main pipeline](../README.md) trains both the 3B proxy and 7B judge,
generates responses, scores pools, and runs the allocation algorithms.
Run the commands below from the repository root.

```bash
python -m pip install -r reward_training/requirements.txt
# Additional dependencies for CUDA QLoRA.
python -m pip install -r reward_training/requirements-qlora.txt
```

Local data must contain `DOMAIN/train.json` and `DOMAIN/validation.json` as
JSON arrays or JSON Lines. Each record has `domain`, `history`, `human_ref_A`,
`human_ref_B`, and `labels`. Set `domain` to `DOMAIN_train` or
`DOMAIN_validation`; label 1 prefers A and label 0 prefers B.
The tokenizer must provide a chat template. Choose precision and LoRA target
modules for your model; all angle-bracket values below are placeholders.

```bash
# Single-device LoRA; use --load_in_4bit for CUDA QLoRA.
bash reward_training/scripts/train_rm.sh \
  --model_path '<BASE_MODEL_DIR>' --dataset_path '<PAIRWISE_DATA_ROOT>' \
  --domain '<DOMAIN>' --output_dir '<OUTPUT_DIR>' \
  --no-load_in_4bit --bf16

# Full-parameter FSDP on one machine with at least two GPUs.
bash reward_training/scripts/train_rm_fsdp.sh \
  --num_processes 2 --wrap_class '<DECODER_LAYER_CLASS>' --mixed_precision bf16 -- \
  --model_path '<BASE_MODEL_DIR>' --dataset_path '<PAIRWISE_DATA_ROOT>' \
  --domain '<DOMAIN>' --output_dir '<OUTPUT_DIR>'

# Validate a trained adapter with matching training settings.
bash reward_training/scripts/validate_rm.sh \
  --model_path '<BASE_MODEL_DIR>' --adapter_path '<ADAPTER_DIR>' \
  --dataset_path '<PAIRWISE_DATA_ROOT>' --domain '<DOMAIN>' \
  --output_dir '<VALIDATION_OUTPUT_DIR>' --no-load_in_4bit --bf16

# Validate a full-parameter checkpoint.
bash reward_training/scripts/validate_rm.sh --full_model \
  --model_path '<CHECKPOINT_DIR>' --dataset_path '<PAIRWISE_DATA_ROOT>' \
  --domain '<DOMAIN>' --output_dir '<VALIDATION_OUTPUT_DIR>' \
  --no-load_in_4bit --bf16

# All options, including configuration checks.
bash reward_training/scripts/train_rm.sh --help
bash reward_training/scripts/train_rm_fsdp.sh --help
bash reward_training/scripts/validate_rm.sh --help

# CPU training and inference checks using temporary tiny models.
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false PYTHONPATH=reward_training \
  python -m unittest discover -s reward_training/tests -v
```

Set `PYTHON_BIN` to choose the scripts' Python interpreter. FSDP saves model
weights only by default; these checkpoints do not include optimizer state.

Pool generation and scoring use the same tokenization and model-loading code as
training. Scoring reads precision and sequence-length settings from the adapter's
saved `training_arguments.json`:

```bash
python reward_training/response_pool.py generate --help
python reward_training/response_pool.py score --help
python reward_training/response_pool.py calibrate --help
python reward_training/response_pool.py export --help
```
