# MAXMIN INFERENCE-TIME ALIGNMENT WITH DIVERSE REWARDS

## Install and test

```bash
python -m pip install -r requirements.txt
PYTHON_BIN=python bash scripts/check_release.sh
```

Numerical tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py' -v
```

## Run the algorithms

```bash
# In-memory synthetic candidates; JSON results go to standard output.
PYTHON_BIN=python bash scripts/run_algorithms.sh --synthetic

# Inspect the complete argument list.
python -B -m pessimism.run --help

# Real inputs must be supplied explicitly; this is a placeholder, not a path.
python -B -m pessimism.run --input '<SCORED_POOL_JSON>' --budget 255 --seed 2026
```

Each input group supplies ordered candidate IDs and normalized proxy scores,
with an optional Judge score used only for evaluation.

See [the algorithm guide](pessimism/README.md) for the complete JSON schema and parameter conventions.
Specify error bounds and normalization using a calibration protocol appropriate
for your experiment.

Fixed-beta runs bypass the certified beta set and are empirical comparisons.

## Reward-model training

Install the optional dependencies and consult the training guide:

```bash
python -m pip install -r reward_training/requirements.txt
bash reward_training/scripts/train_rm.sh --help
bash reward_training/scripts/train_rm_fsdp.sh --help
bash reward_training/scripts/validate_rm.sh --help
```

The [training guide](reward_training/README.md) gives the exact local data schemas,
training, validation, configuration checks, and offline smoke-test commands.
Base-model, dataset, checkpoint, and output locations are always runtime inputs.

For pairwise training, the local dataset root contains `DOMAIN/train.json` and
`DOMAIN/validation.json` (JSON arrays or JSON Lines). Required fields are
`domain`, `history`, `human_ref_A`, `human_ref_B`, and `labels`. The `domain`
field must equal `DOMAIN_train` or `DOMAIN_validation`; label 1 prefers A and
label 0 prefers B.

```bash
# Pairwise LoRA. All angle-bracket values are placeholders to replace.
bash reward_training/scripts/train_rm.sh \
  --model_path '<BASE_MODEL_DIR>' --dataset_path '<PAIRWISE_DATA_ROOT>' \
  --domain '<DOMAIN>' --output_dir '<NEW_OUTPUT_DIR>' \
  --no-load_in_4bit --bf16

# For CUDA QLoRA, install these extra dependencies and use --load_in_4bit.
python -m pip install -r reward_training/requirements-qlora.txt

# Single-machine full-parameter FSDP; choose the model's decoder layer class.
bash reward_training/scripts/train_rm_fsdp.sh \
  --num_processes 2 --wrap_class '<DECODER_LAYER_CLASS>' --mixed_precision bf16 -- \
  --model_path '<BASE_MODEL_DIR>' --dataset_path '<PAIRWISE_DATA_ROOT>' \
  --domain '<DOMAIN>' --output_dir '<NEW_OUTPUT_DIR>'

# Evaluate a trained pairwise adapter.
bash reward_training/scripts/validate_rm.sh \
  --model_path '<BASE_MODEL_DIR>' --adapter_path '<ADAPTER_DIR>' \
  --dataset_path '<PAIRWISE_DATA_ROOT>' --domain '<DOMAIN>' \
  --output_dir '<NEW_VALIDATION_OUTPUT_DIR>' --no-load_in_4bit --bf16
```

Run the optional training tests on CPU with generated text and randomly
initialized tiny models, including actual training and save/reload checks:

```bash
cd reward_training
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
  python -m unittest discover -s tests -v
```

Executed with Python 3.13.2, NumPy 1.26.4, PyTorch 2.6.0, Transformers 4.55.2,
PEFT 0.17.1, Datasets 3.6.0, and Accelerate 1.10.1. This used an existing
environment; a fresh dependency installation was not tested. CUDA QLoRA,
mixed-precision GPU training, and multi-GPU FSDP were not executed. The generated
FSDP launch configuration was checked with Accelerate's configuration parser.
