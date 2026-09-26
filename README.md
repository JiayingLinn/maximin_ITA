# MAXMIN INFERENCE-TIME ALIGNMENT WITH DIVERSE REWARDS

We provide a pipeline that adaptively allocates inference budgets across groups,
with two main algorithms:

- **Greedy-BoN** achieves optimal max–min performance on proxy rewards.
- **Batched Greedy Pessimistic (BGP)** maintains competitive performance when
  proxy rewards contain errors.

## Setup

Run commands from the repository root in the same Bash session. The full pipeline
requires Python 3.12, Linux, and a CUDA GPU.

```bash
# Install dependencies.
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r reward_training/requirements-qlora.txt
export PYTHON_BIN=python

# Set absolute paths for local resources and a new output directory.
export MODEL_3B='<LOCAL_QWEN2_5_3B_INSTRUCT_DIRECTORY>'
export MODEL_7B='<LOCAL_QWEN2_5_7B_INSTRUCT_DIRECTORY>'
export DATA_ROOT='<LOCAL_SHP_DATASET_DIRECTORY>'
export RUN_ROOT='<NEW_OUTPUT_DIRECTORY>'
export SEED=2026
```

```bash
# Download models and data if they are not already available locally.
python - <<'PYTHON'
import os
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen2.5-3B-Instruct", local_dir=os.environ["MODEL_3B"])
snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir=os.environ["MODEL_7B"])
domains = ["askacademia", "askculinary", "askengineers", "asksciencefiction", "changemyview"]
snapshot_download("stanfordnlp/SHP", repo_type="dataset", local_dir=os.environ["DATA_ROOT"],
                  allow_patterns=[f"{domain}/*.json" for domain in domains])
PYTHON
```

## Pipeline

Run the stages in order:

```bash
# 1. Train the proxy and judge reward models.
bash scripts/run_pipeline.sh train

# 2. Generate validation and test response pools.
bash scripts/run_pipeline.sh generate

# 3. Score responses with the proxy and judge models.
bash scripts/run_pipeline.sh score

# 4. Calibrate rewards and export algorithm inputs.
bash scripts/run_pipeline.sh prepare

# 5. Run the main algorithms and baselines; save results under RUN_ROOT/results.
bash scripts/run_pipeline.sh run
```

```bash
# Alternatively, run all five stages with a fresh RUN_ROOT.
bash scripts/run_pipeline.sh all

# Run Greedy-BoN and BGP on one prepared pool.
python -m pessimism.run --input "$RUN_ROOT/algorithm_inputs/pool_00000.json" \
  --methods greedy_argmax greedy_pessimistic_auto --budget 255 --seed "$SEED" \
  --output "$RUN_ROOT/results/custom.json"

# Show pipeline configuration and algorithm options.
bash scripts/run_pipeline.sh --help
python -m pessimism.run --help
```

## Tests

```bash
# Check allocation algorithms and pool preparation.
python -m unittest discover -s tests -v

# Check reward model training and validation.
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false PYTHONPATH=reward_training \
  python -m unittest discover -s reward_training/tests -v
```
