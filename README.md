# MAXMIN INFERENCE-TIME ALIGNMENT WITH DIVERSE REWARDS

Train domain-specific reward models, generate responses with Qwen2.5-3B-Instruct,
score each response with a 3B proxy RM and a 7B evaluation RM, then run the six
allocation and selection algorithms on the scored pools. The 7B judge is a
separately trained scalar reward model. Each model uses the Bradley–Terry
pairwise preference loss.

## Install and choose local resources

Use Python 3.12 on Linux. Run all commands from the repository root, in the same
Bash session. The full pipeline uses one CUDA GPU and loads one model at a time. Model checkpoints,
data, and generated outputs are supplied or created locally.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r reward_training/requirements-qlora.txt
export PYTHON_BIN=python

# Replace these four placeholders with your own absolute directories.
export MODEL_3B='<LOCAL_QWEN2_5_3B_INSTRUCT_DIRECTORY>'
export MODEL_7B='<LOCAL_QWEN2_5_7B_INSTRUCT_DIRECTORY>'
export DATA_ROOT='<LOCAL_SHP_DATASET_DIRECTORY>'
export RUN_ROOT='<NEW_OUTPUT_DIRECTORY>'
export SEED=2026
```

If the base models and SHP are not already available locally, download them:

```bash
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

The dataset layout is `DATA_ROOT/DOMAIN/{train,validation,test}.json`. Files may
contain JSON arrays or JSON Lines. Records require `domain`, `history`,
`human_ref_A`, `human_ref_B`, and `labels`. The domain field is `DOMAIN_SPLIT`;
label 1 prefers A and label 0 prefers B.

## Run the pipeline

Execute these stages in order:

```bash
# 1. Train one 3B proxy and one 7B judge adapter for each of the five domains.
bash scripts/run_pipeline.sh train

# 2. Generate validation and test responses with the unmodified 3B base model.
bash scripts/run_pipeline.sh generate

# 3. Score every response with both trained domain-specific adapters.
bash scripts/run_pipeline.sh score

# 4. Calibrate on validation pools and export test inputs for the algorithms.
bash scripts/run_pipeline.sh prepare

# 5. Run all six methods with a shared budget of 255 responses per input.
bash scripts/run_pipeline.sh run
```

Alternatively, `bash scripts/run_pipeline.sh all` runs all five stages for a
fresh `RUN_ROOT`. Generation, scoring, and preparation refuse to overwrite
existing outputs. Stages can also be invoked independently through
`python reward_training/response_pool.py --help`.

Training uses learning rate `5e-6`, two epochs, maximum length 1,024, AdamW,
weight decay 0.05, cosine scheduling, and 3% warmup. Both models use 4-bit NF4
QLoRA with double quantization, FP16 computation, gradient checkpointing,
rank 64, alpha 32, dropout 0.1, and attention targets `q_proj`, `k_proj`,
`v_proj`, and `o_proj`. The scalar score head is also trained. The 3B model
uses batch size 4 and four accumulation steps; the 7B model uses batch size 2
and eight accumulation steps. Both therefore use 16 preference pairs per
optimizer update. Training and data seeds are 2026 by default.

Each domain has 50 test prompts with 256 responses per prompt, plus 20
validation prompts with 32 responses per prompt for calibration. Prompts are
unique valid SHP histories selected in a deterministic hash order. Generation
uses temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05, at most
1,024 prompt tokens and 512 new tokens. Candidate order is preserved during
scoring and algorithm execution. Scoring reuses each adapter's saved precision,
chat template, and maximum sequence length.

The preparation stage fits a separate min–max scale for each domain and scorer
on validation responses, applies those frozen scales to test scores, and clips
to [0, 1]. It supplies validation proxy–judge RMSE as the algorithms' error
parameter. This is an empirical calibration recipe, not a certified population
error bound. Test judge scores are used only to evaluate the algorithms. Fixed
beta and the default finite-pool settings are empirical comparisons; this
pipeline does not encode every paper-specific sweep.

The exported `pool_00000.json`, for example, contains the first test prompt from
each domain; its budget is shared across those five domain groups. Each method
sees the same ordered candidates. Results are written separately for every
prompt index and include allocations, selected responses, selection
probabilities, and proxy/judge reward summaries.

| Output under `RUN_ROOT` | Contents |
| --- | --- |
| `rm_3b/DOMAIN/`, `judge_7b/DOMAIN/` | Trained adapters, scalar heads, tokenizers, and training settings |
| `pools/DOMAIN/SPLIT.raw.json` | Prompts and generated response text |
| `pools/DOMAIN/SPLIT.proxy.json` | Responses with raw 3B scores |
| `pools/DOMAIN/SPLIT.scored.json` | Responses with both raw 3B and 7B scores |
| `calibration.json` | Validation normalization scales and empirical RMSE |
| `algorithm_inputs/pool_*.json` | Normalized test inputs for all six methods |
| `algorithm_inputs/manifest.json` | Prompt mapping and calibration metadata |
| `results/seed_2026/pool_*.json` | Algorithm outputs for each prompt index |

Use `bash scripts/run_pipeline.sh --help` for optional pool-size and batch-size
environment variables. `SEED=2027 bash scripts/run_pipeline.sh run` repeats
algorithm selection with another seed on the same fixed pools. To customize
algorithm parameters directly:

```bash
python -m pessimism.run --input "$RUN_ROOT/algorithm_inputs/pool_00000.json" \
  --budget 255 --seed 2026 --alpha 1 --fixed-beta 0.1 \
  --output "$RUN_ROOT/results/custom.json"
```

See [algorithm options](pessimism/README.md) and
[training and validation commands](reward_training/README.md).

## Tests

```bash
python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false PYTHONPATH=reward_training \
  python -m unittest discover -s reward_training/tests -v
```

The tests use small in-memory fixtures and temporary tiny models; they do not
require downloading the production checkpoints or launching GPU training.
