# Fair Best-of-N: anonymous core release

This directory is a standalone source release of the allocation algorithms and
reward-model training routines. It contains no checkpoints, real datasets,
precomputed scores, experimental outputs, account credentials, original project Git history, or
configured model/dataset locations. Supply your own local inputs at runtime.

The algorithm package requires only NumPy. Training dependencies are separate.
The synthetic examples check implementation behavior; they do not reproduce the
paper's empirical results.

## Contents

| Location | Purpose |
| --- | --- |
| `pessimism/core/itp.py` | Finite-sample ITP certification and sampling (Algorithm 2). |
| `pessimism/core/lcb_greedy.py` | LCB-Greedy / BGP allocation and beta selection (Algorithm 1). |
| `pessimism/core/radii.py` | Confidence radii, certified beta sets, and initialization. |
| `pessimism/core/validation.py` | Core numerical assumption checks. |
| `pessimism/baselines.py` | Uniform and greedy BoN; uniform allocation with ITP. |
| `pessimism/run.py` | Portable CLI comparing six allocation/selection combinations. |
| `scripts/run_algorithms.sh` | Algorithm run wrapper. |
| `scripts/make_synthetic_pool.py` | Generate an explicitly synthetic input fixture. |
| `tests/` | Core analytical checks and portable runner tests. |
| `reward_training/` | Pairwise LoRA/QLoRA, full-parameter FSDP, and scalar reward-model validation. |
| `scripts/check_release.sh` | CPU algorithm tests, Python/shell syntax checks, and source audit. |
| `scripts/audit_release.py` | Scan source text for common identifying patterns and generated artifacts. |
| `RELEASE_GUIDE.md` | Chinese contents and execution guide. |

## Algorithm names

| CLI method | Allocation | Selection |
| --- | --- | --- |
| `uniform_argmax` | Equal integer budget split | Highest proxy score (Uniform-BoN). |
| `greedy_argmax` | One initial draw per group; help the current lowest maximum | Highest proxy score (Greedy-BoN). |
| `uniform_itp_fixed` | Equal integer budget split | ITP at the supplied fixed beta. |
| `uniform_itp_auto` | Equal integer budget split | ITP with beta chosen from the certified grid. |
| `lcb_greedy_fixed` | Allocate to the current lowest index | Fixed-beta ITP (BGP-fixed / ITP-Fixed). |
| `lcb_greedy_auto` | Allocate to the current lowest index | Automatic-beta ITP (BGP-auto / ITP-Auto). |

The numerical core and baselines retain the original implementation. The CLI,
input format, and shell wrappers are portable release interfaces. Historical
parameter sweeps, plotting code, cluster job submission, pool generation, Judge
inference, and oracle diagnostics are outside this minimal release.

## Install and test

Run all commands from this directory. Python 3.11 is recommended for the pinned
training dependencies. A NumPy-compatible Python is sufficient for algorithms.
Use an environment outside the source directory to keep the upload tree clean.

```bash
python -m pip install -r requirements.txt
PYTHON_BIN=python bash scripts/check_release.sh
```

These checks require no model, dataset, GPU, network service, or credentials
after NumPy has been installed. To run just the numerical tests:

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
with an optional Judge score used only for evaluation. Allocation reads each
group's prefix in the supplied order. The global budget counts all revealed
candidates across groups. The CLI never loads a model and never estimates an
error bound from Judge scores.

See [the algorithm guide](pessimism/README.md) for the complete JSON schema and parameter conventions.
Specify error bounds and normalization using a calibration protocol appropriate
for your experiment. No hidden min-max normalization or tuning is performed.

Fixed-beta runs bypass the certified beta set and are empirical comparisons.
Changing error scales or separating allocation and selection error scales also
changes the assumptions behind the certificate. Small synthetic budgets can
require a vacuous confidence bound. Finite-pool capping and constant-size batches
are explicit variants of the theoretical doubling rule. Run diagnostics record
these settings; a numerical certificate is not automatically a useful guarantee.

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
Experiment tracking is disabled. The release includes neither an RM nor a Judge.

For pairwise training, the local dataset root contains `DOMAIN/train.json` and
`DOMAIN/validation.json` (JSON arrays or JSON Lines). Required fields are
`domain`, `history`, `human_ref_A`, `human_ref_B`, and `labels`. The `domain`
field must equal `DOMAIN_train` or `DOMAIN_validation`; label 1 prefers A and
label 0 prefers B. Only train/validation files are loaded. The tokenizer must
provide a chat template; choose precision and LoRA target modules for your model.

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

FSDP saves model weights only by default, so these checkpoints do not support exact
optimizer-state resumption. The launcher targets a single machine with at least
two GPUs. Full GPU/quantization/distributed behavior requires validation on the
target hardware.

Run the optional training tests on CPU with generated text and randomly
initialized tiny models, including actual training and save/reload checks:

```bash
cd reward_training
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
  python -m unittest discover -s tests -v
```

Return to the release root for the source audit below. Test models and data are
created in temporary directories and are not part of this release.

## Before uploading

```bash
python -B scripts/audit_release.py
```

Upload this source directory to a fresh repository. Generated outputs, local
environments, model files, and datasets do not belong in the upload. The audit
checks common identifying patterns and file types; it does not inspect Git
history or the identity associated with your repository account. No license is
assigned on your behalf.

## Validation performed

- 69 algorithm/core/CLI tests passed.
- 4 CPU reward-training tests passed, including two-step LoRA train/save/reload.
- The complete directory was copied to a separate temporary location. Both
  suites passed there, and the shell demo ran all six methods from another
  working directory, spending 60 candidates per method.
- All three training/validation wrappers passed `--help`; Python and shell
  syntax checks and the source audit passed. Tests left no generated files in
  the release tree.
- The seven copied core/package/baseline Python files are byte-for-byte
  identical to the original implementation.

Executed with Python 3.13.2, NumPy 1.26.4, PyTorch 2.6.0, Transformers 4.55.2,
PEFT 0.17.1, Datasets 3.6.0, and Accelerate 1.10.1. This used an existing
environment; a fresh dependency installation was not tested. CUDA QLoRA,
mixed-precision GPU training, and multi-GPU FSDP were not executed. The generated
FSDP launch configuration was checked with Accelerate's configuration parser.
