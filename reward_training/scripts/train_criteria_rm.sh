#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1
exec "${PYTHON_BIN:-${PYTHON:-python3}}" "$SCRIPT_DIR/../train_criteria_rm.py" "$@"
