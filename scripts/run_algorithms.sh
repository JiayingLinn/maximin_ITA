#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${PYTHON_BIN:-${PYTHON:-python3}}" -m pessimism.run "$@"
