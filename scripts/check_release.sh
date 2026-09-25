#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Compile in memory: this deliberately creates no bytecode or model artifacts.
"$PYTHON_BIN" - <<'PY'
import ast
from pathlib import Path
import subprocess

files = list(Path('.').rglob('*.py'))
for path in files:
    if '.venv' not in path.parts:
        ast.parse(path.read_text(), filename=str(path))
for path in Path('.').rglob('*.sh'):
    if '.venv' not in path.parts:
        subprocess.run(['bash', '-n', str(path)], check=True)
print('Python and shell syntax checks passed.')
PY
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py' -v
"$PYTHON_BIN" scripts/audit_release.py
