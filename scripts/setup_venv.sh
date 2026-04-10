#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found. Install Python 3.10+ and retry."
  exit 1
fi

echo "[1/4] Creating virtual environment at: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[2/4] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[3/4] Installing project dependencies"
pip install -r requirements.txt

echo "[4/4] Verifying key imports"
python - <<'PY'
import importlib
mods = ["torch", "pandas", "numpy", "librosa", "sklearn", "matplotlib", "resamply"]
for m in mods:
    importlib.import_module(m)
print("Dependency import check passed")
PY

echo
echo "Virtual environment is ready."
echo "Activate with: source $VENV_DIR/bin/activate"
