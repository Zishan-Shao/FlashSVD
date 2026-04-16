#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./install_local.sh    # Install the project (requires PyTorch preinstalled)

# 0) Create and activate virtual environment
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip wheel
echo "[Check] CUDA/PyTorch will NOT be installed automatically. Please manually install the correct PyTorch version first."

# 1) Pre-check: verify that PyTorch is already installed
python - <<'PY' || { echo "[Fail] PyTorch not detected. Please install PyTorch manually before running this script."; exit 1; }
try:
    import torch
    print("Found PyTorch:", torch.__version__)
    print("CUDA built?:", torch.backends.cuda.is_built())
    print("CUDA version:", torch.version.cuda)
    print("CUDA available?:", torch.cuda.is_available())
except Exception as e:
    raise SystemExit(e)
PY

# 2) Install this project in editable mode (does not touch CUDA/torch)
echo "[Step] Installing project (editable mode)..."
pip install -e .

echo
echo "Done! Next time, activate the environment using: source .venv/bin/activate"
echo "Note: If you later want optional GPU extras, install them manually, e.g. 'pip install -e .[gpu]' after you've installed a CUDA-enabled PyTorch."
