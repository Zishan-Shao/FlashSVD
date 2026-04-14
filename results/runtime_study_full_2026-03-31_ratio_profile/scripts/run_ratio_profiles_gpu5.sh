#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zs89/FlashSVD"
STUDY_DIR="$ROOT/FlashSVD-v1.5/results/runtime_study_full_2026-03-31_ratio_profile"
PYTHON_BIN="/home/zs89/miniconda3/envs/flashsvd15/bin/python"
LOG_DIR="$STUDY_DIR/raw"
LOG_FILE="$LOG_DIR/run_gpu5.log"
ENV_FILE="$LOG_DIR/env_gpu5.txt"

mkdir -p "$STUDY_DIR/raw" "$STUDY_DIR/profiles" "$STUDY_DIR/tables" "$STUDY_DIR/examples"
rm -f "$STUDY_DIR/raw"/* "$STUDY_DIR/profiles"/* "$STUDY_DIR/tables"/* "$STUDY_DIR/examples"/*

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export FLASH_SVD_DENSE_DECODE_BACKEND=packed
export FLASH_SVD_DENSE_DECODE_GRAPH=1

{
  echo "date=$(date -Iseconds)"
  echo "cwd=$ROOT"
  echo "python=$PYTHON_BIN"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
  "$PYTHON_BIN" - <<'PY'
import flash_attn
import torch
import transformers
import triton
print(f"python={__import__('sys').version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"triton={triton.__version__}")
print(f"flash_attn={flash_attn.__version__}")
PY
} > "$ENV_FILE" 2>&1

cd "$ROOT"
"$PYTHON_BIN" "$STUDY_DIR/scripts/collect_ratio_profiles.py" \
  --ratios 0.5 0.6 0.7 0.8 \
  --gpu 5 \
  --dtype bf16 \
  --profile_prompt_len 512 \
  --profile_new_tokens 32 \
  --profile_decode_steps 16 \
  --warmup 3 \
  --example_decode_steps 64 \
  > "$LOG_FILE" 2>&1
