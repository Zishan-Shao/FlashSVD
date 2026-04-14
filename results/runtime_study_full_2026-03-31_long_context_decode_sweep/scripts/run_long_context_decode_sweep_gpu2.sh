#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zs89/FlashSVD"
STUDY_DIR="$ROOT/FlashSVD-v1.5/results/runtime_study_full_2026-03-31_long_context_decode_sweep"
PYTHON_BIN="/home/zs89/miniconda3/envs/flashsvd15/bin/python"
LOG_DIR="$STUDY_DIR/raw"
LOG_FILE="$LOG_DIR/run_gpu2.log"
ENV_FILE="$LOG_DIR/env_gpu2.txt"

mkdir -p "$STUDY_DIR/raw" "$STUDY_DIR/tables" "$STUDY_DIR/examples"
rm -f "$STUDY_DIR/raw"/* "$STUDY_DIR/tables"/* "$STUDY_DIR/examples"/*

export CUDA_VISIBLE_DEVICES=2
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
"$PYTHON_BIN" "$STUDY_DIR/scripts/run_long_context_decode_sweep.py" \
  --ratios 0.8 0.7 0.6 0.5 \
  --gpu 2 \
  --dtype bf16 \
  --stage_new_tokens 128 \
  --stage_repeats 5 \
  --decode_prompt_len 512 \
  --decode_repeats 2 \
  --warmup 3 \
  > "$LOG_FILE" 2>&1
