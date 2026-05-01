#!/usr/bin/env bash
set -u -o pipefail

GPU_ID="${1:-3}"
ROOT_DIR="/home/zs89/FlashSVD"
PYTHON_BIN="/home/zs89/miniconda3/envs/flashsvdv15/bin/python"
BENCH_SCRIPT="$ROOT_DIR/benchmark/decode/bench_flashsvd_vs_svd_decode.py"
RUN_TAG="${RUN_TAG:-runtime_rerun_2026-04-14_ratio_sweep_v1_perlayer_retry}"
OUTDIR="${OUTDIR:-$ROOT_DIR/results/$RUN_TAG}"
RAW_DIR="$OUTDIR/raw"
POLL_SECONDS="${POLL_SECONDS:-10}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-20}"
LOWRANKARENA_SOURCE_PREFIX="${LOWRANKARENA_SOURCE_PREFIX:-LowRankArena::}"

mkdir -p "$RAW_DIR"

wait_for_gpu_free() {
  while true; do
    local pids
    pids="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sed '/^$/d')"
    if [ -z "$pids" ]; then
      return 0
    fi
    echo ">>> WAIT $(date '+%F %T') gpu=${GPU_ID} busy_pids=${pids//$'\n'/,}" | tee -a "$OUTDIR/driver.log"
    sleep "$POLL_SECONDS"
  done
}

run_case() {
  local ratio="$1"
  local baseline="$2"
  local prompt_len="$3"
  local new_tokens="$4"
  local src="${LOWRANKARENA_SOURCE_PREFIX}llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_${ratio}_hf"
  local baseline_flag=""
  local log="$RAW_DIR/ratio_${ratio}_${baseline}_p${prompt_len}_n${new_tokens}.log"
  local attempt=1

  if [ "$baseline" = "densekv" ]; then
    baseline_flag="--baseline_dense_kvcache"
  else
    baseline_flag="--no-baseline_dense_kvcache"
  fi

  while true; do
    wait_for_gpu_free
    echo ">>> START $(date '+%F %T') ratio=${ratio} baseline=${baseline} prompt=${prompt_len} new_tokens=${new_tokens} attempt=${attempt}" | tee -a "$log" "$OUTDIR/driver.log"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    FLASH_SVD_TRUST_PICKLE=1 \
    FLASH_SVD_DENSE_DECODE_BACKEND=packed \
    FLASH_SVD_DENSE_DECODE_GRAPH=1 \
    PYTHONPATH="$ROOT_DIR" \
    "$PYTHON_BIN" "$BENCH_SCRIPT" \
      --checkpoint "$src" \
      --dtype bf16 \
      --device cuda \
      --prompt_len "$prompt_len" \
      --new_tokens "$new_tokens" \
      --warmup 3 \
      --batch_size 1 \
      --flashsvd_ffn_backend flashsvd_mlp_dual_split_prod \
      --experimental_flash_dense_attn \
      --mlp_cuda_graph \
      --mlp_cuda_graph_scope layer_tail \
      $baseline_flag 2>&1 | tee -a "$log"
    local status="${PIPESTATUS[0]}"
    echo ">>> EXIT_CODE=${status}" | tee -a "$log" "$OUTDIR/driver.log"
    if [ "$status" -eq 0 ]; then
      return 0
    fi
    if rg -q "CUDA-capable device\\(s\\) is/are busy or unavailable" "$log"; then
      echo ">>> RETRY $(date '+%F %T') ratio=${ratio} baseline=${baseline} prompt=${prompt_len} new_tokens=${new_tokens} reason=gpu_busy" | tee -a "$log" "$OUTDIR/driver.log"
      attempt=$((attempt + 1))
      sleep "$COOLDOWN_SECONDS"
      continue
    fi
    return "$status"
  done
}

echo ">>> DRIVER_START $(date '+%F %T') gpu=${GPU_ID} outdir=${OUTDIR}" | tee -a "$OUTDIR/driver.log"

for spec in "512 32" "2048 128"; do
  read -r prompt_len new_tokens <<< "$spec"
  for ratio in 0.4 0.5 0.6 0.7 0.8; do
    for baseline in densekv static; do
      run_case "$ratio" "$baseline" "$prompt_len" "$new_tokens" || {
        echo ">>> DRIVER_ABORT $(date '+%F %T') ratio=${ratio} baseline=${baseline} prompt=${prompt_len} new_tokens=${new_tokens}" | tee -a "$OUTDIR/driver.log"
        exit 1
      }
    done
  done
done

echo ">>> DRIVER_DONE $(date '+%F %T')" | tee -a "$OUTDIR/driver.log"
