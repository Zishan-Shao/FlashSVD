## FlashSVD rope-attn comparisons

This folder contains a small harness to compare **aligned** end-to-end performance and correctness for:

1) **Baseline (unfused)**: `flashsvdropeattn_baseline.py` reconstructs dense Q/K/V from low-rank factors, applies RoPE, then runs causal FlashAttention.
2) **`flashsvdropeattn_v1.5.py` (FA-aligned packed)**: `flashsvd_rope_fwd_packed_R` via `flashsvd_attn_packed`.
3) **`flashsvdropeattn_v1.py` (BMHd)**: `flashsvd_rope_sdpa` kernel (called directly, with prebuilt cos/sin).

### Quick run (example)

From repo root:

```bash
python kernels/flashsvd-v1.5/flashsvdropeattn/compare.py \
  --B 8 --S 2048 --H 32 --Hk 8 --Dh 128 --R 64 --dtype bf16 --causal \
  --warmup 50 --iters 200
```

### Notes

- The baseline uses `kernels/flash_attn_causal.py` (`flash_attn_triton`) for the attention step.
- For correctness checks, use a small `--S` (e.g. `<=256`) to enable the fp32 reference.
- `flashsvdropeattn_v1.5.py` is loaded by file path because its filename contains `.`.

### Decode microbench (q_len=1)

`decode_compare.py` compares single-step decode attention (`q_len=1`, `kv_len=L`) for:

- Dense KV-cache (FA2 / Triton / torch)
- Low-rank KV-cache streaming (PyTorch online softmax; no full K/V materialization inside the timed region)
- Low-rank KV-cache fused Triton decode (RoPE + split-K) via `flashsvdropeattn_v1.5_decode.py`

Example:

```bash
python kernels/flashsvd-v1.5/flashsvdropeattn/decode_compare.py \
  --B 8 --L 2048 --H 32 --Hk 8 --Dh 128 --R 64 --dtype bf16 \
  --dense-backend auto --bn 128 --split-k 512 --br 64 --warmup 50 --iters 200
```

Sweep short/mid/long contexts:

```bash
python kernels/flashsvd-v1.5/flashsvdropeattn/decode_compare.py \
  --B 8 --Ls 256,2048,8192 --H 32 --Hk 8 --Dh 128 --R 64 --dtype bf16 \
  --dense-backend auto --bn 128 --split-k 512 --br 64 --warmup 50 --iters 200
```

Tune fused decode blocking (helpful when long-context is close to FA2):

```bash
python kernels/flashsvd-v1.5/flashsvdropeattn/decode_compare.py \
  --B 8 --L 8192 --H 32 --Hk 8 --Dh 128 --R 64 --dtype bf16 \
  --dense-backend fa2 --no-stream --fused-tune --br 64
```

### LLaMA/global-rank recipes (recommended)

Same-input attention comparison (closer to real decode path):

```bash
python kernels/flashsvd-v1.5/flashsvdropeattn/decode_compare.py \
  --llama llama2-7b \
  --target-param-ratio 0.5 \
  --rank-formula global \
  --rank-round-multiple 64 \
  --B 4 --L 2048 --dtype bf16 \
  --no-fused-vk-ablation \
  --realistic-attn
```

Same-input + equal KV-budget throughput comparison (auto scales low-rank batch by `Dh/R`):

```bash
python kernels/flashsvd-v1.5/flashsvdropeattn/decode_compare.py \
  --llama llama2-7b \
  --target-param-ratio 0.5 \
  --rank-formula global \
  --rank-round-multiple 64 \
  --B 4 --L 2048 --dtype bf16 \
  --no-fused-vk-ablation \
  --compare-kv-budget \
  --realistic-attn
```

Batch sweep (manual), useful for scheduler design:

```bash
python kernels/flashsvd-v1.5/flashsvdropeattn/decode_compare.py \
  --llama llama2-7b \
  --target-param-ratio 0.5 \
  --rank-formula global \
  --rank-round-multiple 64 \
  --Bs 1,2,4,8,16,32,64,128 --L 2048 --dtype bf16 \
  --no-fused-vk-ablation \
  --realistic-attn
```

### Tricks log (decode_compare)

- Rank semantics:
  - `--rank-formula global` means infer total rank for full `D x D` factorization, then map to per-head kernel rank.
  - `--R-total` can be passed directly when you already know total rank.
- Realistic attention path:
  - Use `--realistic-attn` to disable the Python streaming baseline and focus on dense FA vs fused low-rank kernels.
- V2 safety:
  - V2 has auto safety knobs for high-rank and small `rep` (e.g. `pad_to_16=False` when `rep<=2`).
  - If v2 is slower, prefer `lowrank_fused`/`lowrank_fused_v1` as serving candidates.
- Serving kernel dispatch (SVD LLaMA):
  - `flashsvd_component/svd_llama.py` now supports decode variant dispatch across `v1.5`, `v1.6_v1`, and `v1.6_v2`.
  - Default is `FLASH_SVD_DECODE_KERNEL_VARIANT=auto` with a stable heuristic (online variant autotune is opt-in).
  - Override manually with `FLASH_SVD_DECODE_KERNEL_VARIANT={v15|v16_v1|v16_v2}`.
  - Optional batch map: `FLASH_SVD_DECODE_KERNEL_MAP="1:v16_v1,2:v16_v1,8:v15,16:v16_v1,32:v15,64:v15,128:v16_v1"`.
  - Enable online variant autotune only when needed: `FLASH_SVD_DECODE_KERNEL_AUTOTUNE=1`.
- Memory measurement:
  - Per-variant CUDA memory reset is enabled by default.
  - Output now includes both `peak_delta_*` (incremental) and absolute `peak_*`.
  - Use `--no-mem-reset` only if you intentionally want allocator-cached behavior.
- Recompile control:
  - Keep `split_k/bn/br/warps/stages/dtype` fixed across runs.
  - Keep cache shape fixed when benchmarking: in `SVDLLM_flashsvd.py --step 6`, you can pin `--max_cache_len`.
  - For serving stability, prefer `FLASH_SVD_DECODE_KERNEL_AUTOTUNE=0` and fixed `FLASH_SVD_DECODE_KERNEL_VARIANT`.
