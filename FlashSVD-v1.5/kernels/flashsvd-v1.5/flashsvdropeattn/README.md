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
