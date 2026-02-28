## FlashSVD attention (mask-friendly, BHMR layout)

This folder contains rank-aware Flash-SVD attention with 4D mask support:

1) **flashsvdattn_v1.5.py**: Optimized low-rank kernel (FA-aligned).
2) **flashsvdattn_v1.py**: Baseline low-rank kernel.
3) **compare.py**: Benchmark harness comparing dense baseline vs FlashSVD variants.

### Quick run (example)

From repo root:

```bash
python kernels/flashsvd-v1.5/flashsvdattn/compare.py \
  --B 8 --M 512 --d-model 768 --H 12 --R 64 --dtype fp16 \
  --warmup 20 --iters 100
```

### Correctness check

```bash
python kernels/flashsvd-v1.5/flashsvdattn/compare.py \
  --B 4 --M 128 --R 64 --check
```

### Direct module test

Each module has a built-in `if __name__ == "__main__"` test. Run from repo root:

```bash
python kernels/flashsvd-v1.5/flashsvdattn/flashsvdattn_v1.5.py
python kernels/flashsvd-v1.5/flashsvdattn/flashsvdattn_v1.py
```

If you have another `kernels` package installed (e.g. FlashSVD), ensure repo root is first in `PYTHONPATH` or the scripts will add it automatically.

### Notes

- Requires `kernels.utils_mask` (Triton kernel `_demo_attn_kernel`).
- Layout: BHMR (batch, heads, seq, rank) for P; mask is [B,1,1,M] or [B,H,M].
