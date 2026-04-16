# ModernBERT Low-Rank Runtime Notes

Date: `2026-03-18`

This note tracks the corrected `low-rank eager -> FlashSVD` comparison for the ModernBERT-style synthetic encoder benchmark after the short-sequence runtime fix.

## What changed

- `FlashSVDRoPEAttention` now caches flattened `Vq/Vk/Vv`, flattened biases, and RoPE tables across repeated calls with the same tensors and position ids.
- `flashsvd_ffn_geglu_autotuned()` now supports an `eager` runtime variant and defaults to eager low-rank matmuls for short encoder sequences via `FLASH_SVD_GEGLU_EAGER_MAX_SEQ` (default `512`).
- `benchmark/legacy/archive/flashsvdgeglu/encoder_compare.py` now supports `--baseline lowrank`, so the benchmark no longer mixes `dense -> FlashSVD` and `low-rank eager -> FlashSVD`.

## Repro command

```bash
cd /home/zs89/FlashSVD/FlashSVD-v1.5
CUDA_VISIBLE_DEVICES=4 /home/zs89/miniconda3/envs/flashsvd/bin/python \
  benchmark/legacy/archive/flashsvdgeglu/encoder_compare.py \
  --baseline lowrank \
  --B 8 \
  --dtype bf16 \
  --target-param-ratio 0.5 \
  --ffn-variant auto
```

Length-specific runs used:

- `L=128,  --chunk-q 128`
- `L=512,  --chunk-q 256`
- `L=2048, --chunk-q 192`

The synthetic ModernBERT shape is:

- hidden size `768`
- heads `12`
- head dim `64`
- intermediate size `1152`
- target param ratio `0.5`
- ranks: attention `192`, FFN `r1=288`, `r2=224`

## Current results

| Seq Len | Attn Speedup vs Low-Rank | FFN Speedup vs Low-Rank | Layer Speedup vs Low-Rank |
| --- | ---: | ---: | ---: |
| 128  | `1.010x` | `0.965x` | `1.007x` |
| 512  | `1.022x` | `0.974x` | `1.019x` |
| 2048 | `1.944x` | `1.157x` | `1.530x` |

Raw timings:

| Seq Len | Low-Rank Attn (ms) | Flash Attn (ms) | Low-Rank FFN (ms) | Flash FFN (ms) | Low-Rank Layer (ms) | Flash Layer (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128  | `2.6862` | `2.6587` | `2.4677` | `2.5565` | `2.7687` | `2.7505` |
| 512  | `3.0475` | `2.9804` | `2.6018` | `2.6723` | `3.2504` | `3.1898` |
| 2048 | `8.7464` | `4.4999` | `3.5261` | `3.0489` | `12.2806` | `8.0263` |

## Takeaway

The previous concern was real: older measurements mixed outdated GEGLU behavior and a benchmark that only reported `dense -> FlashSVD`. After the runtime cache and short-sequence dispatch fix, the current encoder path is no longer slower than the low-rank eager baseline at `L=128/512`, and becomes clearly advantageous at long context (`L=2048`).
