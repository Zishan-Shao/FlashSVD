# DobiSVD Ratio Sweep

Date: 2026-03-17

## Setup

- Checkpoints:
  - `Qinsi1/DobiSVD_Noremapping-Llama-2-7b-hf-0.4`
  - `Qinsi1/DobiSVD_Noremapping-Llama-2-7b-hf-0.6`
  - `Qinsi1/DobiSVD_Noremapping-Llama-2-7b-hf-0.8`
- Hardware: `A100`
- Config: `bf16`, `batch=1`, `prompt_len=256`, `new_tokens=16`, `warmup=1`
- Harness:
  - default comparison: `bench_flashsvd_vs_svd_decode.py`
  - baseline = low-rank baseline path
  - FlashSVD = current integrated Dobi support

## Layer-0 structure

The public `noremapping` checkpoints are not equally regular:

| Ratio | Attention ranks `(Rq, Rk, Rv, Ro)` | MLP ranks `(Rgate, Rup, Rdown)` |
| --- | --- | --- |
| `0.4` | `(700, 826, 732, 760)` | `(1382, 1292, 1162)` |
| `0.6` | `(1204, 1206, 1294, 1206)` | `(1774, 1778, 1806)` |
| `0.8` | `(1628, 1628, 1628, 1630)` | `(2378, 2378, 2374)` |

This helps explain why the realized speedup is much smaller than SVD-LLM v1/v2:

- `0.4` and `0.6` still have mismatched attention ranks
- `0.8` finally has shared `Rq = Rk = Rv`, but MLP is still not fully uniform

## Main results

| Ratio | Baseline Prefill (s) | FlashSVD Prefill (s) | Prefill Speedup | Baseline Decode (ms/token) | FlashSVD Decode (ms/token) | Decode Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.4` | `0.217` | `0.052` | `4.17x` | `31.049` | `28.121` | `1.10x` |
| `0.6` | `0.267` | `0.044` | `6.07x` | `31.491` | `29.613` | `1.06x` |
| `0.8` | `0.261` | `0.158` | `1.65x` | `32.787` | `28.327` | `1.16x` |

## Extra check: production dense-KV path on 0.8

Because `0.8` has shared attention ranks (`Rq = Rk = Rv = 1628`), it becomes eligible for FlashSVD's dense-KV decode path.

Command:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zs89/FlashSVD/FlashSVD-v1.5 \
python /home/zs89/FlashSVD/FlashSVD-v1.5/benchmark/decode/bench_flashsvd_vs_svd_decode.py \
  --checkpoint /home/zs89/FlashSVD/checkpoints/dobisvd/dobisvd_0.8/DobiSVD_Model.pt \
  --device cuda:0 \
  --dtype bf16 \
  --prompt_len 256 \
  --new_tokens 16 \
  --batch_size 1 \
  --warmup 1 \
  --experimental_flash_dense_attn
```

Observed output:

- baseline decode: `31.815 ms/token`
- FlashSVD decode: `28.010 ms/token`
- speedup: `1.14x`

The runtime does switch to `FlashSVDV15DenseKVCache` on the FlashSVD side, but the end-to-end gain is still modest. This suggests that even when Dobi becomes more attention-friendly at high keep ratios, the overall structure is still less aligned with FlashSVD's strongest fast paths than SVD-LLM v1/v2.

## Takeaway

The Dobi ratio sweep supports three conclusions:

1. FlashSVD v1.5 supports all three public Dobi non-remapping checkpoints (`0.4 / 0.6 / 0.8`) end-to-end.
2. The support is stable and decode remains a positive gain across all three ratios.
3. The gain stays modest (`~1.06x - 1.16x`) because Dobi's non-uniform factorization does not match the shared-rank assumptions behind FlashSVD's largest kernel wins.

This makes Dobi a useful “generalization” result, but not the strongest “speedup” result.
