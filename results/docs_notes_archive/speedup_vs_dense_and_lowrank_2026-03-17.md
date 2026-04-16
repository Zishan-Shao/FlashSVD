# FlashSVD Speedup vs Dense and Low-Rank

Generated: 2026-03-17 19:43:09

## Config

- GPU: `4`
- Device: `cuda:0`
- Dtype: `bf16`
- Prompt length: `512`
- New tokens: `32`
- Batch size: `1`
- FlashSVD MLP backend: `flashsvd_mlp_dual_split_prod`

## Dense Baseline

- Dense prefill: `0.429 s`
- Dense decode: `26.864 ms/token`
- Dense end-to-end: `1.289 s` (`prefill + 32-token decode`)

## Average Speedups

| Compare Against | Prefill | Decode | End-to-End |
| --- | ---: | ---: | ---: |
| Low-rank baseline | 1.25x | 1.48x | 1.44x |
| Dense baseline | 3.76x | 1.12x | 1.45x |

## Per Checkpoint

| Family | Ratio | vs Low-rank Prefill | vs Low-rank Decode | vs Low-rank End-to-End | vs Dense Prefill | vs Dense Decode | vs Dense End-to-End |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVD-LLM v1 | 0.4 | 1.28x | 1.33x | 1.32x | 2.21x | 1.01x | 1.23x |
| SVD-LLM v1 | 0.5 | 1.10x | 1.40x | 1.37x | 4.60x | 1.02x | 1.38x |
| SVD-LLM v1 | 0.6 | 1.07x | 1.53x | 1.44x | 2.17x | 1.16x | 1.37x |
| SVD-LLM v1 | 0.7 | 1.72x | 1.55x | 1.58x | 2.25x | 1.19x | 1.41x |
| SVD-LLM v1 | 0.8 | 2.57x | 1.45x | 1.58x | 4.12x | 1.10x | 1.45x |
| SVD-LLM v2 | 0.4 | 1.00x | 1.52x | 1.47x | 5.99x | 1.15x | 1.57x |
| SVD-LLM v2 | 0.5 | 1.58x | 1.47x | 1.48x | 4.69x | 1.10x | 1.48x |
| SVD-LLM v2 | 0.6 | 1.09x | 1.50x | 1.43x | 2.87x | 1.17x | 1.46x |
| SVD-LLM v2 | 0.7 | 1.04x | 1.53x | 1.44x | 2.47x | 1.16x | 1.40x |
| SVD-LLM v2 | 0.8 | 1.01x | 1.47x | 1.41x | 4.10x | 1.13x | 1.49x |
| Basis Sharing | 0.4 | 1.38x | 1.50x | 1.49x | 5.63x | 1.15x | 1.56x |
| Basis Sharing | 0.5 | 0.63x | 1.45x | 1.35x | 4.06x | 1.13x | 1.48x |
| Basis Sharing | 0.6 | 0.73x | 1.52x | 1.41x | 3.70x | 1.15x | 1.49x |

## Notes

- `End-to-End` is computed as `prefill_time_s + decode_ms_per_token * 32 / 1000`.
- `Dense` uses `jeffwan/llama-7b-hf` measured once under the same config.
- `Low-rank baseline` uses the baseline column from the LowRankArena main table (`dense-KV baseline` for the low-rank checkpoints).
