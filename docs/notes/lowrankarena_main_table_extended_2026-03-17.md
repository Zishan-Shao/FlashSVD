# LowRankArena Main Table Results (Extended)

Generated: 2026-03-17 19:36:16

## Benchmark Config

- GPU: `4`
- Device: `cuda:0`
- Dtype: `bf16`
- Prompt length: `512`
- New tokens: `32`
- Warmup: `3`
- Batch size: `1`
- FlashSVD MLP backend: `flashsvd_mlp_dual_split_prod`
- Compare mode: baseline dense-KV vs FlashSVD production packed path
- Added ratios: `0.7`, `0.8` for `SVD-LLM v1` and `SVD-LLM v2`; `Basis Sharing 0.7/0.8` are unavailable on Hugging Face

## Main Table

| Family | Ratio | Baseline Prefill (s) | FlashSVD Prefill (s) | Prefill Speedup | Baseline Decode (ms/token) | FlashSVD Decode (ms/token) | Decode Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVD-LLM v1 | 0.4 | 0.249 | 0.194 | 1.28x | 35.362 | 26.556 | 1.33x |
| SVD-LLM v1 | 0.5 | 0.103 | 0.093 | 1.10x | 36.877 | 26.369 | 1.40x |
| SVD-LLM v1 | 0.6 | 0.212 | 0.197 | 1.07x | 35.569 | 23.176 | 1.53x |
| SVD-LLM v1 | 0.7 | 0.328 | 0.190 | 1.72x | 34.901 | 22.573 | 1.55x |
| SVD-LLM v1 | 0.8 | 0.268 | 0.104 | 2.57x | 35.394 | 24.469 | 1.45x |
| SVD-LLM v2 | 0.4 | 0.071 | 0.072 | 1.00x | 35.408 | 23.332 | 1.52x |
| SVD-LLM v2 | 0.5 | 0.145 | 0.091 | 1.58x | 35.911 | 24.377 | 1.47x |
| SVD-LLM v2 | 0.6 | 0.163 | 0.150 | 1.09x | 34.506 | 22.987 | 1.50x |
| SVD-LLM v2 | 0.7 | 0.180 | 0.174 | 1.04x | 35.526 | 23.231 | 1.53x |
| SVD-LLM v2 | 0.8 | 0.105 | 0.105 | 1.01x | 34.776 | 23.735 | 1.47x |
| Basis Sharing | 0.4 | 0.105 | 0.076 | 1.38x | 34.939 | 23.351 | 1.50x |
| Basis Sharing | 0.5 | 0.067 | 0.106 | 0.63x | 34.507 | 23.840 | 1.45x |
| Basis Sharing | 0.6 | 0.084 | 0.116 | 0.73x | 35.426 | 23.337 | 1.52x |
| Basis Sharing | 0.7 | N/A | N/A | N/A | N/A | N/A | N/A |
| Basis Sharing | 0.8 | N/A | N/A | N/A | N/A | N/A | N/A |

## Detailed Throughput

| Family | Ratio | Baseline Prefill Tok/s | FlashSVD Prefill Tok/s | Baseline Decode Tok/s | FlashSVD Decode Tok/s | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SVD-LLM v1 | 0.4 | 2056 | 2636 | 28 | 38 | `/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.4/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.4_hf` |
| SVD-LLM v1 | 0.5 | 4988 | 5496 | 27 | 38 | `/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.5/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.5_hf` |
| SVD-LLM v1 | 0.6 | 2421 | 2594 | 28 | 43 | `/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.6/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.6_hf` |
| SVD-LLM v1 | 0.7 | 1562 | 2689 | 29 | 44 | `/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.7/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.7_hf` |
| SVD-LLM v1 | 0.8 | 1913 | 4920 | 28 | 41 | `/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.8/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.8_hf` |
| SVD-LLM v2 | 0.4 | 7163 | 7147 | 28 | 43 | `/home/zs89/FlashSVD/models/lowrankarena/svdllmv2_keep0.4/llama_7b/SVDLLMv2/jeffwan_llama_7b_hf_svdllmv2_keep0.4_hf` |
| SVD-LLM v2 | 0.5 | 3539 | 5599 | 28 | 41 | `/home/zs89/FlashSVD/models/lowrankarena/svdllmv2_keep0.5/llama_7b/SVDLLMv2/jeffwan_llama_7b_hf_svdllmv2_keep0.5_hf` |
| SVD-LLM v2 | 0.6 | 3137 | 3420 | 29 | 44 | `/home/zs89/FlashSVD/models/lowrankarena/svdllmv2_keep0.6/llama_7b/SVDLLMv2/jeffwan_llama_7b_hf_svdllmv2_keep0.6_hf` |
| SVD-LLM v2 | 0.7 | 2838 | 2943 | 28 | 43 | `/home/zs89/FlashSVD/models/lowrankarena/svdllmv2_keep0.7/llama_7b/SVDLLMv2/jeffwan_llama_7b_hf_svdllmv2_keep0.7_hf` |
| SVD-LLM v2 | 0.8 | 4864 | 4894 | 29 | 42 | `/home/zs89/FlashSVD/models/lowrankarena/svdllmv2_keep0.8/llama_7b/SVDLLMv2/jeffwan_llama_7b_hf_svdllmv2_keep0.8_hf` |
| Basis Sharing | 0.4 | 4861 | 6723 | 29 | 43 | `/home/zs89/FlashSVD/models/lowrankarena/basis_sharing_40/llama_7b/Basis_Sharing/share_llama-7b_40` |
| Basis Sharing | 0.5 | 7697 | 4842 | 29 | 42 | `/home/zs89/FlashSVD/models/lowrankarena/basis_sharing_50/llama_7b/Basis_Sharing/share_llama-7b_50` |
| Basis Sharing | 0.6 | 6080 | 4420 | 28 | 43 | `/home/zs89/FlashSVD/models/lowrankarena/basis_sharing_60/llama_7b/Basis_Sharing/share_llama-7b_60` |
| Basis Sharing | 0.7 | N/A | N/A | N/A | N/A | `Checkpoint unavailable in Duke-CEI-SVD/LowRankArena` |
| Basis Sharing | 0.8 | N/A | N/A | N/A | N/A | `Checkpoint unavailable in Duke-CEI-SVD/LowRankArena` |

## Availability Notes

- `Basis Sharing 0.7` and `Basis Sharing 0.8` are not present in `Duke-CEI-SVD/LowRankArena`, so those cells are marked `N/A`.

_Saved to `/home/zs89/FlashSVD/FlashSVD-v1.5/notes/lowrankarena_main_table_extended_2026-03-17.md`._
