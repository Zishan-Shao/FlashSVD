# LowRankArena Main Table Results

Generated: 2026-03-17 19:35:36

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

## Main Table

| Family | Ratio | Baseline Prefill (s) | FlashSVD Prefill (s) | Prefill Speedup | Baseline Decode (ms/token) | FlashSVD Decode (ms/token) | Decode Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVD-LLM v1 | 0.7 | 0.328 | 0.190 | 1.72x | 34.901 | 22.573 | 1.55x |
| SVD-LLM v1 | 0.8 | 0.268 | 0.104 | 2.57x | 35.394 | 24.469 | 1.45x |
| SVD-LLM v2 | 0.7 | 0.180 | 0.174 | 1.04x | 35.526 | 23.231 | 1.53x |
| SVD-LLM v2 | 0.8 | 0.105 | 0.105 | 1.01x | 34.776 | 23.735 | 1.47x |

## Detailed Throughput

| Family | Ratio | Baseline Prefill Tok/s | FlashSVD Prefill Tok/s | Baseline Decode Tok/s | FlashSVD Decode Tok/s | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SVD-LLM v1 | 0.7 | 1562 | 2689 | 29 | 44 | `/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.7/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.7_hf` |
| SVD-LLM v1 | 0.8 | 1913 | 4920 | 28 | 41 | `/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.8/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.8_hf` |
| SVD-LLM v2 | 0.7 | 2838 | 2943 | 28 | 43 | `/home/zs89/FlashSVD/models/lowrankarena/svdllmv2_keep0.7/llama_7b/SVDLLMv2/jeffwan_llama_7b_hf_svdllmv2_keep0.7_hf` |
| SVD-LLM v2 | 0.8 | 4864 | 4894 | 29 | 42 | `/home/zs89/FlashSVD/models/lowrankarena/svdllmv2_keep0.8/llama_7b/SVDLLMv2/jeffwan_llama_7b_hf_svdllmv2_keep0.8_hf` |

_Saved to `/home/zs89/FlashSVD/FlashSVD-v1.5/notes/lowrankarena_main_table_0.7_0.8_2026-03-17.md`._
