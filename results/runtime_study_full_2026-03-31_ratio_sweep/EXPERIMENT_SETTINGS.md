# Ratio Sweep Runtime Study

Date: 2026-03-31

## Goal

This study extends the existing `0.5` FlashSVD-v1.5 runtime recipe to SVD-LLM v1 checkpoints at ratios:

- `0.5`
- `0.6`
- `0.7`
- `0.8`

The purpose is to measure how the active `FlashSVD-v1.5` serving stack scales with compression ratio, using the same end-to-end benchmark recipe as the main `0.5` study.

## Machine / Software

- Host cwd: `/home/zs89/FlashSVD`
- GPU target: `CUDA_VISIBLE_DEVICES=5`
- Python env: `/home/zs89/miniconda3/envs/flashsvd15/bin/python`
- Python: `3.13.2`
- PyTorch: `2.7.1+cu128`
- Triton: `3.3.1`
- Transformers: `4.53.0`
- FlashAttention: `2.8.3`

## Runtime Recipe

FlashSVD runtime knobs:

```bash
FLASH_SVD_DENSE_DECODE_BACKEND=packed
FLASH_SVD_DENSE_DECODE_GRAPH=1
```

Benchmark flags:

```bash
--experimental_flash_dense_attn
--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod
--mlp_cuda_graph
--mlp_cuda_graph_scope layer_tail
```

## Benchmarked Checkpoints

- `0.5`: `/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.5/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.5_hf`
- `0.6`: `/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.6/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.6_hf`
- `0.7`: `/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.7/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.7_hf`
- `0.8`: `/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.8/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.8_hf`

## Benchmark Matrix

For each ratio:

- Baselines:
  - `StaticCache`
  - `DenseKVCacheBaseline`
- Target:
  - `FlashSVD-v1.5`
- Configurations:
  - `prompt_len=512, new_tokens=32`
  - `prompt_len=2048, new_tokens=128`
- Repeats:
  - `n=5` timed runs per mode/configuration
- Warmup:
  - `warmup=3`
  - plus one untimed compile/burn-in run per mode/configuration because this sweep harness reuses a loaded model instance

## Notes

- All checkpoints are loaded from their exported HuggingFace local directories for consistency across ratios.
- This study focuses on repeated end-to-end latency only. It does not re-run the full profiler stack per ratio.
- The absolute `0.5` numbers here should not be mixed directly with the earlier `.pt`-checkpoint runtime study. This sweep is internally apples-to-apples because all four ratios use the same exported-HF checkpoint format, runtime recipe, and GPU.
