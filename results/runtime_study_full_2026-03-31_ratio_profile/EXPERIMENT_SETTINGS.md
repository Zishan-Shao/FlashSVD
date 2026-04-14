# Ratio Profile Study

Date: 2026-03-31

## Goal

Collect ratio-wise profiling beyond end-to-end speed for SVD-LLM v1 exported checkpoints at ratios:

- 0.5, 0.6, 0.7, 0.8

The study emphasizes:

- module-wise decode timing
- op-level CPU/CUDA profiler output
- decode-path launch and staging overhead
- graph-fragmentation counts for FlashSVD runtime variants
- qualitative greedy decode examples for paper figures or appendix

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

Default active path:

```bash
FLASH_SVD_DENSE_DECODE_BACKEND=packed
FLASH_SVD_DENSE_DECODE_GRAPH=1
--experimental_flash_dense_attn
--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod
--mlp_cuda_graph
--mlp_cuda_graph_scope layer_tail
```

## Profiling Matrix

For each ratio and each runtime mode (`StaticCache`, `DenseKVCacheBaseline`, `FlashSVD-v1.5`):

- module-wise decode profile on `prompt_len=512`, `new_tokens=32`, `profile_decode_steps=16`
- op-level `torch.profiler` for:
  - prefill
  - decode

For each ratio and FlashSVD only:

- graph fragmentation profile on variants:
  - `nograph`
  - `split`
  - `layer`

Greedy decode examples:

- prompts: 3
- generated tokens per prompt: `64`
