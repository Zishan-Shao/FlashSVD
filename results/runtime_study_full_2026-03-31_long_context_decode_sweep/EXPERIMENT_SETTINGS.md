# Long-Context Decode Sweep

Date: 2026-03-31

## Goal

This study extends the earlier ratio/runtime sweep in two directions:

1. Long-context stage study with larger prompt lengths while still recording both prefill and decode.
2. Decode-length sweep with fixed prompt length and `new_tokens` ranging from `64` to `16384`.

## Machine / Software

- Host cwd: `/home/zs89/FlashSVD`
- GPU target: `CUDA_VISIBLE_DEVICES=2`
- Python env: `/home/zs89/miniconda3/envs/flashsvd15/bin/python`
- Python: `3.13.2`
- PyTorch: `2.7.1+cu128`
- Triton: `3.3.1`
- Transformers: `4.53.0`
- FlashAttention: `2.8.3`

## Runtime Recipe

```bash
FLASH_SVD_DENSE_DECODE_BACKEND=packed
FLASH_SVD_DENSE_DECODE_GRAPH=1
--experimental_flash_dense_attn
--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod
--mlp_cuda_graph
--mlp_cuda_graph_scope layer_tail
```

## Stage Study

- Ratios: 0.8, 0.7, 0.6, 0.5
- Baselines / target: `StaticCache`, `DenseKVCacheBaseline`, `FlashSVD-v1.5`
- Configurations:
  - `prompt_len=4096, new_tokens=128`
  - `prompt_len=8192, new_tokens=128`
- Repeats: `5` timed runs per ratio/mode/config after one burn-in run

## Decode-Length Sweep

- Fixed prompt length: `prompt_len=512`
- `new_tokens`: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
- Repeats: `2` timed runs per ratio/mode/length after one burn-in run

## Decode Examples

- Mode: `FlashSVD-v1.5`
- Prompt: `Reducing kernel launch overhead helps autoregressive decoding because`
- Example lengths: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
