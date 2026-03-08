# FlashSVD SiLU / SwiGLU Decode Kernels

This folder is the readable landing zone for the current MLP decode work.

## What is active

- Active exact dual-split Triton kernels live in:
  - [`./dual_split_exact.py`](./dual_split_exact.py)
- The main component dispatch is in:
  - [`../../flashsvd_component/svd_llama.py`](../../flashsvd_component/svd_llama.py)

## Backends to care about

- `baseline`
  - original exact SVD-Llama MLP path, no FlashSVD FFN optimization
- `dual_split_kernel`
  - first exact Triton token kernel (`v1`)
- `dual_split_kernel_v2`
  - exact token kernel with direct `S` accumulation
- `dual_split_kernel_v2_sm80`
  - current SM80/A100-oriented exact token kernel (`latest Triton`)
- `dual_split_cublas_legacy`
  - current exact-safe serving default
- `dual_split_cublas`
  - packed exact cuBLAS backend

## Recommended benchmark

Use [`compare_exact_backends.py`](./compare_exact_backends.py) to compare:

- unoptimized exact baseline
- v1 exact kernel
- latest exact Triton kernel
- current exact cuBLAS backends

Example:

```bash
CUDA_VISIBLE_DEVICES=5 \
/home/zs89/miniconda3/envs/flashsvd/bin/python \
/home/zs89/FlashSVD/FlashSVD-v1.5/kernels/flashsvdsilu/compare_exact_backends.py \
--checkpoint /home/zs89/FlashSVD/FlashSVD-v1.5/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
--dtype bf16 \
--device cuda \
--layer 18 \
--batch_size 1 \
--seq_len 1 \
--warmup 20 \
--iters 100
```

For all-layer sweep:

```bash
CUDA_VISIBLE_DEVICES=5 \
/home/zs89/miniconda3/envs/flashsvd/bin/python \
/home/zs89/FlashSVD/FlashSVD-v1.5/kernels/flashsvdsilu/compare_exact_backends.py \
--checkpoint /home/zs89/FlashSVD/FlashSVD-v1.5/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
--dtype bf16 \
--device cuda \
--batch_size 1 \
--seq_len 1 \
--warmup 20 \
--iters 80 \
--all_layers
```

For end-to-end decode backend A/B:

```bash
CUDA_VISIBLE_DEVICES=5 \
/home/zs89/miniconda3/envs/flashsvd/bin/python \
/home/zs89/FlashSVD/FlashSVD-v1.5/kernels/flashsvdsilu/compare_decode_backends.py \
--checkpoint /home/zs89/FlashSVD/FlashSVD-v1.5/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
--dtype bf16 \
--device cuda \
--prompt_len 512 \
--new_tokens 128 \
--warmup 3 \
--batch_size 1
```

## Layout note

- Exact dual-split serving code now lives under this folder.
- Generic SwiGLU and shared-split experiments now live in:
  - [`../flashsvdswiglu/generic.py`](../flashsvdswiglu/generic.py)
  - [`../flashsvdswiglu/shared_split.py`](../flashsvdswiglu/shared_split.py)
