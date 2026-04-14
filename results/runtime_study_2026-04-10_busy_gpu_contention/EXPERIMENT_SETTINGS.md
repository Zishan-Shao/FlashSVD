# FlashSVD-v1.5 Busy-GPU Contention Study

Date: 2026-04-10

## Goal

This study investigates why `FlashSVD-v1.5` can look much slower on a shared GPU:

- is `CUDA graph` failing or being bypassed?
- or is graph still active, while GPU-side work is delayed by contention?

## Machine / Software

- Host cwd: `/home/zs89/FlashSVD`
- GPUs: `NVIDIA A100-SXM4-80GB`
- Compute mode: `Default` on all visible GPUs
- Python env: `/home/zs89/miniconda3/envs/flashsvdv15/bin/python`
- Python: `3.13.5`
- PyTorch: `2.9.1+cu128`
- Triton: `3.5.1`
- Transformers: `4.57.6`
- Git base commit: `c6a067b304b8a541b3f1c4f24d8bbe0ecbe21869`

Important local state:

- this study was run on the current working tree after:
  - removing `flashinfer` from the `FlashSVD-v1.5` mainline
  - fixing `profile_decode=True` to use the same FlashSVD decode step helper as the main decode benchmark path

## Model / Checkpoint

- Checkpoint source:
  `namespace/repo/path/to/export`
- Local cache:
  `/home/zs89/FlashSVD/checkpoints/hf_exports/<cache-entry>`
- Model family: `SVDLLM v1`
- Rank shape:
  - attention rank `R=1024`
  - MLP rank `R=1492`
- Main dtype: `bf16`
- Batch size: `1`

## Runtime Knobs

Unless otherwise stated, FlashSVD runs use:

```bash
FLASH_SVD_DENSE_DECODE_BACKEND=packed
FLASH_SVD_DENSE_DECODE_GRAPH=1
```

Main FlashSVD flags:

```bash
--experimental_flash_dense_attn
--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod
--mlp_cuda_graph
--mlp_cuda_graph_scope layer_tail
```

## GPU State Snapshot

Representative snapshot while the study was running:

```text
0, NVIDIA A100-SXM4-80GB, Default, 17227 MiB, 100 %
1, NVIDIA A100-SXM4-80GB, Default, 19147 MiB, 100 %
2, NVIDIA A100-SXM4-80GB, Default, 74311 MiB, 100 %
3, NVIDIA A100-SXM4-80GB, Default, 50869 MiB, 0 %
4, NVIDIA A100-SXM4-80GB, Default, 16715 MiB, 92 %
5, NVIDIA A100-SXM4-80GB, Default, 24619 MiB, 98 %
6, NVIDIA A100-SXM4-80GB, Default, 16203 MiB, 100 %
7, NVIDIA A100-SXM4-80GB, Default, 17227 MiB, 90 %
```

Representative `pmon` sample during the “busy GPU” phase:

```text
GPU 1: python at ~96% SM
GPU 5: python at ~2% SM
```

So:

- `GPU 1` is a deliberately busy-card measurement
- `GPU 5` is only relatively cleaner, not a perfectly isolated card

## Experiment Matrix

### 1. Busy-GPU Module Profile

Busy card: `CUDA_VISIBLE_DEVICES=1`

Variants:

- `DenseKVCacheBaseline`
- `FlashSVD-v1.5`

Settings:

- `prompt_len=64`
- `new_tokens=8`
- `warmup=1`
- `profile_decode_steps=8`

### 2. Host-vs-GPU Decode Split

For `FlashSVD-v1.5`, measure one-token decode with:

- host wall-clock time around the decode call
- CUDA-event elapsed time for the same decode call

Variants:

- busy `GPU 1`, `graph_on`
- busy `GPU 1`, `graph_off`
- cleaner `GPU 5`, `graph_on`

Settings:

- `prompt_len=64`
- 3 warmup decode tokens
- 12 timed decode tokens

### 3. Short Nsight Systems Trace

Busy card: `CUDA_VISIBLE_DEVICES=1`

Settings:

- `prompt_len=32`
- `new_tokens=4`
- `warmup=1`
- trace: `cuda,osrt`

Caveat:

- this trace includes model load and prefill, so it is not a pure steady-state decode trace

## Interpretation Rules

We use the following logic:

- if graph were being bypassed, host submit time with `graph_on` would remain close to `graph_off`
- if graph still works but GPU is contended, host submit time stays low while GPU elapsed grows

That is the main decision rule behind this study.
