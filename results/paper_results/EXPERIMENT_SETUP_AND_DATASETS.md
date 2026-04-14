# Experiment Setup And Datasets

## Scope

This paper bundle is derived from the existing runtime study at:

- [`results/runtime_study_2026-03-31`](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31)
- [`results/motivation/graph_fusion_2026-03-31`](/home/zs89/FlashSVD/FlashSVD-v1.5/results/motivation/graph_fusion_2026-03-31)

No new measurements were introduced while building `paper_results/`; this folder is a visualization and packaging layer on top of those frozen artifacts.

## Machine And Software

- Host cwd: `/home/zs89/FlashSVD`
- GPU: `A100 80GB`
- Main runtime study GPU: `CUDA_VISIBLE_DEVICES=5`
- Graph-fusion motivation study GPU: `CUDA_VISIBLE_DEVICES=7`
- Python env: `/home/zs89/miniconda3/envs/flashsvd15/bin/python`
- Python: `3.13.2`
- PyTorch: `2.7.1+cu128`
- Triton: `3.3.1`
- Transformers: `4.53.0`
- FlashAttention: `2.8.3`
- Git commit recorded in the runtime study: `c6a067b304b8a541b3f1c4f24d8bbe0ecbe21869`

## Model

- Checkpoint: `/home/zs89/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt`
- Compression family: uniform-rank `SVDLLM v1` style checkpoint
- Attention ranks: `Rq=Rk=Rv=1024` for all 32 layers
- MLP ranks: `Rgate=Rup=Rdown=1492` for all 32 layers
- Main inference dtype: `bf16`
- Batch size: `1`

## Runtime Definitions

- `StaticCache`:
  standard SVD runtime with HuggingFace-style static KV cache.
- `DenseKVCacheBaseline`:
  aligned dense-KV reference path with reference QKV reconstruct, external RoPE, and `flash_attn_with_kvcache`.
- `FlashSVD-v1.5`:
  packed rank projection + token reconstruct + internal RoPE + FA2 KV-cache decode + active per-layer CUDA graph.

Default FlashSVD-v1.5 runtime knobs used in the main study:

```bash
FLASH_SVD_DENSE_DECODE_BACKEND=packed
FLASH_SVD_DENSE_DECODE_GRAPH=1
```

Default benchmark flags:

```bash
--experimental_flash_dense_attn
--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod
--mlp_cuda_graph
--mlp_cuda_graph_scope layer_tail
```

## Datasets And Inputs

### 1. Latency And Ablation Benchmarks

The end-to-end latency measurements are not tied to a natural-text corpus. The decode benchmark uses synthetic token IDs generated with `torch.randint(...)` inside [`decode_kvcache_eval`](/home/zs89/FlashSVD/FlashSVD-v1.5/utils/evaluator.py), with fixed prompt lengths and decode lengths.

Primary settings:

- `prompt_len=512, new_tokens=32`
- `prompt_len=2048, new_tokens=128`

This is appropriate for systems benchmarking because the goal is to isolate serving-path runtime cost from dataset-specific tokenization or sampling behavior.

### 2. Correctness Audit Prompt Set

The correctness audit uses 20 manually curated short prompts. The prompt list is exported in:

- [correctness_prompts.csv](./tables/correctness_prompts.csv)

Gold reference:

- `fp32 no-cache` full recomputation

Correctness anchor:

- `fp32 StaticCache cached`, which matches the no-cache gold `20/20`

Serving-path comparison:

- `bf16 StaticCache`
- `bf16 DenseKVCacheBaseline`
- `bf16 FlashSVD-v1.5`

### 3. Kernel Microbenchmarks

The attention-route and kernel microbenchmarks use the real checkpoint weights and real layer ranks, but synthetic decode-shape inputs. They are best interpreted as operator-level latency studies, not text-generation quality experiments.

## Paper-Facing Usage

- Use `DenseKVCacheBaseline` as the main aligned performance baseline.
- Use `StaticCache` as the practical baseline and `fp32 no-cache` plus `fp32 StaticCache cached` as the correctness anchor.
- Use the graph-fusion figures to motivate why runtime thinness matters even when the underlying algorithm is unchanged.
