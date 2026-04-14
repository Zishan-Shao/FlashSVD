# FlashSVD-v1.5 Runtime Study

Date: 2026-03-31

## Goal

This study measures FlashSVD-v1.5 runtime behavior against two end-to-end SVD baselines:

- `StaticCache`
- `DenseKVCacheBaseline`
- `FlashSVD-v1.5`

The study is designed to support paper writing with:

- end-to-end throughput and latency
- prefill vs decode stage breakdown
- repeated-run stability statistics
- module-wise decode profile
- layer-wise attention reconstruct profile
- layer-wise MLP backend profile
- end-to-end runtime ablations

## Machine / Software

- Host cwd: `/home/zs89/FlashSVD`
- GPU target: `CUDA_VISIBLE_DEVICES=5` unless otherwise stated
- Python: `3.13.2`
- PyTorch: `2.7.1+cu128`
- Triton: `3.3.1`
- Transformers: `4.53.0`
- FlashAttention: `2.8.3`
- Git commit: `c6a067b304b8a541b3f1c4f24d8bbe0ecbe21869`
- Python env: `/home/zs89/miniconda3/envs/flashsvd15/bin/python`

## Model / Checkpoint

- Checkpoint: `/home/zs89/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt`
- Compression family: SVDLLM v1 style uniform-rank checkpoint
- Main dtype: `bf16`
- Batch size: `1`

## Runtime Knobs

Unless an ablation overrides them, FlashSVD-v1.5 runs use:

```bash
FLASH_SVD_DENSE_DECODE_BACKEND=packed
FLASH_SVD_DENSE_DECODE_GRAPH=1
```

Main FlashSVD benchmark flags:

```bash
--experimental_flash_dense_attn
--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod
--mlp_cuda_graph
--mlp_cuda_graph_scope layer_tail
```

## Baseline Definitions

- `StaticCache`:
  normal SVD runtime with `StaticCache`; this is the default `svd` mode in `benchmark/decode/bench_flashsvd_vs_svd_decode.py`.
- `DenseKVCacheBaseline`:
  dense KV cache baseline with reference QKV reconstruct, external RoPE, and `flash_attn_with_kvcache`.
- `FlashSVD-v1.5`:
  packed rank projection + token reconstruct + internal RoPE + FA2 KV-cache decode + per-layer CUDA graph.

## Experiment Matrix

### 1. End-to-End Repeated Runs

Run repeated full-model decode benchmarks for:

- `prompt_len=512, new_tokens=32`
- `prompt_len=2048, new_tokens=128`

For each setting:

- `StaticCache` vs `FlashSVD-v1.5`
- `DenseKVCacheBaseline` vs `FlashSVD-v1.5`

Collected metrics:

- prefill time
- prefill tok/s
- decode ms/token
- decode tok/s
- total time
- repeated-run mean / median / stdev

### 2. Module-Wise Decode Profile

Decode profile is collected with module-level CUDA event timing on a representative configuration:

- `prompt_len=512`
- `new_tokens=32`
- `profile_decode_steps=16`

Variants:

- `StaticCache`
- `DenseKVCacheBaseline`
- `FlashSVD-v1.5`

### 3. Layer-Wise Kernel / Backend Study

Attention reconstruct:

- `benchmark/attn/bench_real_checkpoint_decode_reconstruct.py --layer -1`

MLP backends:

- `benchmark/mlp/bench_real_checkpoint_mlp.py --all_layers`

### 4. End-to-End Ablations

Planned ablations:

- baseline type: `StaticCache` vs `DenseKVCacheBaseline`
- graph scope: `mlp` vs `layer_tail`
- FlashSVD graph on/off
- MLP backend family
- attention-route microbench:
  `FlashSVD-v1.5`, `dense+FA2-only`, `sparse`, `sparse+FA2-only`

## Output Layout

- `raw/`: raw stdout logs
- `profiles/`: profiler outputs and decode profile logs
- `tables/`: parsed JSON / CSV / compact tables
- `SUMMARY.md`: paper-facing summary
