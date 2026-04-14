# Hetero-Rank Runtime Check

Date: 2026-04-11

## Goal

Measure how the current stable FlashSVD v1.5 serving configuration behaves on real heterogeneous-rank checkpoints.

We used the public DobiSVD non-remapping LLaMA-2-7B checkpoints at ratios `0.4 / 0.6 / 0.8`, which have non-uniform per-projection ranks.

## Runtime configuration

- GPU: `CUDA_VISIBLE_DEVICES=5` (`A100 80GB`)
- Python: `/home/zs89/miniconda3/envs/flashsvdv15/bin/python`
- Dtype: `bf16`
- Batch size: `1`
- Prompt length: `256`
- Decode length: `16`
- Warmup: `3`

FlashSVD knobs:

```bash
FLASH_SVD_DENSE_DECODE_BACKEND=packed
FLASH_SVD_DENSE_DECODE_GRAPH=1
FLASH_SVD_DENSE_DECODE_HETERO_FUSED=0
```

Benchmark flags:

```bash
--experimental_flash_dense_attn
--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod
--mlp_cuda_graph
--mlp_cuda_graph_scope layer_tail
```

## Main results

### vs `StaticCache`

| Ratio | Baseline Prefill (s) | FlashSVD Prefill (s) | Baseline Decode (ms/token) | FlashSVD Decode (ms/token) | Decode Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.4` | `0.646` | `0.075` | `34.103` | `20.693` | `1.65x` |
| `0.6` | `0.705` | `0.078` | `34.073` | `26.373` | `1.29x` |
| `0.8` | `0.820` | `0.093` | `37.741` | `27.945` | `1.35x` |

### vs `DenseKVCacheBaseline`

| Ratio | Baseline Prefill (s) | FlashSVD Prefill (s) | Baseline Decode (ms/token) | FlashSVD Decode (ms/token) | Decode Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.4` | `0.282` | `0.064` | `32.079` | `21.380` | `1.50x` |
| `0.6` | `0.354` | `0.074` | `29.949` | `26.163` | `1.14x` |
| `0.8` | `0.382` | `0.064` | `30.964` | `29.295` | `1.06x` |

## Structure check

Layer counts that satisfy the strongest current fast-path assumptions:

| Ratio | Layers with `Rq = Rk = Rv` | Layers with `Rgate = Rup` | Layers with `Rgate = Rup = Rdown` |
| --- | ---: | ---: | ---: |
| `0.4` | `0 / 32` | `1 / 32` | `0 / 32` |
| `0.6` | `0 / 32` | `1 / 32` | `0 / 32` |
| `0.8` | `1 / 32` | `2 / 32` | `0 / 32` |

This explains why Dobi remains much weaker than the uniform-rank SVD-LLM checkpoints: the current shared-rank attention and packed MLP paths are only rarely available.

## Experimental note

We did not include the experimental `FLASH_SVD_DENSE_DECODE_HETERO_FUSED` flag in the reported table above.
The numbers summarized here are for the current stable production-style setting with:

```bash
FLASH_SVD_DENSE_DECODE_HETERO_FUSED=0
```

That keeps the reported results aligned with the currently preferred runtime configuration rather than an extra hetero-only experimental branch.

## Takeaway

- Heterogeneous-rank checkpoints are now clearly better than the old baseline runtime under the current stable FlashSVD v1.5 configuration.
- The gain is real but much smaller than the uniform-rank headline results.
- On this representative Dobi sweep, decode speedup ranges from:
  - `1.29x - 1.65x` vs `StaticCache`
  - `1.06x - 1.50x` vs `DenseKVCacheBaseline`
- The main reason is structural: these checkpoints almost never satisfy the shared-rank assumptions behind FlashSVD's strongest decode fast paths.

## Raw logs

- [ratio_0.4_static_p256_n16.log](./ratio_0.4_static_p256_n16.log)
- [ratio_0.4_densekv_p256_n16.log](./ratio_0.4_densekv_p256_n16.log)
- [ratio_0.6_static_p256_n16.log](./ratio_0.6_static_p256_n16.log)
- [ratio_0.6_densekv_p256_n16.log](./ratio_0.6_densekv_p256_n16.log)
- [ratio_0.8_static_p256_n16.log](./ratio_0.8_static_p256_n16.log)
- [ratio_0.8_densekv_p256_n16.log](./ratio_0.8_densekv_p256_n16.log)
- [ratio_0.4_flashsvd_only_hetero_fused1_p256_n16.log](./ratio_0.4_flashsvd_only_hetero_fused1_p256_n16.log)
