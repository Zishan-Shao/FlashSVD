# Basis Sharing Runtime Sweep

Date: `2026-04-11`

## Setup

- GPU: `GPU 1`
- Launch-time GPU snapshot:
  - `GPU 1: memory.used=19147 MiB, utilization.gpu=0%`
  - other visible GPUs were mostly active (`30%-100%`), so `GPU 1` was the least contended choice
- Python: `/home/zs89/miniconda3/envs/flashsvdv15/bin/python`
- Runtime config:
  - `FLASH_SVD_DENSE_DECODE_BACKEND=packed`
  - `FLASH_SVD_DENSE_DECODE_GRAPH=1`
  - `--experimental_flash_dense_attn`
  - `--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod`
  - `--mlp_cuda_graph`
  - `--mlp_cuda_graph_scope layer_tail`
- Benchmark config:
  - `dtype=bf16`
  - `batch_size=1`
  - `prompt_len=256`
  - `new_tokens=16`
  - `warmup=3`
- Checkpoints:
  - `Duke-CEI-SVD/LowRankArena::llama_7b/Basis_Sharing/share_llama-7b_40`
  - `Duke-CEI-SVD/LowRankArena::llama_7b/Basis_Sharing/share_llama-7b_50`
  - `Duke-CEI-SVD/LowRankArena::llama_7b/Basis_Sharing/share_llama-7b_60`

## Decode Results

### Against `StaticCache`

| Ratio | Baseline decode (ms/token) | FlashSVD decode (ms/token) | Speedup |
| --- | ---: | ---: | ---: |
| `0.4` | `32.079` | `12.938` | `2.48x` |
| `0.5` | `31.995` | `24.977` | `1.28x` |
| `0.6` | `32.258` | `22.460` | `1.44x` |

### Against `DenseKVCacheBaseline`

| Ratio | Baseline decode (ms/token) | FlashSVD decode (ms/token) | Speedup |
| --- | ---: | ---: | ---: |
| `0.4` | `30.021` | `26.399` | `1.14x` |
| `0.5` | `29.330` | `24.791` | `1.18x` |
| `0.6` | `30.296` | `21.809` | `1.39x` |

## Prefill Results

### Against `StaticCache`

| Ratio | Baseline prefill (s) | FlashSVD prefill (s) | Speedup |
| --- | ---: | ---: | ---: |
| `0.4` | `0.202` | `0.061` | `3.31x` |
| `0.5` | `0.576` | `0.084` | `6.86x` |
| `0.6` | `0.270` | `0.085` | `3.18x` |

### Against `DenseKVCacheBaseline`

| Ratio | Baseline prefill (s) | FlashSVD prefill (s) | Speedup |
| --- | ---: | ---: | ---: |
| `0.4` | `0.456` | `0.090` | `5.07x` |
| `0.5` | `0.430` | `0.090` | `4.78x` |
| `0.6` | `0.501` | `0.085` | `5.89x` |

## Rank Structure Check

We also inspected the loaded FlashSVD-native Basis Sharing models:

| Ratio | Layers | `Rq=Rk=Rv` layers | `Rgate=Rup` layers | Unique attention rank values | Unique gate/up rank values |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.4` | `32` | `32/32` | `32/32` | `1` (`1638`) | `1` (`2072`) |
| `0.5` | `32` | `32/32` | `32/32` | `1` (`1365`) | `1` (`1726`) |
| `0.6` | `32` | `32/32` | `32/32` | `1` (`1092`) | `1` (`1381`) |

Under the user's current definition of adaptive rank ("layer rank changes across layers"), these public Basis Sharing checkpoints are **not adaptive-rank checkpoints**. They use Basis Sharing, but the released `0.4 / 0.5 / 0.6` variants are still layer-wise uniform-rank.

## Takeaways

- Basis Sharing is much more FlashSVD-friendly than Dobi's intra-layer hetero-rank checkpoints.
- With the current packed decode + per-layer graph configuration, Basis Sharing reaches:
  - `1.28x - 2.48x` decode speedup vs `StaticCache`
  - `1.14x - 1.39x` decode speedup vs `DenseKVCacheBaseline`
- The public Basis Sharing checkpoints do not exercise true layer-wise adaptive rank; they preserve the regular per-layer shared-rank structure that FlashSVD's strongest fast paths expect.
