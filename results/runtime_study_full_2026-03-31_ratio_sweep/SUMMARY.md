# Ratio Sweep Summary

This study uses the same FlashSVD-v1.5 runtime recipe as the main `0.5` result, but sweeps SVD-LLM v1 ratios `0.5 / 0.6 / 0.7 / 0.8` under the exported HuggingFace checkpoint layout.

## Headline Table

| Ratio | Config | Baseline | Baseline decode (median ms/token) | FlashSVD decode (median ms/token) | Decode speedup | Baseline total (median s) | FlashSVD total (median s) | Total speedup |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `0.5` | `short` | `StaticCache` | 30.839 | 12.163 | 2.54x | 1.033 | 0.434 | 2.38x |
| `0.5` | `long` | `StaticCache` | 30.627 | 12.235 | 2.50x | 4.097 | 1.720 | 2.38x |
| `0.5` | `short` | `DenseKVCacheBaseline` | 26.904 | 12.163 | 2.21x | 0.906 | 0.434 | 2.09x |
| `0.5` | `long` | `DenseKVCacheBaseline` | 26.191 | 12.235 | 2.14x | 3.509 | 1.720 | 2.04x |
| `0.6` | `short` | `StaticCache` | 30.574 | 13.890 | 2.20x | 1.112 | 0.570 | 1.95x |
| `0.6` | `long` | `StaticCache` | 29.734 | 14.687 | 2.02x | 4.281 | 2.309 | 1.85x |
| `0.6` | `short` | `DenseKVCacheBaseline` | 25.871 | 13.890 | 1.86x | 0.960 | 0.570 | 1.69x |
| `0.6` | `long` | `DenseKVCacheBaseline` | 25.641 | 14.687 | 1.75x | 3.736 | 2.309 | 1.62x |
| `0.7` | `short` | `StaticCache` | 30.670 | 16.196 | 1.89x | 1.131 | 0.659 | 1.72x |
| `0.7` | `long` | `StaticCache` | 29.184 | 16.420 | 1.78x | 4.292 | 2.605 | 1.65x |
| `0.7` | `short` | `DenseKVCacheBaseline` | 26.225 | 16.196 | 1.62x | 0.988 | 0.659 | 1.50x |
| `0.7` | `long` | `DenseKVCacheBaseline` | 25.637 | 16.420 | 1.56x | 3.817 | 2.605 | 1.47x |
| `0.8` | `short` | `StaticCache` | 29.751 | 17.560 | 1.69x | 1.040 | 0.646 | 1.61x |
| `0.8` | `long` | `StaticCache` | 29.766 | 18.236 | 1.63x | 4.127 | 2.623 | 1.57x |
| `0.8` | `short` | `DenseKVCacheBaseline` | 25.792 | 17.560 | 1.47x | 0.912 | 0.646 | 1.41x |
| `0.8` | `long` | `DenseKVCacheBaseline` | 25.553 | 18.236 | 1.40x | 3.568 | 2.623 | 1.36x |

## Best Ratios

- Best decode speedup vs `StaticCache`: ratio `0.5`, config `short`, `2.54x`.
- Best decode speedup vs `DenseKVCacheBaseline`: ratio `0.5`, config `short`, `2.21x`.
- Best total speedup vs `StaticCache`: ratio `0.5`, config `long`, `2.38x`.
- Best total speedup vs `DenseKVCacheBaseline`: ratio `0.5`, config `short`, `2.09x`.

## Interpretation

- This sweep is directly comparable across ratios because the runtime recipe, GPU, dtype, and benchmark shapes are fixed.
- The `DenseKVCacheBaseline` remains the cleaner aligned performance baseline.
- `StaticCache` remains the practical baseline.
- The speedup decays monotonically as the keep ratio increases from `0.5 -> 0.8`, which is consistent with larger low-rank projections and reconstruct cost eating into the FlashSVD advantage.
- This sweep uses exported HuggingFace checkpoints for all four ratios. The trend across ratios is the main result; the absolute `0.5` numbers should not be directly mixed with the earlier `.pt`-checkpoint study.

Structured outputs:

- [ratio_sweep_repeated_runs.csv](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_full_2026-03-31_ratio_sweep/tables/ratio_sweep_repeated_runs.csv)
- [ratio_sweep_summary.json](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_full_2026-03-31_ratio_sweep/tables/ratio_sweep_summary.json)
