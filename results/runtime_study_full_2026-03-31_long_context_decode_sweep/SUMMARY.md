# Long-Context Decode Sweep Summary

## Long-Context Stage Study

| Ratio | Config | Baseline | Baseline decode (mean ms/token) | FlashSVD decode (mean ms/token) | Decode speedup | Baseline prefill (mean s) | FlashSVD prefill (mean s) | Baseline total (mean s) | FlashSVD total (mean s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `0.5` | `ctx4096` | `StaticCache` | 30.599 | 12.849 | 2.38x | 0.369 | 0.281 | 4.286 | 1.926 |
| `0.5` | `ctx8192` | `StaticCache` | 30.632 | 14.085 | 2.17x | 0.929 | 0.880 | 4.850 | 2.682 |
| `0.5` | `ctx4096` | `DenseKVCacheBaseline` | 26.512 | 12.849 | 2.06x | 0.284 | 0.281 | 3.678 | 1.926 |
| `0.5` | `ctx8192` | `DenseKVCacheBaseline` | 26.230 | 14.085 | 1.86x | 0.582 | 0.880 | 3.940 | 2.682 |
| `0.6` | `ctx4096` | `StaticCache` | 30.453 | 15.174 | 2.01x | 0.964 | 0.832 | 4.862 | 2.774 |
| `0.6` | `ctx8192` | `StaticCache` | 31.570 | 16.347 | 1.93x | 2.153 | 1.705 | 6.194 | 3.798 |
| `0.6` | `ctx4096` | `DenseKVCacheBaseline` | 25.788 | 15.174 | 1.70x | 0.879 | 0.832 | 4.179 | 2.774 |
| `0.6` | `ctx8192` | `DenseKVCacheBaseline` | 25.626 | 16.347 | 1.57x | 1.800 | 1.705 | 5.080 | 3.798 |
| `0.7` | `ctx4096` | `StaticCache` | 30.115 | 16.954 | 1.78x | 1.111 | 0.963 | 4.965 | 3.134 |
| `0.7` | `ctx8192` | `StaticCache` | 33.782 | 18.099 | 1.87x | 2.419 | 1.952 | 6.743 | 4.269 |
| `0.7` | `ctx4096` | `DenseKVCacheBaseline` | 26.031 | 16.954 | 1.54x | 1.025 | 0.963 | 4.357 | 3.134 |
| `0.7` | `ctx8192` | `DenseKVCacheBaseline` | 25.718 | 18.099 | 1.42x | 2.066 | 1.952 | 5.358 | 4.269 |
| `0.8` | `ctx4096` | `StaticCache` | 29.680 | 18.708 | 1.59x | 0.660 | 0.559 | 4.459 | 2.954 |
| `0.8` | `ctx8192` | `StaticCache` | 34.442 | 19.841 | 1.74x | 1.536 | 3.802 | 5.945 | 6.341 |
| `0.8` | `ctx4096` | `DenseKVCacheBaseline` | 25.593 | 18.708 | 1.37x | 0.575 | 0.559 | 3.851 | 2.954 |
| `0.8` | `ctx8192` | `DenseKVCacheBaseline` | 25.588 | 19.841 | 1.29x | 1.182 | 3.802 | 4.458 | 6.341 |

## Decode-Length Sweep

| Ratio | new_tokens | Baseline | Baseline decode (mean ms/token) | FlashSVD decode (mean ms/token) | Decode speedup |
|---|---:|---|---:|---:|---:|
| `0.5` | `64` | `StaticCache` | 30.632 | 11.468 | 2.67x |
| `0.5` | `128` | `StaticCache` | 30.474 | 11.467 | 2.66x |
| `0.5` | `256` | `StaticCache` | 30.499 | 11.660 | 2.62x |
| `0.5` | `512` | `StaticCache` | 30.537 | 11.701 | 2.61x |
| `0.5` | `1024` | `StaticCache` | 30.619 | 11.857 | 2.58x |
| `0.5` | `2048` | `StaticCache` | 30.479 | 12.074 | 2.52x |
| `0.5` | `4096` | `StaticCache` | 30.546 | 12.528 | 2.44x |
| `0.5` | `8192` | `StaticCache` | 30.672 | 13.303 | 2.31x |
| `0.5` | `16384` | `StaticCache` | 44.456 | 14.874 | 2.99x |
| `0.5` | `64` | `DenseKVCacheBaseline` | 26.330 | 11.468 | 2.30x |
| `0.5` | `128` | `DenseKVCacheBaseline` | 26.163 | 11.467 | 2.28x |
| `0.5` | `256` | `DenseKVCacheBaseline` | 26.278 | 11.660 | 2.25x |
| `0.5` | `512` | `DenseKVCacheBaseline` | 26.276 | 11.701 | 2.25x |
| `0.5` | `1024` | `DenseKVCacheBaseline` | 26.279 | 11.857 | 2.22x |
| `0.5` | `2048` | `DenseKVCacheBaseline` | 26.319 | 12.074 | 2.18x |
| `0.5` | `4096` | `DenseKVCacheBaseline` | 26.321 | 12.528 | 2.10x |
| `0.5` | `8192` | `DenseKVCacheBaseline` | 26.353 | 13.303 | 1.98x |
| `0.5` | `16384` | `DenseKVCacheBaseline` | 26.772 | 14.874 | 1.80x |
| `0.6` | `64` | `StaticCache` | 31.429 | 13.952 | 2.25x |
| `0.6` | `128` | `StaticCache` | 31.493 | 13.953 | 2.26x |
| `0.6` | `256` | `StaticCache` | 31.459 | 14.151 | 2.22x |
| `0.6` | `512` | `StaticCache` | 31.388 | 14.196 | 2.21x |
| `0.6` | `1024` | `StaticCache` | 30.148 | 14.338 | 2.10x |
| `0.6` | `2048` | `StaticCache` | 30.233 | 14.525 | 2.08x |
| `0.6` | `4096` | `StaticCache` | 29.902 | 14.979 | 2.00x |
| `0.6` | `8192` | `StaticCache` | 32.132 | 15.696 | 2.05x |
| `0.6` | `16384` | `StaticCache` | 46.069 | 17.257 | 2.67x |
| `0.6` | `64` | `DenseKVCacheBaseline` | 25.814 | 13.952 | 1.85x |
| `0.6` | `128` | `DenseKVCacheBaseline` | 25.801 | 13.953 | 1.85x |
| `0.6` | `256` | `DenseKVCacheBaseline` | 25.745 | 14.151 | 1.82x |
| `0.6` | `512` | `DenseKVCacheBaseline` | 25.834 | 14.196 | 1.82x |
| `0.6` | `1024` | `DenseKVCacheBaseline` | 25.883 | 14.338 | 1.81x |
| `0.6` | `2048` | `DenseKVCacheBaseline` | 25.678 | 14.525 | 1.77x |
| `0.6` | `4096` | `DenseKVCacheBaseline` | 25.769 | 14.979 | 1.72x |
| `0.6` | `8192` | `DenseKVCacheBaseline` | 25.773 | 15.696 | 1.64x |
| `0.6` | `16384` | `DenseKVCacheBaseline` | 25.796 | 17.257 | 1.49x |
| `0.7` | `64` | `StaticCache` | 30.467 | 15.744 | 1.94x |
| `0.7` | `128` | `StaticCache` | 29.972 | 15.749 | 1.90x |
| `0.7` | `256` | `StaticCache` | 30.310 | 15.949 | 1.90x |
| `0.7` | `512` | `StaticCache` | 30.085 | 15.982 | 1.88x |
| `0.7` | `1024` | `StaticCache` | 30.102 | 16.121 | 1.87x |
| `0.7` | `2048` | `StaticCache` | 30.726 | 16.306 | 1.88x |
| `0.7` | `4096` | `StaticCache` | 30.914 | 16.740 | 1.85x |
| `0.7` | `8192` | `StaticCache` | 34.690 | 17.470 | 1.99x |
| `0.7` | `16384` | `StaticCache` | 48.485 | 19.030 | 2.55x |
| `0.7` | `64` | `DenseKVCacheBaseline` | 25.803 | 15.744 | 1.64x |
| `0.7` | `128` | `DenseKVCacheBaseline` | 25.827 | 15.749 | 1.64x |
| `0.7` | `256` | `DenseKVCacheBaseline` | 25.928 | 15.949 | 1.63x |
| `0.7` | `512` | `DenseKVCacheBaseline` | 25.934 | 15.982 | 1.62x |
| `0.7` | `1024` | `DenseKVCacheBaseline` | 25.919 | 16.121 | 1.61x |
| `0.7` | `2048` | `DenseKVCacheBaseline` | 25.873 | 16.306 | 1.59x |
| `0.7` | `4096` | `DenseKVCacheBaseline` | 26.206 | 16.740 | 1.57x |
| `0.7` | `8192` | `DenseKVCacheBaseline` | 25.888 | 17.470 | 1.48x |
| `0.7` | `16384` | `DenseKVCacheBaseline` | 25.714 | 19.030 | 1.35x |
| `0.8` | `64` | `StaticCache` | 29.570 | 17.506 | 1.69x |
| `0.8` | `128` | `StaticCache` | 29.759 | 17.490 | 1.70x |
| `0.8` | `256` | `StaticCache` | 29.606 | 17.688 | 1.67x |
| `0.8` | `512` | `StaticCache` | 29.612 | 17.732 | 1.67x |
| `0.8` | `1024` | `StaticCache` | 29.637 | 17.886 | 1.66x |
| `0.8` | `2048` | `StaticCache` | 29.674 | 18.085 | 1.64x |
| `0.8` | `4096` | `StaticCache` | 29.683 | 18.494 | 1.61x |
| `0.8` | `8192` | `StaticCache` | 35.112 | 19.240 | 1.82x |
| `0.8` | `16384` | `StaticCache` | 49.029 | 20.809 | 2.36x |
| `0.8` | `64` | `DenseKVCacheBaseline` | 25.569 | 17.506 | 1.46x |
| `0.8` | `128` | `DenseKVCacheBaseline` | 25.686 | 17.490 | 1.47x |
| `0.8` | `256` | `DenseKVCacheBaseline` | 25.546 | 17.688 | 1.44x |
| `0.8` | `512` | `DenseKVCacheBaseline` | 25.650 | 17.732 | 1.45x |
| `0.8` | `1024` | `DenseKVCacheBaseline` | 25.523 | 17.886 | 1.43x |
| `0.8` | `2048` | `DenseKVCacheBaseline` | 25.485 | 18.085 | 1.41x |
| `0.8` | `4096` | `DenseKVCacheBaseline` | 25.547 | 18.494 | 1.38x |
| `0.8` | `8192` | `DenseKVCacheBaseline` | 25.637 | 19.240 | 1.33x |
| `0.8` | `16384` | `DenseKVCacheBaseline` | 25.768 | 20.809 | 1.24x |

Elapsed wall time: `10.13 h`
