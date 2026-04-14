# FlashSVD-v1.5 Runtime Results

## Headline

All results in this folder use:

- checkpoint: `/home/zs89/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt`
- GPU: `A100 80GB`, `CUDA_VISIBLE_DEVICES=5`
- env: `flashsvd15` (`python=3.13.2`, `torch=2.7.1+cu128`, `transformers=4.53.0`, `triton=3.3.1`, `flash_attn=2.8.3`)
- FlashSVD-v1.5 runtime knobs:
  - `FLASH_SVD_DENSE_DECODE_BACKEND=packed`
  - `FLASH_SVD_DENSE_DECODE_GRAPH=1`
  - `--experimental_flash_dense_attn`
  - `--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod`
  - `--mlp_cuda_graph --mlp_cuda_graph_scope layer_tail`

Main result: with the active per-layer graph runtime, FlashSVD-v1.5 is now a stable and large win over both practical and aligned SVD baselines.

## End-to-End Decode

We report repeated full-model runs (`n=5`) and prefer the median when a baseline shows outliers.

| Setting | Baseline | Baseline prefill (median s) | Baseline decode (median ms/token) | FlashSVD prefill (median s) | FlashSVD decode (median ms/token) | Decode speedup (median) | Total speedup (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `512 / 32` | `StaticCache` | `0.452` | `29.814` | `0.079` | `9.980` | `2.97x` | `3.50x` |
| `2048 / 128` | `StaticCache` | `0.578` | `30.001` | `0.152` | `10.097` | `2.94x` | `3.03x` |
| `512 / 32` | `DenseKVCacheBaseline` | `0.424` | `32.794` | `0.066` | `10.022` | `3.32x` | `3.81x` |
| `2048 / 128` | `DenseKVCacheBaseline` | `0.544` | `32.771` | `0.147` | `10.189` | `3.22x` | `3.26x` |

Observations:

- The aligned `DenseKVCacheBaseline` is the cleaner paper baseline because it uses dense KV cache, external RoPE, and FA2 decode just like the reference dense path.
- FlashSVD-v1.5 wins on both decode and total runtime against both baselines.
- Variance is much lower on the FlashSVD path than on the baselines. This matters for serving.

Structured tables:

- [e2e_summary.json](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/tables/e2e_summary.json)
- [e2e_repeated_runs.csv](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/tables/e2e_repeated_runs.csv)

## Prefill vs Decode

The benchmark already separates prefill and decode.

Key stage-wise findings:

- Against `StaticCache`, FlashSVD prefill is `5.7x` faster at `512 / 32` and `3.8x` faster at `2048 / 128` when comparing medians.
- Against `DenseKVCacheBaseline`, FlashSVD prefill is `6.4x` faster at `512 / 32` and `3.7x` faster at `2048 / 128`.
- Decode is also substantially faster:
  - vs `StaticCache`: `2.97x` at `512 / 32`, `2.94x` at `2048 / 128`
  - vs `DenseKVCacheBaseline`: `3.32x` at `512 / 32`, `3.22x` at `2048 / 128`

## Module-Wise Decode Profile

Representative decode profile uses `prompt_len=512`, `new_tokens=32`, `profile_decode_steps=16`.

For the two non-graph baselines:

- `StaticCache`: total forward `37.014 ms/token`
  - `attn_total = 16.530 ms (44.7%)`
  - `mlp_total = 6.109 ms (16.5%)`
  - `ln1_total + ln2_total = 7.489 ms (20.2%)`
  - `other = 6.537 ms (17.7%)`
- `DenseKVCacheBaseline`: total forward `40.893 ms/token`
  - `attn_total = 20.188 ms (49.4%)`
  - `mlp_total = 6.399 ms (15.6%)`
  - `ln1_total + ln2_total = 7.974 ms (19.5%)`
  - `other = 5.983 ms (14.6%)`

For `FlashSVD-v1.5` without graph:

- total forward `36.098 ms/token`
- `attn_total = 15.137 ms (41.9%)`
- `mlp_total = 7.584 ms (21.0%)`
- `other = 5.774 ms (16.0%)`

This is important: before graph fusion, FlashSVD attention already helps, but the total decode path is still not enough to produce a large end-to-end win.

For `FlashSVD-v1.5` with the active per-layer graph:

- total forward `9.348 ms/token`
- submodule hook times collapse into `other = 9.108 ms (97.4%)`

This is expected: the graph path captures the whole decoder-layer hot path, so per-submodule hooks no longer see the internal work. The graph collapse itself is evidence that the serving fast path is active.

Files:

- [module_profile_static.txt](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/profiles/module_profile_static.txt)
- [module_profile_densekv.txt](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/profiles/module_profile_densekv.txt)
- [module_profile_flashsvd_no_graph.txt](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/profiles/module_profile_flashsvd_no_graph.txt)
- [module_profile_flashsvd.txt](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/profiles/module_profile_flashsvd.txt)
- [module_profiles.json](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/tables/module_profiles.json)

Note:

- The dense-KV module-profile prefill time in a fresh process is inflated by first-call compilation and should not be used as the steady-state prefill headline. Use the repeated end-to-end runs for steady-state prefill numbers.

## Op-Level Profile

We also ran `torch.profiler` on `prefill` and `decode` for all three baselines.

Static / DenseKV prefill:

- Both are dominated by `aten::mm` on CUDA:
  - `StaticCache` prefill: `67.73%` self CUDA
  - `DenseKVCacheBaseline` prefill: `69.35%` self CUDA
- `aten::copy_` is secondary:
  - `StaticCache`: `3.65%` self CUDA
  - `DenseKVCacheBaseline`: `5.72%` self CUDA

Static / DenseKV decode:

- `aten::mm` still dominates:
  - `StaticCache` decode: `57.15%` self CUDA
  - `DenseKVCacheBaseline` decode: `56.91%` self CUDA
- `DenseKVCacheBaseline` pays more copy traffic:
  - `aten::copy_`: `10.48%` self CUDA vs `4.54%` for `StaticCache`
- CPU-side overhead also shows `aten::_to_copy` and `aten::clone` as meaningful residual costs.

FlashSVD-v1.5 decode with graph:

- The visible self-CUDA outside graph replay becomes very small.
- The main visible host-side residuals are:
  - `cudaGraphLaunch`: `29.90%` CPU total
  - `aten::copy_`: `45.76%` CPU total
  - `aten::clone`: `12.46%` CPU total
- The remaining visible self-CUDA compute outside graph is tiny:
  - `aten::mm`: `1.68%` self CUDA

Interpretation:

- Before graph fusion, attention and MLP compute are still clearly visible.
- After graph fusion, the remaining optimization targets are no longer major math kernels. They are launch, copy, and clone overhead around graph replay.
- This directly supports the current engineering direction: further gains should come from making the serving path thinner, not from changing the SVD algorithm.

Files:

- [op_profile_static.txt](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/profiles/op_profile_static.txt)
- [op_profile_densekv.txt](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/profiles/op_profile_densekv.txt)
- [op_profile_flashsvd.txt](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/profiles/op_profile_flashsvd.txt)

## Attention Reconstruct: Layer-Wise Kernel Study

All 32 layers in this uniform-rank checkpoint use `Rq=Rk=Rv=1024`.

Per-layer current-token QKV reconstruct:

- exact reference mean: `0.2522 ms`
- packed linear mean: `0.1484 ms`
- packed flat Triton path mean: `0.1278 ms`

Aggregate speedups:

- packed linear vs exact: mean `1.700x`, median `1.699x`
- packed flat vs exact: mean `1.974x`, median `1.976x`

Wins:

- packed linear wins `32 / 32` layers
- packed flat wins `32 / 32` layers

Numerical differences:

- low-precision max diff is small but non-zero, typically `0.0312` to `0.0625`

Files:

- [attn_reconstruct_all_layers.txt](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/raw/attn_reconstruct_all_layers.txt)
- [attn_reconstruct_summary.json](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/tables/attn_reconstruct_summary.json)

## MLP Backend: Layer-Wise Study

Stable all-layer sweep uses:

- `baseline`
- `auto`
- `flashsvd_mlp_dual_split_exact_legacy`
- `flashsvd_mlp_dual_split_prod`

At `B=1, L=1`:

- baseline eager mean: `0.2030 ms`
- prod eager mean is slower than baseline
- exact-legacy eager mean is also slower than baseline
- the important win is from CUDA graph:
  - `flashsvd_mlp_dual_split_prod` graph mean: `0.2018 ms`
  - mean speedup vs baseline: `1.006x`
  - mean graph gain over prod eager: `15.78%`
  - median graph gain over prod eager: `14.7%`

Interpretation:

- MLP backend improvements alone are modest at full decode latency scale.
- MLP graph helps, but it is not the primary source of the 3x end-to-end win.
- The production runtime win is primarily an attention-side and graph-granularity story.

Files:

- [mlp_backend_all_layers_stable.txt](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/raw/mlp_backend_all_layers_stable.txt)
- [mlp_backend_summary.json](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/tables/mlp_backend_summary.json)

Experimental note:

- The broader all-backend sweep in [mlp_backend_all_layers.txt](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/raw/mlp_backend_all_layers.txt) shows `flashsvd_mlp_dual_split_triton_v1` fails for token decode shapes because Triton dot tiles require `M/N/K >= 16`. This is useful negative evidence: the early Triton experimental backend is not a robust token-serving path.

## Runtime Ablation: Graph Granularity

We ran three FlashSVD-only runtime variants:

- `nograph`: no attention graph, no MLP graph
- `split`: attention graph + MLP-only graph
- `layer`: active per-layer graph (`layer_tail`)

`512 / 32` decode:

- `nograph`: `32.424 ms/token` mean
- `split`: `16.587 ms/token` mean
- `layer`: `9.913 ms/token` mean

Speedups:

- `split` vs `nograph`: `1.95x`
- `layer` vs `nograph`: `3.27x`
- `layer` vs `split`: `1.67x`

`2048 / 128` decode:

- `nograph`: `29.938 ms/token` mean
- `split`: `16.401 ms/token` mean
- `layer`: `10.102 ms/token` mean

Speedups:

- `split` vs `nograph`: `1.83x`
- `layer` vs `nograph`: `2.96x`
- `layer` vs `split`: `1.62x`

This is the clearest ablation in the study. The recent per-layer graph fusion is the main runtime unlock.

Files:

- [graph_ablation_summary.json](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/tables/graph_ablation_summary.json)

## MLP-Only End-to-End Ablation

With attention fast path disabled and only MLP backend varied:

- baseline decode: `29.487 ms/token`
- exact_legacy: `27.398 ms/token` (`1.08x`)
- prod: `27.542 ms/token` (`1.07x`)
- auto: `27.433 ms/token` (`1.07x`)

Interpretation:

- MLP-only backend swaps matter, but they do not explain the full system-level gain.

File:

- [mlp_end_to_end_backend_compare.txt](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/raw/mlp_end_to_end_backend_compare.txt)

## Attention-Route Microbench

Shared-rank `llama2-7b`, `target_param_ratio=0.5`, `bf16`, `B=1`.

Direct step winner:

- `L=512`: `FlashSVD-v1.5+graph = 0.1016 ms`
- `L=2048`: `FlashSVD-v1.5+graph = 0.1183 ms`
- `L=4096`: `FlashSVD-v1.5+graph = 0.1390 ms`

Reference routes:

- `dense+FA2-only`: `0.2075 / 0.2137 / 0.2114 ms`
- `sparse+FA2-only`: `0.4389 / 0.4509 / 0.7313 ms`
- legacy `sparse`: `4.3270 / 7.9756 / 13.4465 ms`

Relative to legacy sparse:

- `42.6x` faster at `L=512`
- `67.4x` faster at `L=2048`
- `96.8x` faster at `L=4096`

Notes:

- Direct step winner is `FlashSVD-v1.5+graph` at all tested lengths.
- Experimental `v1.6` sparse variants still fail with `workspace_mismatch`.
- The script prints a conservative “overall” line that includes reconstruct-ablation lower bounds; for the actual online step, the direct winner is the correct metric to report.

Files:

- [attn_route_microbench.txt](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/raw/attn_route_microbench.txt)
- [attn_route_microbench.json](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31/tables/attn_route_microbench.json)

## Main Takeaways

1. `DenseKVCacheBaseline` should be the main paper baseline.
2. FlashSVD-v1.5 is now a stable `~3x` decode win over both `StaticCache` and aligned dense-KV baseline on this checkpoint.
3. The main runtime unlock is not a new SVD algorithm. It is the per-layer graph serving path.
4. Before graph fusion, FlashSVD alone is not enough. After graph fusion, the remaining bottlenecks are launch and copy overhead.
5. Attention reconstruct kernels are strong and uniform across layers.
6. MLP backend changes help, but they are secondary relative to the attention/runtime path.
7. Experimental sparse decode variants remain much weaker or unstable than the current FlashSVD-v1.5 serving path.
