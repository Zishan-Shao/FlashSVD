# FlashSVD-v1.5 Busy-GPU Contention Study

Date: 2026-04-10

## Goal

This study asks a narrow systems question:

- when the target GPU is busy with other jobs, does `FlashSVD-v1.5` lose its win because CUDA graph stops helping, or because GPU-side execution is slowed by contention?

We focus on:

- `DenseKVCacheBaseline`
- `FlashSVD-v1.5`
- host-side submit time vs GPU-side elapsed time
- module-level decode breakdown on a busy GPU

## Headline

The main finding is:

- on a busy shared GPU, `CUDA graph` still reduces CPU-side overhead for `FlashSVD-v1.5`
- the dominant regression comes from GPU-side contention, not from CUDA graph being disabled or frequently broken

Evidence:

- busy `GPU 1`, `graph_on`:
  - host submit mean: `13.75 ms/token`
  - GPU elapsed mean: `25.98 ms/token`
- busy `GPU 1`, `graph_off`:
  - host submit mean: `31.22 ms/token`
  - GPU elapsed mean: `29.76 ms/token`
- cleaner `GPU 5`, `graph_on`:
  - host submit mean: `12.33 ms/token`
  - GPU elapsed mean: `12.47 ms/token`

Interpretation:

- graph still cuts host-side launch cost substantially
- under contention, GPU-side execution nearly doubles while host-side submit time changes only slightly
- this points to shared-GPU scheduling / execution slowdown, not graph failure

## End-to-End Decode Behavior

Representative short decode benchmark on a busy shared GPU:

| GPU | Mode | prompt / decode | Decode ms/token |
|---|---:|---:|---:|
| `1` | `DenseKVCacheBaseline` | `64 / 8` | `31.362` |
| `1` | `FlashSVD-v1.5` | `64 / 8` | `22.374` |

This confirms that FlashSVD still wins on the busy GPU, but the margin is smaller than on a cleaner card.

## Module-Level Decode Breakdown

Busy `GPU 1`, `DenseKVCacheBaseline`:

- total forward: `36.314 ms/token`
- `attn_total = 15.068 ms (41.5%)`
- `mlp_total = 8.260 ms (22.7%)`
- `ln1_total + ln2_total = 8.086 ms (22.3%)`
- `other = 4.696 ms (12.9%)`

Busy `GPU 1`, `FlashSVD-v1.5` with `layer_tail` graph:

- total forward: `20.835 ms/token`
- `attn_total = 9.284 ms (44.6%)`
- `other = 9.401 ms (45.1%)`
- `mlp_total = 0.000 ms`
- `ln2_total = 0.000 ms`

Important note:

- for `FlashSVD-v1.5` with active `layer_tail` CUDA graph, submodule hooks no longer see the MLP / tail work inside the replayed graph
- that work collapses into `other`
- so the right interpretation is not “MLP disappeared”; it is “graph-replayed tail work is hidden from submodule hooks”

## Host vs GPU Split

The most useful measurement in this study is the direct split between:

- host submit time: wall-clock from Python around one decode step
- GPU elapsed time: CUDA-event time for the same decode step

Results:

- `busy_gpu1_graph_on`
  - host mean `13.747 ms`
  - GPU mean `25.977 ms`
- `busy_gpu1_graph_off`
  - host mean `31.215 ms`
  - GPU mean `29.758 ms`
- `cleaner_gpu5_graph_on`
  - host mean `12.327 ms`
  - GPU mean `12.471 ms`

What this means:

- graph-on vs graph-off on the same busy GPU cuts host cost from `31.2` to `13.7 ms`
- graph-on on busy vs cleaner GPU changes host cost only modestly (`13.7` vs `12.3 ms`)
- graph-on on busy vs cleaner GPU changes GPU elapsed massively (`26.0` vs `12.5 ms`)

This is the cleanest evidence in the study:

- the regression is primarily GPU-side contention
- not CPU launch overhead
- not repeated graph failure

## Nsight Systems Snapshot

We also captured one short `nsys` trace on busy `GPU 1`.

The trace includes model load and prefill, so it is not a pure steady-state decode trace. It should be treated as supporting evidence only.

Still, the API summary is useful:

- `cudaMemcpyAsync`: `69.2%`
- `cudaMalloc`: `16.3%`
- `cudaLaunchKernel`: `5.3%`
- `cudaStreamSynchronize`: `4.8%`
- `cudaDeviceSynchronize`: `2.2%`
- `cudaGraphLaunch`: `0.1%`

This does **not** support the story that graph replay itself dominates CPU-side overhead on the busy card.

## Conclusion

For FlashSVD-v1.5 on shared A100s:

- CUDA graph remains active and useful
- the main failure mode under load is GPU contention
- baseline decode remains attention-dominated
- FlashSVD with `layer_tail` graph becomes “attention + graph-replayed tail work” dominated

Practical takeaway:

- if a run is much slower on a shared card, the first thing to suspect is active competing GPU work, not broken graph replay

## Files

- `tables/host_vs_gpu_split.csv`
- `tables/module_breakdown_busy_gpu.csv`
- `raw/nsys_busy_gpu1_flashsvd_cuda_api_sum.txt`
- `raw/nsys_busy_gpu1_flashsvd_cuda_gpu_kern_top.txt`
