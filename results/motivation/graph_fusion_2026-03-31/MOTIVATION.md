# Graph Fusion Motivation: Why Per-Layer Decode Graph Matters

## What We Changed

The recent runtime change is not a new attention algorithm. It is a **graph-granularity change** in the active `FlashSVD-v1.5` decode path.

Before this change, the decode hot path was effectively split into two graphable pieces:

1. attention-side dense decode graph
2. post-attention tail graph for `RMSNorm + MLP`

This meant each layer still paid:

- one attention-side graph replay
- one tail-graph replay
- extra staging copies / tensor materialization between them

The new path captures the whole per-layer token decode body as a single CUDA graph:

`input RMSNorm -> attention token reconstruct + FA2 KV-cache decode -> residual add -> post-attention RMSNorm -> MLP`

Relevant code:

- per-layer graph gating and replay:
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/models/llama.py:713`
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/models/llama.py:761`
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/models/llama.py:887`
- older tail-only graph path:
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/models/llama.py:1425`
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/models/llama.py:1436`
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/models/llama.py:1474`

The key idea is simple:

- old split-graph path still had too many small launches and too many copies
- per-layer graph removes one whole graph boundary per layer
- therefore it also removes a large amount of host launch overhead and graph-to-graph staging noise

## Motivation

We suspected that decode latency was being dominated by **many small kernels and staging ops**, not just by the main math kernels themselves.

In other words, the problem was not only:

- attention compute
- MLP compute

but also:

- `cudaLaunchKernel`
- `cudaGraphLaunch`
- `aten::copy_`
- `aten::clone`
- `aten::to`
- `aten::_to_copy`

This experiment was designed to answer one narrow question:

**Does changing graph granularity reduce kernel fragmentation enough to explain the large end-to-end speedup?**

## Experimental Setting

- GPU: `GPU 7`, A100 80GB
- Env: `flashsvd15`
- Checkpoint: `/home/zs89/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt`
- Precision: `bf16`
- Batch size: `1`
- Warmup decode steps: `4`
- Profiled decode steps: `16`
- Dense decode backend: `packed`

We profiled three FlashSVD-only runtime variants:

- `nograph`: no attention graph, no MLP graph
- `split`: attention graph + MLP-only graph
- `layer`: active per-layer graph (`layer_tail`)

We used two context settings:

- `short`: prompt length `512`
- `long`: prompt length `2048`

For each variant, we recorded:

- decode latency with CUDA events
- `torch.profiler` op summary
- per-token counts for:
  - `cudaLaunchKernel`
  - `cudaGraphLaunch`
  - `aten::copy_`
  - `aten::clone`
  - `aten::to + aten::_to_copy`

## Main Result

### Short Context (`prompt_len=512`)

| Mode | Decode ms/token | `cudaLaunchKernel` / token | `cudaGraphLaunch` / token | `copy_` / token | `to/_to_copy` / token | launch CPU ms / token |
|---|---:|---:|---:|---:|---:|---:|
| `nograph` | `36.00` | `1174` | `0` | `199` | `402` | `6.93` |
| `split` | `20.53` | `630` | `64` | `327` | `402` | `4.65` |
| `layer` | `9.35` | `54` | `32` | `135` | `82` | `2.08` |

Speedups:

- `split` vs `nograph`: `1.75x`
- `layer` vs `nograph`: `3.85x`
- `layer` vs `split`: `2.20x`

### Long Context (`prompt_len=2048`)

| Mode | Decode ms/token | `cudaLaunchKernel` / token | `cudaGraphLaunch` / token | `copy_` / token | `to/_to_copy` / token | launch CPU ms / token |
|---|---:|---:|---:|---:|---:|---:|
| `nograph` | `35.52` | `1174` | `0` | `199` | `402` | `6.77` |
| `split` | `20.24` | `630` | `64` | `327` | `402` | `4.74` |
| `layer` | `10.02` | `54` | `32` | `135` | `82` | `2.07` |

Speedups:

- `split` vs `nograph`: `1.76x`
- `layer` vs `nograph`: `3.54x`
- `layer` vs `split`: `2.02x`

## What This Shows

### 1. The old eager path really was fragmented

`nograph` launches about `1174` CUDA kernels per token on both short and long contexts.

This is the strongest direct evidence that the old path was dominated by fine-grained runtime fragmentation.

### 2. Split-graph helps, but it does not actually solve the staging problem

Compared with `nograph`, `split` is better because it replaces part of the eager launch stream with graph replay.

But it still pays:

- `64` graph launches per token
- `327` `aten::copy_` calls per token
- `402` `aten::to/_to_copy` calls per token

So split-graph is not yet a thin serving path. It still has too many graph boundaries and too much tensor movement.

### 3. Per-layer graph halves graph launches and removes a large amount of copy traffic

Relative to `split`, `layer` gives:

- `50%` fewer graph launches per token: `64 -> 32`
- `58.7%` fewer `copy_` calls per token: `327 -> 135`
- `79.6%` fewer `to/_to_copy` calls per token: `402 -> 82`
- about `55%` lower launch-API CPU time per token

This is exactly what we expected from fusing attention-side graph replay with the post-attention tail graph into one per-layer replay.

### 4. The benefit is a launch-granularity effect, not a sequence-length effect

The launch counts are almost identical for `prompt_len=512` and `prompt_len=2048`.

That means this overhead is not mainly about context length. It is a **per-token, per-layer fixed cost**.

A useful way to say this in the paper:

- the main fragmentation overhead is approximately `O(num_layers)` per token
- the per-layer graph changes the constant factor by collapsing two graph boundaries into one

So this optimization is fundamentally about **runtime thinness**, not about changing the asymptotic attention algorithm.

## Why This Is Fancy But Real

This is not just “CUDA Graph is faster.”

The important part is **what we chose to graph together**.

The per-layer graph wins because it captures a semantically complete serving unit:

- layer input normalization
- current-token FlashSVD attention fast path
- residual
- post-attention normalization
- MLP

That is why the counts move so sharply:

- fewer graph launches
- fewer inter-stage copies
- fewer dtype conversion helpers

The profiler says the change is real, and the speedup says the change matters.

## Files

- aggregate summary:
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/results/motivation/graph_fusion_2026-03-31/graph_fusion_summary.json`
- raw profiler tables:
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/results/motivation/graph_fusion_2026-03-31/raw/short_nograph_profiler.txt`
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/results/motivation/graph_fusion_2026-03-31/raw/short_split_profiler.txt`
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/results/motivation/graph_fusion_2026-03-31/raw/short_layer_profiler.txt`
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/results/motivation/graph_fusion_2026-03-31/raw/long_nograph_profiler.txt`
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/results/motivation/graph_fusion_2026-03-31/raw/long_split_profiler.txt`
  - `/home/zs89/FlashSVD/FlashSVD-v1.5/results/motivation/graph_fusion_2026-03-31/raw/long_layer_profiler.txt`
