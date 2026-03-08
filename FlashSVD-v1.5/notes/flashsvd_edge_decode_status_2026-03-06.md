# FlashSVD Edge Decode Status (2026-03-06)

## Motivation

- Primary serving focus: single-token decode.
- Expected batch size: `1-4`, with `B=1` as the dominant case.
- Expected context: short-to-mid / mid-to-long decode, up to `N <= 8192`.
- Why this focus:
  - Most SVD work mainly compresses weights.
  - Weight compression matters most when the deployment target is edge / memory-limited serving, not large-batch datacenter throughput.
  - The practical target is "at or below A100 80GiB" class deployments, not H100-scale large-batch serving.
- Optimization priority:
  - End-to-end decode latency for `q_len=1`.
  - Avoid small-launch overhead.
  - Prefer schedules that win at `B=1` over schedules that only look good in large-grid benchmarks.

## Attention Decode

### What Changed

- Reworked the attention decode experiment to match the real `svd_llama` shared global-rank setting.
- Important correction:
  - Old experiment effectively mapped `global rank=1024` to a headwise-friendly `per-head R=32`.
  - New experiment uses the real shared-rank cache form: `Pk/Pv: [B, L, R_shared]`.
- Added the dense short-context decode path:
  - `reconstruct_qkv_token_shared(...)`
  - `flash_attn_with_kvcache(...)`
- Added CUDA Graph replay variants in the attention decode benchmark.

### Benchmark Command

```bash
/home/zs89/miniconda3/envs/flashsvd/bin/python \
  /home/zs89/FlashSVD/FlashSVD-v1.5/kernels/flashsvd-v1.5/flashsvdropeattn_short/decode_compare.py \
  --llama llama2-7b \
  --target-param-ratio 0.5 \
  --rank-formula global \
  --factor-layout shared \
  --B 1 \
  --Ls 256,512,1024,2048,4096,8192 \
  --dtype bf16 \
  --dense-backend fa2 \
  --cuda-graph-variants \
  --report compact
```

### Current Results

Config:

- `llama2-7b`
- `B=1`
- `H=32, Hk=32, Dh=128`
- `shared global rank R=1024`
- `dtype=bf16`
- `dense backend=FA2`

Important benchmark correction:

- The old `dense_fa2` / `dense_fa2+graph` numbers were partly driven by an unfair cache-only baseline:
  - dense KV cache for all past tokens was prebuilt outside timing;
  - current-step `K/V` update was not inside the timed region.
- The current compact report now focuses on the real current-token step:
  - `dense_step`: `x -> packed dense qkv -> flash_attn_with_kvcache`
  - `rank_step`: `x -> packed rank qkv -> reconstruct_qkv_token_shared -> flash_attn_with_kvcache`
- CUDA Graph for the rank route was also moved earlier:
  - replay now copies `hidden token` only;
  - it no longer copies `Pq/Pk/Pv` into the graph input every step.

Latest fair step results:

| L | step_winner | step_ms | dense_step_ms | dense_step_g_ms | rank_step_ms | rank_step_g_ms | best_lowrank | lowrank_ms |
|---|---|---:|---:|---:|---:|---:|---|---:|
| 256 | `dense_step+fa2_g` | 0.1132 | 0.1222 | 0.1132 | 0.1427 | 0.1357 | `lowrank_fa2` | 0.4431 |
| 512 | `dense_step+fa2_g` | 0.1135 | 0.1275 | 0.1135 | 0.1869 | 0.1484 | `lowrank_fa2` | 0.5652 |
| 1024 | `rank_tok+fa2_kv+g` | 0.1482 | 0.2492 | 0.3134 | 0.2646 | 0.1482 | `lowrank_fa2` | 0.4534 |
| 2048 | `dense_step+fa2_g` | 0.1344 | 0.1361 | 0.1344 | 0.1853 | 0.1539 | `lowrank_fa2` | 0.4853 |
| 4096 | `dense_step+fa2` | 0.1521 | 0.1521 | 0.1594 | 0.2489 | 0.1798 | `lowrank_fa2` | 0.7914 |
| 8192 | `dense_step+fa2` | 0.1921 | 0.1921 | 0.1989 | 0.2825 | 0.2142 | `lowrank_fa2` | 1.3372 |

Winner counts:

- fair step winner:
  - `dense_step+fa2_g`: `3 / 6`
  - `dense_step+fa2`: `2 / 6`
  - `rank_tok+fa2_kv+g`: `1 / 6`
- overall winner:
  - `dense_step+fa2_g`: `3 / 6`
  - `dense_step+fa2`: `2 / 6`
  - `rank_tok+fa2_kv+g`: `1 / 6`
- low-rank variants: `0 / 6`

Additional low-rank standard baseline:

- Added `lowrank_reconstruct(fa2)`:
  - reconstruct dense `q/k/v` from low-rank factors
  - apply RoPE
  - call regular FA2 (not `with_kvcache`)
- This is the best low-rank baseline across all tested lengths in the current script.
- But it still loses to the best step winner by about `3.1x ~ 7.0x`.

Fair reconstruct-ablation baseline:

- Added `baseline(reconstruct_abl+fa2)` / `baseline(reconstruct_abl+fa2+graph)`:
  - `Q` still comes from low-rank weights
  - `K/V` are assumed already dense and post-RoPE ready
  - regular FA2 is called directly
- This is not a full online decode step.
- It is a lower-bound ablation for:
  - "what if current-token reconstruct/update were free?"

Latest fair-baseline comparison:

| L | rank_step_ms | rank_step_g_ms | recon_abl_ms | recon_abl_g_ms |
|---|---:|---:|---:|---:|
| 256 | 0.3328 | 0.3229 | 0.1865 | 0.1893 |
| 512 | 0.2783 | 0.1529 | 0.2307 | 0.0827 |
| 1024 | 0.2131 | 0.1559 | 0.2230 | 0.0952 |
| 2048 | 0.2983 | 0.2618 | 0.2475 | 0.1044 |
| 4096 | 0.1985 | 0.1982 | 0.2835 | 0.1700 |
| 8192 | 0.2176 | 0.2187 | 0.2670 | 0.1842 |

Interpretation of the fair baseline:

- `rank_step` does not beat the reconstruct-ablation baseline at any tested length.
- Gap vs graph-ablation baseline is still large:
  - about `1.17x` slower at `L=4096`
  - about `2.51x` slower at `L=2048`
- This confirms that the missing win is not mainly about extra graph overhead anymore.
- The remaining gap is dominated by the online reconstruct/update path itself.

Low-rank failures:

- `v1.6_v2(vk_resident=1)`: failed at all tested lengths
- `v1.6_v2(vk_resident=0)`: failed at all tested lengths
- Failure mode:
  - `L=256`: shared-memory OOR
  - `L>=512`: workspace mismatch

### Interpretation

- Moving the graph boundary forward helped benchmark hygiene:
  - rank graph input is now one `hidden token` copy, not three `Pq/Pk/Pv` copies.
- A decode-specific reconstruct config (`BD=128, BR=128, warps=8`) also improved the rank route relative to the first fair-step attempt.
- Adding the standard `lowrank_reconstruct(fa2)` baseline clarified another point:
  - the previous low-rank streaming / fused references were not the strongest non-dense comparison;
  - regular FA2 after low-rank reconstruction is a better low-rank reference here.
- But even after those fixes, `rank_step` still does not beat packed dense qkv on this host GPU.
- That means the limiting factor is no longer graph boundary overhead.
- The limiting factor is now the efficiency of `reconstruct_qkv_token_shared` itself versus vendor dense GEMM.

### Why The Rank Step Still Loses

- Theoretically the rank route does less math than dense packed qkv.
- In practice, dense packed qkv is one highly-optimized vendor GEMM:
  - `x @ W_qkv`
- The current rank route is still two stages:
  - `x @ U_qkv_rank`
  - `reconstruct_qkv_token_shared(...)`
- For `B=1`, `Dh=128`, `R=1024`, that second stage is still not efficient enough to beat the dense packed GEMM implementation.
- So the current conclusion is:
  - graph wrapping and forward-boundary fixes were necessary;
  - but they are not sufficient to make the rank step the winner.

### Current Status

- Shared global-rank benchmark: corrected and stable.
- Fair current-token step benchmark: added.
- Rank-route graph boundary: improved to `hidden token -> graph`.
- Attention dense short-context route in main `svd_llama.py`: not yet integrated.
- Attention graph in main model: not yet integrated.

### Update: Shared Prepacked Reconstruct Backend

What changed:

- Added a new shared-rank reconstruct backend:
  - prepack `Vq/Vk/Vv` once to `[R, H*Dh] / [R, Hk*Dh]`
  - use vendor `matmul` for shared decode reconstruction instead of the old per-head Triton reduction
- This backend now drives:
  - `rank_step`
  - `rank_step + graph`
  - shared-query reconstruction in the fair low-rank baselines

Implementation:

- basis packing and prepacked reconstruct helper:
  - [`flashsvdropeattn_dense_decode.py`](../kernels/flashsvd-v1.5/flashsvdropeattn_short/flashsvdropeattn_dense_decode.py)
- benchmark integration:
  - [`decode_compare.py`](../kernels/flashsvd-v1.5/flashsvdropeattn_short/decode_compare.py)

Representative results after switching shared reconstruct to the prepacked backend:

| L | step_winner | step_ms | dense_step_ms | dense_step_g_ms | rank_step_ms | rank_step_g_ms | recon_abl_ms | recon_abl_g_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | `rank_tok+fa2_kv+g` | 0.0981 | 0.1121 | 0.1153 | 0.1186 | 0.0981 | 0.1783 | 0.0985 |
| 512 | `rank_tok+fa2_kv+g` | 0.1018 | 0.1120 | 0.1156 | 0.1094 | 0.1018 | 0.1909 | 0.0840 |
| 1024 | `rank_tok+fa2_kv+g` | 0.1127 | 0.1215 | 0.1279 | 0.1197 | 0.1127 | 0.1906 | 0.0931 |
| 2048 | `rank_tok+fa2_kv+g` | 0.1779 | 0.2689 | 0.3029 | 0.4205 | 0.1779 | 0.1369 | 0.2303 |
| 4096 | `rank_tok+fa2_kv+g` | 0.1415 | 0.1498 | 0.1549 | 0.1434 | 0.1415 | 0.2033 | 0.1217 |
| 8192 | `rank_tok+fa2_kv+g` | 0.1832 | 0.1914 | 0.1981 | 0.1884 | 0.1832 | 0.1961 | 0.1616 |

Winner counts from the representative run above:

- full online step winner:
  - `rank_tok+fa2_kv+g`: `6 / 6`
- overall winner including the reconstruct-ablation lower bound:
  - `recon_abl+g`: `4 / 6`
  - `rank_tok+fa2_kv+g`: `1 / 6`
  - `recon_abl`: `1 / 6`

Interpretation:

- This is the first attention decode result that clearly flips the full online-step comparison in favor of the low-rank reconstruct path.
- The new backend did what the previous graph-only changes could not:
  - in repeated runs, `rank_step` / `rank_step + graph` are now in the same latency band as `dense_step`;
  - the low-rank online route wins `5-6 / 6` tested lengths depending on run-to-run noise.
- But it still does not beat the reconstruct-ablation lower bound consistently.
- So the status is now:
  - online step competition: won
  - lower-bound ablation competition: not fully won yet

### Next Attention Steps

1. If the goal is to beat dense packed qkv, the next real work item is a stronger `x -> qkv` rank-path implementation, not more graph wrapping.
2. The most likely high-ROI path is a true fused decode kernel that avoids materializing packed rank tokens before Q/K/V reconstruction.
3. Keep `dense + FA2 KVCache` as the short-context serving fallback/reference.
4. Fix the `v1.6_v2` workspace / resource issues before trusting it again.

## SwiGLU / FFN Decode

### What Changed

- Added decode-specialized SwiGLU FFN scheduling.
- Main idea:
  - avoid recomputing gating along the `R2` dimension;
  - use decode-friendly scheduling for tiny batch / tiny sequence;
  - keep `S @ V2 + b2` on vendor GEMM/addmm instead of over-fusing into a monster kernel.
- Also added CUDA Graph support around:
  - MLP-only path
  - decoder layer tail (`post_attention_layernorm -> mlp -> residual`)

### Benchmark Command

```bash
/home/zs89/miniconda3/envs/flashsvd/bin/python \
  /home/zs89/FlashSVD/FlashSVD-v1.5/bench_svd_llama_decode_graph.py \
  --batches 1 4 \
  --seq-len 1 \
  --warmup 20 \
  --runs 80
```

### Current Results

Config:

- Approximate `Llama-2-7B`
- `H=4096`
- `D=11008`
- `ratio=0.5`
- `rank=1492`
- `seq_len=1`

`B=1, L=1`:

- kernel: `3.7929 -> 2.2374 ms` (`1.695x`)
- layer eager: `2.6399 ms`
- `mlp_graph`: `2.5549 ms` (`1.033x`)
- `layer_tail_graph`: `2.4177 ms` (`1.092x`)
- numerical diff:
  - `eager vs mlp_graph`: `0`
  - `eager vs layer_tail_graph`: `0`

`B=4, L=1`:

- kernel: `3.9455 -> 2.2272 ms` (`1.771x`)
- layer eager: `2.6081 ms`
- `mlp_graph`: `2.5405 ms` (`1.027x`)
- `layer_tail_graph`: `2.4474 ms` (`1.066x`)
- numerical diff:
  - `eager vs mlp_graph`: `0`
  - `eager vs layer_tail_graph`: `5.69531`

### Interpretation

- The decode-specialized FFN kernel is solid:
  - about `1.7x` kernel-level improvement on this host GPU.
- CUDA Graph gains exist, but are modest compared to the kernel win:
  - roughly `3%` for `mlp_graph`
  - roughly `7% ~ 9%` for `layer_tail_graph`
- Important caveat:
  - `B=4` layer-tail graph currently shows a correctness issue in the benchmark (`max diff = 5.69531`)
  - so the `B=4 layer_tail_graph` performance number should not be treated as final

### SwiGLU Status

- Kernel work: good and useful.
- MLP CUDA Graph: already integrated.
- Layer-tail CUDA Graph: integrated, but still needs correctness cleanup for the `B=4` case.

## High-Level Takeaways

- For the target regime (`B=1`, single-token decode, context up to `8192`, edge-serving focus), the best current direction is:
  - attention: dense short-context route, not low-rank fused attention
  - FFN: decode-specialized FlashSVD SwiGLU remains worthwhile
- The core motivation remains valid:
  - SVD weight compression is most valuable for edge-style deployments
  - therefore decode latency at tiny batch is the metric that matters most

## Original-vs-Optimized Stack Comparison

### New Benchmark

- Added:
  - [`bench_decode_stack_compare.py`](../kernels/flashsvd-v1.5/flashsvdropeattn_short/bench_decode_stack_compare.py)
- Purpose:
  - compare the original SVD-Llama decode stack against the current optimized attention / MLP pieces using the same low-rank weights.
- Compared variants:
  - original attention + original MLP
  - optimized attention + original MLP
  - original attention + FlashSVDSwiGLU
  - optimized attention + FlashSVDSwiGLU

### Benchmark Command

```bash
/home/zs89/miniconda3/envs/flashsvd/bin/python \
  /home/zs89/FlashSVD/FlashSVD-v1.5/kernels/flashsvd-v1.5/flashsvdropeattn_short/bench_decode_stack_compare.py \
  --llama llama2-7b \
  --ratio 0.5 \
  --B 1 \
  --Ls 256,1024,4096,8192 \
  --dtype bf16 \
  --warmup 10 \
  --iters 40
```

### Current Results

Config:

- `llama2-7b`
- `B=1`
- `q_len=1`
- dense KV cache
- attention rank `1024`
- MLP rank `1492`
- `dtype=bf16`
- `gate_v_proj == up_v_proj` enabled to match shipped FlashSVD checkpoints

Important baseline note:

- the baseline in this stack benchmark is the **original SVD-Llama low-rank MLP**
- it still uses low-rank weights (`up_v/up_u`, `gate_v/gate_u`, `down_v/down_u`)
- it is **not** a densified-weight baseline
- “dense” in this section refers to dense activations / dense KV cache behavior, not densified MLP weights

Representative output:

- `L=256`
  - attention: `0.4348 -> 0.1450 ms` (`3.00x`)
  - MLP: `0.1654 -> 2.2161 ms` (`0.07x`)
  - full layer:
    - original: `0.6391 ms`
    - `+attn`: `0.4284 ms`
    - `+mlp`: `2.4487 ms`
    - `full`: `2.3559 ms`
- `L=1024`
  - attention: `0.3808 -> 0.1525 ms` (`2.50x`)
  - full layer:
    - original: `0.6378 ms`
    - `+attn`: `0.4307 ms`
    - `full`: `2.3727 ms`
- `L=4096`
  - attention: `0.3844 -> 0.1718 ms` (`2.24x`)
  - full layer:
    - original: `0.6175 ms`
    - `+attn`: `0.4157 ms`
    - `full`: `2.3990 ms`
- `L=8192`
  - attention: `0.5410 -> 0.2140 ms` (`2.53x`)
  - full layer:
    - original: `0.9421 ms`
    - `+attn`: `0.4534 ms`
    - `full`: `2.4309 ms`

Summary:

- attention average speedup: `2.57x`
- current integrated FlashSVDSwiGLU average speedup: `0.07x`
- full-layer average speedup with both enabled: `0.30x`

### Interpretation

- Attention:
  - the new short-context dense-KV route clearly beats the original dense-cache decode attention on this host GPU.
  - replacing only attention already gives a meaningful full-layer win (`~1.5x`, and `2.08x` at `L=8192`).
- MLP:
  - the currently integrated `kernels.flashsvdswiglu` path is not competitive in this stack benchmark.
  - so, right now, turning on both attention and MLP together is slower than the original layer.
- Practical conclusion:
  - attention optimization is ready to be taken seriously against the original SVD-Llama decode path.
  - MLP optimization still needs another integration pass before it should be enabled as the default serving path.

### Follow-up: shared-P cuBLAS MLP candidate

To answer whether FlashSVD MLP can still beat the original baseline with a different implementation shape, the stack benchmark was extended with:

- `shared_mlp`:
  - compute `P = up_v_proj(x)` once
  - use a packed `V1 = [gate_u; up_u]`
  - run the rest with vendor GEMMs / `addmm`
- `shared_mlp_g`:
  - the same shared-P cuBLAS path captured with CUDA Graph

Command:

```bash
/home/zs89/miniconda3/envs/flashsvd/bin/python \
  /home/zs89/FlashSVD/FlashSVD-v1.5/kernels/flashsvd-v1.5/flashsvdropeattn_short/bench_decode_stack_compare.py \
  --llama llama2-7b \
  --ratio 0.5 \
  --B 1 \
  --Ls 256,1024,4096,8192 \
  --dtype bf16 \
  --warmup 10 \
  --iters 40
```

Results:

- `L=256`
  - MLP: `orig 0.1769 ms`, `flash 2.2132 ms`, `shared 0.2393 ms`, `shared_g 0.2343 ms`
  - layer:
    - original: `0.6136 ms`
    - `+attn`: `0.4086 ms`
    - `+shared_g`: `0.4392 ms`
    - `best(attn+shared_g)`: `0.3691 ms` (`1.66x`)
- `L=1024`
  - `best(attn+shared_g)`: `0.3841 ms` vs original `0.6206 ms` (`1.62x`)
- `L=4096`
  - `best(attn+shared_g)`: `0.4287 ms` vs original `0.6171 ms` (`1.44x`)
- `L=8192`
  - `best(attn+shared_g)`: `0.4758 ms` vs original `0.7599 ms` (`1.60x`)

Summary:

- attention average speedup: `2.45x`
- current Triton FlashSVDSwiGLU: `0.08x`
- shared-P cuBLAS MLP: `0.71x`
- shared-P cuBLAS MLP + graph: `0.72x`
- best full-layer path (`optimized attention + shared_g MLP`): `1.58x`

Takeaway:

- the path that beats the original single-layer decode baseline is currently:
  - optimized attention
  - plus a shared-P cuBLAS MLP backend
- the current Triton FlashSVDSwiGLU kernel is still not the serving winner for this regime.

### Follow-up: shared-split MLP finally beats baseline

The previous shared-P cuBLAS path still packed `gate/up` into one large GEMM, which remained slower than the original low-rank MLP. A more conservative candidate was added:

- `shared_split_g`
  - compute `P = up_v_proj(x)` once
  - keep two separate `gate_u_proj(P)` / `up_u_proj(P)`
  - keep the rest of the original MLP structure unchanged
  - wrap this path with CUDA Graph

Baseline being beaten here:

- the original SVD-Llama low-rank MLP
- i.e. `up_u(up_v(x))`, `gate_u(gate_v(x))`, `down_u(down_v(silu(gate) * up))`
- still low-rank weights throughout
- not a densified-weight MLP

Representative command:

```bash
/home/zs89/miniconda3/envs/flashsvd/bin/python \
  /home/zs89/FlashSVD/FlashSVD-v1.5/kernels/flashsvd-v1.5/flashsvdropeattn_short/bench_decode_stack_compare.py \
  --llama llama2-7b \
  --ratio 0.5 \
  --B 1 \
  --Ls 256,1024,4096,8192 \
  --dtype bf16 \
  --warmup 10 \
  --iters 40
```

Results:

- MLP-only:
  - `L=256`: `orig 0.3618 ms`, `shared_split_g 0.3005 ms`
  - `L=1024`: `0.3608 -> 0.2816 ms`
  - `L=4096`: `0.3614 -> 0.2822 ms`
  - `L=8192`: `0.3431 -> 0.2980 ms`
- Average MLP speedup:
  - `shared_split_g = 1.23x`

Best full-layer path (`optimized attention + shared_split_g`):

- `L=256`: `1.0200 -> 0.4990 ms` (`2.04x`)
- `L=1024`: `0.8139 -> 0.5745 ms` (`1.42x`)
- `L=4096`: `0.8507 -> 0.6188 ms` (`1.37x`)
- `L=8192`: `1.7682 -> 0.7811 ms` (`2.26x`)

Summary:

- attention average speedup: `1.98x`
- `shared_split_g` average MLP speedup: `1.23x`
- best full-layer average speedup: `1.77x`

Current practical winner for the original single-layer decode baseline:

- optimized attention
- plus `shared_split_g` MLP

Current MLP record:

- winner: `shared_split_g`
- target regime:
  - `B=1`
  - `q_len=1`
  - `llama2-7b`
  - `ratio=0.5`
  - `gate_v_proj == up_v_proj`
- mean MLP speedup vs original low-rank MLP baseline: `1.23x`

### Follow-up: shared-P token kernel for `R≈1492`

To push beyond the Python `shared_split_g` path, a new decode-only shared-P backend was added to:

- [`kernels/flashsvdswiglu/__init__.py`](../kernels/flashsvdswiglu/__init__.py)
  - `flashsvd_ffn_shared_split_token(...)`
  - exact input contract:
    - `P`
    - `gate_u_proj.weight.T`
    - `up_u_proj.weight.T`
    - `down_v_proj.weight.T`
    - `down_u_proj.weight.T`

Design:

- target regime:
  - `B<=4`
  - `q_len=1`
  - Llama-2-7B half-rank-like `R≈1492`
- phase1 Triton schedule:
  - one program owns one token and one `D` tile
  - compute `gate/up` once from shared `P`
  - accumulate partial `S` over all rank outputs
  - final `S @ down_u` remains `torch.matmul` / `addmm`

This avoids the old `R2`-tile recompute problem and, unlike the older decode-only token path, does not require `R<=512`.

Representative command:

```bash
/home/zs89/miniconda3/envs/flashsvd/bin/python \
  /home/zs89/FlashSVD/FlashSVD-v1.5/kernels/flashsvd-v1.5/flashsvdropeattn_short/bench_decode_stack_compare.py \
  --llama llama2-7b \
  --ratio 0.5 \
  --B 1 \
  --Ls 256,1024,4096,8192 \
  --dtype bf16 \
  --warmup 15 \
  --iters 60
```

Latest results:

- MLP-only:
  - `L=256`: `orig 0.1781 ms`, `shared_split_g 0.1423 ms`, `shared_split_kernel_g 0.1450 ms`
  - `L=1024`: `orig 0.1596 ms`, `0.1426 ms`, `0.1437 ms`
  - `L=4096`: `orig 0.1597 ms`, `0.1428 ms`, `0.1443 ms`
  - `L=8192`: `orig 0.1580 ms`, `0.1422 ms`, `0.1436 ms`
- average MLP speedup vs original low-rank baseline:
  - `shared_split_g = 1.15x`
  - `shared_split_kernel_g = 1.14x`

Interpretation:

- the new token kernel does beat the original low-rank MLP baseline
- it is valid on the real half-rank regime (`R≈1492`)
- but it does **not** beat the current Python+CUDA-Graph `shared_split_g` winner yet
- config sweep showed the best current kernel variant is still the workspace-based reduction path with:
  - `BR=128`
  - `BD=128`
  - `BR2=128`
  - `num_warps=8`
  - `num_stages=2`
  - `store_partials_fp32=0`
- an fp32 atomic-accumulation variant was also tested and did not improve over the workspace path

Current MLP takeaway:

- the serving winner is still `shared_split_g`
- the new shared-P token kernel is now a viable experimental backend, but not yet the default decode choice
- to beat `shared_split_g`, the next gain likely needs to come from tighter main-path integration or a different phase1 operator, not another small tile tweak

## Directory Cleanup

The short-context attention folder was reorganized:

- top-level now keeps only the active files:
  - `decode_compare.py`
  - `bench_decode_stack_compare.py`
  - `flashsvdropeattn_dense_decode.py`
  - `flashsvdropeattn_v1.5_decode.py`
  - `flashsvdropeattn_v1.6_decode_opt.py`
- older exploratory scripts were moved to:
  - [`flashsvdropeattn_short/legacy/`](../kernels/flashsvd-v1.5/flashsvdropeattn_short/legacy)
