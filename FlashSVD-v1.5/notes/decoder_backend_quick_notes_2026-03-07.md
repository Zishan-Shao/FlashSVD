# Decoder Backend Quick Notes (2026-03-07)

## Scope

This note is only about the target serving regime we actually care about:

- decode only
- `B = 1` as the main case, `B <= 4` as a secondary case
- `q_len = 1`
- context length `N <= 8192`
- edge-oriented deployment
- low-rank SVD weights remain the baseline

Important baseline convention:

- MLP baseline here is the original SVD-Llama low-rank MLP
- attention baseline here is the original SVD-Llama low-rank decode attention
- this note is **not** comparing against densified MLP weights

## Recommended Decoder Backend

All active decoder recommendations below are exact-only. Approximate shared-`P` MLP routes were useful as experiments, but they are no longer part of the serving path or exposed in the active CLI.

If we install the current best-performing pieces into the decoder backend, the recommendation is:

### Attention: use the optimized dense-KV decode path

Use:

- low-rank `q/k/v` projection
- reconstruct only the current token
- dense KV cache
- `flash_attn_with_kvcache`
- CUDA Graph around the stable decode path

Current practical status:

- this is the clear attention winner in the target regime
- it consistently beats the original attention path
- in the latest stack benchmark it gives about `2.25x ~ 2.64x` attention-only speedup
- average attention speedup in the latest run: `2.52x`

Representative numbers:

- `L=256`: `0.3843 -> 0.1489 ms` (`2.58x`)
- `L=1024`: `0.3692 -> 0.1400 ms` (`2.64x`)
- `L=4096`: `0.3691 -> 0.1641 ms` (`2.25x`)
- `L=8192`: `0.5251 -> 0.2017 ms` (`2.60x`)

### MLP: exact `dual_split_cublas` on token decode

Use:

- packed exact input-side projection:
  - `p_cat = x @ [up_v; gate_v]`
- then split into:
  - `p_up`
  - `p_gate`
- then keep the exact original structure:
  - `gate = gate_u_proj(p_gate)`
  - `up = up_u_proj(p_up)`
  - `down = down_u_proj(down_v_proj(silu(gate) * up))`
- wrap this path with CUDA Graph at `mlp` scope

Current practical status on the real 7B checkpoint:

- this is the exact MLP default in `auto`
- it is structurally exact for checkpoints where `gate_v_proj != up_v_proj`
- it beats the original exact low-rank MLP baseline, but only modestly

Latest representative exact numbers:

- `L=1`, `layer=0`: `0.1597 -> 0.1556 ms`
- real profile `mlp_total`: `9.667 -> 9.010 ms`

Average MLP speedup in the latest real runs:

- microbench: about `1.03x`
- decode profile component: about `1.07x`
- caveat: this is layer-dependent; some real layers still regress slightly, so exact MLP is not yet a universal win

## What Not To Use As Default

### Do not use the generic Triton `flashsvd_ffn_swiglu` as the decode default

Reason:

- it is still bad in the real `llama2-7b`, half-rank, `q_len=1` regime
- latest average result is only about `0.07x` vs the original low-rank MLP baseline

### Do not default to the older low-rank fused attention kernels for shared global-rank serving

Reason:

- in the more realistic shared global-rank decode setting, they are not the latency winner
- the dense-KV + current-token reconstruct + FA2 path is better for the target regime

## Experimental Backend

### `dual_split_kernel`

File:

- [`flashsvdswiglu/__init__.py`](../kernels/flashsvdswiglu/__init__.py)

Function:

- `flashsvd_ffn_dual_split_token(...)`

Input contract:

- `P_up`
- `P_gate`
- `gate_u_proj.weight.T`
- `up_u_proj.weight.T`
- `down_v_proj.weight.T`
- `down_u_proj.weight.T`

Status:

- valid on the real half-rank regime (`R≈1492`)
- exact on the real checkpoint
- currently still loses to `dual_split_cublas`

Latest representative numbers:

- `layer=0`, `B=1`, `L=1`: `0.1915 ms`

Average MLP speedup in the latest run:

- `dual_split_kernel = 0.83x` vs the exact baseline

Decision:

- keep it as an experimental exact backend
- do not make it the default decoder MLP path yet

## Best Current Full-Layer Combination

If we want the best currently validated decoder-layer direction:

1. Attention: optimized dense-KV + reconstruct-current-token + FA2 KVCache + CUDA Graph
2. MLP: exact `dual_split_cublas`

Latest full-layer numbers:

- exact whole-model path now benchmarks closer to `1.34x ~ 1.43x` end-to-end on the real checkpoint

Average full-layer speedup in the latest run:

- see the exact real-checkpoint section below; the older `2.16x` stack number depended on approximate MLP experiments and should not be used as the serving expectation

## Real Checkpoint Default

For the real `jeffwan_llama_7b_hf_whitening_only_0.5.pt` checkpoint, the exact default should now be:

1. Attention: optimized dense-KV + current-token reconstruct + FA2 KVCache
2. MLP:
   - decode token path: `dual_split_cublas` in `auto`
   - non-token path/prefill: exact original low-rank MLP
3. `dual_split_kernel` remains experimental

Real aligned end-to-end smoke run (`prompt_len=512`, `new_tokens=32`, `B=1`, `bf16`):

- SVD baseline: `33.589 ms/token`
- FlashSVD exact auto: `30.048 ms/token`
- decode speedup: `1.12x`

Real exact MLP update (2026-03-07, later pass):

- `dual_split_cublas` now uses a packed input-side exact path:
  - one packed `up_v/gate_v` projection
  - exact original `gate_u/up_u/down_v/down_u` structure
- `dual_split_cublas_legacy` keeps the older exact path for A/B

Real checkpoint microbench (`layer=0`, `B=1`, `L=1`, `bf16`, CUDA Graph on):

- baseline exact low-rank MLP: `0.1597 ms`
- `dual_split_cublas_legacy`: `0.1597 ms`
- new `dual_split_cublas`: `0.1556 ms`
- exact MLP speedup vs baseline: `1.03x`

Real decode profile (`prompt_len=256`, `new_tokens=8`, dense-KV attention winner):

- legacy exact MLP: `mlp_total = 9.667 ms`
- new exact packed-input MLP: `mlp_total = 9.010 ms`
- exact MLP component speedup in profile: `1.07x`

For end-to-end benchmarking, `mlp` CUDA Graph scope is currently better than `layer_tail` on the real 7B checkpoint:

- `bench_flashsvd_vs_svd_decode.py` now defaults to `--mlp_cuda_graph_scope mlp`
- `SVDLLM_flashsvd.py` now exposes `--flashsvd_mlp_cuda_graph` and `--flashsvd_mlp_graph_scope`, defaulting to graph on + `mlp`
- SVD baseline should remain graph-off for fairness to the original implementation; the compare benchmark already forces baseline graph off

Graph alias-output experiment:

- tried making the FlashSVD graph path return the static graph output buffer directly instead of `clone()`
- result on the real 7B checkpoint was effectively noise:
  - clone path: `24.847 ms/token`
  - alias path: `24.892 ms/token`
- decision: keep alias-output as opt-in only, not the default

Real end-to-end smoke run with the better `mlp` graph scope (`prompt_len=512`, `new_tokens=32`, `B=1`, `bf16`):

- SVD baseline: `33.418 ms/token`
- FlashSVD exact auto: `25.013 ms/token`
- decode speedup: `1.34x`

## Integration Guidance

If wiring into `svd_llama.py` now:

1. Put the optimized attention path on the decode fast path for `B<=4`, `q_len=1`, dense KV serving.
2. Put exact `dual_split_cublas` on the MLP decode fast path.
3. Keep `dual_split_kernel` behind an opt-in flag until it clearly beats `dual_split_cublas`.
4. Keep approximate shared-`P` MLP experiments out of the serving path.

## Current Component Wiring

As of 2026-03-07, the real component path under `flashsvd_component/` is wired as follows:

- Attention winner is integrated in:
  - `flashsvd_component/svd_llama.py`
  - decode path: `SVD_LlamaAttention.forward(...)`
  - cache type: `flashsvd_component/dense_cache.py::FlashSVDDenseKVCache`
- MLP winner is integrated in:
  - `flashsvd_component/svd_llama.py`
  - decode path: `SVD_LlamaMLP._forward_flashsvd_core(...)`
  - real 7B exact default: `FLASH_SVD_FFN_BACKEND=auto` (decode token path resolves to `dual_split_cublas`)
  - approximate shared-`P` backends have been removed from the active component/CLI path

Bench entrypoints:

- direct model script:
  - `SVDLLM_flashsvd.py --step 6`
- compare FlashSVD vs normal SVD on the same checkpoint:
  - `bench_flashsvd_vs_svd_decode.py`
- v2 wrapper:
  - `SVDLLM_v2_flashsvd.py`
  - defaults to `--mlp_cuda_graph_scope mlp` when benchmarking, to avoid the older layer-tail patch issue

## Real 7B Smoke Results

Checkpoint used:

- `jeffwan_llama_7b_hf_whitening_only_0.5.pt`

Real decode benchmark through the integrated component path:

```bash
/home/zs89/miniconda3/envs/flashsvd/bin/python \
  /home/zs89/FlashSVD/FlashSVD-v1.5/bench_flashsvd_vs_svd_decode.py \
  --checkpoint /home/zs89/FlashSVD/FlashSVD-v1.5/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --dtype bf16 \
  --device cuda \
  --prompt_len 512 \
  --new_tokens 32 \
  --warmup 3 \
  --batch_size 1 \
  --flashsvd_ffn_backend auto
```

Observed result:

- SVD baseline decode: `33.451 ms/token`
- FlashSVD decode: `29.331 ms/token`
- end-to-end decode speedup on this smoke run: `1.14x`
- prefill also improved:
  - SVD: `0.285 s`
  - FlashSVD: `0.070 s`

Direct script smoke runs also passed:

- `SVDLLM_flashsvd.py --step 6`
  - `prompt_len=256`
  - `new_tokens=8`
  - `decode = 29.625 ms/token`
- `SVDLLM_v2_flashsvd.py`
  - wrapper mode
  - `prompt_len=256`
  - `new_tokens=8`
  - `decode = 25.636 ms/token`
