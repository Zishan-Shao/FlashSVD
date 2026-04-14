# FlashSVD v1.5 Speedup Tricks

Generated: 2026-03-17

This note summarizes the tricks that are actually responsible for the current FlashSVD v1.5 gains on `SVD-LLM v1`, `SVD-LLM v2`, and `Basis Sharing`.

Use this as paper-writing material, not as a changelog.

## One-Sentence Summary

FlashSVD v1.5 is fast not because low-rank factorization is automatically fast, but because the runtime is redesigned around the regimes where low-rank models are actually served:

- decode: `dense KV cache + current-token reconstruction + FA2`
- MLP decode: packed exact `flashsvd_mlp_dual_split_prod`
- prefill: full-sequence factorized attention kernel
- runtime: aggressive prepack, cache reuse, and Basis Sharing-aware weight tying

## Current Measured Headline

Under the current A100 setup (`bf16`, `B=1`, `prompt_len=512`, `new_tokens=32`):

- vs low-rank baseline:
  - prefill: about `1.25x` on average
  - decode: about `1.48x` on average
  - end-to-end: about `1.44x` on average
- vs dense baseline:
  - prefill: about `3.76x` on average
  - decode: about `1.12x` on average
  - end-to-end: about `1.45x` on average

The main consistent gain is still decode. Prefill gains are more sensitive to the baseline definition and model family.

## Main Tricks

### 1. Dense-KV decode instead of low-rank KV decode

This is the most important trick in the current winner path.

- What it does:
  - keep low-rank weights in the model
  - reconstruct only the current token's dense `Q/K/V`
  - store past `K/V` in a dense cache
  - call `flash_attn_with_kvcache` for the actual decode attention
- Why it helps:
  - removes repeated low-rank reconstruction over the entire decode history
  - matches the execution style that FA2 and vendor kernels are good at
  - turns the online step into "small current-token reconstruction + fast dense KV attention"
- Where it lives:
  - runtime direction documented in `notes/CURRENT_STATUS.md`
  - decode implementation in `flashsvd_component/svd_llama.py`
- Why it matters:
  - this is the primary reason decode is consistently `~1.5x` faster than the low-rank runtime baseline

Paper wording:
`FlashSVD v1.5 replaces online low-rank KV-cache decode with a dense-KV serving path that reconstructs only the current token and delegates the historical attention step to FA2.`

### 2. Packed current-token QKV reconstruction

The current-token attention step is not implemented as three separate small low-rank projections.

- What it does:
  - concatenate rank-side `q/k/v` input factors into one packed tensor
  - perform a single `hidden @ packed_qkv_rank`
  - split the rank result into `q_rank`, `k_rank`, `v_rank`
  - reconstruct dense `q/k/v` from prepacked shared bases
- Why it helps:
  - reduces kernel launch count
  - improves memory locality
  - avoids repeated tiny matmuls
  - matches the shared-rank structure of the supported checkpoints
- Where it lives:
  - `flashsvd_component/svd_llama.py`
  - the current-token path starts around `_flashsvd_dense_decode_token_from_hidden`
- Why it matters:
  - this makes low-rank decode look like one packed rank projection plus one efficient reconstruct step instead of a sequence of fragmented operations

Paper wording:
`We pack the rank-space QKV projection and reconstruct dense current-token QKV from shared prepacked bases, reducing decode-time fragmentation.`

### 3. FA2-friendly dense KV cache layout

The cache layout is part of the speedup, not just a bookkeeping detail.

- What it does:
  - stores dense KV as `[B, S, H_k, D_h]`
  - stores `K` after RoPE
  - maintains FA2-compatible sequence-length buffers
- Why it helps:
  - avoids layout conversion at decode time
  - lets FlashSVD and reference dense decode share the same cache contract
  - reduces glue overhead around attention
- Where it lives:
  - `flashsvd_component/flashsvd_v15_attn_dense_kv_cache.py`
- Why it matters:
  - without this cache contract, the dense-KV winner path would lose much of its practical advantage

Paper wording:
`FlashSVD v1.5 uses a dense KV cache layout tailored to FA2 decode, including post-RoPE keys and reusable cache-length buffers.`

### 4. Packed exact MLP decode (`flashsvd_mlp_dual_split_prod`)

The current MLP winner is not a fully fused exotic kernel; it is a decode-specialized packed exact path.

- What it does:
  - concatenate `up_v` and `gate_v` into one cached `v_cat`
  - do one input-side projection `p_cat = x @ v_cat^T`
  - split into `p_up` and `p_gate`
  - keep the exact `U/down` path unchanged
- Why it helps:
  - reduces input-side duplication
  - avoids two separate rank-side projections
  - preserves exact model structure
  - improves the tiny-batch token-decode regime without changing model outputs
- Where it lives:
  - `flashsvd_component/svd_llama.py`
  - `_get_flashsvd_mlp_dual_split_prod_factors`
  - `_forward_flashsvd_mlp_dual_split_prod`
- Why it matters:
  - MLP is not the biggest source of speedup, but this gives a real and stable decode improvement on top of the attention win

Paper wording:
`For exact low-rank MLP decode, we use a packed dual-split path that computes the up/gate input projection once and reuses it across the SwiGLU branches.`

### 5. Decode-regime specialization

FlashSVD v1.5 does not force one runtime path on every sequence shape.

- What it does:
  - identifies the small-batch token-decode regime
  - switches to decode-specialized attention and MLP paths only when it makes sense
  - otherwise keeps the exact non-token path
- Why it helps:
  - avoids paying decode-specific overhead in the wrong regime
  - lets the system be aggressive where the serving bottleneck actually is
- Where it lives:
  - `_prefer_token_decode` in `flashsvd_component/svd_llama.py`
- Why it matters:
  - many good decode tricks are bad general-purpose tricks; this guard keeps the winner path narrow and robust

Paper wording:
`We specialize FlashSVD v1.5 for the token-decode regime instead of applying decode-oriented kernels uniformly to all sequence lengths.`

### 6. Full-sequence factorized prefill kernel

Prefill is accelerated with a different trick from decode.

- What it does:
  - stays in factorized `P/V` form for full-sequence attention
  - applies RoPE and masking inside the FlashSVD full-seq kernel
  - avoids explicit dense `Q/K/V` materialization in the FlashSVD path
- Why it helps:
  - reduces redundant dense reconstruction during prefill
  - keeps the full-sequence path more aligned with the compressed representation
- Where it lives:
  - `_run_flashsvd_prefill_kernel` in `flashsvd_component/svd_llama.py`
  - called from the `q_len > 1` branch in attention forward
- Why it matters:
  - this is the source of the prefill gain when FlashSVD beats the low-rank runtime baseline
  - however, prefill gains are less stable because the comparison baseline matters a lot

Paper wording:
`For prefill, FlashSVD v1.5 uses a full-sequence factorized attention kernel that performs RoPE and causal masking directly on the low-rank factors.`

## Runtime Support Tricks

These are not the whole story by themselves, but they are necessary for the main kernels to win in practice.

### 7. Prepack and cache static factors

- prepack decode tensors once per `(device, dtype)`
- cache `v_cat` for MLP
- cache packed decode bases for attention
- reuse temporary workspaces

Why it matters:

- tiny decode kernels are very sensitive to setup overhead
- without prepacking, the theoretical low-rank FLOP reduction does not translate into latency reduction

### 8. Zero-stride head expansion instead of materialization

- expand rank factors across heads as views
- avoid explicit copies when the factors are shared across heads

Why it matters:

- this saves bandwidth and temporary allocations
- especially important in the prefill path where tensors are larger

### 9. CUDA Graph on the decode tail

- currently used mainly around the stable decode-side MLP path
- reduces per-step launch overhead in the `B <= 4, q_len = 1` regime

Why it matters:

- decode is small enough that launch overhead can dominate
- graph capture is a support trick that helps the packed decode path show up in wall-clock latency

### 10. Basis Sharing-aware parameter tying

Basis Sharing is not handled as "equal tensors loaded independently."

- What it does:
  - restore actual shared `Parameter` objects across layer groups
  - validate that grouped layers are really identical before tying
  - preserve shared-rank assumptions for attention and MLP
- Why it helps:
  - avoids repacking identical bases separately
  - reduces redundant memory traffic and runtime preparation
  - lets Basis Sharing checkpoints enter the same optimized FlashSVD path
- Where it lives:
  - `utils/model_utils.py`
  - `_tie_basis_sharing_layer_groups`
  - `_finalize_basis_sharing_flashsvd_model`
- Why it matters:
  - this is what makes Basis Sharing a real systems integration result instead of a format-only loader

Paper wording:
`For Basis Sharing checkpoints, we restore true parameter sharing across grouped layers so the runtime can reuse packed shared bases instead of handling identical tensors independently.`

## Which Tricks Actually Drove the Current Gains

If the paper needs a priority ordering, use this one:

1. `dense KV + current-token reconstruct + FA2`
2. packed current-token QKV reconstruction
3. packed exact MLP decode (`flashsvd_mlp_dual_split_prod`)
4. prepack/cache/layout hygiene
5. CUDA Graph
6. Basis Sharing-aware tying and reuse
7. full-sequence factorized prefill kernel

Interpretation:

- decode gains are dominated by the first three items
- prefill gains come mostly from the full-sequence kernel, but are more baseline-sensitive
- Basis Sharing support is not just loader breadth; the tied-basis runtime matters

## Important Caveats

### Prefill and decode are accelerated for different reasons

- decode wins mainly because the runtime changes shape completely
- prefill wins mainly because the kernel stays factorized

Do not describe FlashSVD as "one kernel that makes everything faster." That is not the current system story.

### Low-rank factorization alone is not the trick

This is an important writing point.

- low-rank models are not automatically faster
- naive low-rank serving can still be slow
- the speedup comes from matching the compressed structure to the runtime and kernel design

Good wording:
`The compressed representation creates an opportunity for faster serving, but realizing that opportunity requires decode- and prefill-specific kernel/runtime co-design.`

### Some paths are not current winners

These should not be presented as the main production contributions:

- old low-rank KV decode kernels
- generic fused `FlashSVDSiLU` as a headline path
- current CUTLASS RoPEAttn scaffold

Those are useful experiments or fallback infrastructure, but not the core explanation for the paper's current results.

## Recommended Paper Framing

If you want one short paragraph for the paper:

`FlashSVD v1.5 accelerates SVD-compressed LLM inference through a combination of decode- and prefill-specific system tricks. For decode, it replaces online low-rank KV-cache attention with a dense-KV path that reconstructs only the current token from packed low-rank factors and dispatches the historical attention step to FA2. For MLP decode, it uses a packed exact dual-split path that reuses the input-side projection across the SwiGLU branches. For prefill, it uses a full-sequence factorized attention kernel that performs RoPE and masking directly on low-rank factors. These kernels are supported by runtime optimizations including static factor prepacking, FA2-friendly dense cache layout, CUDA Graph capture, zero-stride head expansion, and Basis Sharing-aware parameter tying.` 

## Useful Code Pointers

- decode attention winner path:
  - `flashsvd_component/svd_llama.py`
- packed exact MLP:
  - `flashsvd_component/svd_llama.py`
- dense KV cache:
  - `flashsvd_component/flashsvd_v15_attn_dense_kv_cache.py`
- full-sequence prefill kernel entry:
  - `flashsvd_component/svd_llama.py`
- Basis Sharing tying and finalize logic:
  - `utils/model_utils.py`
