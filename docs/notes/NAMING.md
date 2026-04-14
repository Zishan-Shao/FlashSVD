# FlashSVD-v1.5 Naming

Last updated: 2026-03-30

This note defines the canonical names used in runtime, benchmarks, and docs.

## Top-Level Rule

- `FlashSVD-v1.5`
  Only refers to the current formal production system:
  `dense KV cache + current-token reconstruct + FA2 KV-cache decode` for attention,
  `flashsvd_mlp_dual_split_prod` for decode MLP,
  and the current factorized prefill path.

Everything else should be called out explicitly as `legacy`, `experimental`, or `compare` path.

## Canonical Attention Names

- `FlashSVD-v1.5`
  Current production attention route with dense KV cache.
- `FlashSVD-attn-legacy-v1.5-sparsekv`
  Old low-rank KV-cache decode kernel line.
- `FlashSVD-attn-legacy-v1.6-sparsekv`
  Experimental v1.6 sparse-KV decode line.

## Legacy Decoding Folder Names

- `benchmark/legacy/archive/decoding/`
  Unified benchmark folder for legacy decoding compare scripts and stack microbenches.
- `kernels/flashsvd-archive/v*/decoding/`
  Versioned legacy decoding kernels that are still kept for regression and A/B work.

## Canonical MLP Backend Names

- `FlashSVD-mlp-dual-split-prod`
  Runtime/backend string: `flashsvd_mlp_dual_split_prod`
  Current production decode MLP backend.
- `FlashSVD-mlp-dual-split-exact-legacy`
  Runtime/backend string: `flashsvd_mlp_dual_split_exact_legacy`
  Exact-safe compare/reference backend.
- `FlashSVD-mlp-dual-split-triton-v1`
  Runtime/backend string: `flashsvd_mlp_dual_split_triton_v1`
- `FlashSVD-mlp-dual-split-triton-v2`
  Runtime/backend string: `flashsvd_mlp_dual_split_triton_v2`
- `FlashSVD-mlp-dual-split-triton-v2_sm80`
  Runtime/backend string: `flashsvd_mlp_dual_split_triton_v2_sm80`
- `FlashSVD-mlp-dual-split-triton-v3`
  Runtime/backend string: `flashsvd_mlp_dual_split_triton_v3`

## Compatibility Rule

- Old strings like `dual_split_cublas`, `dual_split_cublas_legacy`, `dual_split_kernel*`, and older attention decode aliases may still be accepted for backward compatibility.
- New docs, benchmark defaults, logs, and discussion should use the canonical names above.
