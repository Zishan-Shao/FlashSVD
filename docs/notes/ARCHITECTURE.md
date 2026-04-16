# FlashSVD-v1.5 Architecture

For the current runbook and verified benchmark numbers, see
[`CURRENT_STATUS.md`](./CURRENT_STATUS.md) and
[`quick_commands.md`](./quick_commands.md).

This directory now follows a production-vs-legacy split instead of a runtime backend registry.

## Runtime Ownership

- `flashsvd_component/svd_llama.py`
  Owns the production semantics for `SVD_LlamaAttention` and `SVD_LlamaMLP`.
  Attention/MLP `forward` stays here, and the production path launches the chosen kernel directly.

- `kernels/`
  Owns low-level Triton/CUDA kernels only.
  The top level should stay focused on current production kernels plus the archived `flashsvd-archive/` history bucket.
  Inside `flashsvd-archive/`, versioned implementations should live under `v1/`, `v1.5/`, and `v1.6/`.
  Current helper kernels that are still used by runtime should live directly under `kernels/`.
  These files should not define model semantics.

- `benchmark/`
  Owns correctness checks, performance benchmarks, and legacy experiment harnesses.
  Comparison code should live here instead of next to production kernels.
  Current attention microbenchmarks belong under `benchmark/attn/`.
  Archived compare harnesses belong under `benchmark/legacy/archive/`.
  Old one-off kernel experiments belong under `benchmark/legacy/experimental/`.

- `backend/`
  Owns backend-facing runtime entrypoints and helper loaders for both attention and MLP.
  Component code may import from here when it needs an explicit implementation helper, but this layer should not own model semantics.
  `backend/attn/` holds attention decode helper loading; `backend/mlp/` holds MLP exact decode implementations.

- `flashsvd_component/legacy/`
  Holds low-rank cache implementations and compatibility re-exports that are still needed for regression.

## Production Rules

- Attention production decode path:
  dense KV cache + reconstruct-current-token + FA2 KV cache

- MLP production decode path:
  exact packed-input `flashsvd_mlp_dual_split_prod`

- Prefill/full-sequence attention:
  component code prepares tensors, then launches the production kernel directly

## Legacy Rules

- Older attention decode kernels stay in `kernels/flashsvd-archive/v1*/...` for testing and benchmarking.
- Experimental/legacy MLP backends stay available under `backend/mlp/`, but should only be selected explicitly.
- Attention helper loading lives under `backend/attn/`; legacy low-rank decode kernels still live under `kernels/flashsvd-archive/v1*/...`.
- Production runtime should stay readable even if legacy paths remain for A/B and correctness checks.
