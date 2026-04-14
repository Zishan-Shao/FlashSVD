# FlashSVD-v1.5 Directory Reorg Plan

This plan makes one rule explicit:

- `kernels/` contains only kernel implementations
- `runtime/` contains dispatch, selection, cache plumbing, and runtime glue
- `models/` contains model integration (`Llama`, `Mistral`, `OPT`)
- `compat/` contains compatibility shims only

## Target Tree

```text
FlashSVD-v1.5/
  kernels/
    attn/
      common/
        flash_attn_causal.py
        mask_utils.py
      prefill/
        flashsvdattn.py
        rope_sdpa.py
      decode/
        densekv_reconstruct_triton.py
    mlp/
      prefill/
        flashsvdffn.py
      decode/
        dual_split_triton.py
    legacy/
      ... existing archive subtree kept as-is for now
  runtime/
    attn/
      __init__.py
      decode_registry.py
    cache/
      __init__.py
      attn_dense_kv.py
      attn_legacy_sparse_kv.py
    legacy/
      __init__.py
      decode.py
    README.md
  models/
    __init__.py
    llama.py
    mistral.py
    opt.py
  compat/
    component/
      __init__.py
      svd_llama.py
      svd_mistral.py
      svd_opt.py
```

## Exact File Migration Targets

| Current file | Target file | Why |
| --- | --- | --- |
| `backend/README.md` | `runtime/README.md` | `backend` currently mixes runtime and kernel language; this README is really about runtime/backend selection. |
| `backend/__init__.py` | delete | The package becomes `runtime/` plus `kernels/`; no need for a fake umbrella. |
| `backend/attn/README.md` | `runtime/attn/README.md` | This folder is dispatch/runtime, not kernels. |
| `backend/attn/__init__.py` | `runtime/attn/__init__.py` | Keep a small export surface for decode-registry helpers. |
| `backend/attn/flashsvd_attn_decode_registry.py` | `runtime/attn/decode_registry.py` | Pure runtime dispatch and capability detection. |
| `backend/mlp/README.md` | `kernels/mlp/README.md` | This README describes the active MLP kernel family. |
| `backend/mlp/__init__.py` | `kernels/mlp/__init__.py` | Re-export active MLP kernel entrypoints from the real kernel package. |
| `backend/mlp/flashsvd_mlp_dual_split_triton.py` | `kernels/mlp/decode/dual_split_triton.py` | Pure Triton kernel implementation. |
| `kernels/flash_attn_causal.py` | `kernels/attn/common/flash_attn_causal.py` | Attention-common helper kernel. |
| `kernels/utils_mask.py` | `kernels/attn/common/mask_utils.py` | Shared attention mask helper. |
| `kernels/flashsvdattn.py` | `kernels/attn/prefill/flashsvdattn.py` | Active attention prefill kernel. |
| `kernels/flashsvdropeattn.py` | `kernels/attn/prefill/rope_sdpa.py` | Active RoPE/prefill attention kernel. |
| `kernels/flashsvd_v15_attn_densekv_decode.py` | `kernels/attn/decode/densekv_reconstruct_triton.py` | Active dense-KV decode reconstruction kernel. |
| `kernels/flashsvdffn.py` | `kernels/mlp/prefill/flashsvdffn.py` | Active MLP/prefill kernel. |
| `kernels/flashsvd-archive/` | `kernels/legacy/` | Keep archive isolated from active kernels without rewriting its internal layout yet. |
| `flashsvd_component/svd_llama.py` | `models/llama.py` | Model integration, runtime decisions, HF-facing logic. |
| `flashsvd_component/svd_mistral.py` | `models/mistral.py` | Same rule as above. |
| `flashsvd_component/svd_opt.py` | `models/opt.py` | Same rule as above. |
| `flashsvd_component/flashsvd_v15_attn_dense_kv_cache.py` | `runtime/cache/attn_dense_kv.py` | Runtime cache implementation, not a model class and not a kernel. |
| `flashsvd_component/legacy/__init__.py` | `runtime/legacy/__init__.py` | Legacy runtime path package. |
| `flashsvd_component/legacy/decode.py` | `runtime/legacy/decode.py` | Legacy runtime decode glue. |
| `flashsvd_component/legacy/flashsvd_attn_legacy_sparse_kv_cache.py` | `runtime/cache/attn_legacy_sparse_kv.py` | Legacy sparse-KV cache runtime object. |
| `component/__init__.py` | `compat/component/__init__.py` | Keep compatibility aliases out of active model/runtime packages. |
| `component/svd_llama.py` | `compat/component/svd_llama.py` | Legacy pickle/import compatibility shim. |
| `component/svd_mistral.py` | `compat/component/svd_mistral.py` | Legacy pickle/import compatibility shim. |
| `component/svd_opt.py` | `compat/component/svd_opt.py` | Legacy pickle/import compatibility shim. |

## Mechanical Import Replacements

These replacements are intentionally chosen so they can be applied repo-wide with simple search/replace.

### Runtime / dispatch

```python
from backend.attn import ...
```

Replace with:

```python
from runtime.attn import ...
```

```python
from backend.attn.flashsvd_attn_decode_registry import ...
```

Replace with:

```python
from runtime.attn.decode_registry import ...
```

### MLP kernels

```python
from backend.mlp import ...
```

Replace with:

```python
from kernels.mlp import ...
```

### Model integration

```python
from flashsvd_component.svd_llama import ...
from flashsvd_component.svd_mistral import ...
from flashsvd_component.svd_opt import ...
```

Replace with:

```python
from models.llama import ...
from models.mistral import ...
from models.opt import ...
```

### Runtime caches / legacy runtime

```python
from flashsvd_component.flashsvd_v15_attn_dense_kv_cache import ...
```

Replace with:

```python
from runtime.cache.attn_dense_kv import ...
```

```python
from flashsvd_component.legacy.flashsvd_attn_legacy_sparse_kv_cache import ...
```

Replace with:

```python
from runtime.cache.attn_legacy_sparse_kv import ...
```

```python
from flashsvd_component.legacy.decode import ...
```

Replace with:

```python
from runtime.legacy.decode import ...
```

### Compatibility shims

```python
from component.svd_llama import ...
from component.svd_mistral import ...
from component.svd_opt import ...
```

Replace with:

```python
from compat.component.svd_llama import ...
from compat.component.svd_mistral import ...
from compat.component.svd_opt import ...
```

### Active kernel imports

```python
from kernels.flash_attn_causal import ...
```

Replace with:

```python
from kernels.attn.common.flash_attn_causal import ...
```

```python
from kernels.utils_mask import ...
```

Replace with:

```python
from kernels.attn.common.mask_utils import ...
```

```python
from kernels.flashsvdattn import ...
```

Replace with:

```python
from kernels.attn.prefill.flashsvdattn import ...
```

```python
from kernels.flashsvdropeattn import ...
```

Replace with:

```python
from kernels.attn.prefill.rope_sdpa import ...
```

```python
from kernels.flashsvdffn import ...
```

Replace with:

```python
from kernels.mlp.prefill.flashsvdffn import ...
```

### Path-based archive loads

```python
"kernels" / "flashsvd-archive"
```

Replace with:

```python
"kernels" / "legacy"
```

## Current Import Sites Covered By The Mechanical Rules

The replacements above cover the active imports currently used in:

- `SVDLLM.py`
- `SVDLLM_v2.py`
- `SVDLLM_v2_hetero.py`
- `utils/model_utils.py`
- `evaluater.py`
- `SVDLLM_flashsvd.py`
- `SVDLLM_v2_flashsvd.py`
- `SVDLLM_v2_hetero_flashsvd.py`
- `flashsvd_component/svd_llama.py`
- `flashsvd_component/legacy/decode.py`
- `benchmark/attn/bench_real_checkpoint_decode_reconstruct.py`
- `benchmark/attn/generate_dense_decode_plan.py`
- `benchmark/decode/check_flashsvd_decode_correctness.py`
- `benchmark/mlp/bench_svd_llama_decode_graph.py`
- `benchmark/mlp/legacy_swiglu/__init__.py`

## Suggested Low-Risk Execution Order

1. Create the new packages and move files physically.
2. Add temporary one-line shim `__init__.py` files so old imports still work.
3. Apply the mechanical import replacements repo-wide.
4. Run smoke imports and benchmark `--help` commands.
5. Delete the old shim packages only after everything imports cleanly.

## Important Non-Goal For This Pass

Do not deeply normalize the internals of `kernels/legacy/` in the same patch.
The first pass should make the active code easy to follow.
Archive cleanup can happen later.
