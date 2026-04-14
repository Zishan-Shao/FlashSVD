from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import math
from typing import Callable, Optional

import torch

from runtime.attn import call_flash_attn_with_kvcache, get_flash_attn_with_kvcache


_FLASHINFER_SINGLE_DECODE = None
_FLASHINFER_SINGLE_DECODE_RESOLVED = False
_FLASHINFER_APPLY_ROPE_INPLACE = None
_FLASHINFER_APPLY_ROPE_INPLACE_RESOLVED = False
_CALLABLE_PARAM_CACHE: dict[int, Optional[frozenset[str]]] = {}


def _maybe_kwargs(fn, kwargs: dict[str, object]) -> dict[str, object]:
    key = id(fn)
    cached = _CALLABLE_PARAM_CACHE.get(key, None)
    if cached is None and key not in _CALLABLE_PARAM_CACHE:
        try:
            cached = frozenset(inspect.signature(fn).parameters)
        except Exception:
            cached = None
        _CALLABLE_PARAM_CACHE[key] = cached
    if cached is None:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in cached}


def get_flashinfer_single_decode_with_kv_cache():
    global _FLASHINFER_SINGLE_DECODE
    global _FLASHINFER_SINGLE_DECODE_RESOLVED
    if _FLASHINFER_SINGLE_DECODE_RESOLVED:
        return _FLASHINFER_SINGLE_DECODE

    fn = None
    try:
        import flashinfer  # type: ignore

        decode_mod = getattr(flashinfer, "decode", None)
        if decode_mod is not None:
            fn = getattr(decode_mod, "single_decode_with_kv_cache", None)
        if fn is None:
            fn = getattr(flashinfer, "single_decode_with_kv_cache", None)
    except Exception:
        fn = None

    _FLASHINFER_SINGLE_DECODE = fn
    _FLASHINFER_SINGLE_DECODE_RESOLVED = True
    return _FLASHINFER_SINGLE_DECODE


def get_flashinfer_apply_rope_inplace():
    global _FLASHINFER_APPLY_ROPE_INPLACE
    global _FLASHINFER_APPLY_ROPE_INPLACE_RESOLVED
    if _FLASHINFER_APPLY_ROPE_INPLACE_RESOLVED:
        return _FLASHINFER_APPLY_ROPE_INPLACE

    fn = None
    try:
        import flashinfer  # type: ignore

        fn = getattr(flashinfer, "apply_rope_inplace", None)
    except Exception:
        fn = None

    _FLASHINFER_APPLY_ROPE_INPLACE = fn
    _FLASHINFER_APPLY_ROPE_INPLACE_RESOLVED = True
    return _FLASHINFER_APPLY_ROPE_INPLACE


def has_flashinfer() -> bool:
    return get_flashinfer_single_decode_with_kv_cache() is not None


@dataclass
class FlashInferDenseKVCacheView:
    """Dense KV cache bindings for a single decode step.

    The current serving winner uses dense KV layout `[B, S, Hk, Dh]`. This view
    carries exactly the tensors a flashinfer-style decode kernel should consume:
    full cache tensors, current-token write views, RoPE tables, and an optional
    cache-advance callback.
    """

    k_cache_bmhd: torch.Tensor
    v_cache_bmhd: torch.Tensor
    cache_seqlens: torch.Tensor
    decode_positions: torch.Tensor
    kv_len_hint: Optional[int] = None
    rope_offsets: Optional[torch.Tensor] = None
    rope_indptr: Optional[torch.Tensor] = None
    rotary_cos: Optional[torch.Tensor] = None
    rotary_sin: Optional[torch.Tensor] = None
    k_write_bmhd: Optional[torch.Tensor] = None
    v_write_bmhd: Optional[torch.Tensor] = None
    flash_attn_with_kvcache: Optional[Callable[..., torch.Tensor]] = None
    advance_after_step: Optional[Callable[[], None]] = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def supports_direct_write(self) -> bool:
        return self.k_write_bmhd is not None and self.v_write_bmhd is not None

    def kv_len(self, batch_idx: int = 0) -> int:
        if int(batch_idx) == 0 and self.kv_len_hint is not None:
            return int(self.kv_len_hint)
        return int(self.cache_seqlens[int(batch_idx)].item())


@dataclass
class FlashInferDenseDecodePlan:
    """Decode plan for a flashinfer-style dense-KV token step."""

    reconstruct_qkv: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    num_heads: int
    head_dim: int
    project_output: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    reconstruct_qkv_into_cache: Optional[
        Callable[[torch.Tensor, FlashInferDenseKVCacheView], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = None
    backend_hint: str = "auto"
    sm_scale: Optional[float] = None
    rope_theta: float = 10000.0
    metadata: dict[str, object] = field(default_factory=dict)


def _gather_llama_rope(
    *,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    decode_positions: torch.Tensor,
    batch_size: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = decode_positions.to(device=rotary_cos.device, dtype=torch.long).reshape(-1)
    if int(pos.numel()) == 1 and int(batch_size) > 1:
        pos = pos.expand(int(batch_size))
    if int(pos.numel()) != int(batch_size):
        raise ValueError(f"decode_positions numel={int(pos.numel())} != batch={int(batch_size)}")
    cos = rotary_cos.index_select(0, pos).view(int(batch_size), 1, 1, int(head_dim) // 2)
    sin = rotary_sin.index_select(0, pos).view(int(batch_size), 1, 1, int(head_dim) // 2)
    return cos, sin


def apply_llama_rope_bmhd(
    x_bmhd: torch.Tensor,
    *,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    decode_positions: torch.Tensor,
) -> torch.Tensor:
    if int(x_bmhd.shape[-1]) % 2 != 0:
        raise ValueError(f"head_dim must be even, got {int(x_bmhd.shape[-1])}")
    cos, sin = _gather_llama_rope(
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        decode_positions=decode_positions,
        batch_size=int(x_bmhd.shape[0]),
        head_dim=int(x_bmhd.shape[-1]),
    )
    half = int(x_bmhd.shape[-1]) // 2
    x0 = x_bmhd[..., :half]
    x1 = x_bmhd[..., half:]
    return torch.cat((x0 * cos - x1 * sin, x1 * cos + x0 * sin), dim=-1)


def _normalize_decode_output(
    out: torch.Tensor,
    *,
    batch_size: int,
    q_len: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if out.shape == (batch_size, q_len, num_heads, head_dim):
        return out.reshape(batch_size, q_len, num_heads * head_dim).contiguous()
    if out.shape == (batch_size, num_heads, q_len, head_dim):
        return out.transpose(1, 2).reshape(batch_size, q_len, num_heads * head_dim).contiguous()
    if out.shape == (num_heads, head_dim):
        return out.reshape(1, 1, num_heads * head_dim).contiguous()
    if out.shape == (1, num_heads, head_dim):
        return out.reshape(1, 1, num_heads * head_dim).contiguous()
    raise ValueError(f"Unexpected decode output shape: {tuple(out.shape)}")


def _tensor_aliases(lhs: Optional[torch.Tensor], rhs: Optional[torch.Tensor]) -> bool:
    if lhs is None or rhs is None:
        return False
    return (
        lhs.data_ptr() == rhs.data_ptr()
        and tuple(lhs.shape) == tuple(rhs.shape)
        and tuple(lhs.stride()) == tuple(rhs.stride())
        and int(lhs.storage_offset()) == int(rhs.storage_offset())
    )


def _write_current_kv_to_cache(
    k_bmhd: torch.Tensor,
    v_bmhd: torch.Tensor,
    cache_view: FlashInferDenseKVCacheView,
) -> None:
    if not cache_view.supports_direct_write:
        raise RuntimeError("Direct-write dense KV cache views are required for external-RoPE flashinfer-style decode.")
    if not _tensor_aliases(k_bmhd, cache_view.k_write_bmhd):
        cache_view.k_write_bmhd.copy_(k_bmhd)
    if not _tensor_aliases(v_bmhd, cache_view.v_write_bmhd):
        cache_view.v_write_bmhd.copy_(v_bmhd)


def _has_external_rope_contract(cache_view: FlashInferDenseKVCacheView) -> bool:
    return (
        cache_view.supports_direct_write
        and cache_view.rotary_cos is not None
        and cache_view.rotary_sin is not None
    )


def _apply_llama_rope_bmhd_inplace(
    x_bmhd: torch.Tensor,
    *,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    decode_positions: torch.Tensor,
) -> torch.Tensor:
    if int(x_bmhd.shape[-1]) % 2 != 0:
        raise ValueError(f"head_dim must be even, got {int(x_bmhd.shape[-1])}")
    cos, sin = _gather_llama_rope(
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        decode_positions=decode_positions,
        batch_size=int(x_bmhd.shape[0]),
        head_dim=int(x_bmhd.shape[-1]),
    )
    half = int(x_bmhd.shape[-1]) // 2
    x0 = x_bmhd[..., :half].clone()
    x1 = x_bmhd[..., half:]
    x_bmhd[..., :half] = x0 * cos - x1 * sin
    x_bmhd[..., half:] = x1 * cos + x0 * sin
    return x_bmhd


def _apply_external_llama_rope_inplace(
    q_bmhd: torch.Tensor,
    k_bmhd: torch.Tensor,
    cache_view: FlashInferDenseKVCacheView,
    *,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope_fn = get_flashinfer_apply_rope_inplace()
    if (
        rope_fn is not None
        and cache_view.rope_indptr is not None
        and cache_view.rope_offsets is not None
        and int(q_bmhd.shape[1]) == 1
        and int(k_bmhd.shape[1]) == 1
    ):
        q_view = q_bmhd.view(int(q_bmhd.shape[0]), int(q_bmhd.shape[2]), int(q_bmhd.shape[3]))
        k_view = k_bmhd.view(int(k_bmhd.shape[0]), int(k_bmhd.shape[2]), int(k_bmhd.shape[3]))
        rope_fn(
            q_view,
            k_view,
            cache_view.rope_indptr,
            cache_view.rope_offsets,
            rotary_dim=int(q_bmhd.shape[-1]),
            interleave=False,
            rope_theta=float(rope_theta),
        )
        return q_bmhd, k_bmhd

    q_rot = apply_llama_rope_bmhd(
        q_bmhd,
        rotary_cos=cache_view.rotary_cos,
        rotary_sin=cache_view.rotary_sin,
        decode_positions=cache_view.decode_positions,
    )
    if _tensor_aliases(k_bmhd, cache_view.k_write_bmhd):
        _apply_llama_rope_bmhd_inplace(
            k_bmhd,
            rotary_cos=cache_view.rotary_cos,
            rotary_sin=cache_view.rotary_sin,
            decode_positions=cache_view.decode_positions,
        )
        return q_rot, k_bmhd
    k_rot = apply_llama_rope_bmhd(
        k_bmhd,
        rotary_cos=cache_view.rotary_cos,
        rotary_sin=cache_view.rotary_sin,
        decode_positions=cache_view.decode_positions,
    )
    return q_rot, k_rot


def _resolve_dense_decode_backend(
    requested_backend: Optional[str],
    cache_view: FlashInferDenseKVCacheView,
    *,
    batch_size: int,
) -> str:
    selected_backend = str(requested_backend or "auto").strip().lower()
    if selected_backend != "auto":
        return selected_backend

    if int(batch_size) == 1 and has_flashinfer() and _has_external_rope_contract(cache_view):
        return "flashinfer"
    if _has_external_rope_contract(cache_view):
        return "fa2_external_rope"
    return "fa2_internal_rope"


def _run_flashinfer_single_decode(
    q_bmhd: torch.Tensor,
    k_bmhd: torch.Tensor,
    v_bmhd: torch.Tensor,
    cache_view: FlashInferDenseKVCacheView,
    *,
    sm_scale: float,
    rope_theta: float,
) -> torch.Tensor:
    fn = get_flashinfer_single_decode_with_kv_cache()
    if fn is None:
        raise RuntimeError("flashinfer.single_decode_with_kv_cache is not available.")
    if int(q_bmhd.shape[0]) != 1:
        raise RuntimeError("Current flashinfer path only supports batch_size=1 with dense KV cache.")
    q_rot, k_rot = _apply_external_llama_rope_inplace(
        q_bmhd,
        k_bmhd,
        cache_view,
        rope_theta=float(rope_theta),
    )
    _write_current_kv_to_cache(k_rot, v_bmhd, cache_view)

    kv_len = cache_view.kv_len(0)
    kwargs = _maybe_kwargs(
        fn,
        {
            "kv_layout": "NHD",
            "pos_encoding_mode": "NONE",
            "sm_scale": float(sm_scale),
        },
    )
    out = fn(
        q_rot[0, 0].contiguous(),
        cache_view.k_cache_bmhd[0, :kv_len].contiguous(),
        cache_view.v_cache_bmhd[0, :kv_len].contiguous(),
        **kwargs,
    )
    return _normalize_decode_output(
        out,
        batch_size=1,
        q_len=int(q_bmhd.shape[1]),
        num_heads=int(q_bmhd.shape[2]),
        head_dim=int(q_bmhd.shape[3]),
    )


def _run_fa2_external_rope(
    q_bmhd: torch.Tensor,
    k_bmhd: torch.Tensor,
    v_bmhd: torch.Tensor,
    cache_view: FlashInferDenseKVCacheView,
    *,
    rope_theta: float,
) -> torch.Tensor:
    fa2 = cache_view.flash_attn_with_kvcache or get_flash_attn_with_kvcache()
    if fa2 is None:
        raise RuntimeError("flash_attn_with_kvcache is required for DenseKVCache decode fallback.")

    q_rot, k_rot = _apply_external_llama_rope_inplace(
        q_bmhd,
        k_bmhd,
        cache_view,
        rope_theta=float(rope_theta),
    )
    _write_current_kv_to_cache(k_rot, v_bmhd, cache_view)
    out = call_flash_attn_with_kvcache(
        fa2,
        q_rot,
        cache_view.k_cache_bmhd,
        cache_view.v_cache_bmhd,
        cache_seqlens=cache_view.cache_seqlens,
        causal=True,
    )
    return _normalize_decode_output(
        out,
        batch_size=int(q_bmhd.shape[0]),
        q_len=int(q_bmhd.shape[1]),
        num_heads=int(q_bmhd.shape[2]),
        head_dim=int(q_bmhd.shape[3]),
    )


def _run_fa2_internal_rope(
    q_bmhd: torch.Tensor,
    k_bmhd: torch.Tensor,
    v_bmhd: torch.Tensor,
    cache_view: FlashInferDenseKVCacheView,
) -> torch.Tensor:
    fa2 = cache_view.flash_attn_with_kvcache or get_flash_attn_with_kvcache()
    if fa2 is None:
        raise RuntimeError("flash_attn_with_kvcache is required for DenseKVCache decode fallback.")
    out = call_flash_attn_with_kvcache(
        fa2,
        q_bmhd,
        cache_view.k_cache_bmhd,
        cache_view.v_cache_bmhd,
        k_bmhd=k_bmhd,
        v_bmhd=v_bmhd,
        cache_seqlens=cache_view.cache_seqlens,
        rotary_cos=cache_view.rotary_cos,
        rotary_sin=cache_view.rotary_sin,
        causal=True,
    )
    return _normalize_decode_output(
        out,
        batch_size=int(q_bmhd.shape[0]),
        q_len=int(q_bmhd.shape[1]),
        num_heads=int(q_bmhd.shape[2]),
        head_dim=int(q_bmhd.shape[3]),
    )


def flashsvd_flashinfer_dense_kv_attend(
    q_bmhd: torch.Tensor,
    k_bmhd: torch.Tensor,
    v_bmhd: torch.Tensor,
    cache_view: FlashInferDenseKVCacheView,
    *,
    num_heads: int,
    head_dim: int,
    project_output: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    backend: str = "auto",
    sm_scale: Optional[float] = None,
    rope_theta: float = 10000.0,
    advance_cache: bool = True,
) -> torch.Tensor:
    """Dense-KV attention step with flashinfer-style direct-write semantics.

    Backends:
    - `flashinfer`: external RoPE + direct cache write + flashinfer single decode
    - `fa2_external_rope`: same layout contract, but uses flash_attn_with_kvcache
    - `fa2_internal_rope`: legacy-compatible fallback that lets FA2 apply RoPE
    - `auto`: prefer `flashinfer`, then `fa2_external_rope`, then `fa2_internal_rope`
    """

    if int(q_bmhd.shape[1]) != 1:
        raise ValueError(f"Decode kernels expect q_len=1, got {int(q_bmhd.shape[1])}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(float(head_dim))

    selected_backend = _resolve_dense_decode_backend(
        backend,
        cache_view,
        batch_size=int(q_bmhd.shape[0]),
    )

    if selected_backend == "flashinfer":
        out_dense = _run_flashinfer_single_decode(
            q_bmhd,
            k_bmhd,
            v_bmhd,
            cache_view,
            sm_scale=float(sm_scale),
            rope_theta=float(rope_theta),
        )
    elif selected_backend == "fa2_external_rope":
        out_dense = _run_fa2_external_rope(
            q_bmhd,
            k_bmhd,
            v_bmhd,
            cache_view,
            rope_theta=float(rope_theta),
        )
    elif selected_backend == "fa2_internal_rope":
        out_dense = _run_fa2_internal_rope(q_bmhd, k_bmhd, v_bmhd, cache_view)
    else:
        raise ValueError(f"Unknown flashinfer dense decode backend: {backend}")

    if advance_cache and cache_view.advance_after_step is not None:
        cache_view.advance_after_step()
    if project_output is not None:
        out_dense = project_output(out_dense)
    return out_dense


def flashsvd_flashinfer_dense_decode_step(
    hidden_states: torch.Tensor,
    plan: FlashInferDenseDecodePlan,
    cache_view: FlashInferDenseKVCacheView,
    *,
    backend: Optional[str] = None,
    advance_cache: bool = True,
) -> torch.Tensor:
    """Flashinfer-style decode step that optionally reconstructs directly into dense cache."""

    selected_backend = _resolve_dense_decode_backend(
        backend or plan.backend_hint,
        cache_view,
        batch_size=int(hidden_states.shape[0]),
    )

    if plan.reconstruct_qkv_into_cache is not None and selected_backend in {"flashinfer", "fa2_external_rope"}:
        q_bmhd, k_bmhd, v_bmhd = plan.reconstruct_qkv_into_cache(hidden_states, cache_view)
    else:
        q_bmhd, k_bmhd, v_bmhd = plan.reconstruct_qkv(hidden_states)

    return flashsvd_flashinfer_dense_kv_attend(
        q_bmhd,
        k_bmhd,
        v_bmhd,
        cache_view,
        num_heads=int(plan.num_heads),
        head_dim=int(plan.head_dim),
        project_output=plan.project_output,
        backend=selected_backend,
        sm_scale=plan.sm_scale,
        rope_theta=plan.rope_theta,
        advance_cache=advance_cache,
    )


__all__ = [
    "FlashInferDenseDecodePlan",
    "FlashInferDenseKVCacheView",
    "apply_llama_rope_bmhd",
    "flashsvd_flashinfer_dense_decode_step",
    "flashsvd_flashinfer_dense_kv_attend",
    "get_flashinfer_apply_rope_inplace",
    "get_flashinfer_single_decode_with_kv_cache",
    "has_flashinfer",
]
