from __future__ import annotations

import importlib.util
import inspect
import math
import os
from pathlib import Path
from typing import Optional

import torch


_LEGACY_SPARSEKV_ATTN_MODS = None
_FLASH_ATTN_WITH_KVCACHE = None
_FLASH_ATTN_WITH_KVCACHE_RESOLVED = False
_FLASHSVD_V15_ATTN_DENSEKV_DECODE_MOD = None
_LEGACY_SPARSEKV_DECODE_VARIANT_CACHE: dict[tuple[object, ...], str] = {}
_LEGACY_SPARSEKV_DECODE_VARIANT_LOGGED: set[tuple[object, ...]] = set()


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def maybe_kwargs(fn, kwargs):
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _flashsvd_kernel_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = (
        root / "legacy" / "kernel_legacy",
        root / "kernels" / "legacy",
        root / "kernels" / "flashsvd-archive",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def get_flashsvd_attn_legacy_sparsekv_decode_mods():
    global _LEGACY_SPARSEKV_ATTN_MODS
    if _LEGACY_SPARSEKV_ATTN_MODS is not None:
        return _LEGACY_SPARSEKV_ATTN_MODS

    archive_root = _flashsvd_kernel_root()
    mods = {}

    path_v15 = archive_root / "v1.5" / "flashsvdropeattn" / "flashsvd_attn_legacy_v15_sparsekv_decode.py"
    if path_v15.exists():
        try:
            mods["flashsvd_attn_legacy_v15_sparsekv"] = load_module_from_path(
                path_v15, "flashsvd_attn_legacy_v15_sparsekv_decode"
            )
        except Exception:
            pass

    path_v16 = archive_root / "v1.6" / "flashsvdropeattn" / "flashsvd_attn_legacy_v16_sparsekv_decode.py"
    if path_v16.exists():
        try:
            mods["flashsvd_attn_legacy_v16_sparsekv"] = load_module_from_path(
                path_v16, "flashsvd_attn_legacy_v16_sparsekv_decode"
            )
        except Exception:
            pass

    if not mods:
        raise ImportError(f"Failed to load any decode attention module from: {archive_root}")

    _LEGACY_SPARSEKV_ATTN_MODS = mods
    return mods


def get_default_flashsvd_attn_legacy_sparsekv_decode_mod():
    mods = get_flashsvd_attn_legacy_sparsekv_decode_mods()
    if "flashsvd_attn_legacy_v16_sparsekv" in mods:
        return mods["flashsvd_attn_legacy_v16_sparsekv"]
    return next(iter(mods.values()))


def canonical_legacy_sparsekv_decode_variant(name: str) -> str:
    raw = str(name or "").strip().lower().replace("-", "_")
    if raw in {"", "auto"}:
        return "auto"
    if raw in {"flashsvd_attn_legacy_v15_sparsekv", "v15", "v1.5", "1.5"}:
        return "flashsvd_attn_legacy_v15_sparsekv"
    if raw in {
        "flashsvd_attn_legacy_v16_sparsekv_v1",
        "v16_v1",
        "v1",
        "legacy",
    }:
        return "flashsvd_attn_legacy_v16_sparsekv_v1"
    if raw in {
        "flashsvd_attn_legacy_v16_sparsekv_v2",
        "flashsvd_attn_legacy_v16_sparsekv",
        "v16_v2",
        "v16",
        "v2",
    }:
        return "flashsvd_attn_legacy_v16_sparsekv_v2"
    return "auto"


def available_legacy_sparsekv_decode_variants(mods) -> list[str]:
    out: list[str] = []
    mod_v15 = mods.get("flashsvd_attn_legacy_v15_sparsekv", None)
    if mod_v15 is not None and hasattr(mod_v15, "DecodePackedFactors") and hasattr(mod_v15, "flashsvd_attn_decode_packed"):
        out.append("flashsvd_attn_legacy_v15_sparsekv")

    mod_v16 = mods.get("flashsvd_attn_legacy_v16_sparsekv", None)
    if mod_v16 is not None and hasattr(mod_v16, "DecodePackedFactors"):
        if hasattr(mod_v16, "flashsvd_attn_decode_packed_v1"):
            out.append("flashsvd_attn_legacy_v16_sparsekv_v1")
        if hasattr(mod_v16, "flashsvd_attn_decode_packed"):
            out.append("flashsvd_attn_legacy_v16_sparsekv_v2")
    return out


def resolve_legacy_sparsekv_decode_variant(mods, variant: str):
    variant = canonical_legacy_sparsekv_decode_variant(variant)
    if variant == "flashsvd_attn_legacy_v15_sparsekv":
        mod = mods.get("flashsvd_attn_legacy_v15_sparsekv", None)
        if mod is None:
            return None, None
        return mod, getattr(mod, "flashsvd_attn_decode_packed", None)
    if variant == "flashsvd_attn_legacy_v16_sparsekv_v1":
        mod = mods.get("flashsvd_attn_legacy_v16_sparsekv", None)
        if mod is None:
            return None, None
        fn = getattr(mod, "flashsvd_attn_decode_packed_v1", None)
        if fn is None:
            fn = getattr(mod, "flashsvd_attn_decode_packed", None)
        return mod, fn
    if variant == "flashsvd_attn_legacy_v16_sparsekv_v2":
        mod = mods.get("flashsvd_attn_legacy_v16_sparsekv", None)
        if mod is None:
            return None, None
        return mod, getattr(mod, "flashsvd_attn_decode_packed", None)
    return None, None


def get_flash_attn_with_kvcache():
    global _FLASH_ATTN_WITH_KVCACHE
    global _FLASH_ATTN_WITH_KVCACHE_RESOLVED
    if _FLASH_ATTN_WITH_KVCACHE_RESOLVED:
        return _FLASH_ATTN_WITH_KVCACHE

    fn = None
    try:
        from flash_attn import flash_attn_with_kvcache  # type: ignore

        fn = flash_attn_with_kvcache
    except Exception:
        try:
            from flash_attn.flash_attn_interface import flash_attn_with_kvcache  # type: ignore

            fn = flash_attn_with_kvcache
        except Exception:
            fn = None

    _FLASH_ATTN_WITH_KVCACHE = fn
    _FLASH_ATTN_WITH_KVCACHE_RESOLVED = True
    return _FLASH_ATTN_WITH_KVCACHE


def call_flash_attn_with_kvcache(
    flash_attn_with_kvcache,
    q_bmhd: torch.Tensor,
    k_cache_bmhd: torch.Tensor,
    v_cache_bmhd: torch.Tensor,
    *,
    k_bmhd: Optional[torch.Tensor] = None,
    v_bmhd: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    causal: bool,
):
    sig = inspect.signature(flash_attn_with_kvcache)
    params = sig.parameters
    kwargs: dict[str, object] = {}
    if "k" in params and k_bmhd is not None:
        kwargs["k"] = k_bmhd
    if "v" in params and v_bmhd is not None:
        kwargs["v"] = v_bmhd
    if "cache_seqlens" in params and cache_seqlens is not None:
        kwargs["cache_seqlens"] = cache_seqlens
    if "rotary_cos" in params and rotary_cos is not None:
        kwargs["rotary_cos"] = rotary_cos
    if "rotary_sin" in params and rotary_sin is not None:
        kwargs["rotary_sin"] = rotary_sin
    if "rotary_interleaved" in params:
        kwargs["rotary_interleaved"] = False
    if "causal" in params:
        kwargs["causal"] = causal
    if "dropout_p" in params:
        kwargs["dropout_p"] = 0.0
    if "window_size" in params:
        kwargs["window_size"] = (-1, -1)
    return flash_attn_with_kvcache(q_bmhd, k_cache_bmhd, v_cache_bmhd, **kwargs)


def get_flashsvd_v15_attn_densekv_decode_mod():
    global _FLASHSVD_V15_ATTN_DENSEKV_DECODE_MOD
    if _FLASHSVD_V15_ATTN_DENSEKV_DECODE_MOD is not None:
        return _FLASHSVD_V15_ATTN_DENSEKV_DECODE_MOD

    root = Path(__file__).resolve().parents[2]
    candidates = (
        root / "kernels" / "attn" / "decode" / "densekv_reconstruct_triton.py",
        root / "kernels" / "flashsvd_v15_attn_densekv_decode.py",
    )
    path = None
    for candidate in candidates:
        if candidate.exists():
            path = candidate
            break
    if path is None:
        raise FileNotFoundError(
            "Failed to locate FlashSVD-v1.5 dense decode kernel. Tried: "
            + ", ".join(str(item) for item in candidates)
        )
    _FLASHSVD_V15_ATTN_DENSEKV_DECODE_MOD = load_module_from_path(
        path, "flashsvd_v15_attn_densekv_decode_component"
    )
    return _FLASHSVD_V15_ATTN_DENSEKV_DECODE_MOD


def parse_legacy_sparsekv_decode_variant_map(raw: str) -> dict[int, str]:
    table: dict[int, str] = {}
    text = str(raw or "").strip()
    if not text:
        return table

    for item in text.split(","):
        token = item.strip()
        if not token or ":" not in token:
            continue
        batch_text, variant_text = token.split(":", 1)
        try:
            batch_size = int(batch_text.strip())
        except Exception:
            continue
        if batch_size <= 0:
            continue
        table[batch_size] = canonical_legacy_sparsekv_decode_variant(variant_text.strip())
    return table


def select_legacy_sparsekv_decode_variant(
    *,
    mods,
    hidden_states: torch.Tensor,
    bsz: int,
    H: int,
    Hk: int,
    R: int,
    Dh: int,
    seqlen_k: int,
    Pq_q: torch.Tensor,
    Pk: torch.Tensor,
    Pv: torch.Tensor,
    Vq: torch.Tensor,
    Vk: torch.Tensor,
    Vv: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    call_kwargs: dict[str, object],
) -> str:
    del seqlen_k
    global _LEGACY_SPARSEKV_DECODE_VARIANT_CACHE
    global _LEGACY_SPARSEKV_DECODE_VARIANT_LOGGED

    available = available_legacy_sparsekv_decode_variants(mods)
    if not available:
        return "flashsvd_attn_legacy_v16_sparsekv_v2"

    requested = canonical_legacy_sparsekv_decode_variant(os.getenv("FLASH_SVD_DECODE_KERNEL_VARIANT", "auto"))
    if requested != "auto":
        if requested in available:
            return requested
        warn_key = ("requested_missing", requested, tuple(available))
        if warn_key not in _LEGACY_SPARSEKV_DECODE_VARIANT_LOGGED:
            print(
                "[FlashSVD][decode_variant] requested "
                f"{requested} not available; fallback to auto (available={','.join(available)})"
            )
            _LEGACY_SPARSEKV_DECODE_VARIANT_LOGGED.add(warn_key)
        requested = "auto"

    variant_map = parse_legacy_sparsekv_decode_variant_map(os.getenv("FLASH_SVD_DECODE_KERNEL_MAP", ""))
    if requested == "auto" and bsz in variant_map and variant_map[bsz] in available:
        return variant_map[bsz]

    include_v2 = os.getenv("FLASH_SVD_DECODE_AUTO_INCLUDE_V2", "0") != "0"
    candidates = [
        variant
        for variant in (
            "flashsvd_attn_legacy_v15_sparsekv",
            "flashsvd_attn_legacy_v16_sparsekv_v1",
        )
        if variant in available
    ]
    if include_v2 and "flashsvd_attn_legacy_v16_sparsekv_v2" in available:
        candidates.append("flashsvd_attn_legacy_v16_sparsekv_v2")
    if not candidates:
        candidates = list(available)
    if len(candidates) == 1:
        return candidates[0]

    if int(R) >= 768 and "flashsvd_attn_legacy_v15_sparsekv" in candidates:
        default_variant = "flashsvd_attn_legacy_v15_sparsekv"
    elif "flashsvd_attn_legacy_v16_sparsekv_v1" in candidates:
        default_variant = "flashsvd_attn_legacy_v16_sparsekv_v1"
    else:
        default_variant = candidates[0]

    if os.getenv("FLASH_SVD_DECODE_KERNEL_AUTOTUNE", "0") == "0":
        return default_variant

    try:
        tune_warmup = max(1, int(os.getenv("FLASH_SVD_DECODE_KERNEL_AUTOTUNE_WARMUP", "1")))
    except Exception:
        tune_warmup = 1
    try:
        tune_iters = max(1, int(os.getenv("FLASH_SVD_DECODE_KERNEL_AUTOTUNE_ITERS", "2")))
    except Exception:
        tune_iters = 2

    split_key = int(call_kwargs.get("split_k", 0) or 0)
    bn_key = int(call_kwargs.get("bn", 0) or 0)
    br_key = int(call_kwargs.get("br", 0) or 0)
    pad_key = int(bool(call_kwargs.get("pad_to_16", False)))
    vk_key = int(bool(call_kwargs.get("vk_resident", False)))
    cache_key = (
        int(getattr(hidden_states.device, "index", -1) or -1),
        str(hidden_states.dtype),
        int(bsz),
        int(H),
        int(Hk),
        int(R),
        int(Dh),
        int(split_key),
        int(bn_key),
        int(br_key),
        int(pad_key),
        int(vk_key),
    )
    cached = _LEGACY_SPARSEKV_DECODE_VARIANT_CACHE.get(cache_key, None)
    if cached in candidates:
        return str(cached)

    best_variant = default_variant
    best_ms = float("inf")

    for variant in candidates:
        mod_variant, fn_variant = resolve_legacy_sparsekv_decode_variant(mods, variant)
        if mod_variant is None or fn_variant is None:
            continue
        try:
            factors = mod_variant.DecodePackedFactors(Pq=Pq_q, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv)
            variant_kwargs = maybe_kwargs(fn_variant, dict(call_kwargs))
            for _ in range(tune_warmup):
                fn_variant(factors, rotary_cos, rotary_sin, **variant_kwargs)
            torch.cuda.synchronize(hidden_states.device)
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            for _ in range(tune_iters):
                fn_variant(factors, rotary_cos, rotary_sin, **variant_kwargs)
            event_end.record()
            torch.cuda.synchronize(hidden_states.device)
            elapsed_ms = float(event_start.elapsed_time(event_end)) / float(tune_iters)
        except Exception:
            continue
        if elapsed_ms < best_ms:
            best_ms = elapsed_ms
            best_variant = variant

    _LEGACY_SPARSEKV_DECODE_VARIANT_CACHE[cache_key] = best_variant
    log_key = ("autotuned", cache_key)
    if log_key not in _LEGACY_SPARSEKV_DECODE_VARIANT_LOGGED:
        extra = f"{best_ms:.3f} ms" if math.isfinite(best_ms) else "n/a"
        print(
            "[FlashSVD][decode_variant] "
            f"B={bsz} H={H} Hk={Hk} R={R} Dh={Dh} "
            f"selected={best_variant} "
            f"candidates={','.join(candidates)} ({extra})"
        )
        _LEGACY_SPARSEKV_DECODE_VARIANT_LOGGED.add(log_key)

    return best_variant


# Backward-compatible aliases for older imports.
get_flashsvd_decode_attn_mods = get_flashsvd_attn_legacy_sparsekv_decode_mods
get_default_flashsvd_decode_attn_mod = get_default_flashsvd_attn_legacy_sparsekv_decode_mod
canonical_decode_variant = canonical_legacy_sparsekv_decode_variant
available_decode_variants = available_legacy_sparsekv_decode_variants
resolve_decode_variant = resolve_legacy_sparsekv_decode_variant
get_dense_token_decode_mod = get_flashsvd_v15_attn_densekv_decode_mod
parse_decode_variant_map = parse_legacy_sparsekv_decode_variant_map
select_decode_variant = select_legacy_sparsekv_decode_variant
