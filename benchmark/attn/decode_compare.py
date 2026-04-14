#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decode-stage microbench: KV-cache decode (q_len=1, kv_len=L) comparisons.

What this measures (single-step decode, q_len=1, kv_len=L):
  - dense_kvcache: Q is dense [B,1,H,Dh], KV cache is dense [B,L,Hk,Dh] (expanded to H if needed),
    and attention is computed with:
      * FA2 (flash-attn) if installed, or
      * repo Triton kernel (flash_attn_triton_kvcache), or
      * torch reference

  - FlashSVD-v1.5 attention route: short-context `dense KV cache + token reconstruct + FA2 KV-cache decode`.
    Current-token Q/K/V are reconstructed from low-rank factors with one Triton token kernel,
    then passed to FlashAttention-2 KV-cache decode (`flash_attn_with_kvcache`) while KV cache stays dense.

  - sparse+stream: KV cache stored as low-rank factors
      * headwise layout: Pk/Pv [B,L,Hk,R] + bases Vk/Vv [Hk,R,Dh]
      * shared layout:   Pk/Pv [B,L,R]    + bases Vk/Vv [Hk,R,Dh]
    Each iteration reconstructs K/V in blocks (BN) and runs FlashAttention-style online softmax in PyTorch.

  - sparse: legacy FlashSVD low-rank KV-cache decode kernels with RoPE + split-K.
  - sparse+FA2-only: reconstruct dense K/V from sparse factors, then call FA2.
  - dense+FA2-only: dense reference path using FA2 without the FlashSVD token-reconstruct route.

Notes:
  - RoPE is included for low-rank (streaming + fused). Dense baseline pre-rotates K once (typical KV-cache behavior).
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import math
import os
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class LlamaPreset:
    hidden_size: int
    num_heads: int
    num_kv_heads: int

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"Invalid preset: hidden_size={self.hidden_size} is not divisible by num_heads={self.num_heads}")
        return self.hidden_size // self.num_heads


LLAMA_PRESETS: dict[str, LlamaPreset] = {
    "llama2-7b": LlamaPreset(hidden_size=4096, num_heads=32, num_kv_heads=32),
    "llama2-13b": LlamaPreset(hidden_size=5120, num_heads=40, num_kv_heads=40),
    "llama2-70b": LlamaPreset(hidden_size=8192, num_heads=64, num_kv_heads=8),
    "llama3-8b": LlamaPreset(hidden_size=4096, num_heads=32, num_kv_heads=8),
    "llama3-70b": LlamaPreset(hidden_size=8192, num_heads=64, num_kv_heads=8),
    "llama3.1-8b": LlamaPreset(hidden_size=4096, num_heads=32, num_kv_heads=8),
    "llama3.1-70b": LlamaPreset(hidden_size=8192, num_heads=64, num_kv_heads=8),
}


def _repo_root() -> str:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "flashsvd_component").exists() and (parent / "kernels").exists():
            return str(parent)
    raise RuntimeError(f"Failed to locate repo root from {here}")


def _archive_root() -> str:
    root = Path(_repo_root())
    candidates = (
        root / "kernels" / "legacy",
        root / "legacy" / "kernel_legacy",
        root / "kernels" / "flashsvd-archive",
    )
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def _format_llama_presets() -> str:
    lines = ["Available --llama presets:"]
    for name in sorted(LLAMA_PRESETS):
        p = LLAMA_PRESETS[name]
        lines.append(
            f"- {name}: hidden_size={p.hidden_size}, H={p.num_heads}, Hk={p.num_kv_heads}, Dh={p.head_dim}"
        )
    return "\n".join(lines)


def _round_rank(raw_rank: float, multiple: int, mode: str) -> int:
    if raw_rank <= 0:
        return 1
    if multiple <= 1:
        if mode == "down":
            return max(1, int(math.floor(raw_rank)))
        if mode == "up":
            return max(1, int(math.ceil(raw_rank)))
        return max(1, int(round(raw_rank)))

    q = raw_rank / float(multiple)
    if mode == "down":
        q_int = math.floor(q)
    elif mode == "up":
        q_int = math.ceil(q)
    else:
        q_int = int(round(q))

    out = int(q_int * multiple)
    if out <= 0:
        out = int(multiple)
    return out


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _best_lowrank_result(
    results: list[tuple[str, float, float, int, int, int, int]]
) -> Optional[tuple[str, float, float, int, int, int, int]]:
    lowrank = [
        r for r in results
        if r[0].startswith("lowrank_")
    ]
    if not lowrank:
        return None
    return min(lowrank, key=lambda x: x[1])


def _is_lowrank_online_variant(name: str) -> bool:
    return name.startswith("lowrank_")


def _rank_from_param_ratio(
    *,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    target_ratio: float,
    mode: str,
) -> float:
    """
    Infer rank R from target parameter ratio.

    mode="headwise":
      dense  = 2 * Hk * hidden_size * Dh
      lowrank= 2 * Hk * R * (hidden_size + Dh)
      ratio  = lowrank / dense

    mode="global":
      dense  = 2 * hidden_size * (Hk*Dh)
      lowrank= 2 * R * (hidden_size + Hk*Dh)
      ratio  = lowrank / dense
    """
    if hidden_size <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError(
            f"Invalid dimensions for rank inference: hidden_size={hidden_size}, Hk={num_kv_heads}, Dh={head_dim}"
        )
    if not (0.0 < target_ratio <= 1.0):
        raise ValueError(f"--target-param-ratio must be in (0, 1], got {target_ratio}")
    kv_dim = num_kv_heads * head_dim
    if mode == "headwise":
        return target_ratio * hidden_size * head_dim / float(hidden_size + head_dim)
    if mode == "global":
        return target_ratio * hidden_size * kv_dim / float(hidden_size + kv_dim)
    raise ValueError(f"Unknown rank formula mode: {mode}")


def _param_ratio_from_rank(
    *,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    rank: int,
    mode: str,
) -> float:
    if hidden_size <= 0 or num_kv_heads <= 0 or head_dim <= 0 or rank <= 0:
        return float("nan")
    kv_dim = num_kv_heads * head_dim
    if mode == "headwise":
        return (rank * (hidden_size + head_dim)) / float(hidden_size * head_dim)
    if mode == "global":
        return (rank * (hidden_size + kv_dim)) / float(hidden_size * kv_dim)
    return float("nan")


def _bench_ms(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    try:
        import triton  # type: ignore

        return float(triton.testing.do_bench(fn, warmup=warmup, rep=iters))
    except Exception:
        import torch

        for _ in range(max(1, warmup)):
            _ = fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max(1, iters)):
            _ = fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / max(1, iters)


def _reset_cuda_memory_state() -> None:
    import torch

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _isolated_peak_bytes(fn: Callable[[], object]) -> tuple[int, int, int, int]:
    import torch

    _reset_cuda_memory_state()
    base_alloc = int(torch.cuda.memory_allocated())
    base_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    _ = fn()
    torch.cuda.synchronize()
    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    delta_alloc = max(0, peak_alloc - base_alloc)
    delta_reserved = max(0, peak_reserved - base_reserved)
    return delta_alloc, delta_reserved, peak_alloc, peak_reserved


def _pretty_bytes(n: int) -> str:
    x = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} PB"


def _compact_variant_name(name: str) -> str:
    if name.startswith("baseline(reconstruct_abl+fa2+graph)"):
        return "dense+FA2-only+graph"
    if name.startswith("baseline(reconstruct_abl+fa2)"):
        return "dense+FA2-only"
    if name.startswith("dense_kvcache(rank_proj+token_reconstruct+fa2_kvcache+graph)"):
        return "FlashSVD-v1.5+graph"
    if name.startswith("dense_kvcache(rank_proj+token_reconstruct+fa2_kvcache)"):
        return "FlashSVD-v1.5"
    if name.startswith("dense_kvcache(token_reconstruct+fa2_kvcache+graph)"):
        return "FlashSVD-v1.5+graph"
    if name.startswith("dense_kvcache(token_reconstruct+fa2_kvcache)"):
        return "FlashSVD-v1.5"
    if name.startswith("dense_kvcache(full_qkv+fa2_kvcache+graph)"):
        return "dense+FA2-kvcache-only+graph"
    if name.startswith("dense_kvcache(full_qkv+fa2_kvcache)"):
        return "dense+FA2-kvcache-only"
    if name.startswith("dense_cache_only(fa2+graph)"):
        return "dense_cache+g"
    if name.startswith("dense_cache_only(fa2)"):
        return "dense_cache"
    if name.startswith("dense_kvcache(fa2+graph)"):
        return "dense_fa2+g"
    if name.startswith("dense_cache_only(triton+graph)"):
        return "dense_cache+g"
    if name.startswith("dense_cache_only(triton)"):
        return "dense_cache"
    if name.startswith("dense_kvcache(triton+graph)"):
        return "dense_triton+g"
    if name.startswith("dense_cache_only(torch+graph)"):
        return "dense_cache+g"
    if name.startswith("dense_cache_only(torch)"):
        return "dense_cache"
    if name.startswith("dense_kvcache(torch+graph)"):
        return "dense_torch+g"
    if name.startswith("dense_kvcache(fa2)"):
        return "dense_fa2"
    if name.startswith("dense_kvcache(triton)"):
        return "dense_triton"
    if name.startswith("dense_kvcache(torch)"):
        return "dense_torch"
    if name == "lowrank_kvcache(streaming_torch_rope)":
        return "sparse+stream"
    if name.startswith("lowrank_reconstruct(fa2)"):
        return "sparse+FA2-only"
    if name.startswith("lowrank_reconstruct(triton)"):
        return "sparse+triton-only"
    if name.startswith("lowrank_reconstruct(torch)"):
        return "sparse+torch-only"
    if name.startswith("lowrank_fused_v2(vk_resident=1)<flashsvd_attn_legacy_v16_sparsekv"):
        return "sparse(v1.6-v2-res)"
    if name.startswith("lowrank_fused_v2(vk_resident=0)<flashsvd_attn_legacy_v16_sparsekv"):
        return "sparse(v1.6-v2-nres)"
    if name.startswith("lowrank_fused_v1<flashsvd_attn_legacy_v16_sparsekv"):
        return "sparse(v1.6-v1)"
    if name.startswith("lowrank_fused<flashsvd_attn_legacy_v15_sparsekv"):
        return "sparse(v1.5)"
    return name.split("<", 1)[0]


def _compact_failure_reason(err: Exception) -> str:
    msg = str(err)
    if "out of resource: shared memory" in msg:
        return "smem_OOR"
    if "workspace M must be" in msg:
        return "workspace_mismatch"
    if not msg:
        return type(err).__name__
    head = msg.splitlines()[0].strip()
    if len(head) > 48:
        head = head[:45] + "..."
    return f"{type(err).__name__}:{head}"


def _find_named_result(
    results: list[tuple[str, float, float, int, int, int, int]],
    prefix: str,
) -> Optional[tuple[str, float, float, int, int, int, int]]:
    return next((r for r in results if r[0].startswith(prefix)), None)


def _best_matching_result(
    results: list[tuple[str, float, float, int, int, int, int]],
    predicate: Callable[[str], bool],
) -> Optional[tuple[str, float, float, int, int, int, int]]:
    matches = [r for r in results if predicate(r[0])]
    if not matches:
        return None
    return min(matches, key=lambda x: x[1])


def _is_flashsvd_v15_variant(name: str) -> bool:
    return (
        name.startswith("dense_kvcache(rank_proj+token_reconstruct+fa2_kvcache")
        or name.startswith("dense_kvcache(token_reconstruct+fa2_kvcache")
    )


def _is_sparse_legacy_variant(name: str) -> bool:
    return name.startswith("lowrank_fused")


def _is_sparse_fa2_only_variant(name: str) -> bool:
    return name.startswith("lowrank_reconstruct(fa2)")


def _is_dense_fa2_only_variant(name: str) -> bool:
    return name.startswith("baseline(reconstruct_abl+fa2)")


def _uses_dense_resident_cache(name: str) -> bool:
    return (
        name.startswith("dense_kvcache(")
        or name.startswith("dense_cache_only(")
        or name.startswith("baseline(")
    )


def _fmt_ms_or_na(item: Optional[tuple[str, float, float, int, int, int, int]]) -> str:
    if item is None:
        return "-"
    return f"{item[1]:.4f}"


def _make_cuda_graph_replay(
    *,
    device: "torch.device",
    prep_inputs: Callable[[], None],
    run_capture: Callable[[], "torch.Tensor"],
    clone_output: bool = True,
) -> Callable[[], "torch.Tensor"]:
    import torch

    prep_inputs()
    warm_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device=device)
    warm_stream.wait_stream(current_stream)
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            y = run_capture()
            _ = y.reshape(-1)[0]
    current_stream.wait_stream(warm_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_static = run_capture()

    def _run():
        prep_inputs()
        graph.replay()
        if clone_output:
            return y_static.clone()
        return y_static

    return _run


def _import_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_mod_path(p: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    p = p.strip()
    if not p:
        raise ValueError("Empty module path")
    if os.path.isabs(p):
        return p
    script_path = os.path.join(here, p)
    if os.path.exists(script_path):
        return script_path
    return os.path.join(_archive_root(), p)


def _load_fused_decode_modules(paths_csv: str) -> list[tuple[str, object]]:
    """
    Load one or more python modules that provide FlashSVD decode entrypoints.
    Each module is expected to define:
      - DecodePackedFactors
      - flashsvd_attn_decode_packed
    Optionally:
      - flashsvd_attn_decode_packed_v1
    """
    paths = [x.strip() for x in paths_csv.split(",") if x.strip()]
    if not paths:
        return []
    mods: list[tuple[str, object]] = []
    for i, p in enumerate(paths):
        p_abs = _resolve_mod_path(p)
        if not os.path.exists(p_abs):
            raise FileNotFoundError(p_abs)
        name = os.path.basename(p_abs)
        mod = _import_from_path(f"flashsvd_decode_mod_{i}", p_abs)
        mods.append((name, mod))
    return mods


def _maybe_kwargs(fn: Callable[..., object], kwargs: dict[str, object]) -> dict[str, object]:
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _load_flash_attn_triton_kvcache() -> Callable[..., "torch.Tensor"]:
    kernels_dir = os.path.join(_repo_root(), "kernels")
    fa_path = os.path.join(kernels_dir, "flash_attn_causal.py")
    if not os.path.exists(fa_path):
        raise FileNotFoundError(fa_path)
    fa = _import_from_path("flash_attn_causal_local_decode", fa_path)
    return fa.flash_attn_triton_kvcache


def _try_load_flash_attn2() -> Optional[Callable[..., "torch.Tensor"]]:
    try:
        from flash_attn import flash_attn_func  # type: ignore

        return flash_attn_func
    except Exception:
        pass
    try:
        from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore

        return flash_attn_func
    except Exception:
        return None


def _try_load_flash_attn2_kvcache() -> Optional[Callable[..., "torch.Tensor"]]:
    try:
        from flash_attn import flash_attn_with_kvcache  # type: ignore

        return flash_attn_with_kvcache
    except Exception:
        pass
    try:
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache  # type: ignore

        return flash_attn_with_kvcache
    except Exception:
        return None


def _call_flash_attn2(
    flash_attn_func: Callable[..., "torch.Tensor"],
    q_bmhd: "torch.Tensor",  # [B, Mq, H, Dh]
    k_bmhd: "torch.Tensor",  # [B, Mk, Hk|H, Dh]
    v_bmhd: "torch.Tensor",
    *,
    causal: bool,
    softmax_scale: Optional[float],
    window_size: Tuple[int, int],
) -> "torch.Tensor":
    sig = inspect.signature(flash_attn_func)
    params = sig.parameters
    kwargs = {}
    if "dropout_p" in params:
        kwargs["dropout_p"] = 0.0
    if "softmax_scale" in params and softmax_scale is not None:
        kwargs["softmax_scale"] = softmax_scale
    if "causal" in params:
        kwargs["causal"] = causal
    if "window_size" in params:
        kwargs["window_size"] = window_size
    return flash_attn_func(q_bmhd, k_bmhd, v_bmhd, **kwargs)


def _call_flash_attn2_kvcache(
    flash_attn_with_kvcache: Callable[..., "torch.Tensor"],
    q_bmhd: "torch.Tensor",
    k_cache_bmhd: "torch.Tensor",
    v_cache_bmhd: "torch.Tensor",
    *,
    k_bmhd: "torch.Tensor",
    v_bmhd: "torch.Tensor",
    cache_seqlens: "torch.Tensor",
    rotary_cos: "torch.Tensor",
    rotary_sin: "torch.Tensor",
    causal: bool,
    softmax_scale: Optional[float],
    window_size: Tuple[int, int],
) -> "torch.Tensor":
    sig = inspect.signature(flash_attn_with_kvcache)
    params = sig.parameters
    kwargs: dict[str, object] = {}
    if "k" in params:
        kwargs["k"] = k_bmhd
    if "v" in params:
        kwargs["v"] = v_bmhd
    if "cache_seqlens" in params:
        kwargs["cache_seqlens"] = cache_seqlens
    if "rotary_cos" in params:
        kwargs["rotary_cos"] = rotary_cos
    if "rotary_sin" in params:
        kwargs["rotary_sin"] = rotary_sin
    if "rotary_interleaved" in params:
        kwargs["rotary_interleaved"] = False
    if "causal" in params:
        kwargs["causal"] = causal
    if "softmax_scale" in params and softmax_scale is not None:
        kwargs["softmax_scale"] = softmax_scale
    if "window_size" in params:
        kwargs["window_size"] = window_size
    return flash_attn_with_kvcache(q_bmhd, k_cache_bmhd, v_cache_bmhd, **kwargs)


def _dense_decode_attn_torch(q_bh1d: "torch.Tensor", k_bhld: "torch.Tensor", v_bhld: "torch.Tensor", *, causal: bool) -> "torch.Tensor":
    import torch

    B, H, q_len, Dh = q_bh1d.shape
    assert q_len == 1
    L = k_bhld.shape[2]
    scale = 1.0 / math.sqrt(Dh)
    # scores: [B,H,1,L]
    scores = torch.matmul(q_bh1d, k_bhld.transpose(-1, -2)) * scale
    if causal:
        # In decode (q_len=1, query at last position), all keys are valid.
        pass
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_bhld)  # [B,H,1,Dh]
    return out


def _lowrank_decode_stream(
    *,
    q_bh1d: "torch.Tensor",  # [B,H,1,Dh]
    Pk_blhr: "torch.Tensor",  # [B,L,Hk,R]
    Pv_blhr: "torch.Tensor",  # [B,L,Hk,R]
    Vk_hrd: "torch.Tensor",  # [Hk,R,Dh]
    Vv_hrd: "torch.Tensor",  # [Hk,R,Dh]
    H: int,
    Hk: int,
    BN: int,
    causal: bool,
    cos_half: Optional["torch.Tensor"] = None,  # [L, Dh/2]
    sin_half: Optional["torch.Tensor"] = None,  # [L, Dh/2]
) -> "torch.Tensor":
    """
    FlashAttention-style online softmax, reconstructing K/V from (P*, V*) in blocks of BN.
    Uses GQA mapping without expanding K/V to H.
    """
    import torch

    assert H % Hk == 0
    rep = H // Hk
    B, H_q, q_len, Dh = q_bh1d.shape
    assert H_q == H and q_len == 1
    _, L, Hk_in, R = Pk_blhr.shape
    assert Hk_in == Hk
    assert Pv_blhr.shape == (B, L, Hk, R)
    assert Vk_hrd.shape == (Hk, R, Dh)
    assert Vv_hrd.shape == (Hk, R, Dh)

    scale = 1.0 / math.sqrt(Dh)

    if (cos_half is None) != (sin_half is None):
        raise ValueError("cos_half and sin_half must be both set or both None")

    # reshape Q to [B,Hk,rep,Dh] to share KV heads
    q_bhgd = q_bh1d[:, :, 0, :].reshape(B, Hk, rep, Dh).to(torch.float32)

    m_i = torch.full((B, Hk, rep), -float("inf"), device=q_bh1d.device, dtype=torch.float32)
    l_i = torch.zeros((B, Hk, rep), device=q_bh1d.device, dtype=torch.float32)
    acc = torch.zeros((B, Hk, rep, Dh), device=q_bh1d.device, dtype=torch.float32)

    for nk in range(0, L, BN):
        n1 = min(L, nk + BN)
        bn = n1 - nk

        Pk_blk = Pk_blhr[:, nk:n1, :, :]  # [B,bn,Hk,R]
        Pv_blk = Pv_blhr[:, nk:n1, :, :]  # [B,bn,Hk,R]

        # Reconstruct K/V tiles for this block: [B,Hk,bn,Dh]
        K_blk = torch.einsum("blhr,hrd->blhd", Pk_blk, Vk_hrd).permute(0, 2, 1, 3).contiguous()
        V_blk = torch.einsum("blhr,hrd->blhd", Pv_blk, Vv_hrd).permute(0, 2, 1, 3).contiguous()
        K_blk = K_blk.to(torch.float32)
        V_blk = V_blk.to(torch.float32)

        if cos_half is not None:
            # Apply RoPE to K for positions [nk, n1)
            half = Dh // 2
            cos_k = cos_half[nk:n1].to(torch.float32)  # [bn, half]
            sin_k = sin_half[nk:n1].to(torch.float32)
            k0 = K_blk[..., :half]
            k1 = K_blk[..., half:]
            cos = cos_k[None, None, :, :]  # [1,1,bn,half]
            sin = sin_k[None, None, :, :]
            K_blk = torch.cat([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1)

        # scores: [B,Hk,rep,bn]
        scores = torch.einsum("bhgd,bhnd->bhgn", q_bhgd, K_blk) * scale
        if causal:
            # decode with query at the last position: all keys (0..L-1) are <= query_pos
            pass

        block_max = scores.max(dim=-1).values
        m_new = torch.maximum(m_i, block_max)
        exp_diff = torch.exp(m_i - m_new)

        p = torch.exp(scores - m_new.unsqueeze(-1))
        l_new = l_i * exp_diff + p.sum(dim=-1)

        # acc update: [B,Hk,rep,Dh]
        acc = acc * exp_diff.unsqueeze(-1) + torch.einsum("bhgn,bhnd->bhgd", p, V_blk)
        m_i = m_new
        l_i = l_new

    out = acc / l_i.unsqueeze(-1).clamp_min(1e-20)  # [B,Hk,rep,Dh]
    out = out.reshape(B, H, Dh).to(q_bh1d.dtype)
    return out[:, :, None, :]  # [B,H,1,Dh]


def _lowrank_decode_stream_shared(
    *,
    q_bh1d: "torch.Tensor",  # [B,H,1,Dh]
    Pk_blr: "torch.Tensor",  # [B,L,R]
    Pv_blr: "torch.Tensor",  # [B,L,R]
    Vk_hrd: "torch.Tensor",  # [Hk,R,Dh]
    Vv_hrd: "torch.Tensor",  # [Hk,R,Dh]
    H: int,
    Hk: int,
    BN: int,
    causal: bool,
    cos_half: Optional["torch.Tensor"] = None,  # [L, Dh/2]
    sin_half: Optional["torch.Tensor"] = None,  # [L, Dh/2]
) -> "torch.Tensor":
    import torch

    assert H % Hk == 0
    rep = H // Hk
    B, H_q, q_len, Dh = q_bh1d.shape
    assert H_q == H and q_len == 1
    _, L, R = Pk_blr.shape
    assert Pv_blr.shape == (B, L, R)
    assert Vk_hrd.shape == (Hk, R, Dh)
    assert Vv_hrd.shape == (Hk, R, Dh)

    scale = 1.0 / math.sqrt(Dh)

    if (cos_half is None) != (sin_half is None):
        raise ValueError("cos_half and sin_half must be both set or both None")

    q_bhgd = q_bh1d[:, :, 0, :].reshape(B, Hk, rep, Dh).to(torch.float32)

    m_i = torch.full((B, Hk, rep), -float("inf"), device=q_bh1d.device, dtype=torch.float32)
    l_i = torch.zeros((B, Hk, rep), device=q_bh1d.device, dtype=torch.float32)
    acc = torch.zeros((B, Hk, rep, Dh), device=q_bh1d.device, dtype=torch.float32)

    for nk in range(0, L, BN):
        n1 = min(L, nk + BN)
        bn = n1 - nk

        Pk_blk = Pk_blr[:, nk:n1, :]  # [B,bn,R]
        Pv_blk = Pv_blr[:, nk:n1, :]  # [B,bn,R]

        K_blk = torch.einsum("blr,krd->blkd", Pk_blk, Vk_hrd).permute(0, 2, 1, 3).contiguous()
        V_blk = torch.einsum("blr,krd->blkd", Pv_blk, Vv_hrd).permute(0, 2, 1, 3).contiguous()
        K_blk = K_blk.to(torch.float32)
        V_blk = V_blk.to(torch.float32)

        if cos_half is not None:
            half = Dh // 2
            cos_k = cos_half[nk:n1].to(torch.float32)
            sin_k = sin_half[nk:n1].to(torch.float32)
            k0 = K_blk[..., :half]
            k1 = K_blk[..., half:]
            cos = cos_k[None, None, :, :]
            sin = sin_k[None, None, :, :]
            K_blk = torch.cat([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1)

        scores = torch.einsum("bhgd,bhnd->bhgn", q_bhgd, K_blk) * scale
        if causal:
            pass

        block_max = scores.max(dim=-1).values
        m_new = torch.maximum(m_i, block_max)
        exp_diff = torch.exp(m_i - m_new)

        p = torch.exp(scores - m_new.unsqueeze(-1))
        l_new = l_i * exp_diff + p.sum(dim=-1)

        acc = acc * exp_diff.unsqueeze(-1) + torch.einsum("bhgn,bhnd->bhgd", p, V_blk)
        m_i = m_new
        l_i = l_new

    out = acc / l_i.unsqueeze(-1).clamp_min(1e-20)
    out = out.reshape(B, H, Dh).to(q_bh1d.dtype)
    return out[:, :, None, :]


def _build_rope_tables_half(
    seqlen: int,
    head_dim: int,
    base: float,
    *,
    device: "torch.device",
    dtype: "torch.dtype",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    assert head_dim % 2 == 0
    half = head_dim // 2
    pos = torch.arange(seqlen, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = torch.einsum("m,d->md", pos, inv_freq)  # [seqlen, half]
    cos = torch.cos(ang).to(dtype).contiguous()
    sin = torch.sin(ang).to(dtype).contiguous()
    return cos, sin


def _rope_apply_bh1d(
    x_bh1d: "torch.Tensor",
    cos_half: "torch.Tensor",  # [half] or [1,half]
    sin_half: "torch.Tensor",
) -> "torch.Tensor":
    import torch

    Dh = x_bh1d.shape[-1]
    half = Dh // 2
    x0 = x_bh1d[..., :half]
    x1 = x_bh1d[..., half:]
    cos = cos_half.reshape(1, 1, 1, half)
    sin = sin_half.reshape(1, 1, 1, half)
    return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)


def _rope_apply_blhd(
    x_blhd: "torch.Tensor",
    cos_half: "torch.Tensor",  # [L, half]
    sin_half: "torch.Tensor",
) -> "torch.Tensor":
    import torch

    Dh = x_blhd.shape[-1]
    half = Dh // 2
    x0 = x_blhd[..., :half]
    x1 = x_blhd[..., half:]
    cos = cos_half[None, :, None, :]  # [1,L,1,half]
    sin = sin_half[None, :, None, :]
    return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)


def _load_flashsvd_rope_decode() -> object:
    mod_path = os.path.join(_archive_root(), "v1.5", "decoding", "flashsvd_attn_legacy_v15_sparsekv_short_decode.py")
    if not os.path.exists(mod_path):
        raise FileNotFoundError(mod_path)
    return _import_from_path("flashsvd_attn_legacy_v15_sparsekv_decode_local", mod_path)


def _load_dense_token_decode() -> object:
    import sys

    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from runtime.attn.decode_registry import get_flashsvd_v15_attn_densekv_decode_mod

    return get_flashsvd_v15_attn_densekv_decode_mod()


def _make_fused_decode_variants(
    *,
    mod_name: str,
    mod: object,
    B: int,
    H: int,
    Hk: int,
    Dh: int,
    R: int,
    L: int,
    dtype: "torch.dtype",
    dev: "torch.device",
    Pq_b1hr: "torch.Tensor",
    Pk_blhr: "torch.Tensor",
    Pv_blhr: "torch.Tensor",
    Vq_hrd: "torch.Tensor",
    Vk_hkrd: "torch.Tensor",
    Vv_hkrd: "torch.Tensor",
    cos_half: "torch.Tensor",
    sin_half: "torch.Tensor",
    causal: bool,
    split_k: int,
    bn: int,
    br: int,
    warps1: int,
    stages1: int,
    warps2: int,
    stages2: int,
    split_k_v2: Optional[int] = None,
    bn_v2: Optional[int] = None,
    br_v2: Optional[int] = None,
    warps1_v2: Optional[int] = None,
    stages1_v2: Optional[int] = None,
    pad_to_16_v2: bool = True,
    ablate_vk_resident: bool,
) -> list[tuple[str, Callable[[], object]]]:
    import torch

    if split_k % bn != 0:
        raise ValueError(f"split_k ({split_k}) must be a multiple of bn ({bn})")

    variants: list[tuple[str, Callable[[], object]]] = []
    if not (hasattr(mod, "DecodePackedFactors") and hasattr(mod, "flashsvd_attn_decode_packed")):
        return variants

    Pq_bhr = Pq_b1hr[:, 0, :, :]
    f_fused = mod.DecodePackedFactors(
        Pq=Pq_bhr,
        Pk=Pk_blhr,
        Pv=Pv_blhr,
        Vq=Vq_hrd,
        Vk=Vk_hkrd,
        Vv=Vv_hkrd,
        bq=None,
        bk=None,
        bv=None,
    )

    num_splits = max(1, (L + split_k - 1) // split_k)
    M_ws = torch.empty((B, H, num_splits), device=dev, dtype=torch.float32)
    L_ws = torch.empty((B, H, num_splits), device=dev, dtype=torch.float32)
    Acc_ws = torch.empty((B, H, num_splits, R), device=dev, dtype=torch.float32)
    O_ws = torch.empty((B, H, Dh), device=dev, dtype=dtype)

    half = Dh // 2
    Q0_ws = torch.empty((B, H, half), device=dev, dtype=dtype)
    Q1_ws = torch.empty((B, H, half), device=dev, dtype=dtype)

    call_common: dict[str, object] = dict(
        seqlen_k=int(L),
        causal=bool(causal),
        split_k=int(split_k),
        bn=int(bn),
        br=int(br),
        num_warps_stage1=int(warps1),
        num_stages_stage1=int(stages1),
        num_warps_stage2=int(warps2),
        num_stages_stage2=int(stages2),
        workspace=(M_ws, L_ws, Acc_ws),
        out=O_ws,
    )

    def _wrap(fn: Callable[..., object], *, name: str, extra: dict[str, object]):
        kw = dict(call_common)
        kw.update(extra)
        kw = _maybe_kwargs(fn, kw)

        def _run():
            return fn(f_fused, cos_half, sin_half, **kw)  # type: ignore[arg-type]

        variants.append((f"{name}<{mod_name}>", _run))

    # Legacy entrypoints (if present)
    if hasattr(mod, "flashsvd_attn_decode_packed_v1"):
        _wrap(getattr(mod, "flashsvd_attn_decode_packed_v1"), name="lowrank_fused_v1", extra={})

    fn = getattr(mod, "flashsvd_attn_decode_packed")
    params = inspect.signature(fn).parameters
    supports_vk = ("vk_resident" in params) and ("q_buffers" in params)

    if supports_vk:
        v2_split_k = int(split_k if split_k_v2 is None else split_k_v2)
        v2_bn = int(bn if bn_v2 is None else bn_v2)
        v2_br = int(br if br_v2 is None else br_v2)
        v2_warps1 = int(warps1 if warps1_v2 is None else warps1_v2)
        v2_stages1 = int(stages1 if stages1_v2 is None else stages1_v2)
        base_v2: dict[str, object] = dict(
            q_buffers=(Q0_ws, Q1_ws),
            precompute_q=True,
            pad_to_16=bool(pad_to_16_v2),
            writethrough=True,
            split_k=v2_split_k,
            bn=v2_bn,
            br=v2_br,
            num_warps_stage1=v2_warps1,
            num_stages_stage1=v2_stages1,
        )
        if ablate_vk_resident:
            _wrap(fn, name="lowrank_fused_v2(vk_resident=1)", extra={**base_v2, "vk_resident": True})
            _wrap(fn, name="lowrank_fused_v2(vk_resident=0)", extra={**base_v2, "vk_resident": False})
        else:
            _wrap(fn, name="lowrank_fused_v2", extra={**base_v2, "vk_resident": True})
    else:
        _wrap(fn, name="lowrank_fused", extra={})

    return variants


def main() -> int:
    ap = argparse.ArgumentParser("Decode-stage KV-cache comparison (dense vs low-rank)")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--Bs", type=str, default="", help="Optional comma-separated batch sizes to sweep (overrides --B).")
    ap.add_argument("--L", type=int, default=2048, help="KV cache length")
    ap.add_argument("--Ls", type=str, default="", help="Comma-separated KV lengths to sweep (overrides --L)")
    ap.add_argument("--llama", type=str, default="", help="Use LLaMA preset config (e.g., llama2-7b, llama2-13b, llama2-70b, llama3-8b).")
    ap.add_argument("--list-llama", action="store_true", help="List built-in --llama presets and exit.")
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--Hk", type=int, default=8)
    ap.add_argument("--Dh", type=int, default=128)
    ap.add_argument(
        "--R",
        type=int,
        default=0,
        help="Manual rank. In headwise layout this is per-head R; in shared layout this is the global shared rank.",
    )
    ap.add_argument(
        "--R-total",
        type=int,
        default=0,
        help="Optional total rank override. In shared layout this is the shared rank directly; in headwise layout it is mapped to per-head R.",
    )
    ap.add_argument("--hidden-size", type=int, default=0, help="Hidden size used for rank inference. Default: H*Dh or preset hidden_size.")
    ap.add_argument("--target-param-ratio", type=float, default=0.5, help="Target low-rank parameter ratio (vs dense KV projections).")
    ap.add_argument(
        "--rank-formula",
        choices=["headwise", "global"],
        default="headwise",
        help="Rank inference formula. With --factor-layout=auto, global uses shared-rank cache form and headwise uses per-head cache form.",
    )
    ap.add_argument(
        "--factor-layout",
        choices=["auto", "headwise", "shared"],
        default="auto",
        help="Layout for low-rank decode factors. auto: global->shared, headwise->headwise.",
    )
    ap.add_argument("--rank-round-multiple", type=int, default=64, help="Round inferred rank to a multiple of this value.")
    ap.add_argument("--rank-rounding", choices=["down", "nearest", "up"], default="down", help="Rounding mode for inferred rank.")
    ap.add_argument("--rank-cap", type=int, default=0, help="Optional cap for inferred rank (0 disables cap).")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--bn", type=int, default=128, help="KV block size for low-rank streaming")
    ap.add_argument("--split-k", type=int, default=512, help="Split-K chunk length for fused low-rank decode")
    ap.add_argument("--br", type=int, default=64, help="Rank tile size for fused low-rank decode")
    ap.add_argument("--no-fused", action="store_true", help="Disable fused low-rank decode kernel")
    ap.add_argument(
        "--fused-modules",
        type=str,
        default="v1.5/decoding/flashsvd_attn_legacy_v15_sparsekv_short_decode.py,v1.6/decoding/flashsvd_attn_legacy_v16_sparsekv_short_decode.py",
        help="Comma-separated fused decode module paths (relative to this folder or absolute).",
    )
    ap.add_argument("--no-fused-vk-ablation", action="store_true", help="Disable vk_resident on/off ablation (if supported).")
    ap.add_argument("--no-dense", action="store_true", help="Disable dense KV-cache baseline")
    ap.add_argument(
        "--no-dense-token-reconstruct",
        action="store_true",
        help="Disable short-context dense-KV path: token reconstruct kernel + FA2 KV-cache decode.",
    )
    ap.add_argument(
        "--cuda-graph-variants",
        action="store_true",
        help="Add CUDA Graph replay variants for dense decode baselines.",
    )
    ap.add_argument("--no-stream", action="store_true", help="Disable low-rank streaming baseline")
    ap.add_argument("--fused-warps1", type=int, default=4)
    ap.add_argument("--fused-stages1", type=int, default=2)
    ap.add_argument("--fused-warps2", type=int, default=4)
    ap.add_argument("--fused-stages2", type=int, default=1)
    ap.add_argument("--fused-tune", action="store_true", help="Tune fused split-k/bn/warps1 for each L")
    ap.add_argument("--fused-tune-warmup", type=int, default=20)
    ap.add_argument("--fused-tune-iters", type=int, default=50)
    ap.add_argument("--fused-tune-splitks", type=str, default="", help="Comma list, e.g. 512,1024,2048,4096")
    ap.add_argument("--fused-tune-bns", type=str, default="", help="Comma list, e.g. 128,256")
    ap.add_argument("--fused-tune-warps1s", type=str, default="", help="Comma list, e.g. 4,8")
    ap.add_argument("--causal", action="store_true", default=True)
    ap.add_argument("--no-causal", dest="causal", action="store_false")

    ap.add_argument("--dense-backend", choices=["fa2", "triton", "torch", "auto"], default="auto")
    ap.add_argument("--compare-kv-budget", action="store_true", help="Also run a KV-budget-equalized comparison by scaling batch for low-rank.")
    ap.add_argument(
        "--kv-budget-scale",
        type=float,
        default=0.0,
        help="Manual low-rank batch scaling for KV-budget test. <=0 uses auto scale (dense_kv/lowrank_cache).",
    )
    ap.add_argument(
        "--realistic-attn",
        action="store_true",
        help="Use a closer-to-real-attention setup: disable streaming baseline unless explicitly requested.",
    )
    ap.add_argument(
        "--no-mem-reset",
        action="store_true",
        help="Disable CUDA memory reset before each variant benchmark/peak measurement.",
    )
    ap.add_argument(
        "--report",
        choices=["compact", "full"],
        default="compact",
        help="compact: concise setup + per-L summary rows; full: detailed per-variant dump.",
    )
    ap.add_argument("--check", action="store_true", help="run a small reference check (recommend --L <= 256)")
    args = ap.parse_args()

    if args.list_llama:
        print(_format_llama_presets())
        return 0

    if args.realistic_attn and not args.no_stream:
        args.no_stream = True
    mem_reset_each_variant = not args.no_mem_reset

    import torch

    if not torch.cuda.is_available():
        print("[error] CUDA is required.")
        return 2

    llama_name = args.llama.strip().lower()
    preset_used: Optional[LlamaPreset] = None
    if llama_name:
        preset_used = LLAMA_PRESETS.get(llama_name)
        if preset_used is None:
            print(f"[error] Unknown --llama preset: {args.llama}")
            print(_format_llama_presets())
            return 2
        args.H = preset_used.num_heads
        args.Hk = preset_used.num_kv_heads
        args.Dh = preset_used.head_dim

    if args.H % args.Hk != 0:
        print(f"[error] GQA requires H divisible by Hk, got H={args.H}, Hk={args.Hk}")
        return 2
    if args.Dh <= 0:
        print(f"[error] Invalid Dh={args.Dh}")
        return 2

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")

    B_base, H, Hk, Dh = args.B, args.H, args.Hk, args.Dh
    hidden_size = args.hidden_size if args.hidden_size > 0 else (preset_used.hidden_size if preset_used is not None else H * Dh)
    kv_dim = Hk * Dh
    rank_formula = str(args.rank_formula)
    factor_layout_req = str(args.factor_layout)
    if factor_layout_req == "auto":
        factor_layout = "shared" if rank_formula == "global" else "headwise"
    else:
        factor_layout = factor_layout_req

    rank_total_target = 0
    raw_rank = None
    if factor_layout == "shared":
        if args.R_total > 0:
            R = int(args.R_total)
            rank_total_target = R
            rank_policy = f"manual(shared --R-total={rank_total_target})"
        elif args.R > 0:
            R = int(args.R)
            rank_total_target = R
            rank_policy = f"manual(shared --R={R})"
        else:
            try:
                raw_rank = _rank_from_param_ratio(
                    hidden_size=hidden_size,
                    num_kv_heads=Hk,
                    head_dim=Dh,
                    target_ratio=float(args.target_param_ratio),
                    mode=rank_formula,
                )
            except ValueError as e:
                print(f"[error] {e}")
                return 2

            if rank_formula == "global":
                raw_rank_shared = raw_rank
            else:
                raw_rank_shared = raw_rank * Hk

            rank_total_target = _round_rank(
                raw_rank_shared,
                multiple=max(1, int(args.rank_round_multiple)),
                mode=args.rank_rounding,
            )
            if args.rank_cap > 0:
                rank_total_target = min(rank_total_target, int(args.rank_cap))
            rank_total_target = max(1, rank_total_target)
            R = rank_total_target
            rank_policy = (
                f"auto(layout=shared, formula={rank_formula}, target_ratio={float(args.target_param_ratio):.3f}, "
                f"raw_shared={raw_rank_shared:.2f}, rounded_shared={R})"
            )
        R = max(1, R)
        rank_total_effective = R
        achieved_ratio_headwise = float("nan")
        achieved_ratio_global = _param_ratio_from_rank(
            hidden_size=hidden_size,
            num_kv_heads=Hk,
            head_dim=Dh,
            rank=rank_total_effective,
            mode="global",
        )
    else:
        if args.R_total > 0:
            rank_total_target = int(args.R_total)
            R = max(1, _ceil_div(rank_total_target, Hk))
            rank_policy = f"manual(headwise --R-total={rank_total_target} -> per_head={R})"
        elif args.R > 0:
            R = int(args.R)
            rank_total_target = R * Hk
            rank_policy = f"manual(headwise --R={R} per_head, total={rank_total_target})"
        else:
            try:
                raw_rank = _rank_from_param_ratio(
                    hidden_size=hidden_size,
                    num_kv_heads=Hk,
                    head_dim=Dh,
                    target_ratio=float(args.target_param_ratio),
                    mode=rank_formula,
                )
            except ValueError as e:
                print(f"[error] {e}")
                return 2

            if rank_formula == "global":
                raw_rank_total = raw_rank
                rank_total_target = _round_rank(
                    raw_rank_total,
                    multiple=max(1, int(args.rank_round_multiple)),
                    mode=args.rank_rounding,
                )
                if args.rank_cap > 0:
                    rank_total_target = min(rank_total_target, int(args.rank_cap))
                rank_total_target = max(1, rank_total_target)
                R = max(1, _ceil_div(rank_total_target, Hk))
                rank_policy = (
                    f"auto(layout=headwise, formula=global, target_ratio={float(args.target_param_ratio):.3f}, "
                    f"raw_total={raw_rank_total:.2f}, rounded_total={rank_total_target}, per_head={R})"
                )
            else:
                R = _round_rank(raw_rank, multiple=max(1, int(args.rank_round_multiple)), mode=args.rank_rounding)
                if args.rank_cap > 0:
                    R = min(R, int(args.rank_cap))
                R = max(1, R)
                rank_total_target = R * Hk
                rank_policy = (
                    f"auto(layout=headwise, formula=headwise, target_ratio={float(args.target_param_ratio):.3f}, "
                    f"raw={raw_rank:.2f}, round={args.rank_rounding}/{max(1, int(args.rank_round_multiple))}, per_head={R})"
                )

        rank_total_effective = R * Hk
        achieved_ratio_headwise = _param_ratio_from_rank(
            hidden_size=hidden_size,
            num_kv_heads=Hk,
            head_dim=Dh,
            rank=R,
            mode="headwise",
        )
        achieved_ratio_global = _param_ratio_from_rank(
            hidden_size=hidden_size,
            num_kv_heads=Hk,
            head_dim=Dh,
            rank=rank_total_effective,
            mode="global",
        )

    rep = H // Hk
    BN = max(1, args.bn)  # streaming baseline block size

    # Base fused config (used by v1/v1.5 and as default for v2 unless overridden below)
    fused_split_k_eff = int(args.split_k)
    fused_bn_eff = max(1, int(args.bn))
    fused_br_eff = max(1, int(args.br))
    fused_warps1_eff = max(1, int(args.fused_warps1))
    fused_stages1_eff = max(1, int(args.fused_stages1))

    # v2-specific config: keep v1 fast, only apply conservative knobs to v2 path.
    v2_split_k_eff = fused_split_k_eff
    v2_bn_eff = fused_bn_eff
    v2_br_eff = fused_br_eff
    v2_warps1_eff = fused_warps1_eff
    v2_stages1_eff = fused_stages1_eff
    v2_pad_to_16 = True
    auto_blocksize_changes: list[str] = []
    v2_auto_changes: list[str] = []

    # rep=1/2 under-utilizes GROUP_M=16 and can make v2 much slower.
    if rep <= 2:
        v2_pad_to_16 = False
        v2_auto_changes.append("pad_to_16:True->False(rep<=2)")

    # High-R can still stress v2 resources; apply safer v2-only defaults when user keeps defaults.
    if R >= 512:
        if args.bn == 128:
            v2_bn_eff = 64
            v2_auto_changes.append("v2_bn:128->64")
        if args.br == 64:
            v2_br_eff = 32
            v2_auto_changes.append("v2_br:64->32")
        if args.split_k == 512:
            v2_split_k_eff = 256
            v2_auto_changes.append("v2_split_k:512->256")
        if args.fused_warps1 == 4:
            v2_warps1_eff = 2
            v2_auto_changes.append("v2_warps1:4->2")
        if args.fused_stages1 == 2:
            v2_stages1_eff = 1
            v2_auto_changes.append("v2_stages1:2->1")

    if fused_split_k_eff % fused_bn_eff != 0:
        old = fused_split_k_eff
        fused_split_k_eff = ((fused_split_k_eff + fused_bn_eff - 1) // fused_bn_eff) * fused_bn_eff
        auto_blocksize_changes.append(f"split_k:{old}->{fused_split_k_eff}(aligned_to_bn)")
    if v2_split_k_eff % v2_bn_eff != 0:
        old = v2_split_k_eff
        v2_split_k_eff = ((v2_split_k_eff + v2_bn_eff - 1) // v2_bn_eff) * v2_bn_eff
        v2_auto_changes.append(f"v2_split_k:{old}->{v2_split_k_eff}(aligned_to_v2_bn)")

    if args.Ls.strip():
        Ls = _parse_csv_ints(args.Ls)
    else:
        Ls = [int(args.L)]
    if not Ls:
        print("[error] Empty --Ls")
        return 2

    if args.Bs.strip():
        Bs = _parse_csv_ints(args.Bs)
    else:
        Bs = [int(B_base)]
    Bs = [b for b in Bs if b > 0]
    if not Bs:
        print("[error] Empty/invalid --Bs")
        return 2

    kv_budget_scale = 0.0
    if args.compare_kv_budget and len(Bs) == 1:
        if args.kv_budget_scale > 0:
            kv_budget_scale = float(args.kv_budget_scale)
        else:
            # Cache-only ratio for decode:
            #   headwise: lowrank/dense = R / Dh
            #   shared:   lowrank/dense = R / (Hk * Dh)
            dense_to_lowrank = float(kv_dim if factor_layout == "shared" else Dh) / float(max(1, R))
            kv_budget_scale = max(1.0, dense_to_lowrank)
        b0 = Bs[0]
        b1 = max(b0 + 1, int(math.floor(b0 * kv_budget_scale + 1e-9)))
        Bs = [b0, b1]
    elif args.compare_kv_budget and len(Bs) >= 2:
        kv_budget_scale = float(Bs[1]) / float(Bs[0])

    if args.compare_kv_budget and args.no_dense:
        print("[warn] --compare-kv-budget requested but dense baseline is disabled (--no-dense); budget summary will be skipped.")

    # Load fused decode module(s) once (optional)
    fused_mods: list[tuple[str, object]] = []
    if not args.no_fused:
        try:
            fused_mods = _load_fused_decode_modules(args.fused_modules)
        except Exception as e:
            print(f"[warn] Failed to load fused decode module(s), disabling fused: {e}")
            fused_mods = []

    # backend selection (once)
    dense_backend_req = args.dense_backend
    fa2 = _try_load_flash_attn2() if dense_backend_req in ("fa2", "auto") else None
    fa2_kvcache = _try_load_flash_attn2_kvcache() if dense_backend_req in ("fa2", "auto") else None
    if dense_backend_req == "fa2" and fa2 is None:
        raise SystemExit("FlashAttention-2 not found but --dense-backend=fa2 was requested.")

    dense_token_decode_mod = None
    if not args.no_dense_token_reconstruct:
        try:
            dense_token_decode_mod = _load_dense_token_decode()
        except Exception as e:
            print(f"[warn] Failed to load dense token decode module, disabling token reconstruct path: {e}")
            dense_token_decode_mod = None

    fa_triton_kvcache = None
    if dense_backend_req in ("triton", "auto"):
        try:
            fa_triton_kvcache = _load_flash_attn_triton_kvcache()
        except Exception:
            fa_triton_kvcache = None

    if dense_backend_req == "auto":
        if fa2 is not None:
            dense_backend = "fa2"
        elif fa_triton_kvcache is not None:
            dense_backend = "triton"
        else:
            dense_backend = "torch"
    else:
        dense_backend = dense_backend_req

    if args.report == "compact":
        llama_label = llama_name if preset_used is not None else "custom"
        print("==== Decode KV-cache comparison ====")
        print(
            f"Setup: llama={llama_label} | Bs={','.join(str(x) for x in Bs)} | "
            f"Ls={','.join(str(x) for x in Ls)} | H={H} Hk={Hk} rep={rep} Dh={Dh} | "
            f"layout={factor_layout} | R={R} | dtype={args.dtype} | backend={dense_backend} | "
            f"cuda_graph_variants={int(args.cuda_graph_variants)} | causal={int(args.causal)}"
        )
        print(
            f"Rank: total_effective={rank_total_effective}"
            + (f", total_target={rank_total_target}" if rank_total_target > 0 else "")
            + f" | policy={rank_policy}"
        )
        print(
            f"Params/layer: dense={2 * hidden_size * kv_dim:,} | "
            f"lowrank_global_ref={2 * rank_total_effective * (hidden_size + kv_dim):,}"
        )
        print(
            "Columns: L | direct_winner | winner_ms | FlashSVD-v1.5 | sparse | "
            "sparse+FA2-only | dense+FA2-only | sparse/FlashSVD | fail"
        )

    case_reports: dict[tuple[int, int], dict[str, object]] = {}
    cases = [(b, l) for b in Bs for l in Ls if l > 0]
    for B, L in cases:
        # ----------------------------
        # Generate low-rank KV cache + query factors
        # ----------------------------
        cos_half, sin_half = _build_rope_tables_half(L, Dh, base=10000.0, device=dev, dtype=dtype)
        q_pos = L - 1

        Vq_hrd = torch.randn(H, R, Dh, device=dev, dtype=dtype).contiguous()
        Vk_hkrd = torch.randn(Hk, R, Dh, device=dev, dtype=dtype).contiguous()
        Vv_hkrd = torch.randn(Hk, R, Dh, device=dev, dtype=dtype).contiguous()

        hidden_tok_bm = None
        Uq_mr = None
        Uk_mr = None
        Uv_mr = None
        Uqkv_cat_m3r = None
        Wqkv_flat = None
        shared_reconstruct_kwargs: dict[str, int] = {}
        Vq_shared_flat = None
        Vk_shared_flat = None
        Vv_shared_flat = None
        q_dense_dim = H * Dh
        kv_dense_dim = Hk * Dh

        if factor_layout == "shared":
            hidden_tok_bm = torch.randn(B, hidden_size, device=dev, dtype=dtype).contiguous()
            Uq_mr = torch.randn(hidden_size, R, device=dev, dtype=dtype).contiguous()
            Uk_mr = torch.randn(hidden_size, R, device=dev, dtype=dtype).contiguous()
            Uv_mr = torch.randn(hidden_size, R, device=dev, dtype=dtype).contiguous()
            Uqkv_cat_m3r = torch.cat((Uq_mr, Uk_mr, Uv_mr), dim=1).contiguous()

            Pq_tok_br = torch.matmul(hidden_tok_bm, Uq_mr).contiguous()
            Pk_tok_br_seed = torch.matmul(hidden_tok_bm, Uk_mr).contiguous()
            Pv_tok_br_seed = torch.matmul(hidden_tok_bm, Uv_mr).contiguous()

            Pq_b1r = Pq_tok_br[:, None, :].contiguous()
            Pk_blr = torch.randn(B, L, R, device=dev, dtype=dtype).contiguous()
            Pv_blr = torch.randn(B, L, R, device=dev, dtype=dtype).contiguous()
            Pk_blr[:, q_pos, :].copy_(Pk_tok_br_seed)
            Pv_blr[:, q_pos, :].copy_(Pv_tok_br_seed)

            # Match svd_llama decode: shared-rank cache expanded as zero-stride views for kernels.
            Pq_b1hr = Pq_b1r.unsqueeze(2).expand(B, 1, H, R)
            Pk_blhr = Pk_blr.unsqueeze(2).expand(B, L, Hk, R)
            Pv_blhr = Pv_blr.unsqueeze(2).expand(B, L, Hk, R)

            Wq_flat = torch.einsum("mr,hrd->mhd", Uq_mr, Vq_hrd).reshape(hidden_size, q_dense_dim).contiguous()
            Wk_flat = torch.einsum("mr,krd->mkd", Uk_mr, Vk_hkrd).reshape(hidden_size, kv_dense_dim).contiguous()
            Wv_flat = torch.einsum("mr,krd->mkd", Uv_mr, Vv_hkrd).reshape(hidden_size, kv_dense_dim).contiguous()
            Wqkv_flat = torch.cat((Wq_flat, Wk_flat, Wv_flat), dim=1).contiguous()
            if dense_token_decode_mod is not None:
                Vq_shared_flat, Vk_shared_flat, Vv_shared_flat = dense_token_decode_mod.pack_qkv_shared_bases(
                    Vq_hrd,
                    Vk_hkrd,
                    Vv_hkrd,
                )
            shared_reconstruct_kwargs = {
                "BD": 128 if Dh >= 128 else 64,
                "BR": 128 if R >= 512 else 64,
                "num_warps": 8 if R >= 512 else 4,
                "num_stages": 2,
            }

            def build_q_dense_raw_bh1d():
                if Vq_shared_flat is not None:
                    q_flat = torch.matmul(Pq_b1r[:, 0, :], Vq_shared_flat)
                    return q_flat.reshape(B, 1, H, Dh).permute(0, 2, 1, 3).contiguous()
                return torch.einsum("bmr,hrd->bmhd", Pq_b1r, Vq_hrd).permute(0, 2, 1, 3).contiguous()

        else:
            Pq_b1hr = torch.randn(B, 1, H, R, device=dev, dtype=dtype).contiguous()
            Pk_blhr = torch.randn(B, L, Hk, R, device=dev, dtype=dtype).contiguous()
            Pv_blhr = torch.randn(B, L, Hk, R, device=dev, dtype=dtype).contiguous()
            Pq_b1r = None
            Pk_blr = None
            Pv_blr = None

            def build_q_dense_raw_bh1d():
                return torch.einsum("bmhr,hrd->bmhd", Pq_b1hr, Vq_hrd).permute(0, 2, 1, 3).contiguous()

        # Query dense: [B,H,1,Dh]
        def build_q_dense_bh1d():
            q = build_q_dense_raw_bh1d()
            q = _rope_apply_bh1d(q, cos_half[q_pos], sin_half[q_pos])
            return q

        # ----------------------------
        # Dense KV cache baseline (K pre-rotated outside timing)
        # ----------------------------
        K_rope_blhd = None
        V_blhd = None
        K_bhld = None
        V_bhld = None
        if not args.no_dense:
            with torch.no_grad():
                # K/V: [B,L,Hk,Dh]
                if factor_layout == "shared":
                    assert Pk_blr is not None and Pv_blr is not None
                    K_blhd = torch.einsum("blr,krd->blkd", Pk_blr, Vk_hkrd).contiguous()
                    V_blhd = torch.einsum("blr,krd->blkd", Pv_blr, Vv_hkrd).contiguous()
                else:
                    K_blhd = torch.einsum("blhr,hrd->blhd", Pk_blhr, Vk_hkrd).contiguous()
                    V_blhd = torch.einsum("blhr,hrd->blhd", Pv_blhr, Vv_hkrd).contiguous()
                K_rope_blhd = _rope_apply_blhd(K_blhd, cos_half, sin_half)

                if dense_backend in ("torch", "triton"):
                    # Expand KV-cache to H heads once (cache-like) to avoid timing repeat_interleave.
                    K_bhld = K_rope_blhd.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1).contiguous()
                    V_bhld = V_blhd.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1).contiguous()

        dense_full_token_name = ""
        _dense_full_token_decode = None
        token_reconstruct_dense_name = ""
        _token_reconstruct_dense_decode = None
        K_cache_bmhd = None
        V_cache_bmhd = None
        cache_seqlens = None
        Pq_tok_br = None
        Pk_tok_br = None
        Pv_tok_br = None
        Pq_tok_bhr = None
        Pk_tok_bkr = None
        Pv_tok_bkr = None
        if fa2_kvcache is not None and not args.no_dense:
            K_cache_bmhd = torch.empty((B, L, Hk, Dh), device=dev, dtype=dtype)
            V_cache_bmhd = torch.empty((B, L, Hk, Dh), device=dev, dtype=dtype)
            if q_pos > 0:
                assert K_rope_blhd is not None and V_blhd is not None
                K_cache_bmhd[:, :q_pos].copy_(K_rope_blhd[:, :q_pos])
                V_cache_bmhd[:, :q_pos].copy_(V_blhd[:, :q_pos])
            K_cache_bmhd[:, q_pos:].zero_()
            V_cache_bmhd[:, q_pos:].zero_()
            cache_seqlens = torch.full((B,), q_pos, device=dev, dtype=torch.int32)

            if factor_layout == "shared":
                assert hidden_tok_bm is not None and Uqkv_cat_m3r is not None and Wqkv_flat is not None

                def _dense_full_token_decode():
                    packed_qkv = torch.matmul(hidden_tok_bm, Wqkv_flat)
                    q_flat, k_flat, v_flat = torch.split(packed_qkv, (q_dense_dim, kv_dense_dim, kv_dense_dim), dim=1)
                    q_bmhd = q_flat.reshape(B, 1, H, Dh).contiguous()
                    k_bmhd = k_flat.reshape(B, 1, Hk, Dh).contiguous()
                    v_bmhd = v_flat.reshape(B, 1, Hk, Dh).contiguous()
                    out = _call_flash_attn2_kvcache(
                        fa2_kvcache,
                        q_bmhd,
                        K_cache_bmhd,
                        V_cache_bmhd,
                        k_bmhd=k_bmhd,
                        v_bmhd=v_bmhd,
                        cache_seqlens=cache_seqlens,
                        rotary_cos=cos_half,
                        rotary_sin=sin_half,
                        causal=args.causal,
                        softmax_scale=None,
                        window_size=(-1, -1),
                    )
                    return out.permute(0, 2, 1, 3).contiguous()

                dense_full_token_name = "dense_kvcache(full_qkv+fa2_kvcache)"
            if dense_token_decode_mod is not None:
                if factor_layout == "shared":
                    def _token_reconstruct_dense_decode():
                        packed_rank = torch.matmul(hidden_tok_bm, Uqkv_cat_m3r)
                        q_rank, k_rank, v_rank = torch.split(packed_rank, R, dim=1)
                        if Vq_shared_flat is not None:
                            q_bhd, k_bkd, v_bkd = dense_token_decode_mod.reconstruct_qkv_token_shared_prepacked(
                                q_rank,
                                k_rank,
                                v_rank,
                                Vq_shared_flat,
                                Vk_shared_flat,
                                Vv_shared_flat,
                                H=H,
                                Hk=Hk,
                                Dh=Dh,
                            )
                        else:
                            q_bhd, k_bkd, v_bkd = dense_token_decode_mod.reconstruct_qkv_token_shared(
                                q_rank,
                                k_rank,
                                v_rank,
                                Vq_hrd,
                                Vk_hkrd,
                                Vv_hkrd,
                                **shared_reconstruct_kwargs,
                            )
                        q_bmhd = q_bhd[:, None, :, :].contiguous()
                        k_bmhd = k_bkd[:, None, :, :].contiguous()
                        v_bmhd = v_bkd[:, None, :, :].contiguous()
                        out = _call_flash_attn2_kvcache(
                            fa2_kvcache,
                            q_bmhd,
                            K_cache_bmhd,
                            V_cache_bmhd,
                            k_bmhd=k_bmhd,
                            v_bmhd=v_bmhd,
                            cache_seqlens=cache_seqlens,
                            rotary_cos=cos_half,
                            rotary_sin=sin_half,
                            causal=args.causal,
                            softmax_scale=None,
                            window_size=(-1, -1),
                        )
                        return out.permute(0, 2, 1, 3).contiguous()
                    token_reconstruct_dense_name = "dense_kvcache(rank_proj+token_reconstruct+fa2_kvcache)"
                else:
                    Pq_tok_bhr = Pq_b1hr[:, 0, :, :].contiguous()
                    Pk_tok_bkr = Pk_blhr[:, q_pos, :, :].contiguous()
                    Pv_tok_bkr = Pv_blhr[:, q_pos, :, :].contiguous()

                    def _token_reconstruct_dense_decode():
                        q_bhd, k_bkd, v_bkd = dense_token_decode_mod.reconstruct_qkv_token(
                            Pq_tok_bhr,
                            Pk_tok_bkr,
                            Pv_tok_bkr,
                            Vq_hrd,
                            Vk_hkrd,
                            Vv_hkrd,
                        )
                        q_bmhd = q_bhd[:, None, :, :].contiguous()
                        k_bmhd = k_bkd[:, None, :, :].contiguous()
                        v_bmhd = v_bkd[:, None, :, :].contiguous()
                        out = _call_flash_attn2_kvcache(
                            fa2_kvcache,
                            q_bmhd,
                            K_cache_bmhd,
                            V_cache_bmhd,
                            k_bmhd=k_bmhd,
                            v_bmhd=v_bmhd,
                            cache_seqlens=cache_seqlens,
                            rotary_cos=cos_half,
                            rotary_sin=sin_half,
                            causal=args.causal,
                            softmax_scale=None,
                            window_size=(-1, -1),
                        )
                        return out.permute(0, 2, 1, 3).contiguous()
                    token_reconstruct_dense_name = "dense_kvcache(token_reconstruct+fa2_kvcache)"
        elif dense_token_decode_mod is not None and fa2_kvcache is None and dense_backend == "fa2" and not args.no_dense_token_reconstruct:
            print("[warn] flash_attn_with_kvcache is unavailable; token reconstruct dense-KV path is skipped.")

        # We may need to expand K/V to H heads depending on backend support.
        def _dense_decode_attn():
            q_bh1d = build_q_dense_bh1d()
            if dense_backend == "torch":
                # Expand K/V to H for a fair MHA-style reference
                assert K_bhld is not None and V_bhld is not None
                return _dense_decode_attn_torch(q_bh1d, K_bhld, V_bhld, causal=args.causal)

            if dense_backend == "triton":
                if fa_triton_kvcache is None:
                    raise RuntimeError("Triton FlashAttention KV-cache backend is unavailable.")
                assert K_bhld is not None and V_bhld is not None
                return fa_triton_kvcache(q_bh1d, K_bhld, V_bhld, mask=None, BLOCK_M=32)

            # dense_backend == "fa2"
            assert fa2 is not None
            q_bmhd = q_bh1d.permute(0, 2, 1, 3).contiguous()  # [B,1,H,Dh]
            assert K_rope_blhd is not None and V_blhd is not None
            k_bmhd = K_rope_blhd  # [B,L,Hk,Dh]
            v_bmhd = V_blhd
            try:
                out = _call_flash_attn2(fa2, q_bmhd, k_bmhd, v_bmhd, causal=args.causal, softmax_scale=None, window_size=(-1, -1))
                if out.shape == (B, 1, H, Dh):
                    return out.permute(0, 2, 1, 3).contiguous()
            except Exception:
                pass
            # fallback: expand K/V to H
            k_full = k_bmhd.repeat_interleave(rep, dim=2).contiguous()
            v_full = v_bmhd.repeat_interleave(rep, dim=2).contiguous()
            out = _call_flash_attn2(fa2, q_bmhd, k_full, v_full, causal=args.causal, softmax_scale=None, window_size=(-1, -1))
            return out.permute(0, 2, 1, 3).contiguous()

        _lowrank_reconstruct_attn = None
        if fa2 is not None:
            if factor_layout == "shared":
                assert Pk_blr is not None and Pv_blr is not None

                def _lowrank_reconstruct_attn():
                    q_bh1d = build_q_dense_bh1d()
                    k_bmhd = torch.einsum("blr,krd->blkd", Pk_blr, Vk_hkrd).contiguous()
                    v_bmhd = torch.einsum("blr,krd->blkd", Pv_blr, Vv_hkrd).contiguous()
                    k_bmhd = _rope_apply_blhd(k_bmhd, cos_half, sin_half)
                    q_bmhd = q_bh1d.permute(0, 2, 1, 3).contiguous()
                    out = _call_flash_attn2(fa2, q_bmhd, k_bmhd, v_bmhd, causal=args.causal, softmax_scale=None, window_size=(-1, -1))
                    if out.shape == (B, 1, H, Dh):
                        return out.permute(0, 2, 1, 3).contiguous()
                    k_full = k_bmhd.repeat_interleave(rep, dim=2).contiguous()
                    v_full = v_bmhd.repeat_interleave(rep, dim=2).contiguous()
                    out = _call_flash_attn2(fa2, q_bmhd, k_full, v_full, causal=args.causal, softmax_scale=None, window_size=(-1, -1))
                    return out.permute(0, 2, 1, 3).contiguous()
            else:
                def _lowrank_reconstruct_attn():
                    q_bh1d = build_q_dense_bh1d()
                    k_bmhd = torch.einsum("blhr,hrd->blhd", Pk_blhr, Vk_hkrd).contiguous()
                    v_bmhd = torch.einsum("blhr,hrd->blhd", Pv_blhr, Vv_hkrd).contiguous()
                    k_bmhd = _rope_apply_blhd(k_bmhd, cos_half, sin_half)
                    q_bmhd = q_bh1d.permute(0, 2, 1, 3).contiguous()
                    out = _call_flash_attn2(fa2, q_bmhd, k_bmhd, v_bmhd, causal=args.causal, softmax_scale=None, window_size=(-1, -1))
                    if out.shape == (B, 1, H, Dh):
                        return out.permute(0, 2, 1, 3).contiguous()
                    k_full = k_bmhd.repeat_interleave(rep, dim=2).contiguous()
                    v_full = v_bmhd.repeat_interleave(rep, dim=2).contiguous()
                    out = _call_flash_attn2(fa2, q_bmhd, k_full, v_full, causal=args.causal, softmax_scale=None, window_size=(-1, -1))
                    return out.permute(0, 2, 1, 3).contiguous()

        _dense_cache_only_graph = None
        _dense_full_token_decode_graph = None
        _token_reconstruct_dense_decode_graph = None
        if args.cuda_graph_variants:
            if not args.no_dense:
                if factor_layout == "shared":
                    assert hidden_tok_bm is not None and Wqkv_flat is not None
                    static_hidden_tok_bm = torch.empty_like(hidden_tok_bm)

                    def _prep_dense_graph_inputs():
                        static_hidden_tok_bm.copy_(hidden_tok_bm)

                    def _dense_decode_attn_graph_body():
                        q_flat = torch.matmul(static_hidden_tok_bm, Wqkv_flat[:, :q_dense_dim])
                        q_bh1d = q_flat.reshape(B, 1, H, Dh).permute(0, 2, 1, 3).contiguous()
                        q_bh1d = _rope_apply_bh1d(q_bh1d, cos_half[q_pos], sin_half[q_pos])
                        if dense_backend == "torch":
                            assert K_bhld is not None and V_bhld is not None
                            return _dense_decode_attn_torch(q_bh1d, K_bhld, V_bhld, causal=args.causal)
                        if dense_backend == "triton":
                            if fa_triton_kvcache is None:
                                raise RuntimeError("Triton FlashAttention KV-cache backend is unavailable.")
                            assert K_bhld is not None and V_bhld is not None
                            return fa_triton_kvcache(q_bh1d, K_bhld, V_bhld, mask=None, BLOCK_M=32)
                        assert fa2 is not None
                        assert K_rope_blhd is not None and V_blhd is not None
                        q_bmhd = q_bh1d.permute(0, 2, 1, 3).contiguous()
                        try:
                            out = _call_flash_attn2(
                                fa2, q_bmhd, K_rope_blhd, V_blhd, causal=args.causal, softmax_scale=None, window_size=(-1, -1)
                            )
                            if out.shape == (B, 1, H, Dh):
                                return out.permute(0, 2, 1, 3).contiguous()
                        except Exception:
                            pass
                        k_full = K_rope_blhd.repeat_interleave(rep, dim=2).contiguous()
                        v_full = V_blhd.repeat_interleave(rep, dim=2).contiguous()
                        out = _call_flash_attn2(fa2, q_bmhd, k_full, v_full, causal=args.causal, softmax_scale=None, window_size=(-1, -1))
                        return out.permute(0, 2, 1, 3).contiguous()

                else:
                    static_Pq_b1hr = torch.empty_like(Pq_b1hr)

                    def _prep_dense_graph_inputs():
                        static_Pq_b1hr.copy_(Pq_b1hr)

                    def _dense_decode_attn_graph_body():
                        q_bh1d = torch.einsum("bmhr,hrd->bmhd", static_Pq_b1hr, Vq_hrd).permute(0, 2, 1, 3).contiguous()
                        q_bh1d = _rope_apply_bh1d(q_bh1d, cos_half[q_pos], sin_half[q_pos])
                        if dense_backend == "torch":
                            assert K_bhld is not None and V_bhld is not None
                            return _dense_decode_attn_torch(q_bh1d, K_bhld, V_bhld, causal=args.causal)
                        if dense_backend == "triton":
                            if fa_triton_kvcache is None:
                                raise RuntimeError("Triton FlashAttention KV-cache backend is unavailable.")
                            assert K_bhld is not None and V_bhld is not None
                            return fa_triton_kvcache(q_bh1d, K_bhld, V_bhld, mask=None, BLOCK_M=32)
                        assert fa2 is not None
                        assert K_rope_blhd is not None and V_blhd is not None
                        q_bmhd = q_bh1d.permute(0, 2, 1, 3).contiguous()
                        try:
                            out = _call_flash_attn2(
                                fa2, q_bmhd, K_rope_blhd, V_blhd, causal=args.causal, softmax_scale=None, window_size=(-1, -1)
                            )
                            if out.shape == (B, 1, H, Dh):
                                return out.permute(0, 2, 1, 3).contiguous()
                        except Exception:
                            pass
                        k_full = K_rope_blhd.repeat_interleave(rep, dim=2).contiguous()
                        v_full = V_blhd.repeat_interleave(rep, dim=2).contiguous()
                        out = _call_flash_attn2(fa2, q_bmhd, k_full, v_full, causal=args.causal, softmax_scale=None, window_size=(-1, -1))
                        return out.permute(0, 2, 1, 3).contiguous()

                _dense_cache_only_graph = _make_cuda_graph_replay(
                    device=dev,
                    prep_inputs=_prep_dense_graph_inputs,
                    run_capture=_dense_decode_attn_graph_body,
                    clone_output=False,
                )

            if _dense_full_token_decode is not None:
                assert K_cache_bmhd is not None and V_cache_bmhd is not None and cache_seqlens is not None
                assert hidden_tok_bm is not None and Wqkv_flat is not None
                static_K_cache_full_bmhd = K_cache_bmhd.clone()
                static_V_cache_full_bmhd = V_cache_bmhd.clone()
                static_cache_seqlens_full = cache_seqlens.clone()
                static_hidden_tok_full_bm = torch.empty_like(hidden_tok_bm)

                def _prep_dense_full_graph_inputs():
                    static_hidden_tok_full_bm.copy_(hidden_tok_bm)

                def _dense_full_token_decode_graph_body():
                    packed_qkv = torch.matmul(static_hidden_tok_full_bm, Wqkv_flat)
                    q_flat, k_flat, v_flat = torch.split(packed_qkv, (q_dense_dim, kv_dense_dim, kv_dense_dim), dim=1)
                    q_bmhd = q_flat.reshape(B, 1, H, Dh).contiguous()
                    k_bmhd = k_flat.reshape(B, 1, Hk, Dh).contiguous()
                    v_bmhd = v_flat.reshape(B, 1, Hk, Dh).contiguous()
                    out = _call_flash_attn2_kvcache(
                        fa2_kvcache,
                        q_bmhd,
                        static_K_cache_full_bmhd,
                        static_V_cache_full_bmhd,
                        k_bmhd=k_bmhd,
                        v_bmhd=v_bmhd,
                        cache_seqlens=static_cache_seqlens_full,
                        rotary_cos=cos_half,
                        rotary_sin=sin_half,
                        causal=args.causal,
                        softmax_scale=None,
                        window_size=(-1, -1),
                    )
                    return out.permute(0, 2, 1, 3).contiguous()

                _dense_full_token_decode_graph = _make_cuda_graph_replay(
                    device=dev,
                    prep_inputs=_prep_dense_full_graph_inputs,
                    run_capture=_dense_full_token_decode_graph_body,
                    clone_output=False,
                )

            if _token_reconstruct_dense_decode is not None:
                assert K_cache_bmhd is not None and V_cache_bmhd is not None and cache_seqlens is not None
                static_K_cache_bmhd = K_cache_bmhd.clone()
                static_V_cache_bmhd = V_cache_bmhd.clone()
                static_cache_seqlens = cache_seqlens.clone()
                if factor_layout == "shared":
                    assert hidden_tok_bm is not None and Uqkv_cat_m3r is not None
                    static_hidden_tok_rank_bm = torch.empty_like(hidden_tok_bm)

                    def _prep_token_graph_inputs():
                        static_hidden_tok_rank_bm.copy_(hidden_tok_bm)

                    def _token_reconstruct_dense_decode_graph_body():
                        packed_rank = torch.matmul(static_hidden_tok_rank_bm, Uqkv_cat_m3r)
                        q_rank, k_rank, v_rank = torch.split(packed_rank, R, dim=1)
                        if Vq_shared_flat is not None:
                            q_bhd, k_bkd, v_bkd = dense_token_decode_mod.reconstruct_qkv_token_shared_prepacked(
                                q_rank,
                                k_rank,
                                v_rank,
                                Vq_shared_flat,
                                Vk_shared_flat,
                                Vv_shared_flat,
                                H=H,
                                Hk=Hk,
                                Dh=Dh,
                            )
                        else:
                            q_bhd, k_bkd, v_bkd = dense_token_decode_mod.reconstruct_qkv_token_shared(
                                q_rank,
                                k_rank,
                                v_rank,
                                Vq_hrd,
                                Vk_hkrd,
                                Vv_hkrd,
                                **shared_reconstruct_kwargs,
                            )
                        q_bmhd = q_bhd[:, None, :, :].contiguous()
                        k_bmhd = k_bkd[:, None, :, :].contiguous()
                        v_bmhd = v_bkd[:, None, :, :].contiguous()
                        out = _call_flash_attn2_kvcache(
                            fa2_kvcache,
                            q_bmhd,
                            static_K_cache_bmhd,
                            static_V_cache_bmhd,
                            k_bmhd=k_bmhd,
                            v_bmhd=v_bmhd,
                            cache_seqlens=static_cache_seqlens,
                            rotary_cos=cos_half,
                            rotary_sin=sin_half,
                            causal=args.causal,
                            softmax_scale=None,
                            window_size=(-1, -1),
                        )
                        return out.permute(0, 2, 1, 3).contiguous()

                else:
                    assert Pq_tok_bhr is not None and Pk_tok_bkr is not None and Pv_tok_bkr is not None
                    static_Pq_tok_bhr = torch.empty_like(Pq_tok_bhr)
                    static_Pk_tok_bkr = torch.empty_like(Pk_tok_bkr)
                    static_Pv_tok_bkr = torch.empty_like(Pv_tok_bkr)

                    def _prep_token_graph_inputs():
                        static_Pq_tok_bhr.copy_(Pq_tok_bhr)
                        static_Pk_tok_bkr.copy_(Pk_tok_bkr)
                        static_Pv_tok_bkr.copy_(Pv_tok_bkr)

                    def _token_reconstruct_dense_decode_graph_body():
                        q_bhd, k_bkd, v_bkd = dense_token_decode_mod.reconstruct_qkv_token(
                            static_Pq_tok_bhr,
                            static_Pk_tok_bkr,
                            static_Pv_tok_bkr,
                            Vq_hrd,
                            Vk_hkrd,
                            Vv_hkrd,
                        )
                        q_bmhd = q_bhd[:, None, :, :].contiguous()
                        k_bmhd = k_bkd[:, None, :, :].contiguous()
                        v_bmhd = v_bkd[:, None, :, :].contiguous()
                        out = _call_flash_attn2_kvcache(
                            fa2_kvcache,
                            q_bmhd,
                            static_K_cache_bmhd,
                            static_V_cache_bmhd,
                            k_bmhd=k_bmhd,
                            v_bmhd=v_bmhd,
                            cache_seqlens=static_cache_seqlens,
                            rotary_cos=cos_half,
                            rotary_sin=sin_half,
                            causal=args.causal,
                            softmax_scale=None,
                            window_size=(-1, -1),
                        )
                        return out.permute(0, 2, 1, 3).contiguous()

                _token_reconstruct_dense_decode_graph = _make_cuda_graph_replay(
                    device=dev,
                    prep_inputs=_prep_token_graph_inputs,
                    run_capture=_token_reconstruct_dense_decode_graph_body,
                    clone_output=False,
                )

        # ----------------------------
        # Low-rank KV cache decode simulation (streaming)
        # ----------------------------
        def _lowrank_decode_streaming():
            q_bh1d = build_q_dense_bh1d()
            if factor_layout == "shared":
                assert Pk_blr is not None and Pv_blr is not None
                return _lowrank_decode_stream_shared(
                    q_bh1d=q_bh1d,
                    Pk_blr=Pk_blr,
                    Pv_blr=Pv_blr,
                    Vk_hrd=Vk_hkrd,
                    Vv_hrd=Vv_hkrd,
                    H=H,
                    Hk=Hk,
                    BN=BN,
                    causal=args.causal,
                    cos_half=cos_half,
                    sin_half=sin_half,
                )
            return _lowrank_decode_stream(
                q_bh1d=q_bh1d,
                Pk_blhr=Pk_blhr,
                Pv_blhr=Pv_blhr,
                Vk_hrd=Vk_hkrd,
                Vv_hrd=Vv_hkrd,
                H=H,
                Hk=Hk,
                BN=BN,
                causal=args.causal,
                cos_half=cos_half,
                sin_half=sin_half,
            )

        # ----------------------------
        # Fused low-rank decode (Triton) + RoPE + split-K
        # ----------------------------
        if args.fused_tune:
            print("[warn] --fused-tune is currently ignored when using --fused-modules (using provided --split-k/--bn/--fused-warps*).")

        fused_variants: list[tuple[str, Callable[[], object]]] = []
        for mod_name, fs_decode_mod in fused_mods:
            try:
                fused_variants.extend(
                    _make_fused_decode_variants(
                        mod_name=mod_name,
                        mod=fs_decode_mod,
                        B=B,
                        H=H,
                        Hk=Hk,
                        Dh=Dh,
                        R=R,
                        L=L,
                        dtype=dtype,
                        dev=dev,
                        Pq_b1hr=Pq_b1hr,
                        Pk_blhr=Pk_blhr,
                        Pv_blhr=Pv_blhr,
                        Vq_hrd=Vq_hrd,
                        Vk_hkrd=Vk_hkrd,
                        Vv_hkrd=Vv_hkrd,
                        cos_half=cos_half,
                        sin_half=sin_half,
                        causal=args.causal,
                        split_k=int(fused_split_k_eff),
                        bn=int(fused_bn_eff),
                        br=int(fused_br_eff),
                        warps1=int(fused_warps1_eff),
                        stages1=int(fused_stages1_eff),
                        warps2=int(args.fused_warps2),
                        stages2=int(args.fused_stages2),
                        split_k_v2=int(v2_split_k_eff),
                        bn_v2=int(v2_bn_eff),
                        br_v2=int(v2_br_eff),
                        warps1_v2=int(v2_warps1_eff),
                        stages1_v2=int(v2_stages1_eff),
                        pad_to_16_v2=bool(v2_pad_to_16),
                        ablate_vk_resident=(not args.no_fused_vk_ablation),
                    )
                )
            except Exception as e:
                print(f"[warn] Fused decode disabled for module={mod_name}, L={L}: {e}")

        # ----------------------------
        # Report theoretical cache sizes
        # ----------------------------
        bytes_per = 2  # fp16/bf16
        dense_kv_bytes = B * L * Hk * Dh * bytes_per * 2
        if factor_layout == "shared":
            lowrank_cache_factor_bytes = B * L * R * bytes_per * 2
            lowrank_kv_proj_params_layout = 2 * R * (hidden_size + kv_dim)
            layout_ratio_den = kv_dim
            layout_rank_desc = f"shared_rank={R}"
            ratio_desc = f"R/(Hk*Dh)={R}/({Hk}*{Dh})"
            achieved_ratio_layout_str = f"achieved_ratio_shared(global)={achieved_ratio_global:.4f}"
        else:
            lowrank_cache_factor_bytes = B * L * Hk * R * bytes_per * 2
            lowrank_kv_proj_params_layout = 2 * Hk * R * (hidden_size + Dh)
            layout_ratio_den = Dh
            layout_rank_desc = f"per_head={R}"
            ratio_desc = f"R/Dh={R}/{Dh}"
            achieved_ratio_layout_str = (
                f"achieved_ratio_headwise={achieved_ratio_headwise:.4f}, "
                f"achieved_ratio_global={achieved_ratio_global:.4f}"
            )
        lowrank_basis_bytes = Hk * R * Dh * bytes_per * 2
        lowrank_kv_bytes = lowrank_cache_factor_bytes + lowrank_basis_bytes
        io_ratio_cache_only = dense_kv_bytes / max(1, lowrank_cache_factor_bytes)
        io_ratio_conservative = dense_kv_bytes / max(1, lowrank_kv_bytes)
        dense_kv_proj_params = 2 * hidden_size * kv_dim
        lowrank_kv_proj_params_headwise = 2 * Hk * R * (hidden_size + Dh)
        lowrank_kv_proj_params_global = 2 * rank_total_effective * (hidden_size + kv_dim)
        kv_cache_ratio = lowrank_cache_factor_bytes / max(1, dense_kv_bytes)

        fused_cfg_str = (
            "disabled"
            if not fused_variants
            else (
                f"{len(fused_variants)} variants "
                f"(base: split_k={int(fused_split_k_eff)} bn={int(fused_bn_eff)} br={int(fused_br_eff)} warps1={int(fused_warps1_eff)} stages1={int(fused_stages1_eff)}; "
                f"v2: split_k={int(v2_split_k_eff)} bn={int(v2_bn_eff)} br={int(v2_br_eff)} warps1={int(v2_warps1_eff)} stages1={int(v2_stages1_eff)} pad_to_16={int(v2_pad_to_16)})"
            )
        )
        cache_den = kv_dim if factor_layout == "shared" else Dh
        if args.report == "full":
            print("==== Decode KV-cache comparison (single-step, q_len=1) ====")
            print(
                f"Shape: B={B}, L={L}, H={H}, Hk={Hk} (rep={rep}), Dh={Dh}, R={R}, "
                f"layout={factor_layout}, dtype={args.dtype}, causal={args.causal}"
            )
            if preset_used is not None:
                print(f"Preset: llama={llama_name} (hidden_size={preset_used.hidden_size}, H={preset_used.num_heads}, Hk={preset_used.num_kv_heads}, Dh={preset_used.head_dim})")
            print(
                f"Rank mapping: {layout_rank_desc}, total_effective={rank_total_effective}"
                + (f", total_target={rank_total_target}" if rank_total_target > 0 else "")
            )
            print(
                f"Rank policy: {rank_policy} | hidden_size={hidden_size}, kv_dim={kv_dim}, "
                f"{achieved_ratio_layout_str}"
            )
            print(
                f"KV proj params/layer: dense={dense_kv_proj_params:,} | "
                f"lowrank_layout={lowrank_kv_proj_params_layout:,} | "
                f"lowrank_headwise_ref={lowrank_kv_proj_params_headwise:,} | "
                f"lowrank_global_ref={lowrank_kv_proj_params_global:,}"
            )
            print(f"Expected cache ratio (lowrank/dense, cache-only): {ratio_desc}={kv_cache_ratio:.4f}x")
            print(
                f"Config: factor_layout={factor_layout} | dense_backend={dense_backend} | lowrank_stream_bn={BN} | fused={fused_cfg_str} "
                f"| dense_token_reconstruct={int(_token_reconstruct_dense_decode is not None)} "
                f"| cuda_graph_variants={int(args.cuda_graph_variants)} "
                f"| mem_reset_each_variant={int(mem_reset_each_variant)}"
            )
            if auto_blocksize_changes:
                print(f"[auto-blocksize] {'; '.join(auto_blocksize_changes)}")
            if v2_auto_changes:
                print(f"[auto-v2] {'; '.join(v2_auto_changes)}")
            print(f"Theoretical KV read/step: dense≈{_pretty_bytes(dense_kv_bytes)}")
            print(
                f"Theoretical KV read/step (lowrank, cache-only)=≈{_pretty_bytes(lowrank_cache_factor_bytes)} "
                f"| IO upper-bound x{io_ratio_cache_only:.2f}"
            )
            print(
                f"Theoretical KV read/step (lowrank, cache+basis)=≈{_pretty_bytes(lowrank_kv_bytes)} "
                f"(basis≈{_pretty_bytes(lowrank_basis_bytes)}) | IO upper-bound x{io_ratio_conservative:.2f}"
            )
            if R >= cache_den:
                print(
                    f"[note] R ({R}) >= cache_den ({cache_den}): "
                    "low-rank KV cache can be larger than dense KV cache at decode-time."
                )
            if Hk != H and dense_backend in ("torch", "triton"):
                print("[note] dense_backend=torch/triton expands K/V from Hk to H via repeat_interleave (not true GQA). Prefer --dense-backend fa2.")

        variants: list[tuple[str, Callable[[], object]]] = [
            *([] if args.no_dense else [(f"baseline(reconstruct_abl+{dense_backend})", _dense_decode_attn)]),
            *(
                []
                if _dense_cache_only_graph is None
                else [(f"baseline(reconstruct_abl+{dense_backend}+graph)", _dense_cache_only_graph)]
            ),
            *([] if _dense_full_token_decode is None else [(dense_full_token_name, _dense_full_token_decode)]),
            *(
                []
                if _dense_full_token_decode_graph is None
                else [("dense_kvcache(full_qkv+fa2_kvcache+graph)", _dense_full_token_decode_graph)]
            ),
            *([] if _token_reconstruct_dense_decode is None else [(token_reconstruct_dense_name, _token_reconstruct_dense_decode)]),
            *(
                []
                if _token_reconstruct_dense_decode_graph is None
                else [("dense_kvcache(rank_proj+token_reconstruct+fa2_kvcache+graph)", _token_reconstruct_dense_decode_graph)]
            ),
            *([] if _lowrank_reconstruct_attn is None else [(f"lowrank_reconstruct({dense_backend})", _lowrank_reconstruct_attn)]),
            *([] if args.no_stream else [("lowrank_kvcache(streaming_torch_rope)", _lowrank_decode_streaming)]),
        ]
        variants.extend(fused_variants)

        results: list[tuple[str, float, float, int, int, int, int]] = []
        failures: list[tuple[str, Exception]] = []
        dense_cache_released = False
        for name, fn in variants:
            try:
                if (
                    (not args.check)
                    and (not dense_cache_released)
                    and (not args.no_dense)
                    and (not _uses_dense_resident_cache(name))
                ):
                    # Release dense-only resident cache before low-rank variants so memory peak
                    # reflects each path more independently in this shared script.
                    K_rope_blhd = None
                    V_blhd = None
                    K_bhld = None
                    V_bhld = None
                    dense_cache_released = True
                    _reset_cuda_memory_state()

                if mem_reset_each_variant:
                    _reset_cuda_memory_state()
                ms = _bench_ms(fn, warmup=args.warmup, iters=args.iters)
                tok_s = B / (ms / 1e3)
                delta_alloc, delta_res, peak_alloc, peak_res = _isolated_peak_bytes(fn)
                results.append((name, ms, tok_s, delta_alloc, delta_res, peak_alloc, peak_res))
            except Exception as e:
                failures.append((name, e))

        if not results:
            print("[error] All variants failed for this L.")
            for name, e in failures:
                print(f"- {name}: FAILED ({type(e).__name__}: {e})")
            continue

        best_result = min(results, key=lambda x: x[1])
        best_ms = best_result[1]
        fair_baseline_ref = _find_named_result(results, f"baseline(reconstruct_abl+{dense_backend})")
        fair_baseline_graph_ref = _find_named_result(results, f"baseline(reconstruct_abl+{dense_backend}+graph)")
        dense_step_ref = _find_named_result(results, "dense_kvcache(full_qkv+fa2_kvcache)")
        dense_step_graph_ref = _find_named_result(results, "dense_kvcache(full_qkv+fa2_kvcache+graph)")
        rank_step_ref = _find_named_result(results, "dense_kvcache(rank_proj+token_reconstruct+fa2_kvcache)")
        rank_step_graph_ref = _find_named_result(results, "dense_kvcache(rank_proj+token_reconstruct+fa2_kvcache+graph)")
        step_candidates = [
            item
            for item in (dense_step_ref, dense_step_graph_ref, rank_step_ref, rank_step_graph_ref)
            if item is not None
        ]
        step_best_result = min(step_candidates, key=lambda x: x[1]) if step_candidates else best_result
        step_best_ms = step_best_result[1]
        best_lowrank = _best_lowrank_result(results)
        flashsvd_ref = _best_matching_result(results, _is_flashsvd_v15_variant)
        sparse_ref = _best_matching_result(results, _is_sparse_legacy_variant)
        sparse_fa2_ref = _best_matching_result(results, _is_sparse_fa2_only_variant)
        dense_fa2_ref = _best_matching_result(results, _is_dense_fa2_only_variant)
        direct_candidates = [
            item for item in (flashsvd_ref, sparse_ref, sparse_fa2_ref, dense_fa2_ref) if item is not None
        ]
        direct_best_result = min(direct_candidates, key=lambda x: x[1]) if direct_candidates else best_result

        if args.report == "full":
            for name, ms, tok_s, delta_alloc, delta_res, peak_alloc, peak_res in results:
                rel = ms / best_ms
                print(
                    f"- {name}: {ms:.4f} ms | {tok_s:,.0f} tok/s | x{rel:.2f} vs best | "
                    f"peak_delta_alloc={_pretty_bytes(delta_alloc)} peak_delta_res={_pretty_bytes(delta_res)} | "
                    f"peak_alloc={_pretty_bytes(peak_alloc)} peak_res={_pretty_bytes(peak_res)}"
                )
            for name, e in failures:
                print(f"- {name}: FAILED ({type(e).__name__}: {e})")

            dense_baseline_ref = dense_step_ref if dense_step_ref is not None else fair_baseline_ref
            if dense_baseline_ref is not None:
                dense_ms = dense_baseline_ref[1]
                print(f"[speedup_vs_dense] baseline={dense_baseline_ref[0]} ({dense_ms:.4f} ms)")
                for name, ms, tok_s, _, _, _, _ in results:
                    if name.startswith("dense_"):
                        continue
                    sp = dense_ms / ms
                    print(f"  - {name}: x{sp:.2f} speedup | {tok_s:,.0f} tok/s")

            stream_ref = next((r for r in results if r[0] == "lowrank_kvcache(streaming_torch_rope)"), None)
            if stream_ref is not None:
                stream_ms = stream_ref[1]
                print(f"[speedup_vs_lowrank_stream] baseline={stream_ref[0]} ({stream_ms:.4f} ms)")
                for name, ms, tok_s, _, _, _, _ in results:
                    if name == stream_ref[0]:
                        continue
                    sp = stream_ms / ms
                    print(f"  - {name}: x{sp:.2f} speedup | {tok_s:,.0f} tok/s")
        else:
            sparse_gap = (
                "-"
                if sparse_ref is None or flashsvd_ref is None
                else f"{sparse_ref[1] / max(flashsvd_ref[1], 1e-9):.2f}x"
            )
            fail_summary = (
                "-"
                if not failures
                else ", ".join(
                    f"{_compact_variant_name(name)}:{_compact_failure_reason(err)}"
                    for name, err in failures
                )
            )
            print(
                f"{L:>5} | {_compact_variant_name(direct_best_result[0]):<18} | {direct_best_result[1]:>9.4f} | "
                f"{_fmt_ms_or_na(flashsvd_ref):>14} | {_fmt_ms_or_na(sparse_ref):>7} | "
                f"{_fmt_ms_or_na(sparse_fa2_ref):>16} | {_fmt_ms_or_na(dense_fa2_ref):>15} | "
                f"{sparse_gap:>15} | {fail_summary}"
            )

        case_reports[(B, L)] = {
            "results": list(results),
            "failures": list(failures),
            "dense_kv_bytes": int(dense_kv_bytes),
            "lowrank_cache_factor_bytes": int(lowrank_cache_factor_bytes),
            "best_result": best_result,
            "step_best_result": step_best_result,
            "best_lowrank": best_lowrank,
            "fair_baseline_ref": fair_baseline_ref,
            "fair_baseline_graph_ref": fair_baseline_graph_ref,
            "dense_step_ref": dense_step_ref,
            "dense_step_graph_ref": dense_step_graph_ref,
            "rank_step_ref": rank_step_ref,
            "rank_step_graph_ref": rank_step_graph_ref,
            "flashsvd_ref": flashsvd_ref,
            "sparse_ref": sparse_ref,
            "sparse_fa2_ref": sparse_fa2_ref,
            "dense_fa2_ref": dense_fa2_ref,
            "direct_best_result": direct_best_result,
        }

        if args.check:
            if L > 256:
                print("[check] Skipped: recommend --L <= 256 for correctness check.")
            else:
                with torch.no_grad():
                    # Reference via torch dense attention (expanded to H)
                    q_bh1d = build_q_dense_bh1d()
                    K_bhld = K_rope_blhd.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1).contiguous()
                    V_bhld = V_blhd.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1).contiguous()
                    ref = _dense_decode_attn_torch(q_bh1d, K_bhld, V_bhld, causal=args.causal).to(torch.float32)

                    out_stream = _lowrank_decode_streaming().to(torch.float32)
                    diff = out_stream - ref
                    rel = (torch.linalg.norm(diff) / (torch.linalg.norm(ref) + 1e-12)).item()
                    max_abs = diff.abs().max().item()
                    finite = torch.isfinite(out_stream).all().item()
                    print(f"[check] lowrank_stream vs torch_ref: finite={finite} max_abs={max_abs:.3e} rel_fro={rel:.3e}")

                    if _token_reconstruct_dense_decode is not None:
                        out_token = _token_reconstruct_dense_decode().to(torch.float32)
                        diff_tok = out_token - ref
                        rel_tok = (torch.linalg.norm(diff_tok) / (torch.linalg.norm(ref) + 1e-12)).item()
                        max_abs_tok = diff_tok.abs().max().item()
                        finite_tok = torch.isfinite(out_token).all().item()
                        print(
                            f"[check] {token_reconstruct_dense_name} vs torch_ref: "
                            f"finite={finite_tok} max_abs={max_abs_tok:.3e} rel_fro={rel_tok:.3e}"
                        )

                    for fused_name, fused_fn in fused_variants:
                        out_fused = fused_fn().to(torch.float32)[:, :, None, :]
                        diff2 = out_fused - ref
                        rel2 = (torch.linalg.norm(diff2) / (torch.linalg.norm(ref) + 1e-12)).item()
                        max_abs2 = diff2.abs().max().item()
                        finite2 = torch.isfinite(out_fused).all().item()
                        print(f"[check] {fused_name} vs torch_ref: finite={finite2} max_abs={max_abs2:.3e} rel_fro={rel2:.3e}")

    if args.report == "compact" and case_reports:
        win_counts: dict[str, int] = {}
        overall_win_counts: dict[str, int] = {}
        failure_counts: dict[str, int] = {}
        lowrank_wins = 0
        for _, report in sorted(case_reports.items()):
            direct_best_result = report.get("direct_best_result")
            if direct_best_result is not None:
                step_name = _compact_variant_name(direct_best_result[0])  # type: ignore[index]
                win_counts[step_name] = win_counts.get(step_name, 0) + 1
            best_result = report.get("best_result")
            if best_result is not None:
                overall_name = _compact_variant_name(best_result[0])  # type: ignore[index]
                overall_win_counts[overall_name] = overall_win_counts.get(overall_name, 0) + 1
                if _is_lowrank_online_variant(str(best_result[0])):  # type: ignore[index]
                    lowrank_wins += 1
            for fail_name, _ in report.get("failures", []):  # type: ignore[assignment]
                short = _compact_variant_name(fail_name)
                failure_counts[short] = failure_counts.get(short, 0) + 1

        win_summary = ", ".join(f"{k}={v}" for k, v in sorted(win_counts.items(), key=lambda x: (-x[1], x[0])))
        print(f"Step winner counts (full online step): {win_summary}")
        overall_summary = ", ".join(
            f"{k}={v}" for k, v in sorted(overall_win_counts.items(), key=lambda x: (-x[1], x[0]))
        )
        print(f"Overall winner counts (including reconstruct-ablation lower bound): {overall_summary}")
        if lowrank_wins == 0:
            print("Overall (including reconstruct-ablation lower bound): no low-rank online variant wins.")
        else:
            print(f"Overall (including reconstruct-ablation lower bound): low-rank online variants win {lowrank_wins}/{len(case_reports)} cases.")
        if failure_counts:
            fail_summary = ", ".join(f"{k}={v}" for k, v in sorted(failure_counts.items()))
            print(f"Failure counts: {fail_summary}")

    if args.compare_kv_budget:
        print("==== KV-Budget Throughput Summary ====")
        if len(Bs) < 2:
            print("[warn] Need at least two batch sizes to compare KV-budget mode.")
        else:
            b_dense = Bs[0]
            b_lowrank = Bs[1]
            scale = float(b_lowrank) / float(max(1, b_dense))
            scale_note = f" (auto_scale={kv_budget_scale:.2f})" if kv_budget_scale > 0 else ""
            print(f"Compare batches: dense/ref B={b_dense}, lowrank/budget B={b_lowrank}, scale={scale:.2f}{scale_note}")
            if args.realistic_attn:
                print("[note] realistic-attn enabled: streaming baseline disabled for closer kernel-path comparison.")

            for L in Ls:
                base = case_reports.get((b_dense, L))
                budget = case_reports.get((b_lowrank, L))
                if base is None or budget is None:
                    print(f"- L={L}: missing case(s), skipped.")
                    continue

                dense_ref = base.get("dense_step_ref")
                low_base = base.get("best_lowrank")
                low_budget = budget.get("best_lowrank")
                if dense_ref is None or low_base is None or low_budget is None:
                    print(f"- L={L}: missing dense or lowrank result, skipped.")
                    continue

                dense_name, _, dense_tok_s, _, _, _, _ = dense_ref
                low_base_name, _, low_base_tok_s, _, _, _, _ = low_base
                low_budget_name, _, low_budget_tok_s, _, _, _, _ = low_budget
                same_input_gain = low_base_tok_s / max(1e-12, dense_tok_s)
                kv_budget_gain = low_budget_tok_s / max(1e-12, dense_tok_s)

                dense_kv = int(base.get("dense_kv_bytes", 0))
                lowrank_kv = int(budget.get("lowrank_cache_factor_bytes", 0))
                budget_ratio = (lowrank_kv / dense_kv) if dense_kv > 0 else float("nan")
                print(
                    f"- L={L}: same_input_gain={same_input_gain:.3f}x "
                    f"({low_base_name} @B={b_dense} vs {dense_name} @B={b_dense}); "
                    f"kv_budget_gain={kv_budget_gain:.3f}x "
                    f"({low_budget_name} @B={b_lowrank} vs {dense_name} @B={b_dense}); "
                    f"budget_match(lowrank/dense)={budget_ratio:.3f}x"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
