#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FlashSVD "held blocks" baseline (unfused pipeline).

This baseline intentionally materializes dense Q/K/V (i.e. holds dense blocks in memory):
  1) reconstruct Q/K/V from low-rank factors (P @ V + bias)
  2) apply RoPE to Q and K
  3) run causal FlashAttention (kernels/flash_attn_causal.py::flash_attn_triton)

It is meant to be compared against the fused kernels in:
  - flashsvdropeattn_v1.5.py
  - flashsvdropeattn_v1.py
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


def _import_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_flash_attn_triton() -> Callable[..., torch.Tensor]:
    here = os.path.dirname(os.path.abspath(__file__))
    kernels_dir = os.path.abspath(os.path.join(here, "..", ".."))  # kernels/
    fa_path = os.path.join(kernels_dir, "flash_attn_causal.py")
    if not os.path.exists(fa_path):
        raise FileNotFoundError(fa_path)
    fa = _import_from_path("flash_attn_causal_local", fa_path)
    return fa.flash_attn_triton


def _try_load_flash_attn2() -> Optional[Callable[..., torch.Tensor]]:
    """
    Try to load FlashAttention-2 python API if installed in the environment.
    Returns a callable like flash_attn_func(q, k, v, ...) or None.
    """
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


def _call_flash_attn2(
    flash_attn_func: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: Optional[float],
    window_size: Tuple[int, int],
) -> torch.Tensor:
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
    return flash_attn_func(q, k, v, **kwargs)


@dataclass
class PackedFactors:
    # Canonical packed layout (matches flashsvdropeattn_v1.5.py)
    Pq: torch.Tensor  # [B, S, H,  R]
    Pk: torch.Tensor  # [B, S, Hk, R]
    Pv: torch.Tensor  # [B, S, Hk, R]
    Vq: torch.Tensor  # [H,  R, Dh]
    Vk: torch.Tensor  # [Hk, R, Dh]
    Vv: torch.Tensor  # [Hk, R, Dh]
    bq: Optional[torch.Tensor] = None  # [H,  Dh]
    bk: Optional[torch.Tensor] = None  # [Hk, Dh]
    bv: Optional[torch.Tensor] = None  # [Hk, Dh]


@torch.no_grad()
def build_rope_tables(
    seqlen: int,
    head_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0
    half = head_dim // 2
    pos = torch.arange(seqlen, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = torch.einsum("m,d->md", pos, inv_freq)  # [seqlen, half]
    cos = torch.cos(ang).to(dtype)
    sin = torch.sin(ang).to(dtype)
    return cos.contiguous(), sin.contiguous()


def _rope_apply_bshd(x_bshd: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
    # x: [B,S,H,Dh], cos/sin: [S, Dh/2]
    Dh = x_bshd.shape[-1]
    half = Dh // 2
    cos = cos_half[None, :, None, :]
    sin = sin_half[None, :, None, :]
    x0 = x_bshd[..., :half]
    x1 = x_bshd[..., half:]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.cat([y0, y1], dim=-1)


@torch.no_grad()
def reconstruct_qkv_packed(factors: object) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reconstruct dense Q,K,V from packed low-rank factors.
    Accepts any object with attributes: Pq,Pk,Pv,Vq,Vk,Vv,(optional) bq,bk,bv.
    """
    Pq = factors.Pq
    Pk = factors.Pk
    Pv = factors.Pv
    Vq = factors.Vq
    Vk = factors.Vk
    Vv = factors.Vv
    bq = getattr(factors, "bq", None)
    bk = getattr(factors, "bk", None)
    bv = getattr(factors, "bv", None)

    Q = torch.einsum("bshr,hrd->bshd", Pq, Vq)
    K = torch.einsum("bskr,krd->bskd", Pk, Vk)
    V = torch.einsum("bskr,krd->bskd", Pv, Vv)

    if bq is not None:
        Q = Q + bq[None, None, :, :]
    if bk is not None:
        K = K + bk[None, None, :, :]
    if bv is not None:
        V = V + bv[None, None, :, :]

    return Q.contiguous(), K.contiguous(), V.contiguous()


@torch.no_grad()
def flashsvd_attn_packed_baseline(
    factors: object,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
    *,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    fa_block_m: int = 32,
    attn_backend: str = "triton",
    flash_attn_fn: Optional[Callable[..., torch.Tensor]] = None,
) -> torch.Tensor:
    """
    End-to-end baseline: reconstruct dense Q/K/V, apply RoPE, then causal FlashAttention.
    Returns O: [B,S,H,Dh]
    """
    Pq = factors.Pq
    Pk = factors.Pk
    Vq = factors.Vq
    Vk = factors.Vk

    if Pq.dim() != 4 or Pk.dim() != 4:
        raise ValueError(f"Expected packed Pq/Pk [B,S,H|Hk,R], got {tuple(Pq.shape)}/{tuple(Pk.shape)}")

    B, S, H, _ = Pq.shape
    _, _, Hk, _ = Pk.shape
    Dh = Vq.shape[-1]

    if H % Hk != 0:
        raise ValueError(f"GQA requires H divisible by Hk, got H={H}, Hk={Hk}")
    rep = H // Hk

    softmax_scale = 1.0 / math.sqrt(Dh)

    Q, K, V = reconstruct_qkv_packed(factors)

    Q = _rope_apply_bshd(Q, cos_half, sin_half)
    K = _rope_apply_bshd(K, cos_half, sin_half)

    attn_backend = attn_backend.lower().strip()
    if attn_backend not in ("triton", "fa2", "auto"):
        raise ValueError("attn_backend must be one of: triton, fa2, auto")

    # ----------------------------
    # Backend A: FlashAttention-2 (if installed)
    # ----------------------------
    if attn_backend in ("fa2", "auto"):
        fa2 = _try_load_flash_attn2()
        if fa2 is None:
            if attn_backend == "fa2":
                raise ImportError("FlashAttention-2 not found (pip install flash-attn).")
        else:
            # Prefer native GQA if supported by the installed FA2; fallback to K/V repeat if needed.
            try:
                out = _call_flash_attn2(
                    fa2,
                    Q.contiguous(),
                    K.contiguous(),
                    V.contiguous(),
                    causal=causal,
                    softmax_scale=softmax_scale,
                    window_size=window_size,
                )
                if out.shape == (B, S, H, Dh):
                    return out.contiguous()
            except Exception:
                pass

            # Fallback: expand K/V to match H heads (works for any attention backend)
            if rep != 1:
                K_full = K.repeat_interleave(rep, dim=2)
                V_full = V.repeat_interleave(rep, dim=2)
            else:
                K_full, V_full = K, V
            out = _call_flash_attn2(
                fa2,
                Q.contiguous(),
                K_full.contiguous(),
                V_full.contiguous(),
                causal=causal,
                softmax_scale=softmax_scale,
                window_size=window_size,
            )
            if out.shape != (B, S, H, Dh):
                raise RuntimeError(f"flash_attn_func returned {tuple(out.shape)}, expected {(B, S, H, Dh)}")
            return out.contiguous()

    # ----------------------------
    # Backend B: repo Triton FlashAttention (causal-only)
    # ----------------------------
    if not causal:
        raise NotImplementedError("Triton baseline backend only supports causal=True.")
    if window_size != (-1, -1):
        raise NotImplementedError("Triton baseline backend only supports window_size=(-1,-1).")

    if rep != 1:
        K = K.repeat_interleave(rep, dim=2)
        V = V.repeat_interleave(rep, dim=2)

    Q_bhmd = Q.permute(0, 2, 1, 3).contiguous()
    K_bhmd = K.permute(0, 2, 1, 3).contiguous()
    V_bhmd = V.permute(0, 2, 1, 3).contiguous()

    if flash_attn_fn is None:
        flash_attn_fn = _load_flash_attn_triton()

    O_bhmd = flash_attn_fn(Q_bhmd, K_bhmd, V_bhmd, mask=None, BLOCK_M=fa_block_m)
    if O_bhmd.shape != (B, H, S, Dh):
        raise RuntimeError(f"flash_attn_triton returned {tuple(O_bhmd.shape)}, expected {(B, H, S, Dh)}")
    return O_bhmd.permute(0, 2, 1, 3).contiguous()  # [B,S,H,Dh]


def _bench_ms(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    try:
        import triton  # type: ignore

        return float(triton.testing.do_bench(fn, warmup=warmup, rep=iters))
    except Exception:
        for _ in range(max(1, warmup)):
            _ = fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max(1, iters)):
            _ = fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / max(1, iters)


def _isolated_peak_bytes(fn: Callable[[], object]) -> tuple[int, int]:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def _pretty_bytes(n: int) -> str:
    x = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} PB"


def main() -> int:
    ap = argparse.ArgumentParser("FlashSVD held-blocks baseline (unfused)")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--S", type=int, default=2048)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--Hk", type=int, default=8)
    ap.add_argument("--Dh", type=int, default=128)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--fa-block-m", type=int, default=32)
    ap.add_argument("--attn-backend", choices=["triton", "fa2", "auto"], default="triton")
    ap.add_argument("--rope-base", type=float, default=10000.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[error] CUDA is required.")
        return 2

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")

    B, S, H, Hk, Dh, R = args.B, args.S, args.H, args.Hk, args.Dh, args.R
    if H % Hk != 0:
        raise SystemExit(f"H must be divisible by Hk for GQA, got H={H}, Hk={Hk}")

    Pq = torch.randn(B, S, H, R, device=dev, dtype=dtype).contiguous()
    Pk = torch.randn(B, S, Hk, R, device=dev, dtype=dtype).contiguous()
    Pv = torch.randn(B, S, Hk, R, device=dev, dtype=dtype).contiguous()
    Vq = torch.randn(H, R, Dh, device=dev, dtype=dtype).contiguous()
    Vk = torch.randn(Hk, R, Dh, device=dev, dtype=dtype).contiguous()
    Vv = torch.randn(Hk, R, Dh, device=dev, dtype=dtype).contiguous()
    bq = torch.randn(H, Dh, device=dev, dtype=dtype).contiguous()
    bk = torch.randn(Hk, Dh, device=dev, dtype=dtype).contiguous()
    bv = torch.randn(Hk, Dh, device=dev, dtype=dtype).contiguous()

    cos_half, sin_half = build_rope_tables(S, Dh, base=args.rope_base, device=dev, dtype=dtype)

    fac = PackedFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv, bq=bq, bk=bk, bv=bv)
    flash_attn_fn = _load_flash_attn_triton()

    def run():
        return flashsvd_attn_packed_baseline(
            fac,
            cos_half,
            sin_half,
            causal=True,
            window_size=(-1, -1),
            fa_block_m=args.fa_block_m,
            attn_backend=args.attn_backend,
            flash_attn_fn=flash_attn_fn,
        )

    ms = _bench_ms(run, warmup=args.warmup, iters=args.iters)
    peak_alloc, peak_reserved = _isolated_peak_bytes(run)

    toks = B * S
    toks_per_s = toks / (ms / 1e3)
    print(
        f"baseline_unfused({args.attn_backend}): {ms:.3f} ms | {toks_per_s:,.0f} tok/s | "
        f"peak_alloc={_pretty_bytes(peak_alloc)} peak_rsv={_pretty_bytes(peak_reserved)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
