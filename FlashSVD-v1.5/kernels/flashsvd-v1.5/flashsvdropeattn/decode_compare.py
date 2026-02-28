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

  - lowrank_kvcache_stream: KV cache stored as low-rank factors Pk/Pv [B,L,Hk,R] + bases Vk/Vv [Hk,R,Dh].
    Each iteration reconstructs K/V in blocks (BN) and runs FlashAttention-style online softmax in PyTorch.

  - lowrank_kvcache_fused(triton): FlashSVD low-rank KV-cache decode kernel with RoPE + split-K.

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
import time
from typing import Callable, Optional, Tuple


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


def _isolated_peak_bytes(fn: Callable[[], object]) -> tuple[int, int]:
    import torch

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
    # Treat relative paths as relative to this folder.
    return os.path.join(here, p)


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
    here = os.path.dirname(os.path.abspath(__file__))
    kernels_dir = os.path.abspath(os.path.join(here, "..", ".."))  # kernels/
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
    here = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.join(here, "flashsvdropeattn_v1.5_decode.py")
    if not os.path.exists(mod_path):
        raise FileNotFoundError(mod_path)
    return _import_from_path("flashsvdropeattn_v15_decode_local", mod_path)


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
    ablate_vk_resident: bool,
) -> list[tuple[str, Callable[[], object]]]:
    import torch

    if split_k % bn != 0:
        raise ValueError(f"split_k ({split_k}) must be a multiple of bn ({bn})")

    variants: list[tuple[str, Callable[[], object]]] = []
    if not (hasattr(mod, "DecodePackedFactors") and hasattr(mod, "flashsvd_attn_decode_packed")):
        return variants

    Pq_bhr = Pq_b1hr[:, 0, :, :].contiguous()
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

    if supports_vk and ablate_vk_resident:
        base_v2: dict[str, object] = dict(
            q_buffers=(Q0_ws, Q1_ws),
            precompute_q=True,
            pad_to_16=True,
            writethrough=True,
        )
        _wrap(fn, name="lowrank_fused_v2(vk_resident=1)", extra={**base_v2, "vk_resident": True})
        _wrap(fn, name="lowrank_fused_v2(vk_resident=0)", extra={**base_v2, "vk_resident": False})
    else:
        _wrap(fn, name="lowrank_fused", extra={})

    return variants


def main() -> int:
    ap = argparse.ArgumentParser("Decode-stage KV-cache comparison (dense vs low-rank)")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--L", type=int, default=2048, help="KV cache length")
    ap.add_argument("--Ls", type=str, default="", help="Comma-separated KV lengths to sweep (overrides --L)")
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--Hk", type=int, default=8)
    ap.add_argument("--Dh", type=int, default=128)
    ap.add_argument("--R", type=int, default=64)
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
        default="flashsvdropeattn_v1.5_decode.py,flashsvdropeattn_v1.6_decode_opt.py",
        help="Comma-separated fused decode module paths (relative to this folder or absolute).",
    )
    ap.add_argument("--no-fused-vk-ablation", action="store_true", help="Disable vk_resident on/off ablation (if supported).")
    ap.add_argument("--no-dense", action="store_true", help="Disable dense KV-cache baseline")
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
    ap.add_argument("--check", action="store_true", help="run a small reference check (recommend --L <= 256)")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("[error] CUDA is required.")
        return 2

    if args.H % args.Hk != 0:
        print(f"[error] GQA requires H divisible by Hk, got H={args.H}, Hk={args.Hk}")
        return 2

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")

    B, H, Hk, Dh, R = args.B, args.H, args.Hk, args.Dh, args.R
    rep = H // Hk
    BN = max(1, args.bn)

    if args.Ls.strip():
        Ls = [int(x) for x in args.Ls.split(",") if x.strip()]
    else:
        Ls = [int(args.L)]
    if not Ls:
        print("[error] Empty --Ls")
        return 2

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
    if dense_backend_req == "fa2" and fa2 is None:
        raise SystemExit("FlashAttention-2 not found but --dense-backend=fa2 was requested.")

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

    for L in Ls:
        if L <= 0:
            print(f"[warn] Skipping non-positive L={L}")
            continue

        # ----------------------------
        # Generate low-rank KV cache + query factors
        # ----------------------------
        Pq_b1hr = torch.randn(B, 1, H, R, device=dev, dtype=dtype).contiguous()
        Pk_blhr = torch.randn(B, L, Hk, R, device=dev, dtype=dtype).contiguous()
        Pv_blhr = torch.randn(B, L, Hk, R, device=dev, dtype=dtype).contiguous()

        Vq_hrd = torch.randn(H, R, Dh, device=dev, dtype=dtype).contiguous()
        Vk_hkrd = torch.randn(Hk, R, Dh, device=dev, dtype=dtype).contiguous()
        Vv_hkrd = torch.randn(Hk, R, Dh, device=dev, dtype=dtype).contiguous()

        cos_half, sin_half = _build_rope_tables_half(L, Dh, base=10000.0, device=dev, dtype=dtype)
        q_pos = L - 1

        # Query dense: [B,H,1,Dh]
        def build_q_dense_bh1d():
            # Note: torch.einsum subscripts must be letters (no digits).
            q = torch.einsum("bmhr,hrd->bmhd", Pq_b1hr, Vq_hrd).permute(0, 2, 1, 3).contiguous()
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
                K_blhd = torch.einsum("blhr,hrd->blhd", Pk_blhr, Vk_hkrd).contiguous()
                V_blhd = torch.einsum("blhr,hrd->blhd", Pv_blhr, Vv_hkrd).contiguous()
                K_rope_blhd = _rope_apply_blhd(K_blhd, cos_half, sin_half)

                if dense_backend in ("torch", "triton"):
                    # Expand KV-cache to H heads once (cache-like) to avoid timing repeat_interleave.
                    K_bhld = K_rope_blhd.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1).contiguous()
                    V_bhld = V_blhd.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1).contiguous()

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

        # ----------------------------
        # Low-rank KV cache decode simulation (streaming)
        # ----------------------------
        def _lowrank_decode_streaming():
            q_bh1d = build_q_dense_bh1d()
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
                        split_k=int(args.split_k),
                        bn=int(BN),
                        br=int(args.br),
                        warps1=int(args.fused_warps1),
                        stages1=int(args.fused_stages1),
                        warps2=int(args.fused_warps2),
                        stages2=int(args.fused_stages2),
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
        lowrank_kv_bytes = B * L * Hk * R * bytes_per * 2 + Hk * R * Dh * bytes_per * 2

        print("==== Decode KV-cache comparison (single-step, q_len=1) ====")
        print(f"Shape: B={B}, L={L}, H={H}, Hk={Hk} (rep={rep}), Dh={Dh}, R={R}, dtype={args.dtype}, causal={args.causal}")
        fused_cfg_str = "disabled" if not fused_variants else f"{len(fused_variants)} variants (split_k={int(args.split_k)} bn={BN} br={int(args.br)} warps1={int(args.fused_warps1)})"
        print(f"Config: dense_backend={dense_backend} | lowrank_stream_bn={BN} | fused={fused_cfg_str}")
        print(f"Theoretical KV cache size: dense≈{_pretty_bytes(dense_kv_bytes)} | lowrank≈{_pretty_bytes(lowrank_kv_bytes)}")
        if Hk != H and dense_backend in ("torch", "triton"):
            print("[note] dense_backend=torch/triton expands K/V from Hk to H via repeat_interleave (not true GQA). Prefer --dense-backend fa2.")

        variants: list[tuple[str, Callable[[], object]]] = [
            *([] if args.no_dense else [(f"dense_kvcache({dense_backend})", _dense_decode_attn)]),
            *([] if args.no_stream else [("lowrank_kvcache(streaming_torch_rope)", _lowrank_decode_streaming)]),
        ]
        variants.extend(fused_variants)

        results: list[tuple[str, float, float, int, int]] = []
        failures: list[tuple[str, Exception]] = []
        for name, fn in variants:
            try:
                ms = _bench_ms(fn, warmup=args.warmup, iters=args.iters)
                tok_s = B / (ms / 1e3)
                alloc, res = _isolated_peak_bytes(fn)
                results.append((name, ms, tok_s, alloc, res))
            except Exception as e:
                failures.append((name, e))

        if not results:
            print("[error] All variants failed for this L.")
            for name, e in failures:
                print(f"- {name}: FAILED ({type(e).__name__}: {e})")
            continue

        best_ms = min(r[1] for r in results)
        for name, ms, tok_s, alloc, res in results:
            rel = ms / best_ms
            print(f"- {name}: {ms:.4f} ms | {tok_s:,.0f} tok/s | x{rel:.2f} vs best | peak_alloc={_pretty_bytes(alloc)} peak_res={_pretty_bytes(res)}")
        for name, e in failures:
            print(f"- {name}: FAILED ({type(e).__name__}: {e})")

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

                    for fused_name, fused_fn in fused_variants:
                        out_fused = fused_fn().to(torch.float32)[:, :, None, :]
                        diff2 = out_fused - ref
                        rel2 = (torch.linalg.norm(diff2) / (torch.linalg.norm(ref) + 1e-12)).item()
                        max_abs2 = diff2.abs().max().item()
                        finite2 = torch.isfinite(out_fused).all().item()
                        print(f"[check] {fused_name} vs torch_ref: finite={finite2} max_abs={max_abs2:.3e} rel_fro={rel2:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
