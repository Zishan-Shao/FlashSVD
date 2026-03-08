#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare aligned baselines vs FlashSVD fused kernels.

Variants:
  1) baseline_unfused: (P@V)->Q/K/V dense, RoPE, then causal FlashAttention (held dense blocks)
  2) flashsvd_py: flashsvdropeattn_v1.5.py (FA-aligned packed) fused kernel (loaded by path)
  3) ropeattn_v1: flashsvdropeattn_v1.py fused kernel (loaded by path)

This script is meant to be run on a CUDA machine with torch+triton installed.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import itertools
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


def _import_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure the module is visible during execution (e.g. dataclasses with
    # `from __future__ import annotations` expects `sys.modules[__name__]`).
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pick_first_existing_path(candidates: list[str], *, what: str) -> str:
    tried = []
    for p in candidates:
        p = os.path.abspath(p)
        tried.append(p)
        if os.path.exists(p):
            return p
    msg = f"Could not find {what}. Tried:\n" + "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(msg)


def _repo_paths() -> tuple[str, str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    flashsvd_dir = os.path.abspath(os.path.join(here, ".."))  # kernels/flashsvd-v1.5
    kernels_dir = os.path.abspath(os.path.join(flashsvd_dir, ".."))  # kernels/
    repo_root = os.path.abspath(os.path.join(kernels_dir, ".."))
    return repo_root, kernels_dir, flashsvd_dir


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


def _rope_apply_bshd(x, cos_half, sin_half):
    # x: [B,S,H,Dh], cos/sin: [S, Dh/2]
    import torch

    Dh = x.shape[-1]
    half = Dh // 2
    cos = cos_half[None, :, None, :]
    sin = sin_half[None, :, None, :]
    x0 = x[..., :half]
    x1 = x[..., half:]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.cat([y0, y1], dim=-1)


@dataclass
class Inputs:
    # Canonical packed layout (flashsvdropeattn_v1.5.py)
    Pq_bshr: object
    Pk_bskr: object
    Pv_bskr: object
    Vq_hrd: object
    Vk_hrd: object
    Vv_hrd: object
    bq_hd: object
    bk_hkd: object
    bv_hkd: object
    cos_half: object
    sin_half: object

    # Derived layout for ropeattn kernel (BMHd)
    Pq_bhmr: object
    Pk_bhkmr: object
    Pv_bhkmr: object
    bq_flat: object
    bk_flat: object
    bv_flat: object
    cos_bmd: object
    sin_bmd: object


@dataclass
class ProjBuffers:
    # Projection buffers for including X@U cost in the benchmark.
    # Shapes (2D GEMM form):
    #   X2d:  [B*S, D]
    #   Uq:   [D,   H*R]
    #   Uk/Uv:[D,   Hk*R]
    #   Pq2d: [B*S, H*R]   (viewed as [B,S,H,R])
    #   Pk2d: [B*S, Hk*R]  (viewed as [B,S,Hk,R])
    #   Pv2d: [B*S, Hk*R]  (viewed as [B,S,Hk,R])
    X2d: object
    Uq: object
    Uk: object
    Uv: object
    Pq2d: object
    Pk2d: object
    Pv2d: object
    D: int


def _make_inputs(*, fs_mod, B: int, S: int, H: int, Hk: int, Dh: int, R: int, dtype, seed: int) -> Inputs:
    import torch

    assert H % Hk == 0
    rep = H // Hk
    torch.manual_seed(seed)
    dev = torch.device("cuda")

    Pq = torch.randn(B, S, H, R, device=dev, dtype=dtype).contiguous()
    Pk = torch.randn(B, S, Hk, R, device=dev, dtype=dtype).contiguous()
    Pv = torch.randn(B, S, Hk, R, device=dev, dtype=dtype).contiguous()
    Vq = torch.randn(H, R, Dh, device=dev, dtype=dtype).contiguous()
    Vk = torch.randn(Hk, R, Dh, device=dev, dtype=dtype).contiguous()
    Vv = torch.randn(Hk, R, Dh, device=dev, dtype=dtype).contiguous()
    bq = torch.randn(H, Dh, device=dev, dtype=dtype).contiguous()
    bk = torch.randn(Hk, Dh, device=dev, dtype=dtype).contiguous()
    bv = torch.randn(Hk, Dh, device=dev, dtype=dtype).contiguous()

    cos_half, sin_half = fs_mod.build_rope_tables(S, Dh, base=10000.0, device=dev, dtype=dtype)

    # ropeattn layout: [B,H,M,R] + [B,Hk,M,R]
    Pq_bhmr = Pq.permute(0, 2, 1, 3).contiguous()
    Pk_bhkmr = Pk.permute(0, 2, 1, 3).contiguous()
    Pv_bhkmr = Pv.permute(0, 2, 1, 3).contiguous()

    # ropeattn biases are indexed as a flat [H*Dh] vector (hid*Dh + d).
    bq_flat = bq.reshape(-1).contiguous()
    bk_full = bk.repeat_interleave(rep, dim=0).contiguous()
    bv_full = bv.repeat_interleave(rep, dim=0).contiguous()
    bk_flat = bk_full.reshape(-1).contiguous()
    bv_flat = bv_full.reshape(-1).contiguous()

    # ropeattn expects cos/sin in BMd with full Dh (half duplicated).
    cos_full = torch.cat([cos_half, cos_half], dim=-1).contiguous()
    sin_full = torch.cat([sin_half, sin_half], dim=-1).contiguous()
    cos_bmd = cos_full[None, :, :].expand(B, -1, -1).contiguous()
    sin_bmd = sin_full[None, :, :].expand(B, -1, -1).contiguous()

    return Inputs(
        Pq_bshr=Pq,
        Pk_bskr=Pk,
        Pv_bskr=Pv,
        Vq_hrd=Vq,
        Vk_hrd=Vk,
        Vv_hrd=Vv,
        bq_hd=bq,
        bk_hkd=bk,
        bv_hkd=bv,
        cos_half=cos_half,
        sin_half=sin_half,
        Pq_bhmr=Pq_bhmr,
        Pk_bhkmr=Pk_bhkmr,
        Pv_bhkmr=Pv_bhkmr,
        bq_flat=bq_flat,
        bk_flat=bk_flat,
        bv_flat=bv_flat,
        cos_bmd=cos_bmd,
        sin_bmd=sin_bmd,
    )


def _make_inputs_with_projection(
    *,
    fs_mod,
    B: int,
    S: int,
    H: int,
    Hk: int,
    Dh: int,
    R: int,
    D: Optional[int],
    dtype,
    seed: int,
    ropeattn_contig: bool,
) -> tuple[Inputs, ProjBuffers]:
    import torch

    assert H % Hk == 0
    rep = H // Hk
    torch.manual_seed(seed)
    dev = torch.device("cuda")

    if D is None:
        D = H * Dh
    if D <= 0:
        raise ValueError(f"D must be positive, got {D}")

    # X: [B*S, D] (2D GEMM layout)
    X2d = torch.randn(B * S, D, device=dev, dtype=dtype).contiguous()
    Uq = torch.randn(D, H * R, device=dev, dtype=dtype).contiguous()
    Uk = torch.randn(D, Hk * R, device=dev, dtype=dtype).contiguous()
    Uv = torch.randn(D, Hk * R, device=dev, dtype=dtype).contiguous()

    # Pre-allocated P buffers (overwritten each iteration via torch.mm(..., out=...))
    Pq2d = torch.empty(B * S, H * R, device=dev, dtype=dtype)
    Pk2d = torch.empty(B * S, Hk * R, device=dev, dtype=dtype)
    Pv2d = torch.empty(B * S, Hk * R, device=dev, dtype=dtype)

    Pq = Pq2d.view(B, S, H, R)
    Pk = Pk2d.view(B, S, Hk, R)
    Pv = Pv2d.view(B, S, Hk, R)

    Vq = torch.randn(H, R, Dh, device=dev, dtype=dtype).contiguous()
    Vk = torch.randn(Hk, R, Dh, device=dev, dtype=dtype).contiguous()
    Vv = torch.randn(Hk, R, Dh, device=dev, dtype=dtype).contiguous()
    bq = torch.randn(H, Dh, device=dev, dtype=dtype).contiguous()
    bk = torch.randn(Hk, Dh, device=dev, dtype=dtype).contiguous()
    bv = torch.randn(Hk, Dh, device=dev, dtype=dtype).contiguous()

    cos_half, sin_half = fs_mod.build_rope_tables(S, Dh, base=10000.0, device=dev, dtype=dtype)

    # ropeattn layout: [B,H,M,R] + [B,Hk,M,R]
    Pq_bhmr = Pq.permute(0, 2, 1, 3)
    Pk_bhkmr = Pk.permute(0, 2, 1, 3)
    Pv_bhkmr = Pv.permute(0, 2, 1, 3)
    if ropeattn_contig:
        Pq_bhmr = Pq_bhmr.contiguous()
        Pk_bhkmr = Pk_bhkmr.contiguous()
        Pv_bhkmr = Pv_bhkmr.contiguous()

    # ropeattn biases are indexed as a flat [H*Dh] vector (hid*Dh + d).
    bq_flat = bq.reshape(-1).contiguous()
    bk_full = bk.repeat_interleave(rep, dim=0).contiguous()
    bv_full = bv.repeat_interleave(rep, dim=0).contiguous()
    bk_flat = bk_full.reshape(-1).contiguous()
    bv_flat = bv_full.reshape(-1).contiguous()

    # ropeattn expects cos/sin in BMd with full Dh (half duplicated).
    cos_full = torch.cat([cos_half, cos_half], dim=-1).contiguous()
    sin_full = torch.cat([sin_half, sin_half], dim=-1).contiguous()
    cos_bmd = cos_full[None, :, :].expand(B, -1, -1).contiguous()
    sin_bmd = sin_full[None, :, :].expand(B, -1, -1).contiguous()

    inp = Inputs(
        Pq_bshr=Pq,
        Pk_bskr=Pk,
        Pv_bskr=Pv,
        Vq_hrd=Vq,
        Vk_hrd=Vk,
        Vv_hrd=Vv,
        bq_hd=bq,
        bk_hkd=bk,
        bv_hkd=bv,
        cos_half=cos_half,
        sin_half=sin_half,
        Pq_bhmr=Pq_bhmr,
        Pk_bhkmr=Pk_bhkmr,
        Pv_bhkmr=Pv_bhkmr,
        bq_flat=bq_flat,
        bk_flat=bk_flat,
        bv_flat=bv_flat,
        cos_bmd=cos_bmd,
        sin_bmd=sin_bmd,
    )
    proj = ProjBuffers(X2d=X2d, Uq=Uq, Uk=Uk, Uv=Uv, Pq2d=Pq2d, Pk2d=Pk2d, Pv2d=Pv2d, D=D)
    return inp, proj


def main() -> int:
    ap = argparse.ArgumentParser("Compare FlashSVD kernels vs aligned baseline")
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
    ap.add_argument("--causal", action="store_true", default=True)
    ap.add_argument("--no-causal", dest="causal", action="store_false")
    ap.add_argument("--baseline-attn", choices=["triton", "fa2", "auto"], default="triton")
    ap.add_argument(
        "--include-proj",
        action="store_true",
        help="include projection cost by computing Pq/Pk/Pv = (X @ U*) each iteration (still compares low-rank pipelines)",
    )
    ap.add_argument("--D", type=int, default=None, help="model hidden dim for --include-proj (default: H*Dh)")
    ap.add_argument("--skip-ropeattn", action="store_true", help="skip flashsvdropeattn_v1 variant")

    # flashsvdropeattn_v1.5.py tiling
    ap.add_argument("--fs-bm", type=int, default=64)
    ap.add_argument("--fs-bn", type=int, default=64)
    ap.add_argument("--fs-br", type=int, default=64)
    ap.add_argument("--fs-warps", type=int, default=8)
    ap.add_argument("--fs-stages", type=int, default=3)
    ap.add_argument(
        "--fs-value-in-rank",
        action="store_true",
        help="use flashsvdropeattn_v1.5.py value_in_rank (accumulate Pv in rank-space, lift once with Vv)",
    )
    ap.add_argument(
        "--fs-autotune",
        action="store_true",
        help="sweep (bn,br,warps,stages) for flashsvdropeattn_v1.5.py and use the best config",
    )
    ap.add_argument("--fs-at-warmup", type=int, default=10, help="flashsvd autotune warmup iters per config")
    ap.add_argument("--fs-at-iters", type=int, default=50, help="flashsvd autotune bench iters per config")
    ap.add_argument("--fs-at-bn", type=int, nargs="+", default=[32, 64, 128], help="BN candidates")
    ap.add_argument("--fs-at-br", type=int, nargs="*", default=None, help="BR candidates (default derived from R)")
    ap.add_argument("--fs-at-warps", type=int, nargs="+", default=[4, 8], help="num_warps candidates")
    ap.add_argument("--fs-at-stages", type=int, nargs="+", default=[2, 3, 4], help="num_stages candidates")
    ap.add_argument("--fs-at-topk", type=int, default=5, help="print top-k autotune configs")

    # ropeattn tiling
    ap.add_argument("--ra-bm", type=int, default=64)
    ap.add_argument("--ra-bn", type=int, default=64)
    ap.add_argument("--ra-br", type=int, default=64)
    ap.add_argument("--ra-warps", type=int, default=4)
    ap.add_argument("--ra-stages", type=int, default=2)

    ap.add_argument("--check", action="store_true", help="run fp32 reference check (recommend --S <= 256)")
    args = ap.parse_args()

    try:
        import torch
    except Exception as e:
        print(f"[error] torch import failed: {e}")
        return 2

    if not torch.cuda.is_available():
        print("[error] CUDA is required for this comparison.")
        return 2

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    repo_root, kernels_dir, flashsvd_dir = _repo_paths()
    here = os.path.dirname(os.path.abspath(__file__))

    fs_path = _pick_first_existing_path(
        [
            os.path.join(here, "flashsvdropeattn_v1.5.py"),
            os.path.join(flashsvd_dir, "flashsvdropeattn", "flashsvdropeattn_v1.5.py"),
        ],
        what="flashsvdropeattn_v1.5.py",
    )
    fs = _import_from_path("flashsvdropeattn_v1_5_local", fs_path)

    ropeattn_path = _pick_first_existing_path(
        [
            os.path.join(here, "flashsvdropeattn_v1.py"),
            os.path.join(flashsvd_dir, "flashsvdropeattn", "flashsvdropeattn_v1.py"),
        ],
        what="flashsvdropeattn_v1.py",
    )
    ra = _import_from_path("flashsvdropeattn_v1_local", ropeattn_path)

    fa_path = os.path.join(kernels_dir, "flash_attn_causal.py")
    if not os.path.exists(fa_path):
        raise FileNotFoundError(fa_path)
    fa = _import_from_path("flash_attn_causal_local", fa_path)

    baseline_path = _pick_first_existing_path(
        [
            os.path.join(here, "flashsvdropeattn_baseline.py"),
            os.path.join(flashsvd_dir, "flashsvdropeattn", "flashsvdropeattn_baseline.py"),
        ],
        what="flashsvdropeattn_baseline.py",
    )
    bl = _import_from_path("flashsvdropeattn_baseline_local", baseline_path)

    B, S, H, Hk, Dh, R = args.B, args.S, args.H, args.Hk, args.Dh, args.R
    assert H % Hk == 0, "H must be divisible by Hk for GQA."
    rep = H // Hk

    # Cheap safety: avoid half-masked rank tiles when user forgets to change --fs-br/--ra-br for small R.
    if args.fs_br > R:
        print(f"[note] --fs-br ({args.fs_br}) > R ({R}); clamping to {R} to avoid masked rank tiles.")
        args.fs_br = R
    if args.ra_br > R:
        print(f"[note] --ra-br ({args.ra_br}) > R ({R}); clamping to {R} to avoid masked rank tiles.")
        args.ra_br = R

    proj: Optional[ProjBuffers]
    if args.include_proj:
        inp, proj = _make_inputs_with_projection(
            fs_mod=fs,
            B=B,
            S=S,
            H=H,
            Hk=Hk,
            Dh=Dh,
            R=R,
            D=args.D,
            dtype=dtype,
            seed=args.seed,
            # If we include projection in the benchmark, we prefer to avoid extra permute+copy work.
            # Users can still measure ropeattn separately if they care.
            ropeattn_contig=False,
        )
    else:
        inp = _make_inputs(fs_mod=fs, B=B, S=S, H=H, Hk=Hk, Dh=Dh, R=R, dtype=dtype, seed=args.seed)
        proj = None

    def _proj_step():
        if proj is None:
            return
        # Pq/Pk/Pv are overwritten in-place via the out= buffers.
        torch.mm(proj.X2d, proj.Uq, out=proj.Pq2d)
        torch.mm(proj.X2d, proj.Uk, out=proj.Pk2d)
        torch.mm(proj.X2d, proj.Uv, out=proj.Pv2d)

    # ----------------------------
    # Variant 1: baseline (unfused pipeline)
    # ----------------------------
    def _baseline_body():
        return bl.flashsvd_attn_packed_baseline(
            packed,
            inp.cos_half,
            inp.sin_half,
            causal=args.causal,
            window_size=(-1, -1),
            fa_block_m=32,
            attn_backend=args.baseline_attn,
            flash_attn_fn=fa.flash_attn_triton,
        )

    def baseline_unfused():
        _proj_step()
        return _baseline_body()

    # ----------------------------
    # Variant 2: flashsvdropeattn_v1.5.py fused kernel
    # ----------------------------
    packed = fs.PackedFactors(
        Pq=inp.Pq_bshr,
        Pk=inp.Pk_bskr,
        Pv=inp.Pv_bskr,
        Vq=inp.Vq_hrd,
        Vk=inp.Vk_hrd,
        Vv=inp.Vv_hrd,
        bq=inp.bq_hd,
        bk=inp.bk_hkd,
        bv=inp.bv_hkd,
    )

    def _flashsvd_body(*, bn: int, br: int, warps: int, stages: int):
        return fs.flashsvd_attn_packed(
            packed,
            inp.cos_half,
            inp.sin_half,
            causal=args.causal,
            window_size=(-1, -1),
            bm=args.fs_bm,
            bn=bn,
            br=br,
            num_warps=warps,
            num_stages=stages,
            value_in_rank=args.fs_value_in_rank,
        )

    def _flashsvd_call(*, bn: int, br: int, warps: int, stages: int):
        _proj_step()
        return _flashsvd_body(bn=bn, br=br, warps=warps, stages=stages)

    if args.fs_autotune:
        if args.fs_at_br is None:
            br_cands = [v for v in (16, 32, 64, 128) if v <= R]
            if R not in br_cands:
                br_cands.append(R)
        else:
            br_cands = [v for v in args.fs_at_br if v <= R]
            if not br_cands:
                raise ValueError(f"--fs-at-br produced no valid candidates <= R={R}.")
        bn_cands = [v for v in args.fs_at_bn if v > 0]
        warps_cands = [v for v in args.fs_at_warps if v > 0]
        stages_cands = [v for v in args.fs_at_stages if v > 0]

        combos = list(itertools.product(bn_cands, br_cands, warps_cands, stages_cands))
        print(f"[fs-autotune] Sweeping {len(combos)} configs for flashsvdropeattn_v1.5.py (fixed BM={args.fs_bm}, R={R}) ...")
        tuned = []
        for bn, br, warps, stages in combos:
            ms = _bench_ms(
                lambda bn=bn, br=br, warps=warps, stages=stages: _flashsvd_call(bn=bn, br=br, warps=warps, stages=stages),
                warmup=args.fs_at_warmup,
                iters=args.fs_at_iters,
            )
            tuned.append((ms, bn, br, warps, stages))
        tuned.sort(key=lambda x: x[0])

        best_ms, best_bn, best_br, best_w, best_st = tuned[0]
        print("[fs-autotune] Top configs:")
        for i, (ms, bn, br, w, st) in enumerate(tuned[: max(1, args.fs_at_topk)], start=1):
            print(f"  {i:2d}) {ms:.4f} ms  (BN={bn}, BR={br}, warps={w}, stages={st})")
        print(f"[fs-autotune] Selected: BN={best_bn}, BR={best_br}, warps={best_w}, stages={best_st} (ms={best_ms:.4f})")

        args.fs_bn = best_bn
        args.fs_br = best_br
        args.fs_warps = best_w
        args.fs_stages = best_st

    def flashsvd_py():
        return _flashsvd_call(bn=args.fs_bn, br=args.fs_br, warps=args.fs_warps, stages=args.fs_stages)

    # ----------------------------
    # Variant 3: flashsvdropeattn_v1 fused kernel (called directly)
    # ----------------------------
    def _ropeattn_body():
        O = torch.empty((B, S, H, Dh), device="cuda", dtype=dtype)  # BMHd

        sPq_b, sPq_h, sPq_m, sPq_r = inp.Pq_bhmr.stride()
        sPk_b, sPk_h, sPk_m, sPk_r = inp.Pk_bhkmr.stride()
        sPv_b, sPv_h, sPv_m, sPv_r = inp.Pv_bhkmr.stride()
        sVq_h, sVq_r, sVq_dh = inp.Vq_hrd.stride()
        sVk_h, sVk_r, sVk_dh = inp.Vk_hrd.stride()
        sVv_h, sVv_r, sVv_dh = inp.Vv_hrd.stride()
        sbq_hd = inp.bq_flat.stride(0)
        sbk_hd = inp.bk_flat.stride(0)
        sbv_hd = inp.bv_flat.stride(0)
        sCOS_b, sCOS_m, sCOS_dh = inp.cos_bmd.stride()
        sSIN_b, sSIN_m, sSIN_dh = inp.sin_bmd.stride()
        sO_b, sO_m, sO_h, sO_dh = O.stride()

        grid = (B * H, (S + args.ra_bm - 1) // args.ra_bm)

        ra.flashsvd_rope_sdpa[grid](
            inp.Pq_bhmr, inp.Pk_bhkmr, inp.Pv_bhkmr,
            inp.Vq_hrd, inp.Vk_hrd, inp.Vv_hrd,
            inp.bq_flat, inp.bk_flat, inp.bv_flat,
            inp.cos_bmd, inp.sin_bmd,
            O,
            O,  # pad_mask_ptr (unused)
            O,  # add_mask_ptr (unused)
            B, H, Hk, S, R, Dh,
            sPq_b, sPq_h, sPq_m, sPq_r,
            sPk_b, sPk_h, sPk_m, sPk_r,
            sPv_b, sPv_h, sPv_m, sPv_r,
            sVq_h, sVq_r, sVq_dh,
            sVk_h, sVk_r, sVk_dh,
            sVv_h, sVv_r, sVv_dh,
            sbq_hd, sbk_hd, sbv_hd,
            sCOS_b, sCOS_m, sCOS_dh,
            sSIN_b, sSIN_m, sSIN_dh,
            sO_b, sO_m, sO_h, sO_dh,
            0, 0,  # sPM_b, sPM_m
            0, 0, 0,  # sAM_b, sAM_mq, sAM_mk
            BM=args.ra_bm, BN=args.ra_bn, BDH=Dh, BR=args.ra_br,
            HAS_PAD=0, HAS_ADD=0, CAUSAL=int(args.causal),
            num_warps=args.ra_warps, num_stages=args.ra_stages,
        )
        return O

    def ropeattn_v15():
        _proj_step()
        return _ropeattn_body()

    # ----------------------------
    # Run benchmarks
    # ----------------------------
    tokens = B * S
    title = "==== FlashSVD comparison (aligned, end-to-end) ===="
    if args.include_proj:
        title = "==== FlashSVD comparison (aligned, incl. projection X@U) ===="
    print(title)
    if args.include_proj:
        print(
            f"Shape: B={B}, S={S}, H={H}, Hk={Hk} (rep={rep}), Dh={Dh}, R={R}, D={proj.D if proj is not None else args.D}, "
            f"dtype={args.dtype}, causal={args.causal}"
        )
    else:
        print(f"Shape: B={B}, S={S}, H={H}, Hk={Hk} (rep={rep}), Dh={Dh}, R={R}, dtype={args.dtype}, causal={args.causal}")
    print(
        f"Config: baseline_attn={args.baseline_attn} | "
        f"flashsvd(BM={args.fs_bm},BN={args.fs_bn},BR={args.fs_br},warps={args.fs_warps},stages={args.fs_stages},value_in_rank={args.fs_value_in_rank}) | "
        f"ropeattn(BM={args.ra_bm},BN={args.ra_bn},BR={args.ra_br},warps={args.ra_warps},stages={args.ra_stages})"
    )

    variants: list[tuple[str, Callable[[], object]]] = []
    if args.include_proj:
        variants.append(("proj_only(X@U -> Pq/Pk/Pv)", lambda: (_proj_step() or 0)))
    variants.append((f"baseline_unfused(recon+RoPE+{args.baseline_attn})", baseline_unfused))
    variants.append(("flashsvdropeattn_v1.5(fused packed)", flashsvd_py))
    if not args.skip_ropeattn:
        variants.append(("flashsvdropeattn_v1(fused BMHd)", ropeattn_v15))

    results = []
    for name, fn in variants:
        ms = _bench_ms(fn, warmup=args.warmup, iters=args.iters)
        tok_s = tokens / (ms / 1e3)
        alloc, res = _isolated_peak_bytes(fn)
        results.append((name, ms, tok_s, alloc, res))

    best_ms = min(r[1] for r in results)
    for name, ms, tok_s, alloc, res in results:
        rel = ms / best_ms
        print(f"- {name}: {ms:.4f} ms | {tok_s:,.0f} tok/s | x{rel:.2f} vs best | peak_alloc={_pretty_bytes(alloc)} peak_res={_pretty_bytes(res)}")

    # ----------------------------
    # Optional correctness check (fp32 reference; small S only)
    # ----------------------------
    if args.check:
        if S > 256:
            print("[check] Skipped: S too large for fp32 reference; use --S <= 256.")
        else:
            if proj is not None:
                # Ensure a consistent Pq/Pk/Pv snapshot for all check variants.
                _proj_step()
            with torch.no_grad():
                O_ref = fs.reference_packed_fp32(
                    packed,
                    inp.cos_half,
                    inp.sin_half,
                    causal=args.causal,
                    window_left=-1,
                    window_right=-1,
                ).to(torch.float32)
                O_fs = _flashsvd_body(bn=args.fs_bn, br=args.fs_br, warps=args.fs_warps, stages=args.fs_stages).to(torch.float32)
                O_bl = _baseline_body().to(torch.float32)

                def _err(x):
                    diff = x - O_ref
                    rel = (torch.linalg.norm(diff) / (torch.linalg.norm(O_ref) + 1e-12)).item()
                    max_abs = diff.abs().max().item()
                    finite = torch.isfinite(x).all().item()
                    return finite, max_abs, rel

                to_report = [("flashsvdropeattn_v1.5", O_fs), ("baseline_unfused", O_bl)]
                if not args.skip_ropeattn:
                    O_ra = _ropeattn_body().to(torch.float32)
                    to_report.insert(1, ("flashsvdropeattn_v1", O_ra))
                for tag, out in to_report:
                    finite, max_abs, rel = _err(out)
                    print(f"[check] {tag}: finite={finite} max_abs={max_abs:.3e} rel_fro={rel:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
