#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare dense FFN vs FlashSVD FFN (low-rank) kernels.

Variants:
  1) baseline_dense:   X @ (U1 V1) -> GELU -> @ (U2 V2)
  2) lowrank_torch:    (X @ U1) -> @ V1 -> GELU -> @ U2 -> @ V2
  3) flashsvdffn_v1:   fused kernel (baseline)
  4) flashsvdffn_v1.5: fused kernel (rank-space accumulate + single lift)

Run from repo root with:
  CUDA_VISIBLE_DEVICES=0 python kernels/flashsvd-v1.5/flashsvdffn/compare.py --help
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import math
import os
import sys
import time
from typing import Callable


def _import_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def _isolated_peak_bytes(fn: Callable[[], object]) -> int:
    import torch

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def _pretty_bytes(n: int) -> str:
    x = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} PB"


def main() -> int:
    ap = argparse.ArgumentParser("Compare FlashSVD FFN variants")
    ap.add_argument("--B", type=int, default=16, help="batch size")
    ap.add_argument("--L", type=int, default=2048, help="sequence length")
    ap.add_argument("--H", type=int, default=768, help="hidden dim")
    ap.add_argument("--D", type=int, default=3072, help="FFN intermediate dim")
    ap.add_argument("--R1", type=int, default=64, help="rank for first linear (U1,V1)")
    ap.add_argument("--R2", type=int, default=64, help="rank for second linear (U2,V2)")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--weight-std", type=float, default=0.02, help="init std for equivalent dense weights (approx)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--check", action="store_true", help="correctness check (dense vs low-rank)")
    ap.add_argument("--skip-v1", action="store_true")
    ap.add_argument("--skip-v15", action="store_true")
    ap.add_argument("--skip-lowrank-torch", action="store_true")
    ap.add_argument("--BL", type=int, default=64)
    ap.add_argument("--BD", type=int, default=128)
    ap.add_argument("--BH", type=int, default=64)
    ap.add_argument("--BR1", type=int, default=32)
    ap.add_argument("--BR2", type=int, default=32)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--stages", type=int, default=2)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        print("[error] CUDA is required.")
        return 2

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    B, L, H, D, R1, R2 = args.B, args.L, args.H, args.D, args.R1, args.R2
    dev = torch.device("cuda")

    here = os.path.dirname(os.path.abspath(__file__))
    v15_path = os.path.join(here, "flashsvdffn_v1.5.py")
    v1_path = os.path.join(here, "flashsvdffn_v1.py")
    mod_v15 = _import_from_path("flashsvdffn_v1_5_local", v15_path)
    mod_v1 = _import_from_path("flashsvdffn_v1_local", v1_path)

    torch.manual_seed(0)
    X = torch.randn((B, L, H), device=dev, dtype=dtype)

    # Low-rank factors
    # Choose factor std so that W = U@V has roughly args.weight_std.
    # If U,V ~ N(0, s^2), then each entry of W has std ~= sqrt(R) * s^2.
    s1 = math.sqrt(args.weight_std / math.sqrt(max(1, R1)))
    s2 = math.sqrt(args.weight_std / math.sqrt(max(1, R2)))
    U1 = (torch.randn((H, R1), device=dev, dtype=torch.float32) * s1).to(dtype)
    V1 = (torch.randn((R1, D), device=dev, dtype=torch.float32) * s1).to(dtype)
    U2 = (torch.randn((D, R2), device=dev, dtype=torch.float32) * s2).to(dtype)
    V2 = (torch.randn((R2, H), device=dev, dtype=torch.float32) * s2).to(dtype)
    b1 = (torch.randn((D,), device=dev, dtype=torch.float32) * args.weight_std).to(dtype)
    b2 = (torch.randn((H,), device=dev, dtype=torch.float32) * args.weight_std).to(dtype)

    # Dense weights for fair baseline
    W1 = (U1 @ V1).contiguous()
    W2 = (U2 @ V2).contiguous()

    tokens = B * L
    print("==== FlashSVD FFN comparison ====")
    print(f"Shape: B={B}, L={L}, H={H}, D={D}, R1={R1}, R2={R2}, dtype={args.dtype}")
    print(f"Config: warmup={args.warmup}, iters={args.iters} | tiles(BL={args.BL} BD={args.BD} BH={args.BH} BR1={args.BR1} BR2={args.BR2}) warps={args.warps} stages={args.stages}")

    def dense_fn():
        with torch.no_grad():
            Z = X.matmul(W1) + b1.view(1, 1, -1)
            H1 = F.gelu(Z)
            Y = H1.matmul(W2) + b2.view(1, 1, -1)
            return Y

    def lowrank_torch_fn():
        with torch.no_grad():
            P = X.matmul(U1)
            Z = P.matmul(V1) + b1.view(1, 1, -1)
            H1 = F.gelu(Z)
            Y = H1.matmul(U2).matmul(V2) + b2.view(1, 1, -1)
            return Y

    def v15_fn():
        with torch.no_grad():
            P = X.matmul(U1)
            return mod_v15.flashsvd_ffn(
                P, V1, U2, V2, b1, b2,
                BL=args.BL, BD=args.BD, BH=args.BH, BR1=args.BR1, BR2=args.BR2,
                num_warps=args.warps, num_stages=args.stages,
            )

    def v1_fn():
        with torch.no_grad():
            P = X.matmul(U1)
            return mod_v1.flashsvd_ffn(P, V1, U2, V2, b1, b2, args.BL, args.BD, args.BH, args.BR1, args.BR2)

    variants: list[tuple[str, Callable[[], object]]] = [("baseline_dense", dense_fn)]
    if not args.skip_lowrank_torch:
        variants.append(("lowrank_torch", lowrank_torch_fn))
    if not args.skip_v15:
        variants.append(("flashsvdffn_v1.5(fused)", v15_fn))
    if not args.skip_v1:
        variants.append(("flashsvdffn_v1(fused)", v1_fn))

    results = []
    for name, fn in variants:
        ms = _bench_ms(fn, warmup=args.warmup, iters=args.iters)
        tok_s = tokens / (ms / 1e3)
        peak = _isolated_peak_bytes(fn)
        results.append((name, ms, tok_s, peak))

    best_ms = min(r[1] for r in results)
    for name, ms, tok_s, peak in results:
        rel = ms / best_ms
        print(f"- {name}: {ms:.4f} ms | {tok_s:,.0f} tok/s | x{rel:.2f} vs best | peak_alloc={_pretty_bytes(peak)}")

    if args.check:
        print("\n[check] Correctness:")
        with torch.no_grad():
            yd = dense_fn().float()
            yl = lowrank_torch_fn().float()
        rel_l = (yl - yd).norm() / (yd.norm() + 1e-12)
        print(f"  lowrank_torch vs dense: rel_err={rel_l:.4e}")
        if not args.skip_v15:
            with torch.no_grad():
                y15 = v15_fn().float()
            rel_15 = (y15 - yd).norm() / (yd.norm() + 1e-12)
            print(f"  flashsvdffn_v1.5 vs dense: rel_err={rel_15:.4e}")
        if not args.skip_v1:
            with torch.no_grad():
                y1 = v1_fn().float()
            rel_1 = (y1 - yd).norm() / (yd.norm() + 1e-12)
            print(f"  flashsvdffn_v1 vs dense:   rel_err={rel_1:.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
