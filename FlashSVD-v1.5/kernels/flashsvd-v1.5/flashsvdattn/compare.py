#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare dense baseline vs FlashSVD attention (mask-friendly, BHMR layout).

Variants:
  1) baseline_dense: nn.MultiheadAttention (dense)
  2) flashsvdattn_v1.5: flashsvdattn_v1.5.py (optimized low-rank kernel)
  3) flashsvdattn_v1: flashsvdattn_v1.py (baseline low-rank kernel)

This script is meant to be run on a CUDA machine with torch+triton installed.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
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
    flashsvd_dir = os.path.abspath(os.path.join(here, ".."))
    kernels_dir = os.path.abspath(os.path.join(flashsvd_dir, ".."))
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare flashsvdattn variants")
    parser.add_argument("--B", type=int, default=8, help="batch size")
    parser.add_argument("--M", type=int, default=512, help="sequence length")
    parser.add_argument("--d-model", type=int, default=768, help="model dim")
    parser.add_argument("--H", type=int, default=12, help="num heads")
    parser.add_argument("--d-ff", type=int, default=3072, help="FFN dim")
    parser.add_argument("--R", type=int, default=64, help="low rank")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="correctness check (dense vs flashsvd)")
    parser.add_argument("--skip-v1", action="store_true", help="skip flashsvdattn_v1 variant")
    parser.add_argument("--skip-v15", action="store_true", help="skip flashsvdattn_v1.5 variant")
    parser.add_argument("--with-fa2", action="store_true", help="also benchmark dense baseline using FlashAttention-2 (flash_attn)")
    parser.add_argument("--svd-block-m", type=int, default=64, help="FlashSVD v1.5 query tile (BLOCK_M)")
    parser.add_argument("--svd-block-n", type=int, default=64, help="FlashSVD v1.5 key tile (BLOCK_N)")
    parser.add_argument("--svd-warps", type=int, default=4, help="FlashSVD v1.5 num_warps")
    parser.add_argument("--svd-stages", type=int, default=2, help="FlashSVD v1.5 num_stages")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    B, M = args.B, args.M
    d_model, H, d_ff, R = args.d_model, args.H, args.d_ff, args.R

    repo_root, kernels_dir, flashsvd_dir = _repo_paths()
    here = os.path.dirname(os.path.abspath(__file__))

    # Ensure kernels can be imported (add repo to path)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Load flashsvdattn_v1.5 (optimized)
    v15_path = _pick_first_existing_path(
        [
            os.path.join(here, "flashsvdattn_v1.5.py"),
            os.path.join(flashsvd_dir, "flashsvdattn", "flashsvdattn_v1.5.py"),
        ],
        what="flashsvdattn_v1.5.py",
    )
    mod_v15 = _import_from_path("flashsvdattn_v1_5_local", v15_path)

    # Load flashsvdattn_v1 (baseline)
    v1_path = _pick_first_existing_path(
        [
            os.path.join(here, "flashsvdattn_v1.py"),
            os.path.join(flashsvd_dir, "flashsvdattn", "flashsvdattn_v1.py"),
        ],
        what="flashsvdattn_v1.py",
    )
    mod_v1 = _import_from_path("flashsvdattn_v1_local", v1_path)

    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(B, M, d_model, device="cuda", dtype=dtype)
    mask4 = torch.zeros(B, 1, 1, M, device="cuda", dtype=torch.bool)
    mask4[..., : M] = True  # all valid for simplicity

    # Dense baseline
    dense = mod_v15.BaselineBlock(d_model, H, d_ff).to("cuda").to(dtype)

    # FlashSVD blocks (v1.5 and v1)
    flash_v15 = mod_v15.FlashSVDBlock(
        d_model,
        H,
        R,
        d_ff,
        attn_block_m=args.svd_block_m,
        attn_block_n=args.svd_block_n,
        attn_num_warps=args.svd_warps,
        attn_num_stages=args.svd_stages,
    ).to("cuda").to(dtype)
    flash_v1 = mod_v1.FlashSVDBlock(d_model, H, R, d_ff).to("cuda").to(dtype)
    mod_v15.transplant_weights(dense, flash_v15)
    mod_v1.transplant_weights(dense, flash_v1)

    tokens = B * M
    print("==== FlashSVD attention comparison (mask-friendly, BHMR) ====")
    print(f"Shape: B={B}, M={M}, d_model={d_model}, H={H}, d_ff={d_ff}, R={R}, dtype={args.dtype}")
    print(
        f"Config: warmup={args.warmup}, iters={args.iters} | svd_v1.5(BM={args.svd_block_m} BN={args.svd_block_n} warps={args.svd_warps} stages={args.svd_stages})"
    )

    variants: list[tuple[str, Callable[[], object]]] = []

    def baseline_fn():
        with torch.no_grad():
            return dense(x, mask4)

    fa2 = None
    fa2_kwargs: dict[str, object] = {}
    if args.with_fa2:
        try:
            from flash_attn import flash_attn_func as _f  # type: ignore

            fa2 = _f
        except Exception:
            try:
                from flash_attn.flash_attn_interface import flash_attn_func as _f  # type: ignore

                fa2 = _f
            except Exception:
                fa2 = None
        if fa2 is None:
            raise RuntimeError("FlashAttention-2 not found (pip install flash-attn).")
        if (~mask4[:, 0, 0]).any().item():
            raise RuntimeError("--with-fa2 currently assumes all tokens are valid (no padding).")
        params = inspect.signature(fa2).parameters
        if "dropout_p" in params:
            fa2_kwargs["dropout_p"] = 0.0
        if "causal" in params:
            fa2_kwargs["causal"] = False

    def baseline_fa2_fn():
        assert fa2 is not None

        # Re-implement nn.MultiheadAttention forward with FA2 attention.
        dh = d_model // H
        qkv = F.linear(x, dense.mha.in_proj_weight, dense.mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, M, H, dh).contiguous()
        k = k.view(B, M, H, dh).contiguous()
        v = v.view(B, M, H, dh).contiguous()
        out = fa2(q, k, v, **fa2_kwargs)  # [B,M,H,dh]
        out = out.reshape(B, M, d_model)
        out = F.linear(out, dense.mha.out_proj.weight, dense.mha.out_proj.bias)

        y = dense.ln1(x + out)
        return dense.ln2(y + dense.ffn(y))

    def flash_v15_fn():
        with torch.no_grad():
            return flash_v15(x, mask4)

    def flash_v1_fn():
        with torch.no_grad():
            return flash_v1(x, mask4)

    variants.append(("baseline_dense(MHA-SDPA)", baseline_fn))
    if args.with_fa2:
        variants.append(("baseline_dense(FA2)", baseline_fa2_fn))
    if not args.skip_v15:
        variants.append(("flashsvdattn_v1.5(low-rank)", flash_v15_fn))
    if not args.skip_v1:
        variants.append(("flashsvdattn_v1(low-rank)", flash_v1_fn))

    results = []
    for name, fn in variants:
        ms = _bench_ms(fn, warmup=args.warmup, iters=args.iters)
        tok_s = tokens / (ms / 1e3)
        alloc, res = _isolated_peak_bytes(fn)
        results.append((name, ms, tok_s, alloc, res))

    best_ms = min(r[1] for r in results)
    for name, ms, tok_s, alloc, res in results:
        rel = ms / best_ms
        print(f"- {name}: {ms:.4f} ms | {tok_s:,.0f} tok/s | x{rel:.2f} vs best | peak_alloc={_pretty_bytes(alloc)}")

    if args.check:
        print("\n[check] Correctness (dense vs FlashSVD):")
        with torch.no_grad():
            yd = baseline_fn().float()
            yf15 = flash_v15_fn().float()
            yf1 = flash_v1_fn().float()
        rel15 = (yf15 - yd).norm() / (yd.norm() + 1e-12)
        rel1 = (yf1 - yd).norm() / (yd.norm() + 1e-12)
        print(f"  flashsvdattn_v1.5 vs dense: rel_err={rel15:.4e}")
        print(f"  flashsvdattn_v1 vs dense:   rel_err={rel1:.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
