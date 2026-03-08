#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decode kernel sweep benchmark for FlashSVD low-rank KV-cache decode (Q_len=1).

This script compares *current* decode implementations in this folder:
  - flashsvdropeattn_v1.5_decode.py
  - flashsvdropeattn_v1.6_decode_opt.py (v1 wrapper + v2 wrapper)

It also includes an explicit Vk-resident ablation for v1.6 v2:
  - v2 (vk_resident=True)
  - v2 (vk_resident=False)

Notes:
  - Preallocates workspace/out/q_buffers (when supported) to avoid Python-side alloc overhead in timing.
  - Requires torch + triton + CUDA.

Example:
  CUDA_VISIBLE_DEVICES=1 python bench_flashsvd_decode_sweep.py \
    --B 8 --Smax 2048 --seqlen_k 2048 --H 32 --Hk 8 --Dh 128 --R 64 \
    --dtype bf16 --split_k 512 --bn 64 --br 64 \
    --warps1 4 --stages1 2 --warps2 4 --stages2 1 \
    --causal --warmup 50 --iters 1000
"""

from __future__ import annotations

import argparse
import gc
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import triton


def load_module(path: str):
    path = str(Path(path).resolve())
    import importlib.util

    spec = importlib.util.spec_from_file_location("flashsvd_mod_sweep", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def do_bench_ms(fn: Callable[[], object], *, warmup: int, rep: int) -> float:
    try:
        return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))
    except Exception:
        torch.cuda.synchronize()
        for _ in range(max(1, warmup)):
            _ = fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max(1, rep)):
            _ = fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / max(1, rep)


def pretty_bytes(n: int) -> str:
    x = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} PB"


def isolated_peak(fn: Callable[..., object], *a: Any, **k: Any) -> tuple[object, int, int]:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn(*a, **k)
    torch.cuda.synchronize()
    return out, int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def _maybe_kwargs(fn: Callable[..., object], kwargs: dict[str, object]) -> dict[str, object]:
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


@dataclass(frozen=True)
class Variant:
    name: str
    mod_name: str
    f: object
    fn: Callable[..., torch.Tensor]
    kwargs: dict[str, object]


def main() -> None:
    ap = argparse.ArgumentParser("FlashSVD decode sweep benchmark (packed)")
    ap.add_argument(
        "--modules",
        type=str,
        default="flashsvdropeattn_v1.5_decode.py,flashsvdropeattn_v1.6_decode_opt.py",
        help="Comma-separated module paths to compare",
    )
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--Smax", type=int, default=2048)
    ap.add_argument("--seqlen_k", type=int, default=2048)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--Hk", type=int, default=8)
    ap.add_argument("--Dh", type=int, default=128)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    ap.add_argument("--split_k", type=int, default=512)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--br", type=int, default=64)
    ap.add_argument("--warps1", type=int, default=4)
    ap.add_argument("--stages1", type=int, default=2)
    ap.add_argument("--warps2", type=int, default=4)
    ap.add_argument("--stages2", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--window_left", type=int, default=-1)
    ap.add_argument("--window_right", type=int, default=-1)
    ap.add_argument("--no_vk_ablation", action="store_true", help="Disable v2 vk_resident on/off ablation")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    B, Smax, seqlen_k = args.B, args.Smax, args.seqlen_k
    H, Hk, Dh, R = args.H, args.Hk, args.Dh, args.R
    assert H % Hk == 0
    assert Dh % 2 == 0
    assert 0 <= seqlen_k <= Smax

    split_k = int(args.split_k)
    bn = int(args.bn)
    br = int(args.br)
    assert split_k % bn == 0, "split_k must be a multiple of bn"

    torch.manual_seed(0)

    # RoPE tables: [Smax, Dh/2]
    half = Dh // 2
    pos = torch.arange(Smax, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = torch.einsum("m,d->md", pos, inv_freq)
    cos = torch.cos(ang).to(dtype).contiguous()
    sin = torch.sin(ang).to(dtype).contiguous()

    # Shared tensors across modules
    Pq_q = torch.randn(B, H, R, device=device, dtype=dtype).contiguous()
    Pk = torch.randn(B, Smax, Hk, R, device=device, dtype=dtype).contiguous()
    Pv = torch.randn(B, Smax, Hk, R, device=device, dtype=dtype).contiguous()
    Vq = torch.randn(H, R, Dh, device=device, dtype=dtype).contiguous()
    Vk = torch.randn(Hk, R, Dh, device=device, dtype=dtype).contiguous()
    Vv = torch.randn(Hk, R, Dh, device=device, dtype=dtype).contiguous()
    bq = torch.randn(H, Dh, device=device, dtype=dtype).contiguous()
    bk = torch.randn(Hk, Dh, device=device, dtype=dtype).contiguous()
    bv = torch.randn(Hk, Dh, device=device, dtype=dtype).contiguous()

    common_call = dict(
        seqlen_k=int(seqlen_k),
        causal=bool(args.causal),
        window_size=(int(args.window_left), int(args.window_right)),
        split_k=int(split_k),
        bn=int(bn),
        br=int(br),
        num_warps_stage1=int(args.warps1),
        num_stages_stage1=int(args.stages1),
        num_warps_stage2=int(args.warps2),
        num_stages_stage2=int(args.stages2),
    )

    module_paths = [p.strip() for p in args.modules.split(",") if p.strip()]
    if not module_paths:
        raise ValueError("Empty --modules")

    variants: list[Variant] = []

    for mod_path in module_paths:
        mod = load_module(mod_path)
        mod_name = Path(mod_path).name

        f = mod.DecodePackedFactors(Pq=Pq_q, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv, bq=bq, bk=bk, bv=bv)

        # Preallocate output
        O = torch.empty((B, H, Dh), device=device, dtype=dtype)

        # Workspace prealloc: try to match each wrapper's split logic.
        # v1 wrappers typically use num_splits = cdiv(Smax, split_k); v2 uses cdiv(seqlen_k, split_k).
        num_splits_v1 = triton.cdiv(Smax, split_k)
        M_v1 = torch.empty((B, H, num_splits_v1), device=device, dtype=torch.float32)
        L_v1 = torch.empty((B, H, num_splits_v1), device=device, dtype=torch.float32)
        Acc_v1 = torch.empty((B, H, num_splits_v1, R), device=device, dtype=torch.float32)
        ws_v1 = (M_v1, L_v1, Acc_v1)

        num_splits_v2 = max(1, triton.cdiv(seqlen_k if seqlen_k > 0 else 1, split_k))
        M_v2 = torch.empty((B, H, num_splits_v2), device=device, dtype=torch.float32)
        L_v2 = torch.empty((B, H, num_splits_v2), device=device, dtype=torch.float32)
        Acc_v2 = torch.empty((B, H, num_splits_v2, R), device=device, dtype=torch.float32)
        ws_v2 = (M_v2, L_v2, Acc_v2)
        Q0 = torch.empty((B, H, half), device=device, dtype=dtype)
        Q1 = torch.empty((B, H, half), device=device, dtype=dtype)
        qbuf = (Q0, Q1)

        if hasattr(mod, "flashsvd_attn_decode_packed_v1"):
            fn_v1 = mod.flashsvd_attn_decode_packed_v1
            kwargs = dict(common_call)
            kwargs.update(dict(workspace=ws_v1, out=O))
            variants.append(
                Variant(
                    name="v1",
                    mod_name=mod_name,
                    f=f,
                    fn=fn_v1,
                    kwargs=_maybe_kwargs(fn_v1, kwargs),
                )
            )

        if hasattr(mod, "flashsvd_attn_decode_packed"):
            fn = mod.flashsvd_attn_decode_packed

            # If this module supports v2 knobs, add explicit vk_resident ablations.
            params = inspect.signature(fn).parameters
            if ("vk_resident" in params) and (not args.no_vk_ablation):
                v2_common = dict(common_call)
                v2_common.update(
                    dict(
                        workspace=ws_v2,
                        out=O,
                        q_buffers=qbuf,
                        precompute_q=True,
                        pad_to_16=True,
                        writethrough=True,
                    )
                )

                kwargs_vk1 = dict(v2_common)
                kwargs_vk1["vk_resident"] = True
                variants.append(
                    Variant(
                        name="v2(vk_resident=1)",
                        mod_name=mod_name,
                        f=f,
                        fn=fn,
                        kwargs=_maybe_kwargs(fn, kwargs_vk1),
                    )
                )

                kwargs_vk0 = dict(v2_common)
                kwargs_vk0["vk_resident"] = False
                variants.append(
                    Variant(
                        name="v2(vk_resident=0)",
                        mod_name=mod_name,
                        f=f,
                        fn=fn,
                        kwargs=_maybe_kwargs(fn, kwargs_vk0),
                    )
                )
            else:
                # Default decode path for modules without v2 knobs.
                base_kwargs = dict(common_call)
                base_kwargs.update(dict(workspace=ws_v1, out=O))
                variants.append(
                    Variant(
                        name="default",
                        mod_name=mod_name,
                        f=f,
                        fn=fn,
                        kwargs=_maybe_kwargs(fn, base_kwargs),
                    )
                )

    # De-dup exact (mod, name) in case a module exposes both v1 and default pointing to same function.
    seen: set[tuple[str, str]] = set()
    uniq: list[Variant] = []
    for v in variants:
        key = (v.mod_name, v.name)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(v)
    variants = uniq

    # Warmup all variants once to compile kernels.
    for v in variants:
        _ = v.fn(v.f, cos, sin, **v.kwargs)  # type: ignore[arg-type]
    torch.cuda.synchronize()

    print("==== FlashSVD decode sweep (packed) ====")
    print(f"Shape: B={B}, Smax={Smax}, seqlen_k={seqlen_k}, H={H}, Hk={Hk}, Dh={Dh}, R={R}, dtype={dtype}")
    print(f"Mask: causal={bool(args.causal)}, window=({int(args.window_left)},{int(args.window_right)})")
    print(f"split_k={split_k}, bn={bn}, br={br} | warps1={args.warps1}, stages1={args.stages1}, warps2={args.warps2}, stages2={args.stages2}")

    results: list[tuple[float, Variant, int, int]] = []

    for v in variants:
        # Peak memory for this variant (isolated)
        _, peak_alloc, peak_res = isolated_peak(v.fn, v.f, cos, sin, **v.kwargs)  # type: ignore[arg-type]
        ms = do_bench_ms(
            lambda: v.fn(v.f, cos, sin, **v.kwargs),  # type: ignore[arg-type]
            warmup=max(10, args.warmup // 2),
            rep=max(1, args.iters),
        )
        results.append((ms, v, peak_alloc, peak_res))

    results.sort(key=lambda x: x[0])

    best_ms = results[0][0] if results else float("nan")
    for ms, v, peak_alloc, peak_res in results:
        tok_s = B / (ms / 1e3)
        rel = (ms / best_ms) if best_ms > 0 else float("nan")
        print(
            f"[{v.mod_name} :: {v.name}] {ms:.4f} ms | tok/s {tok_s:,.0f} | vs_best {rel:.2f}x | "
            f"peak alloc {pretty_bytes(peak_alloc)} | peak res {pretty_bytes(peak_res)}"
        )

    if results:
        ms, v, _, _ = results[0]
        print(f"Best: {v.mod_name} :: {v.name}  ({ms:.4f} ms)")


if __name__ == "__main__":
    main()
