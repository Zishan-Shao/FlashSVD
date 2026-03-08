#!/usr/bin/env python3
"""Kernel-level microbenchmark for FlashSVD encoder kernels.

This script benchmarks kernel primitives directly (not end-to-end encoder wiring):
- GEGLU low-rank kernel (two-stage / fused) vs PyTorch baseline
- RoPE attention Triton kernel (flashsvdropeattn.py) vs reference path
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
import statistics as stats
import sys
from pathlib import Path
from typing import Dict

import torch


def _load_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dtype_from_str(x: str):
    x = x.lower()
    if x in {"fp16", "float16", "half"}:
        return torch.float16
    if x in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if x in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {x}")


def _mib(x: int) -> float:
    return x / (1024 ** 2)


def _parse_int_csv(x: str):
    vals = []
    for tok in x.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    if not vals:
        raise ValueError(f"Empty integer list: {x}")
    return vals


def _bench(fn, warmup: int, iters: int):
    for _ in range(max(1, warmup)):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(max(1, iters)):
        start.record()
        y = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        _ = y.reshape(-1)[0].item()

    mean_ms = float(sum(times) / len(times))
    med_ms = float(stats.median(times))
    p95_ms = float(sorted(times)[max(0, int(len(times) * 0.95) - 1)])
    return {"mean_ms": mean_ms, "median_ms": med_ms, "p95_ms": p95_ms}


def _pick_ranks(hidden: int, inter: int, ratio: float, round_multiple: int) -> Dict[str, int]:
    def rd(x: float, m: int, minv: int = 8) -> int:
        if m <= 1:
            return max(int(x), minv)
        return max((int(x) // m) * m, minv)

    # D*D -> 2*D*R
    r_attn = rd(ratio * hidden / 2.0, round_multiple)
    # D*(2F) -> R1*(D+2F)
    r1 = rd(ratio * (hidden * (2 * inter)) / (hidden + 2 * inter), round_multiple)
    # F*D -> R2*(F+D)
    r2 = rd(ratio * (inter * hidden) / (inter + hidden), round_multiple)
    return {"r_attn": r_attn, "r1": r1, "r2": r2}


def _make_geglu_runner(geglu, target: str, tensors: Dict[str, torch.Tensor], cfg: Dict[str, int]):
    P = tensors["P"]
    V1 = tensors["V1"]
    U2 = tensors["U2"]
    V2 = tensors["V2"]
    G = tensors["G"]
    b1 = tensors["b1"]
    b2 = tensors["b2"]
    if target == "two_stage":
        return lambda: geglu.flashsvd_ffn_geglu_two_stage(
            P, V1, U2, V2, b1, b2,
            BL=cfg["BL"], BD=cfg["BD"], BR1=cfg["BR1"], BR2=cfg["BR2"],
            num_warps=cfg["warps"], num_stages=cfg["stages"],
        )
    if target == "fused":
        return lambda: geglu.flashsvd_ffn_geglu_fused(
            P, V1, U2, V2, b1, b2,
            BL=cfg["BL"], BD=cfg["BD"], BR1=cfg["BR1"], BH=cfg["BH"], BR2=cfg["BR2"],
            num_warps=cfg["warps"], num_stages=cfg["stages"],
        )
    if target == "preg_rebuild":
        return lambda: geglu.flashsvd_ffn_geglu_fused_preg(
            P, V1, geglu.precompute_ffn_g(U2, V2), b1, b2,
            BL=cfg["BL"], BD=cfg["BD"], BR1=cfg["BR1"], BH=cfg["BH"],
            num_warps=cfg["warps"], num_stages=cfg["stages"],
        )
    # default: "preg_cache"
    return lambda: geglu.flashsvd_ffn_geglu_fused_preg(
        P, V1, G, b1, b2,
        BL=cfg["BL"], BD=cfg["BD"], BR1=cfg["BR1"], BH=cfg["BH"],
        num_warps=cfg["warps"], num_stages=cfg["stages"],
    )


def bench_geglu(args, base_dir: Path):
    geglu = _load_module(base_dir / "flashsvdgeglu_extreme.py", "flashsvdgeglu_extreme_mod")

    B, L = args.B, args.L
    H = args.hidden_size
    F = args.intermediate_size
    dtype = _dtype_from_str(args.dtype)

    ranks = _pick_ranks(H, F, args.target_param_ratio, args.rank_round_multiple)
    R1, R2 = ranks["r1"], ranks["r2"]
    g_bl = int(args.geglu_bl)
    g_bd = int(args.geglu_bd)
    g_br1 = int(args.geglu_br1)
    g_br2 = int(args.geglu_br2)
    g_bh = int(args.geglu_bh)
    g_warps = int(args.geglu_warps)
    g_stages = int(args.geglu_stages)

    device = torch.device("cuda")
    X = torch.randn((B, L, H), device=device, dtype=dtype)
    U1 = (torch.randn((H, R1), device=device, dtype=dtype) / math.sqrt(H)).contiguous()
    V1 = (torch.randn((R1, 2 * F), device=device, dtype=dtype) / math.sqrt(max(1, R1))).contiguous()
    U2 = (torch.randn((F, R2), device=device, dtype=dtype) / math.sqrt(F)).contiguous()
    V2 = (torch.randn((R2, H), device=device, dtype=dtype) / math.sqrt(max(1, R2))).contiguous()
    b1 = torch.zeros((2 * F,), device=device, dtype=dtype).contiguous()
    b2 = torch.zeros((H,), device=device, dtype=dtype).contiguous()
    G = geglu.precompute_ffn_g(U2, V2)

    P = X.matmul(U1).contiguous()
    tensors = {"P": P, "V1": V1, "U2": U2, "V2": V2, "G": G, "b1": b1, "b2": b2}

    if args.geglu_autotune:
        tune_bls = _parse_int_csv(args.geglu_autotune_bl)
        tune_bds = _parse_int_csv(args.geglu_autotune_bd)
        tune_br1s = _parse_int_csv(args.geglu_autotune_br1)
        tune_br2s = _parse_int_csv(args.geglu_autotune_br2)
        tune_target = args.geglu_autotune_target
        tune_bhs = _parse_int_csv(args.geglu_autotune_bh) if tune_target != "two_stage" else [g_bh]
        tune_warps = _parse_int_csv(args.geglu_autotune_warps)
        tune_stages = _parse_int_csv(args.geglu_autotune_stages)

        all_cfgs = []
        for bl, bd, br1, br2, bh, w, s in itertools.product(
            tune_bls, tune_bds, tune_br1s, tune_br2s, tune_bhs, tune_warps, tune_stages
        ):
            if bd > F or br1 > R1 or br2 > R2 or bh > H:
                continue
            if bl <= 0 or bd <= 0 or br1 <= 0 or br2 <= 0 or bh <= 0 or w <= 0 or s <= 0:
                continue
            all_cfgs.append(
                {"BL": bl, "BD": bd, "BR1": br1, "BR2": br2, "BH": bh, "warps": w, "stages": s}
            )
        if not all_cfgs:
            raise RuntimeError("No valid GeGLU autotune candidates after shape filtering")
        max_cfg = max(1, int(args.geglu_autotune_max_configs))
        if len(all_cfgs) > max_cfg:
            # Keep coverage across the full cartesian space instead of prefix truncation.
            if max_cfg == 1:
                keep_idx = [0]
            else:
                keep_idx = [int(round(i * (len(all_cfgs) - 1) / (max_cfg - 1))) for i in range(max_cfg)]
            # Dedup in case of rounding collisions.
            seen_idx = set()
            keep_idx = [i for i in keep_idx if not (i in seen_idx or seen_idx.add(i))]
            all_cfgs = [all_cfgs[i] for i in keep_idx]

        # Always include current manual config as one candidate.
        manual_cfg = {"BL": g_bl, "BD": g_bd, "BR1": g_br1, "BR2": g_br2, "BH": g_bh, "warps": g_warps, "stages": g_stages}
        if manual_cfg not in all_cfgs:
            all_cfgs.insert(0, manual_cfg)
            if len(all_cfgs) > max_cfg:
                all_cfgs = all_cfgs[:max_cfg]

        print("---- GEGLU autotune ----")
        print(
            f"target={tune_target}, candidates={len(all_cfgs)}, tune_warmup={args.geglu_tune_warmup}, tune_iters={args.geglu_tune_iters}"
        )
        best_cfg = None
        best_ms = float("inf")
        seen_ok = 0
        for i, cfg in enumerate(all_cfgs, 1):
            fn = _make_geglu_runner(geglu, tune_target, tensors, cfg)
            try:
                t = _bench(fn, args.geglu_tune_warmup, args.geglu_tune_iters)["mean_ms"]
                seen_ok += 1
                if t < best_ms:
                    best_ms = t
                    best_cfg = cfg
                if args.geglu_autotune_verbose:
                    print(
                        f"[autotune {i:03d}/{len(all_cfgs)}] BL={cfg['BL']} BD={cfg['BD']} BR1={cfg['BR1']} BR2={cfg['BR2']} "
                        f"BH={cfg['BH']} w={cfg['warps']} s={cfg['stages']} -> {t:.4f} ms"
                    )
            except Exception as e:
                if args.geglu_autotune_verbose:
                    em = str(e).splitlines()[0]
                    print(
                        f"[autotune {i:03d}/{len(all_cfgs)}] BL={cfg['BL']} BD={cfg['BD']} BR1={cfg['BR1']} BR2={cfg['BR2']} "
                        f"BH={cfg['BH']} w={cfg['warps']} s={cfg['stages']} -> FAIL ({em})"
                    )
                continue
        if best_cfg is None:
            print(
                f"[autotune-fallback] all {len(all_cfgs)} candidates failed; "
                f"use manual tiles BL={g_bl} BD={g_bd} BR1={g_br1} BR2={g_br2} BH={g_bh} warps={g_warps} stages={g_stages}"
            )
        else:
            g_bl = best_cfg["BL"]
            g_bd = best_cfg["BD"]
            g_br1 = best_cfg["BR1"]
            g_br2 = best_cfg["BR2"]
            g_bh = best_cfg["BH"]
            g_warps = best_cfg["warps"]
            g_stages = best_cfg["stages"]
            print(
                f"[autotune-best] BL={g_bl} BD={g_bd} BR1={g_br1} BR2={g_br2} BH={g_bh} warps={g_warps} stages={g_stages} "
                f"| {tune_target}={best_ms:.4f} ms | ok={seen_ok}/{len(all_cfgs)}"
            )
            print(
                f"[autotune-args] --geglu-bl {g_bl} --geglu-bd {g_bd} --geglu-br1 {g_br1} "
                f"--geglu-br2 {g_br2} --geglu-bh {g_bh} --geglu-warps {g_warps} --geglu-stages {g_stages}"
            )
        if args.geglu_autotune_only:
            print(
                f"[autotune-only] selected BL={g_bl} BD={g_bd} BR1={g_br1} BR2={g_br2} BH={g_bh} warps={g_warps} stages={g_stages}"
            )
            return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Compile warmup for fair timing
    _ = geglu.flashsvd_ffn_geglu_two_stage(
        P, V1, U2, V2, b1, b2,
        BL=g_bl, BD=g_bd, BR1=g_br1, BR2=g_br2,
        num_warps=g_warps, num_stages=g_stages,
    )
    _ = geglu.flashsvd_ffn_geglu_fused(
        P, V1, U2, V2, b1, b2,
        BL=g_bl, BD=g_bd, BR1=g_br1, BH=g_bh, BR2=g_br2,
        num_warps=g_warps, num_stages=g_stages,
    )
    _ = geglu.flashsvd_ffn_geglu_fused_preg(
        P, V1, G, b1, b2,
        BL=g_bl, BD=g_bd, BR1=g_br1, BH=g_bh,
        num_warps=g_warps, num_stages=g_stages,
    )
    _ = geglu._pt_baseline(P, V1, U2, V2, b1, b2)
    torch.cuda.synchronize()

    pt = _bench(lambda: geglu._pt_baseline(P, V1, U2, V2, b1, b2), args.warmup, args.iters)
    two = _bench(
        lambda: geglu.flashsvd_ffn_geglu_two_stage(
            P, V1, U2, V2, b1, b2,
            BL=g_bl, BD=g_bd, BR1=g_br1, BR2=g_br2,
            num_warps=g_warps, num_stages=g_stages,
        ),
        args.warmup,
        args.iters,
    )
    fus = _bench(
        lambda: geglu.flashsvd_ffn_geglu_fused(
            P, V1, U2, V2, b1, b2,
            BL=g_bl, BD=g_bd, BR1=g_br1, BH=g_bh, BR2=g_br2,
            num_warps=g_warps, num_stages=g_stages,
        ),
        args.warmup,
        args.iters,
    )
    preg = _bench(
        lambda: geglu.flashsvd_ffn_geglu_fused_preg(
            P, V1, G, b1, b2,
            BL=g_bl, BD=g_bd, BR1=g_br1, BH=g_bh,
            num_warps=g_warps, num_stages=g_stages,
        ),
        args.warmup,
        args.iters,
    )
    preg_with_g = _bench(
        lambda: geglu.flashsvd_ffn_geglu_fused_preg(
            P, V1, geglu.precompute_ffn_g(U2, V2), b1, b2,
            BL=g_bl, BD=g_bd, BR1=g_br1, BH=g_bh,
            num_warps=g_warps, num_stages=g_stages,
        ),
        args.warmup,
        args.iters,
    )

    peak_alloc = _mib(torch.cuda.max_memory_allocated())
    peak_res = _mib(torch.cuda.max_memory_reserved())

    toks = B * L
    print("==== GEGLU Kernel Microbench ====")
    print(f"shape: B={B}, L={L}, H={H}, F={F}, R1={R1}, R2={R2}, dtype={args.dtype}")
    print(f"tiles: BL={g_bl}, BD={g_bd}, BR1={g_br1}, BR2={g_br2}, BH={g_bh}, warps={g_warps}, stages={g_stages}")
    print(f"- torch_baseline: {pt['mean_ms']:.4f} ms | {toks/(pt['mean_ms']/1e3):,.0f} tok/s")
    print(f"- triton_two_stage: {two['mean_ms']:.4f} ms | {toks/(two['mean_ms']/1e3):,.0f} tok/s | speedup x{pt['mean_ms']/two['mean_ms']:.3f}")
    print(f"- triton_fused: {fus['mean_ms']:.4f} ms | {toks/(fus['mean_ms']/1e3):,.0f} tok/s | speedup x{pt['mean_ms']/fus['mean_ms']:.3f}")
    print(f"- triton_fused_preG(cache-G): {preg['mean_ms']:.4f} ms | {toks/(preg['mean_ms']/1e3):,.0f} tok/s | speedup x{pt['mean_ms']/preg['mean_ms']:.3f}")
    print(f"- triton_fused_preG(rebuild-G): {preg_with_g['mean_ms']:.4f} ms | {toks/(preg_with_g['mean_ms']/1e3):,.0f} tok/s | speedup x{pt['mean_ms']/preg_with_g['mean_ms']:.3f}")
    print(f"- peak mem (process scope): alloc={peak_alloc:.1f} MiB reserved={peak_res:.1f} MiB")


def bench_rope(args, base_dir: Path):
    rope = _load_module(base_dir / "flashsvdropeattn.py", "flashsvdropeattn_mod")

    B, M = args.B, args.L
    H = args.num_heads
    dh = args.hidden_size // args.num_heads
    dtype = _dtype_from_str(args.dtype)
    if args.rope_ref_dtype == "match":
        ref_dtype = dtype
        ref_dtype_name = args.dtype
    else:
        ref_dtype = _dtype_from_str(args.rope_ref_dtype)
        ref_dtype_name = args.rope_ref_dtype

    ranks = _pick_ranks(args.hidden_size, args.intermediate_size, args.target_param_ratio, args.rank_round_multiple)
    R = ranks["r_attn"]  # shown for context only

    device = torch.device("cuda")

    # NOTE:
    # flashsvdropeattn.py Triton kernel currently expects:
    #   P*: [B, M, H, dh], V*: [H, dh, dh], b*: [H, dh]
    # so we benchmark the kernel in its native contract (pure kernel microbench).
    Pq = torch.randn(B, M, H, dh, device=device, dtype=dtype).contiguous()
    Pk = torch.randn(B, M, H, dh, device=device, dtype=dtype).contiguous()
    Pv = torch.randn(B, M, H, dh, device=device, dtype=dtype).contiguous()
    Vq = torch.randn(H, dh, dh, device=device, dtype=dtype).contiguous()
    Vk = torch.randn(H, dh, dh, device=device, dtype=dtype).contiguous()
    Vv = torch.randn(H, dh, dh, device=device, dtype=dtype).contiguous()
    bq = torch.zeros(H, dh, device=device, dtype=dtype).contiguous()
    bk = torch.zeros(H, dh, device=device, dtype=dtype).contiguous()
    bv = torch.zeros(H, dh, device=device, dtype=dtype).contiguous()

    attn_mask = None
    if args.pad_fraction > 0:
        valid = max(1, int(M * (1.0 - args.pad_fraction)))
        am = torch.zeros(B, M, device=device, dtype=torch.int32)
        am[:, :valid] = 1
        attn_mask = am

    # Precompute RoPE tables [B,H,M,dh]
    cos_tab, sin_tab = rope._rotary_emb_make(M, dh, base=10000.0, device=device, dtype=dtype)
    cos = cos_tab[None, None, :, :].expand(B, H, M, dh).contiguous()
    sin = sin_tab[None, None, :, :].expand(B, H, M, dh).contiguous()

    O = torch.empty((B, H, M, dh), device=device, dtype=dtype)

    has_pad = 1 if attn_mask is not None else 0
    has_add = 0
    pad_ptr = attn_mask if has_pad else O
    add_ptr = O

    BM = int(args.attn_bm)
    BN = int(args.attn_bn)
    warps = int(args.attn_warps)
    stages = int(args.attn_stages)

    grid = (B * H, (M + BM - 1) // BM)

    def run_kernel():
        rope.flashsvd_rope_sdpa[grid](
            Pq, Pk, Pv,
            Vq, Vk, Vv,
            bq, bk, bv,
            cos, sin,
            O,
            pad_ptr, add_ptr,
            B, H, M, dh,
            Pq.stride(0), Pq.stride(1), Pq.stride(2), Pq.stride(3),
            Pk.stride(0), Pk.stride(1), Pk.stride(2), Pk.stride(3),
            Pv.stride(0), Pv.stride(1), Pv.stride(2), Pv.stride(3),
            Vq.stride(0), Vq.stride(1), Vq.stride(2),
            Vk.stride(0), Vk.stride(1), Vk.stride(2),
            Vv.stride(0), Vv.stride(1), Vv.stride(2),
            bq.stride(0), bq.stride(1),
            bk.stride(0), bk.stride(1),
            bv.stride(0), bv.stride(1),
            cos.stride(0), cos.stride(1), cos.stride(2), cos.stride(3),
            sin.stride(0), sin.stride(1), sin.stride(2), sin.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            pad_ptr.stride(0) if has_pad else 0, pad_ptr.stride(1) if has_pad else 0,
            0, 0, 0,
            BM=BM, BN=BN, BDH=dh,
            HAS_PAD=has_pad, HAS_ADD=has_add,
            USE_TANH=1,
            num_warps=warps, num_stages=stages,
        )
        return O

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # compile
    _ = run_kernel()
    torch.cuda.synchronize()

    ker = _bench(run_kernel, args.warmup, args.iters)

    pq_ref = Pq if Pq.dtype == ref_dtype else Pq.to(ref_dtype)
    pk_ref = Pk if Pk.dtype == ref_dtype else Pk.to(ref_dtype)
    pv_ref = Pv if Pv.dtype == ref_dtype else Pv.to(ref_dtype)
    vq_ref = Vq if Vq.dtype == ref_dtype else Vq.to(ref_dtype)
    vk_ref = Vk if Vk.dtype == ref_dtype else Vk.to(ref_dtype)
    vv_ref = Vv if Vv.dtype == ref_dtype else Vv.to(ref_dtype)
    bq_ref = bq if bq.dtype == ref_dtype else bq.to(ref_dtype)
    bk_ref = bk if bk.dtype == ref_dtype else bk.to(ref_dtype)
    bv_ref = bv if bv.dtype == ref_dtype else bv.to(ref_dtype)
    cos_ref = cos if cos.dtype == ref_dtype else cos.to(ref_dtype)
    sin_ref = sin if sin.dtype == ref_dtype else sin.to(ref_dtype)

    # Reference for the same kernel contract; default dtype matches kernel dtype.
    def run_ref():
        q = torch.einsum("bmhd,hde->bmhe", pq_ref, vq_ref) + bq_ref[None, None, :, :]
        k = torch.einsum("bmhd,hde->bmhe", pk_ref, vk_ref) + bk_ref[None, None, :, :]
        v = torch.einsum("bmhd,hde->bmhe", pv_ref, vv_ref) + bv_ref[None, None, :, :]

        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        q = rope.apply_rotary(q, cos_ref, sin_ref)
        k = rope.apply_rotary(k, cos_ref, sin_ref)

        add_mask = None
        if attn_mask is not None:
            valid = attn_mask.to(torch.bool)
            allow = valid[:, None, :, None] & valid[:, None, None, :]
            add_mask = torch.zeros((B, 1, M, M), device=device, dtype=ref_dtype)
            add_mask.masked_fill_(~allow, torch.finfo(ref_dtype).min)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=add_mask, dropout_p=0.0, is_causal=False
        )

    ref = _bench(
        run_ref,
        max(1, args.warmup // 2),
        max(1, args.iters // 5),
    )

    peak_alloc = _mib(torch.cuda.max_memory_allocated())
    peak_res = _mib(torch.cuda.max_memory_reserved())

    toks = B * M
    print("==== RoPEAttn Kernel Microbench ====")
    print(
        f"shape: B={B}, M={M}, H={H}, dh={dh}, rank_target={R} (kernel-contract uses per-head dh), dtype={args.dtype}, "
        f"BM={args.attn_bm}, BN={args.attn_bn}, BR={args.attn_br}, warps={args.attn_warps}, stages={args.attn_stages}"
    )
    print(f"- triton_kernel: {ker['mean_ms']:.4f} ms | {toks/(ker['mean_ms']/1e3):,.0f} tok/s")
    print(f"- torch_ref({ref_dtype_name}): {ref['mean_ms']:.4f} ms | {toks/(ref['mean_ms']/1e3):,.0f} tok/s")
    print(f"- speedup vs ref: x{ref['mean_ms']/ker['mean_ms']:.3f}")
    print(f"- peak mem (process scope): alloc={peak_alloc:.1f} MiB reserved={peak_res:.1f} MiB")


def main():
    parser = argparse.ArgumentParser("FlashSVD kernel microbench")
    parser.add_argument("--kernel", type=str, default="both", choices=["geglu", "rope", "both"])

    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--L", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=1152)

    parser.add_argument("--target-param-ratio", type=float, default=0.5)
    parser.add_argument("--rank-round-multiple", type=int, default=64)

    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--rope-ref-dtype", type=str, default="match", choices=["match", "fp16", "bf16", "fp32"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)

    parser.add_argument("--pad-fraction", type=float, default=0.0)

    parser.add_argument("--geglu-bl", type=int, default=64)
    parser.add_argument("--geglu-bd", type=int, default=128)
    parser.add_argument("--geglu-br1", type=int, default=64)
    parser.add_argument("--geglu-br2", type=int, default=64)
    parser.add_argument("--geglu-bh", type=int, default=128)
    parser.add_argument("--geglu-warps", type=int, default=4)
    parser.add_argument("--geglu-stages", type=int, default=2)
    parser.add_argument("--geglu-autotune", action="store_true")
    parser.add_argument("--geglu-autotune-only", action="store_true")
    parser.add_argument("--geglu-autotune-target", type=str, default="preg_cache", choices=["preg_cache", "preg_rebuild", "fused", "two_stage"])
    parser.add_argument("--geglu-tune-warmup", type=int, default=4)
    parser.add_argument("--geglu-tune-iters", type=int, default=12)
    parser.add_argument("--geglu-autotune-max-configs", type=int, default=128)
    parser.add_argument("--geglu-autotune-verbose", action="store_true")
    parser.add_argument("--geglu-autotune-bl", type=str, default="32,64,128")
    parser.add_argument("--geglu-autotune-bd", type=str, default="64,128")
    parser.add_argument("--geglu-autotune-br1", type=str, default="32,64,128")
    parser.add_argument("--geglu-autotune-br2", type=str, default="64,128")
    parser.add_argument("--geglu-autotune-bh", type=str, default="64,128,256")
    parser.add_argument("--geglu-autotune-warps", type=str, default="4,8")
    parser.add_argument("--geglu-autotune-stages", type=str, default="1,2")

    parser.add_argument("--attn-bm", type=int, default=64)
    parser.add_argument("--attn-bn", type=int, default=64)
    parser.add_argument("--attn-br", type=int, default=32)
    parser.add_argument("--attn-warps", type=int, default=4)
    parser.add_argument("--attn-stages", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    base_dir = Path(__file__).resolve().parent

    if args.kernel in {"geglu", "both"}:
        bench_geglu(args, base_dir)
    if args.kernel in {"rope", "both"}:
        bench_rope(args, base_dir)


if __name__ == "__main__":
    main()
