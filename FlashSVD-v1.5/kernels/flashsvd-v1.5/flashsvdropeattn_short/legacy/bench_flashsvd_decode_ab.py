#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A/B benchmark for FlashSVD RoPE low-rank *decode* (Q_len=1) kernels.

It compares:
  - v1: flashsvd_attn_decode_packed_v1 (original)
  - v2: flashsvd_attn_decode_packed (upgraded: q-precompute + pad16 + Vk-resident + dynamic splits + writethrough)

Usage example (packed decode, causal):
  CUDA_VISIBLE_DEVICES=1 python bench_flashsvd_decode_ab.py \
    --B 8 --Smax 2048 --seqlen_k 2048 --H 32 --Hk 8 --Dh 128 --R 64 \
    --dtype bf16 --split_k 512 --bn 64 --br 64 --warps1 4 --stages1 2 --warps2 4 --stages2 1 \
    --causal --warmup 50 --iters 1000 --check

Notes:
- Requires Triton + CUDA.
- For correctness check, keep seqlen_k <= 256 to avoid heavy reference cost.
"""

import math
import time
import argparse
import gc
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton

# import the module under test
import importlib.util
from pathlib import Path

def load_module(path: str):
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location("flashsvd_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def build_rope_tables(seqlen: int, head_dim: int, base: float, device, dtype):
    assert head_dim % 2 == 0
    half = head_dim // 2
    pos = torch.arange(seqlen, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = torch.einsum("m,d->md", pos, inv_freq)
    cos = torch.cos(ang).to(dtype)
    sin = torch.sin(ang).to(dtype)
    return cos.contiguous(), sin.contiguous()


def do_bench_ms(fn, warmup=50, rep=200) -> float:
    try:
        return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))
    except Exception:
        torch.cuda.synchronize()
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / rep


def pretty_bytes(n: int) -> str:
    x = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} PB"


@torch.no_grad()
def reference_decode_fp32(
    # factors
    Pq_q, Pk, Pv, Vq, Vk, Vv,
    bq, bk, bv,
    cos, sin,
    *,
    seqlen_k: int,
    causal: bool,
    window_left: int,
    window_right: int,
):
    """
    Reference decode output in fp32.
    Shapes:
      Pq_q: [B,H,R]
      Pk/Pv: [B,Smax,Hk,R]
      Vq: [H,R,Dh]
      Vk/Vv: [Hk,R,Dh]
      cos/sin: [Smax, Dh/2]
    """
    B, H, R = Pq_q.shape
    Smax = Pk.shape[1]
    Hk = Pk.shape[2]
    Dh = Vq.shape[-1]
    half = Dh // 2
    rep = H // Hk
    scale = 1.0 / math.sqrt(Dh)

    assert seqlen_k <= Smax
    q_pos = max(seqlen_k - 1, 0)

    # dense Q
    Q = torch.einsum("bhr,hrd->bhd", Pq_q.float(), Vq.float())
    if bq is not None:
        Q = Q + bq.float()[None, :, :]
    # RoPE @ q_pos
    cos_q = cos[q_pos].float()[None, None, :]
    sin_q = sin[q_pos].float()[None, None, :]
    Q0, Q1 = Q[..., :half], Q[..., half:]
    Q = torch.cat([Q0 * cos_q - Q1 * sin_q, Q0 * sin_q + Q1 * cos_q], dim=-1)

    # dense K,V up to seqlen_k (Hk heads)
    K = torch.einsum("bskr,krd->bskd", Pk[:, :seqlen_k].float(), Vk.float())
    V = torch.einsum("bskr,krd->bskd", Pv[:, :seqlen_k].float(), Vv.float())
    if bk is not None:
        K = K + bk.float()[None, None, :, :]
    if bv is not None:
        V = V + bv.float()[None, None, :, :]

    # RoPE for all keys
    cos_k = cos[:seqlen_k].float()[None, :, None, :]
    sin_k = sin[:seqlen_k].float()[None, :, None, :]
    K0, K1 = K[..., :half], K[..., half:]
    K = torch.cat([K0 * cos_k - K1 * sin_k, K0 * sin_k + K1 * cos_k], dim=-1)

    # expand to H heads
    K_full = K.repeat_interleave(rep, dim=2)  # [B, S, H, Dh]
    V_full = V.repeat_interleave(rep, dim=2)

    # score: Q (B,H,D) vs K_full (B,S,H,D) -> (B,H,S)
    scores = (Q[:, :, None, :] * K_full.transpose(1, 2)).sum(dim=-1) * scale  # [B,H,S]

    kpos = torch.arange(seqlen_k, device=scores.device)[None, None, :]
    qpos = torch.full_like(kpos, q_pos)

    if causal:
        scores = scores.masked_fill(kpos > qpos, float("-inf"))
    if window_left != -1:
        scores = scores.masked_fill(kpos < (qpos - window_left), float("-inf"))
    if window_right != -1:
        scores = scores.masked_fill(kpos > (qpos + window_right), float("-inf"))

    attn = torch.softmax(scores, dim=-1)  # [B,H,S]
    out = torch.einsum("bhs,bshd->bhd", attn, V_full)  # [B,H,Dh]
    return out


def isolated_peak(fn, *a, **k):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn(*a, **k)
    torch.cuda.synchronize()
    return out, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module_path", type=str, default="flashsvdropeattn_v1.6_decode_opt.py",
                    help="path to the module file (python) containing the kernels")
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
    ap.add_argument("--disable_pad16", action="store_true")
    ap.add_argument("--disable_vk_resident", action="store_true")
    ap.add_argument("--disable_writethrough", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    mod = load_module(args.module_path)

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    B, Smax, seqlen_k = args.B, args.Smax, args.seqlen_k
    H, Hk, Dh, R = args.H, args.Hk, args.Dh, args.R
    assert H % Hk == 0
    assert seqlen_k <= Smax and seqlen_k >= 0
    assert Dh % 2 == 0

    torch.manual_seed(0)

    cos, sin = build_rope_tables(Smax, Dh, base=10000.0, device=device, dtype=dtype)

    Pq_q = torch.randn(B, H, R, device=device, dtype=dtype).contiguous()
    Pk = torch.randn(B, Smax, Hk, R, device=device, dtype=dtype).contiguous()
    Pv = torch.randn(B, Smax, Hk, R, device=device, dtype=dtype).contiguous()
    Vq = torch.randn(H, R, Dh, device=device, dtype=dtype).contiguous()
    Vk = torch.randn(Hk, R, Dh, device=device, dtype=dtype).contiguous()
    Vv = torch.randn(Hk, R, Dh, device=device, dtype=dtype).contiguous()
    bq = torch.randn(H, Dh, device=device, dtype=dtype).contiguous()
    bk = torch.randn(Hk, Dh, device=device, dtype=dtype).contiguous()
    bv = torch.randn(Hk, Dh, device=device, dtype=dtype).contiguous()

    f = mod.DecodePackedFactors(Pq=Pq_q, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv, bq=bq, bk=bk, bv=bv)

    # Warmup
    for _ in range(args.warmup):
        _ = mod.flashsvd_attn_decode_packed_v1(
            f, cos, sin,
            seqlen_k=seqlen_k,
            causal=args.causal,
            window_size=(args.window_left, args.window_right),
            split_k=args.split_k,
            bn=args.bn,
            br=args.br,
            num_warps_stage1=args.warps1,
            num_stages_stage1=args.stages1,
            num_warps_stage2=args.warps2,
            num_stages_stage2=args.stages2,
        )
        _ = mod.flashsvd_attn_decode_packed(
            f, cos, sin,
            seqlen_k=seqlen_k,
            causal=args.causal,
            window_size=(args.window_left, args.window_right),
            split_k=args.split_k,
            bn=args.bn,
            br=args.br,
            num_warps_stage1=args.warps1,
            num_stages_stage1=args.stages1,
            num_warps_stage2=args.warps2,
            num_stages_stage2=args.stages2,
            pad_to_16=(not args.disable_pad16),
            vk_resident=(not args.disable_vk_resident),
            writethrough=(not args.disable_writethrough),
        )
    torch.cuda.synchronize()

    # Peak + perf
    _, peak_alloc_v1, peak_res_v1 = isolated_peak(
        mod.flashsvd_attn_decode_packed_v1,
        f, cos, sin,
        seqlen_k=seqlen_k,
        causal=args.causal,
        window_size=(args.window_left, args.window_right),
        split_k=args.split_k,
        bn=args.bn,
        br=args.br,
        num_warps_stage1=args.warps1,
        num_stages_stage1=args.stages1,
        num_warps_stage2=args.warps2,
        num_stages_stage2=args.stages2,
    )
    ms_v1 = do_bench_ms(
        lambda: mod.flashsvd_attn_decode_packed_v1(
            f, cos, sin,
            seqlen_k=seqlen_k,
            causal=args.causal,
            window_size=(args.window_left, args.window_right),
            split_k=args.split_k,
            bn=args.bn,
            br=args.br,
            num_warps_stage1=args.warps1,
            num_stages_stage1=args.stages1,
            num_warps_stage2=args.warps2,
            num_stages_stage2=args.stages2,
        ),
        warmup=max(10, args.warmup // 2),
        rep=args.iters,
    )

    _, peak_alloc_v2, peak_res_v2 = isolated_peak(
        mod.flashsvd_attn_decode_packed,
        f, cos, sin,
        seqlen_k=seqlen_k,
        causal=args.causal,
        window_size=(args.window_left, args.window_right),
        split_k=args.split_k,
        bn=args.bn,
        br=args.br,
        num_warps_stage1=args.warps1,
        num_stages_stage1=args.stages1,
        num_warps_stage2=args.warps2,
        num_stages_stage2=args.stages2,
        pad_to_16=(not args.disable_pad16),
        vk_resident=(not args.disable_vk_resident),
        writethrough=(not args.disable_writethrough),
    )
    ms_v2 = do_bench_ms(
        lambda: mod.flashsvd_attn_decode_packed(
            f, cos, sin,
            seqlen_k=seqlen_k,
            causal=args.causal,
            window_size=(args.window_left, args.window_right),
            split_k=args.split_k,
            bn=args.bn,
            br=args.br,
            num_warps_stage1=args.warps1,
            num_stages_stage1=args.stages1,
            num_warps_stage2=args.warps2,
            num_stages_stage2=args.stages2,
            pad_to_16=(not args.disable_pad16),
            vk_resident=(not args.disable_vk_resident),
            writethrough=(not args.disable_writethrough),
        ),
        warmup=max(10, args.warmup // 2),
        rep=args.iters,
    )

    tok_s_v1 = B / (ms_v1 / 1e3)
    tok_s_v2 = B / (ms_v2 / 1e3)

    print("==== FlashSVD decode A/B (packed) ====")
    print(f"Shape: B={B}, Smax={Smax}, seqlen_k={seqlen_k}, H={H}, Hk={Hk}, Dh={Dh}, R={R}, dtype={dtype}")
    print(f"Mask: causal={args.causal}, window=({args.window_left},{args.window_right})")
    print(f"split_k={args.split_k}, bn={args.bn}, br={args.br}")
    print(f"[v1] latency {ms_v1:.4f} ms | tok/s {tok_s_v1:,.0f} | peak alloc {pretty_bytes(peak_alloc_v1)} | peak res {pretty_bytes(peak_res_v1)}")
    print(f"[v2] latency {ms_v2:.4f} ms | tok/s {tok_s_v2:,.0f} | peak alloc {pretty_bytes(peak_alloc_v2)} | peak res {pretty_bytes(peak_res_v2)}")
    print(f"Speedup: {ms_v1 / ms_v2:.2f}x  (tok/s {tok_s_v2 / tok_s_v1:.2f}x)")

    if args.check:
        if seqlen_k > 256:
            print("[check] seqlen_k too large for reference; re-run with --seqlen_k <= 256")
        else:
            out_v1 = mod.flashsvd_attn_decode_packed_v1(
                f, cos, sin,
                seqlen_k=seqlen_k,
                causal=args.causal,
                window_size=(args.window_left, args.window_right),
                split_k=args.split_k,
                bn=args.bn,
                br=args.br,
            ).float()
            out_v2 = mod.flashsvd_attn_decode_packed(
                f, cos, sin,
                seqlen_k=seqlen_k,
                causal=args.causal,
                window_size=(args.window_left, args.window_right),
                split_k=args.split_k,
                bn=args.bn,
                br=args.br,
                pad_to_16=(not args.disable_pad16),
                vk_resident=(not args.disable_vk_resident),
                writethrough=(not args.disable_writethrough),
            ).float()
            ref = reference_decode_fp32(
                Pq_q, Pk, Pv, Vq, Vk, Vv, bq, bk, bv, cos, sin,
                seqlen_k=seqlen_k,
                causal=args.causal,
                window_left=args.window_left,
                window_right=args.window_right,
            )
            diff1 = (out_v1 - ref).abs()
            diff2 = (out_v2 - ref).abs()
            print(f"[check v1] max_abs={diff1.max().item():.3e}  rel_fro={torch.linalg.norm(out_v1-ref)/ (torch.linalg.norm(ref)+1e-12):.3e}")
            print(f"[check v2] max_abs={diff2.max().item():.3e}  rel_fro={torch.linalg.norm(out_v2-ref)/ (torch.linalg.norm(ref)+1e-12):.3e}")


if __name__ == "__main__":
    main()
