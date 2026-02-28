#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys
from typing import List

import torch

'''
CUDA_VISIBLE_DEVICES=1 python mock_gqa_check.py --B 2 --S 128 --H 32 --Hk 8 --Dh 128 --R 64 --dtype bf16 --bm 64 --bn 64 --br 64 --warps 8 --stages 3 --causal

'''

def _add_flashsvd_to_path():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    kern_dir = os.path.join(repo_root, "kernels", "flashsvd-v1.5")
    sys.path.insert(0, kern_dir)


def _parse_int_list(s: str) -> List[int]:
    items = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _decode_positions(raw: str, seq_len: int) -> List[int]:
    if not raw:
        return [seq_len - 1]
    pos = _parse_int_list(raw)
    out = []
    for p in pos:
        if p < 0:
            p = seq_len + p
        if 0 <= p < seq_len:
            out.append(p)
    if not out:
        out = [seq_len - 1]
    return out


def _reference_decode_fp32(fs, f, cos, sin, pos, causal, window_left, window_right):
    Pq, Pk, Pv = f.Pq.float(), f.Pk.float(), f.Pv.float()
    Vq, Vk, Vv = f.Vq.float(), f.Vk.float(), f.Vv.float()
    bq = f.bq.float() if f.bq is not None else None
    bk = f.bk.float() if f.bk is not None else None
    bv = f.bv.float() if f.bv is not None else None

    B, S, H, R = Pq.shape
    Hk = Pk.shape[2]
    Dh = Vq.shape[-1]
    rep = H // Hk
    scale = 1.0 / math.sqrt(Dh)

    Q = torch.einsum("bshr,hrd->bshd", Pq, Vq)
    K = torch.einsum("bskr,krd->bskd", Pk, Vk)
    V = torch.einsum("bskr,krd->bskd", Pv, Vv)
    if bq is not None:
        Q = Q + bq[None, None, :, :]
    if bk is not None:
        K = K + bk[None, None, :, :]
    if bv is not None:
        V = V + bv[None, None, :, :]

    Q = fs.rope_apply_bshd(Q, cos.float(), sin.float())
    K = fs.rope_apply_bshd(K, cos.float(), sin.float())

    K_full = K.repeat_interleave(rep, dim=2)
    V_full = V.repeat_interleave(rep, dim=2)

    q = Q[:, pos, :, :]  # [B, H, Dh]
    scores = torch.einsum("bhd,bshd->bhs", q, K_full) * scale  # [B,H,S]

    if causal:
        kpos = torch.arange(S, device=scores.device)
        scores = scores.masked_fill(kpos[None, None, :] > pos, float("-inf"))

    if window_left != -1 or window_right != -1:
        kpos = torch.arange(S, device=scores.device)
        left_ok = True if window_left == -1 else (kpos >= (pos - window_left))
        right_ok = True if window_right == -1 else (kpos <= (pos + window_right))
        scores = scores.masked_fill(~(left_ok & right_ok)[None, None, :], float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,bshd->bhd", attn, V_full)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Mock GQA check for flashsvd attention (packed).")
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--Hk", type=int, default=4)
    ap.add_argument("--group-sizes", type=str, default="", help="comma list of group sizes (H/Hk)")
    ap.add_argument("--Dh", type=int, default=64)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--br", type=int, default=32)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--causal", action="store_true", default=True)
    ap.add_argument("--no-causal", dest="causal", action="store_false")
    ap.add_argument("--window-left", type=int, default=-1)
    ap.add_argument("--window-right", type=int, default=-1)
    ap.add_argument(
        "--scenarios",
        type=str,
        default="prefill,decode,eval",
        help="comma list: prefill,decode,eval",
    )
    ap.add_argument("--decode-pos", type=str, default="-1", help="comma list of positions; -1 means last")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this check.")

    if args.S > 512:
        print("[warn] S is large; full reference is O(S^2). Consider S<=256 for quick checks.")

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    _add_flashsvd_to_path()
    import flashsvd as fs  # type: ignore

    scenarios = {s.strip() for s in args.scenarios.split(",") if s.strip()}
    group_sizes = _parse_int_list(args.group_sizes) if args.group_sizes.strip() else []
    if not group_sizes:
        if args.H % args.Hk != 0:
            raise SystemExit(f"H ({args.H}) must be divisible by Hk ({args.Hk}) for GQA.")
        group_sizes = [args.H // args.Hk]

    decode_positions = _decode_positions(args.decode_pos, args.S)

    for g in group_sizes:
        Hk = args.Hk
        H = Hk * g
        torch.manual_seed(args.seed + g)

        B, S, Dh, R = args.B, args.S, args.Dh, args.R
        Pq = torch.randn(B, S, H, R, device=device, dtype=dtype).contiguous()
        Pk = torch.randn(B, S, Hk, R, device=device, dtype=dtype).contiguous()
        Pv = torch.randn(B, S, Hk, R, device=device, dtype=dtype).contiguous()
        Vq = torch.randn(H, R, Dh, device=device, dtype=dtype).contiguous()
        Vk = torch.randn(Hk, R, Dh, device=device, dtype=dtype).contiguous()
        Vv = torch.randn(Hk, R, Dh, device=device, dtype=dtype).contiguous()
        bq = torch.randn(H, Dh, device=device, dtype=dtype).contiguous()
        bk = torch.randn(Hk, Dh, device=device, dtype=dtype).contiguous()
        bv = torch.randn(Hk, Dh, device=device, dtype=dtype).contiguous()

        cos, sin = fs.build_rope_tables(S, Dh, base=10000.0, device=device, dtype=dtype)
        f = fs.PackedFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv, bq=bq, bk=bk, bv=bv)

        with torch.no_grad():
            out = fs.flashsvd_attn_packed(
                f,
                cos,
                sin,
                causal=args.causal,
                window_size=(args.window_left, args.window_right),
                bm=args.bm,
                bn=args.bn,
                br=args.br,
                num_warps=args.warps,
                num_stages=args.stages,
            )

        if "prefill" in scenarios or "eval" in scenarios:
            with torch.no_grad():
                ref = fs.reference_packed_fp32(
                    f,
                    cos,
                    sin,
                    causal=args.causal,
                    window_left=args.window_left,
                    window_right=args.window_right,
                ).to(out.dtype)
            diff = (out - ref).float()
            rel_fro = (torch.linalg.norm(diff) / (torch.linalg.norm(ref.float()) + 1e-12)).item()
            max_abs = diff.abs().max().item()
            finite = torch.isfinite(out).all().item()
            if "prefill" in scenarios:
                print(f"[group={g}] prefill: finite={finite} max|diff|={max_abs:.3e} rel_fro={rel_fro:.3e}")
            if "eval" in scenarios:
                print(f"[group={g}] eval(no-kv): finite={finite} max|diff|={max_abs:.3e} rel_fro={rel_fro:.3e}")

        if "decode" in scenarios:
            for pos in decode_positions:
                with torch.no_grad():
                    ref_d = _reference_decode_fp32(
                        fs,
                        f,
                        cos,
                        sin,
                        pos=pos,
                        causal=args.causal,
                        window_left=args.window_left,
                        window_right=args.window_right,
                    ).to(out.dtype)
                out_d = out[:, pos, :, :]
                diff_d = (out_d - ref_d).float()
                rel_fro_d = (torch.linalg.norm(diff_d) / (torch.linalg.norm(ref_d.float()) + 1e-12)).item()
                max_abs_d = diff_d.abs().max().item()
                finite_d = torch.isfinite(out_d).all().item()
                print(
                    f"[group={g}] decode@{pos}: finite={finite_d} max|diff|={max_abs_d:.3e} rel_fro={rel_fro_d:.3e}"
                )


if __name__ == "__main__":
    main()
