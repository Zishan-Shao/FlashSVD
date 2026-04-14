#!/usr/bin/env python3
"""
benchmark_long_context_decoder_attn.py — Decoder attention benchmark (causal; prefill and KV-cache decode)

Compares:
- PyTorch SDPA (math kernel forced; is_causal=True)
- Triton FlashAttention (decoder causal kernel from benchmark/decoder_kernel)

Modes:
- prefill: Q,K,V all length S (causal + padding)
- decode: Q length T, KV length S (KV-cache)

Records latency stats, measured peak memory, theoretical activation peaks, estimated traffic, and FLOPs.

python benchmark/benchmark_long_context_decoder_attn.py --mode prefill --heads 12 --head-dim 64 --attn-rank 32 --batch-sizes 8 --seq-lens 1024 2048 4096 --dtype float16 --warmup 10 --iters 50 --csv benchmark/decoder_attn_prefill.csv


python benchmark_long_context_decoder_attn.py --mode decode --decode-tokens 1 --heads 12 --head-dim 64 --batch-sizes 1 --seq-lens 1024 2048 4096 --dtype float16 --warmup 10 --iters 50 --csv benchmark/decoder_attn_decode.csv


"""

import argparse
import math
import os
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DEC_DIR = os.path.join(BENCH_DIR, "decoder_kernel")
if DEC_DIR not in sys.path:
    sys.path.insert(0, DEC_DIR)

from flash_attn_causal import (
    flash_attn_triton,
    flash_attn_triton_kvcache,
)
from flashsvdropeattn import FlashSVDRoPEAttention, QKVFactors, _SimpleRotary


@torch.no_grad()
def bench_cuda(fn, n_warmup: int = 10, n_runs: int = 50) -> Dict[str, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(n_warmup):
        _ = fn()
    torch.cuda.synchronize()
    for _ in range(n_runs):
        start.record(); out = fn(); end.record(); torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))
        _ = out.view(-1)[0].item()
    ts.sort()
    mean = sum(ts) / len(ts)
    p50 = ts[len(ts)//2]
    p95 = ts[int(0.95*len(ts)) - 1]
    std = (sum((t-mean)*(t-mean) for t in ts)/max(1,len(ts)-1))**0.5
    return {"mean_ms": mean, "std_ms": std, "p50_ms": p50, "p95_ms": p95, "all_ms": ts}


def elem_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


# FLOPs
def flops_prefill(B,H,S,D):
    return 4*B*H*S*S*D + 5*B*H*S*S

def flops_decode(B,H,T,K,D):
    return 4*B*H*T*K*D + 5*B*H*T*K


# Traffic (rough)
def traffic_prefill_bytes(B,H,S,D,dtype):
    e = elem_size(dtype)
    return e * (4*B*H*S*D + 2*B*H*S*S)

def traffic_flash_bytes(B,H,S,D,dtype):
    e = elem_size(dtype)
    return e * (4*B*H*S*D)

def traffic_decode_bytes(B,H,T,K,D,dtype):
    e = elem_size(dtype)
    return e * (B*H*((T+2*K)*D + T*D) + B*H*T*K)  # reads Q,K,V + writes O + attn RW


# Peaks (activations)
def peak_prefill_sdpa(B,H,S,D,dtype):
    e = elem_size(dtype)
    return e * (B*H*S*S + B*H*S*D)

def peak_prefill_flash(B,H,S,D,dtype):
    e = elem_size(dtype)
    return e * (B*H*S*D + B*H*S)

# FlashSVD+RoPE (rank-space) estimates
def flops_flashsvd_rope(B,H,S,D,R):
    # Same model-level as FlashSVD attention (rank-space) + softmax
    return 8*B*H*S*R*D + 2*B*H*S*S*R + 5*B*H*S*S

def traffic_flashsvd_rope_bytes(B,H,S,D,R,dtype):
    e = elem_size(dtype)
    # Read P/V/b for q,k,v; write out. Ignore small cos/sin.
    per = (B*H*S*R) + (B*H*R*D) + (B*H*D)
    out = B*H*S*D
    return e * (3*per + out)

def peak_flashsvd_rope_bytes(B,H,S,D,R,dtype):
    e = elem_size(dtype)
    out = B*H*S*D
    rank = B*H*S*R
    return e * (out + rank)

def peak_decode_sdpa(B,H,T,K,D,dtype):
    e = elem_size(dtype)
    return e * (B*H*T*K + B*H*T*D)

def peak_decode_flash(B,H,T,D,dtype):
    e = elem_size(dtype)
    return e * (B*H*T*D + B*H*T)


def run():
    p = argparse.ArgumentParser("Decoder attention long-context benchmark (prefill/decode)")
    p.add_argument("--mode", type=str, default="prefill", choices=["prefill","decode"], help="Benchmark mode")
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    p.add_argument("--seq-lens", type=int, nargs="+", default=[1024,2048,4096], help="S (prefill) or KV length K (decode)")
    p.add_argument("--decode-tokens", type=int, default=1, help="Query length T for decode mode")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16","bfloat16","float32"])
    p.add_argument("--attn-rank", type=int, default=32, help="FlashSVDRoPE per-head rank (<= head-dim, <=64)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--csv", type=str, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    dtype = {"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[args.dtype]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)

    B_list = args.batch_sizes
    S_list = args.seq_lens
    H, D = args.heads, args.head_dim
    R = max(1, min(int(args.attn_rank), D, 64))

    print(f"GPU: {torch.cuda.get_device_name()} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"mode={args.mode} B={B_list} S/K={S_list} H={H} D={D} dtype={args.dtype}")
    print("")

    rows: List[str] = []
    if args.csv:
        header = [
            "mode","method","B","H","S_or_K","T","D","dtype",
            "mean_ms","std_ms","p50_ms","p95_ms",
            "peak_mem_bytes","theoretical_peak_bytes","est_traffic_bytes","flops","tflops_per_s",
        ]
        rows.append(",".join(header))

    for B in B_list:
        for S in S_list:
            if args.mode == "prefill":
                # Inputs
                Q = torch.randn(B,H,S,D, device=device, dtype=dtype)
                K = torch.randn(B,H,S,D, device=device, dtype=dtype)
                V = torch.randn(B,H,S,D, device=device, dtype=dtype)
                mask = torch.ones(B,H,1,S, device=device, dtype=torch.bool)

                # PyTorch SDPA causal
                def torch_prefill():
                    q = Q.reshape(B*H,S,D)
                    k = K.reshape(B*H,S,D)
                    v = V.reshape(B*H,S,D)
                    return F.scaled_dot_product_attention(q,k,v, is_causal=True).reshape(B,H,S,D)
                peak = lambda fn: (torch.cuda.reset_peak_memory_stats(), fn(), torch.cuda.synchronize(), int(torch.cuda.max_memory_allocated()))[-1]
                peak_t = peak(torch_prefill)
                st_t = bench_cuda(torch_prefill, args.warmup, args.iters)
                fl_t = flops_prefill(B,H,S,D)
                tr_t = traffic_prefill_bytes(B,H,S,D,dtype)
                pk_theo_t = peak_prefill_sdpa(B,H,S,D,dtype)
                tfl_t = fl_t / (st_t['mean_ms']/1e3) / 1e12
                print(f"[prefill S={S}] PyTorch SDPA: mean {st_t['mean_ms']:.2f} ms | peak {peak_t/1024**2:.2f} MiB (theor {pk_theo_t/1024**2:.2f})")
                if args.csv:
                    rows.append(
                        ",".join(map(str,["prefill","torch_sdpa",B,H,S,0,D,args.dtype,
                                         f"{st_t['mean_ms']:.4f}",f"{st_t['std_ms']:.4f}",f"{st_t['p50_ms']:.4f}",f"{st_t['p95_ms']:.4f}",
                                         peak_t, pk_theo_t, tr_t, fl_t, f"{tfl_t:.6f}"]))
                    )

                # Triton FlashAttention
                def triton_prefill():
                    return flash_attn_triton(Q, K, V, mask)
                peak_f = peak(triton_prefill)
                st_f = bench_cuda(triton_prefill, args.warmup, args.iters)
                fl_f = flops_prefill(B,H,S,D)
                tr_f = traffic_flash_bytes(B,H,S,D,dtype)
                pk_theo_f = peak_prefill_flash(B,H,S,D,dtype)
                tfl_f = fl_f / (st_f['mean_ms']/1e3) / 1e12
                print(f"[prefill S={S}] FlashAttention: mean {st_f['mean_ms']:.2f} ms | peak {peak_f/1024**2:.2f} MiB (theor {pk_theo_f/1024**2:.2f})")
                if args.csv:
                    rows.append(
                        ",".join(map(str,["prefill","flashattention_triton",B,H,S,0,D,args.dtype,
                                         f"{st_f['mean_ms']:.4f}",f"{st_f['std_ms']:.4f}",f"{st_f['p50_ms']:.4f}",f"{st_f['p95_ms']:.4f}",
                                         peak_f, pk_theo_f, tr_f, fl_f, f"{tfl_f:.6f}"]))
                    )

                # --- FlashSVD + RoPE (rank-space, prefill only) ---
                # Build rank factors [B,H,S,R] and lifts [H,R,dh]; biases flattened [H*dh]
                Pq = torch.randn(B,H,S,R, device=device, dtype=dtype)
                Pk = torch.randn(B,H,S,R, device=device, dtype=dtype)
                Pv = torch.randn(B,H,S,R, device=device, dtype=dtype)
                Vq = torch.randn(H,R,D,  device=device, dtype=dtype).contiguous()
                Vk = torch.randn(H,R,D,  device=device, dtype=dtype).contiguous()
                Vv = torch.randn(H,R,D,  device=device, dtype=dtype).contiguous()
                bq = torch.zeros(H*D, device=device, dtype=dtype)
                bk = torch.zeros(H*D, device=device, dtype=dtype)
                bv = torch.zeros(H*D, device=device, dtype=dtype)

                # Rotary: simple base 10000 implementation
                rotary = _SimpleRotary(base=10000.0)
                position_ids = torch.arange(S, device=device)[None, :].expand(B, -1)
                flashsvd = FlashSVDRoPEAttention(H, D, rotary, bm=64, bn=64, bdh=D, br=R)

                qkv = QKVFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv, bq=bq, bk=bk, bv=bv)

                def svd_prefill():
                    return flashsvd(qkv, attention_mask=mask.view(B, S).to(torch.bool), position_ids=position_ids)

                # Peak mem and latency
                torch.cuda.reset_peak_memory_stats(); _ = svd_prefill(); torch.cuda.synchronize()
                peak_s = int(torch.cuda.max_memory_allocated())
                st_s = bench_cuda(svd_prefill, args.warmup, args.iters)
                fl_s = flops_flashsvd_rope(B,H,S,D,R)
                tr_s = traffic_flashsvd_rope_bytes(B,H,S,D,R,dtype)
                pk_theo_s = peak_flashsvd_rope_bytes(B,H,S,D,R,dtype)
                tfl_s = fl_s / (st_s['mean_ms']/1e3) / 1e12
                print(f"[prefill S={S}] FlashSVD+RoPE: mean {st_s['mean_ms']:.2f} ms | peak {peak_s/1024**2:.2f} MiB (theor {pk_theo_s/1024**2:.2f})")
                if args.csv:
                    rows.append(
                        ",".join(map(str,["prefill","flashsvd_rope",B,H,S,0,D,args.dtype,
                                         f"{st_s['mean_ms']:.4f}",f"{st_s['std_ms']:.4f}",f"{st_s['p50_ms']:.4f}",f"{st_s['p95_ms']:.4f}",
                                         peak_s, pk_theo_s, tr_s, fl_s, f"{tfl_s:.6f}"]))
                    )

            else:  # decode
                T = int(args.decode_tokens)
                Q = torch.randn(B,H,T,D, device=device, dtype=dtype)
                K = torch.randn(B,H,S,D, device=device, dtype=dtype)
                V = torch.randn(B,H,S,D, device=device, dtype=dtype)
                mask = torch.ones(B,H,1,T, device=device, dtype=torch.bool)

                def torch_decode():
                    q = Q.reshape(B*H,T,D)
                    k = K.reshape(B*H,S,D)
                    v = V.reshape(B*H,S,D)
                    return F.scaled_dot_product_attention(q,k,v, is_causal=True).reshape(B,H,T,D)
                peak_t = (torch.cuda.reset_peak_memory_stats(), torch_decode(), torch.cuda.synchronize(), int(torch.cuda.max_memory_allocated()))[-1]
                st_t = bench_cuda(torch_decode, args.warmup, args.iters)
                fl_t = flops_decode(B,H,T,S,D)
                tr_t = traffic_decode_bytes(B,H,T,S,D,dtype)
                pk_theo_t = peak_decode_sdpa(B,H,T,S,D,dtype)
                tfl_t = fl_t / (st_t['mean_ms']/1e3) / 1e12
                print(f"[decode K={S},T={T}] PyTorch SDPA: mean {st_t['mean_ms']:.2f} ms | peak {peak_t/1024**2:.2f} MiB (theor {pk_theo_t/1024**2:.2f})")
                if args.csv:
                    rows.append(
                        ",".join(map(str,["decode","torch_sdpa",B,H,S,T,D,args.dtype,
                                         f"{st_t['mean_ms']:.4f}",f"{st_t['std_ms']:.4f}",f"{st_t['p50_ms']:.4f}",f"{st_t['p95_ms']:.4f}",
                                         peak_t, pk_theo_t, tr_t, fl_t, f"{tfl_t:.6f}"]))
                    )

                def triton_decode():
                    return flash_attn_triton_kvcache(Q, K, V, mask)
                peak_f = (torch.cuda.reset_peak_memory_stats(), triton_decode(), torch.cuda.synchronize(), int(torch.cuda.max_memory_allocated()))[-1]
                st_f = bench_cuda(triton_decode, args.warmup, args.iters)
                fl_f = flops_decode(B,H,T,S,D)
                tr_f = traffic_decode_bytes(B,H,T,S,D,dtype)
                pk_theo_f = peak_decode_flash(B,H,T,D,dtype)
                tfl_f = fl_f / (st_f['mean_ms']/1e3) / 1e12
                print(f"[decode K={S},T={T}] FlashAttention: mean {st_f['mean_ms']:.2f} ms | peak {peak_f/1024**2:.2f} MiB (theor {pk_theo_f/1024**2:.2f})")
                if args.csv:
                    rows.append(
                        ",".join(map(str,["decode","flashattention_triton",B,H,S,T,D,args.dtype,
                                         f"{st_f['mean_ms']:.4f}",f"{st_f['std_ms']:.4f}",f"{st_f['p50_ms']:.4f}",f"{st_f['p95_ms']:.4f}",
                                         peak_f, pk_theo_f, tr_f, fl_f, f"{tfl_f:.6f}"]))
                    )

            print("")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True) if args.csv and os.path.dirname(args.csv) else None
        with open(args.csv, "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    run()
