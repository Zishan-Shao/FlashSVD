#!/usr/bin/env python3
"""
benchmark_long_context_decoder_ffn.py — Decoder FFN (SwiGLU) long-context benchmark

Compares:
- PyTorch baseline SwiGLU FFN (rank-space: P -> (V1 split) -> SiLU* -> U2 -> V2)
- Triton FlashSVD rank-space SwiGLU from benchmark/decoder_kernel/flashsvdswiglu.py

Measures latency, peak memory, theoretical peaks, estimated activation traffic, and FLOPs.

python benchmark_long_context_decoder_ffn.py --d-model 768 --d-ff 3072 --ranks 96 192 384 512 768 --batch-sizes 8 --seq-lens 1024 2048 4096 --dtype float16 --warmup 10 --iters 50 --csv benchmark/decoder_ffn_long_context.csv

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

from flashsvdswiglu import (
    flashsvd_ffn_swiglu,
    _pt_baseline_swiglu,
    theoretical_peak_bytes_baseline,
    theoretical_peak_bytes_triton,
)


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
    mean = sum(ts)/len(ts)
    p50 = ts[len(ts)//2]
    p95 = ts[int(0.95*len(ts))-1]
    std = (sum((t-mean)*(t-mean) for t in ts)/max(1,len(ts)-1))**0.5
    return {"mean_ms": mean, "std_ms": std, "p50_ms": p50, "p95_ms": p95, "all_ms": ts}


def elem_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def flops_swiglu_dense(B,L,H,D,R2):
    # rank-space baseline here: P has shape [B,L,R1], but dense baseline cost similar to two matmuls around GELU in encoder style.
    # We provide model-level estimate comparable across variants.
    return 2*B*L*H*D + 2*B*L*D*R2 + B*L*D


def traffic_swiglu_dense_bytes(B,L,H,D,dtype):
    e = elem_size(dtype)
    return e * (2*B*L*H + 2*B*L*D)


def traffic_swiglu_triton_bytes(B,L,H,R2,dtype):
    e = elem_size(dtype)
    # P read, S write(optional) skipped (kernel streams), output write
    return e * (B*L*H + B*L*R2 + B*L*H)


# ============================
# Memory helpers
# ============================

def mib(nbytes: int) -> float:
    return nbytes / (1024.0**2)


def nbytes_of(*tensors: torch.Tensor) -> int:
    total = 0
    for t in tensors:
        if t is None:
            continue
        total += t.numel() * t.element_size()
    return int(total)


def peak_details(fn, device: torch.device):
    torch.cuda.synchronize()
    base_alloc = int(torch.cuda.memory_allocated(device))
    base_res   = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    _ = fn()
    torch.cuda.synchronize()
    peak_alloc = int(torch.cuda.max_memory_allocated(device))
    peak_res   = int(torch.cuda.max_memory_reserved(device))
    return {
        "base_alloc": base_alloc,
        "base_res": base_res,
        "peak_alloc": peak_alloc,
        "peak_res": peak_res,
        "delta_alloc": peak_alloc - base_alloc,
        "delta_res": peak_res - base_res,
    }


def main():
    p = argparse.ArgumentParser("Decoder FFN (SwiGLU) long-context benchmark")
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--ranks", type=int, nargs="+", default=[384, 512])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    p.add_argument("--seq-lens", type=int, nargs="+", default=[1024,2048,4096])
    p.add_argument("--dtype", type=str, default="float16", choices=["float16","bfloat16","float32"])
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

    H = int(args.d_model)
    D = int(args.d_ff)
    ranks = sorted({max(1, min(int(r), 768)) for r in args.ranks})

    rows: List[str] = []
    if args.csv:
        header = [
            "method","B","L","H","D","R","dtype",
            "mean_ms","std_ms","p50_ms","p95_ms",
            "peak_mem_bytes","theoretical_peak_bytes","est_traffic_bytes","flops","tflops_per_s",
        ]
        rows.append(",".join(header))

    print(f"GPU: {torch.cuda.get_device_name()} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"H={H} D={D} dtype={args.dtype} ranks={ranks} B={args.batch_sizes} L={args.seq_lens}")
    print("")

    for B in args.batch_sizes:
        for L in args.seq_lens:
            # Base tensors
            X  = torch.randn(B,L,H, device=device, dtype=dtype)
            for R in ranks:
                # Factors
                U1 = torch.randn(H, R,   device=device, dtype=dtype) / math.sqrt(H)
                V1 = torch.randn(R, 2*D, device=device, dtype=dtype) / math.sqrt(R)
                U2 = torch.randn(D, R,   device=device, dtype=dtype) / math.sqrt(D)
                V2 = torch.randn(R, H,   device=device, dtype=dtype) / math.sqrt(R)
                b1 = torch.zeros(2*D, device=device, dtype=dtype)
                b2 = torch.zeros(H,   device=device, dtype=dtype)

                P = X.matmul(U1)

                # Resident memory breakdown (before calls)
                resident_bytes = {
                    "X": nbytes_of(X),
                    "P": nbytes_of(P),
                    "U1": nbytes_of(U1),
                    "V1": nbytes_of(V1),
                    "U2": nbytes_of(U2),
                    "V2": nbytes_of(V2),
                    "b1": nbytes_of(b1),
                    "b2": nbytes_of(b2),
                }
                total_resident = sum(resident_bytes.values())

                def fn_pt():
                    return _pt_baseline_swiglu(P, V1, U2, V2, b1, b2)
                def fn_tr():
                    return flashsvd_ffn_swiglu(P, V1, U2, V2, b1, b2, use_autotune=True)

                # Accuracy (relative Frobenius) once per config
                with torch.no_grad():
                    Y_pt0 = fn_pt().float()
                    Y_tr0 = fn_tr().float()
                    diff0 = Y_tr0 - Y_pt0
                    rel_frob = (diff0.norm() / (Y_pt0.norm() + 1e-12)).item()
                    max_abs  = diff0.abs().max().item()

                # PyTorch baseline (peak details after warm compile via accuracy path)
                det_pt = peak_details(fn_pt, device)
                st_pt = bench_cuda(fn_pt, args.warmup, args.iters)
                fl_pt = flops_swiglu_dense(B,L,H,D,R)
                tr_pt = traffic_swiglu_dense_bytes(B,L,H,D,dtype)
                pk_theo_pt = int(theoretical_peak_bytes_baseline(B,L,H,D,R, dtype))
                tfl_pt = fl_pt / (st_pt['mean_ms']/1e3) / 1e12

                print(f"[L={L} R={R}] Resident: "
                      f"X {mib(resident_bytes['X']):.2f} MiB, P {mib(resident_bytes['P']):.2f}, "
                      f"U1 {mib(resident_bytes['U1']):.2f}, V1 {mib(resident_bytes['V1']):.2f}, "
                      f"U2 {mib(resident_bytes['U2']):.2f}, V2 {mib(resident_bytes['V2']):.2f}, "
                      f"b1 {mib(resident_bytes['b1']):.2f}, b2 {mib(resident_bytes['b2']):.2f} | "
                      f"total {mib(total_resident):.2f} MiB")

                print(f"[L={L} R={R}] PyTorch SwiGLU: mean {st_pt['mean_ms']:.2f} ms | "
                      f"peak Δalloc {mib(det_pt['delta_alloc']):.2f} MiB (abs {mib(det_pt['peak_alloc']):.2f}) | "
                      f"Δres {mib(det_pt['delta_res']):.2f} MiB (abs {mib(det_pt['peak_res']):.2f}) | "
                      f"theor {mib(pk_theo_pt):.2f} MiB")
                if args.csv:
                    rows.append(
                        ",".join(map(str,[
                            "pt_swiglu",B,L,H,D,R,args.dtype,
                            f"{st_pt['mean_ms']:.4f}",f"{st_pt['std_ms']:.4f}",f"{st_pt['p50_ms']:.4f}",f"{st_pt['p95_ms']:.4f}",
                            det_pt['peak_alloc'], pk_theo_pt, tr_pt, fl_pt, f"{tfl_pt:.6f}",
                        ]))
                    )

                # Triton FlashSVD SwiGLU (peak details after warm compile via accuracy path)
                det_tr = peak_details(fn_tr, device)
                st_tr = bench_cuda(fn_tr, args.warmup, args.iters)
                fl_tr = flops_swiglu_dense(B,L,H,D,R)  # comparable model-level
                tr_tr = traffic_swiglu_triton_bytes(B,L,H,R,dtype)
                pk_theo_tr = int(theoretical_peak_bytes_triton(B,L,H,R, dtype, store_s_fp32=False))
                tfl_tr = fl_tr / (st_tr['mean_ms']/1e3) / 1e12

                print(f"[L={L} R={R}] FlashSVD SwiGLU: mean {st_tr['mean_ms']:.2f} ms | "
                      f"peak Δalloc {mib(det_tr['delta_alloc']):.2f} MiB (abs {mib(det_tr['peak_alloc']):.2f}) | "
                      f"Δres {mib(det_tr['delta_res']):.2f} MiB (abs {mib(det_tr['peak_res']):.2f}) | "
                      f"theor {mib(pk_theo_tr):.2f} MiB | rel_frob {rel_frob:.3e} (max_abs {max_abs:.3e})")
                if args.csv:
                    rows.append(
                        ",".join(map(str,[
                            "flashsvd_swiglu",B,L,H,D,R,args.dtype,
                            f"{st_tr['mean_ms']:.4f}",f"{st_tr['std_ms']:.4f}",f"{st_tr['p50_ms']:.4f}",f"{st_tr['p95_ms']:.4f}",
                            det_tr['peak_alloc'], pk_theo_tr, tr_tr, fl_tr, f"{tfl_tr:.6f}",
                        ]))
                    )

            print("")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True) if args.csv and os.path.dirname(args.csv) else None
        with open(args.csv, "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
