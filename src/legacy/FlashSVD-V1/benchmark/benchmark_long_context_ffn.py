#!/usr/bin/env python3
"""
benchmark_long_context_ffn.py — Long-context FFN benchmark (Dense vs FlashSVD FFN)

Profiles FFN compute for long sequence lengths (e.g., 1024, 2048, 4096) comparing:
- Dense PyTorch FFN (Linear -> GELU -> Linear)
- FlashSVD FFN kernels from benchmark/encoder_kernel (fused and v1)

For each configuration, records:
- Latency: mean, std, p50, p95 (CUDA events; warmup + repeated runs)
- Peak memory (measured)
- Estimated memory traffic (activations only)
- Theoretical FLOPs (model-level)

Notes:
- Weights are generated so that Dense and FlashSVD produce identical results: W1 = U1@V1, W2 = U2@V2.
- Memory traffic ignores weights; it estimates activation tensor movement for a forward pass.


python benchmark_long_context_ffn.py --batch-sizes 8 --d-model 768 --d-ff 3072 --ranks 96 192 384 768 --batch-sizes 2 16 --seq-lens 1024 2048 4096 --dtype float16 --warmup 10 --iters 50 --kernels fused,v1 --bl 64 --bd 128 --bh 64 --br1 32 --br2 32 --csv benchmark/long_context_ffn.csv

"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F

# Make benchmark/encoder_kernel importable
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
ENC_DIR = os.path.join(BENCH_DIR, "encoder_kernel")
if ENC_DIR not in sys.path:
    sys.path.insert(0, ENC_DIR)

try:
    # Both variants available under encoder_kernel
    from flashsvdffn import flashsvd_ffn as flashsvd_ffn_fused
    from flashsvdffnv1 import flashsvd_ffn_v1
except Exception as e:
    print("Error importing FlashSVD FFN kernels from benchmark/encoder_kernel:", e)
    sys.exit(1)


# ============================
# Utilities: timing & stats
# ============================

@torch.no_grad()
def bench_cuda(fn, n_warmup: int = 10, n_runs: int = 50) -> Dict[str, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(n_warmup):
        _ = fn()
    torch.cuda.synchronize()

    for _ in range(n_runs):
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
        _ = out.view(-1)[0].item()  # touch result

    times.sort()
    mean_ms = float(sum(times) / len(times))
    p50_ms = float(times[len(times) // 2])
    p95_ms = float(times[int(0.95 * len(times)) - 1])
    if len(times) > 1:
        var = sum((t - mean_ms) ** 2 for t in times) / (len(times) - 1)
        std_ms = var ** 0.5
    else:
        std_ms = 0.0
    return {"mean_ms": mean_ms, "std_ms": std_ms, "p50_ms": p50_ms, "p95_ms": p95_ms, "all_ms": times}


def elem_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def run_once_peakmem(fn) -> int:
    torch.cuda.reset_peak_memory_stats()
    _ = fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


# ============================
# Theoretical FLOPs estimates
# ============================

def flops_ffn_dense(B: int, L: int, H: int, D: int) -> int:
    # Two GEMMs + GELU
    return 2 * B * L * H * D + 2 * B * L * D * H + B * L * D


def flops_ffn_flashsvd(B: int, L: int, H: int, D: int, R1: int, R2: int) -> int:
    # x@U1 + P@V1 + GELU + @U2 + @V2
    return 2 * B * L * (H * R1 + R1 * D + D * R2 + R2 * H) + B * L * D


# ============================
# Estimated memory traffic (activations only)
# ============================

def traffic_ffn_dense_bytes(B: int, L: int, H: int, D: int, dtype: torch.dtype) -> int:
    e = elem_size(dtype)
    # Read X and write/read hidden and write out
    return e * (2 * B * L * H + 2 * B * L * D)


def traffic_ffn_flashsvd_v1_bytes(B: int, L: int, H: int, R1: int, R2: int, dtype: torch.dtype) -> int:
    e = elem_size(dtype)
    # Read X, write P, write S, write C
    return e * (2 * B * L * H + B * L * (R1 + R2))


def traffic_ffn_flashsvd_fused_bytes(B: int, L: int, H: int, R1: int, dtype: torch.dtype) -> int:
    e = elem_size(dtype)
    # Read X, write P, write C (no S materialization)
    return e * (2 * B * L * H + B * L * R1)


# ============================
# Theoretical peak memory (activations only)
# ============================

def peak_ffn_dense_bytes(B: int, L: int, H: int, D: int, dtype: torch.dtype) -> int:
    """Dense FFN theoretical activation peak: max(Z+Ht, Y+Ht)."""
    e = elem_size(dtype)
    z = B * L * D
    ht = B * L * D
    y = B * L * H
    return e * max(z + ht, y + ht)


def peak_ffn_flashsvd_v1_bytes(B: int, L: int, H: int, R: int, dtype: torch.dtype) -> int:
    """FlashSVD v1: stores S [B,L,R] then Y [B,L,H] (S+Y)."""
    e = elem_size(dtype)
    s = B * L * R
    y = B * L * H
    return e * (s + y)


def peak_ffn_flashsvd_fused_bytes(B: int, L: int, H: int, dtype: torch.dtype) -> int:
    """FlashSVD fused: no S materialization; dominant is Y."""
    e = elem_size(dtype)
    y = B * L * H
    return e * y


# ============================
# Workloads
# ============================

def dense_ffn_forward(X, W1, b1, W2, b2):
    return (F.gelu(X @ W1 + b1.view(1, 1, -1)) @ W2) + b2.view(1, 1, -1)


def main():
    parser = argparse.ArgumentParser("Long-context FFN benchmark: Dense vs FlashSVD (fused/v1)")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension H")
    parser.add_argument("--d-ff", type=int, default=3072, help="FFN hidden dimension D")
    parser.add_argument("--ranks", type=int, nargs="+", default=[192, 384], help="Ranks R (R1=R2=R)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1], help="Batch sizes")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096], help="Sequence lengths")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Data type")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--kernels", type=str, default="fused,v1", choices=[
        "fused", "v1", "v2",
        "fused,v1", "v1,fused",
        "v1,v2", "v2,v1",
        "fused,v2", "v2,fused",
    ], help="Which FlashSVD kernels to benchmark (legacy; use --ffn-version instead)")
    parser.add_argument("--ffn-version", type=str, default=None, choices=["v1", "v2", "both"], help="Choose FlashSVDFFN version(s) to compare; overrides --kernels")
    # Optional tiling controls (useful for very long sequences)
    parser.add_argument("--bl", type=int, default=64, help="Tile size L (BL)")
    parser.add_argument("--bd", type=int, default=128, help="Tile size D (BD)")
    parser.add_argument("--bh", type=int, default=64, help="Tile size H/output (BH) for fused kernel")
    parser.add_argument("--br1", type=int, default=32, help="Tile size rank-1 (BR1)")
    parser.add_argument("--br2", type=int, default=32, help="Tile size rank-2 (BR2)")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("CUDA not available; please run on a CUDA-enabled machine.")
        sys.exit(1)

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    H = int(args.d_model)
    D = int(args.d_ff)
    # Enforce rank cap at most 768
    ranks = sorted({max(1, min(int(r), 768)) for r in args.ranks})
    batch_sizes = [int(x) for x in args.batch_sizes]
    seq_lens = [int(x) for x in args.seq_lens]
    # Determine which FlashSVD versions to run
    if args.ffn_version is not None:
        do_v1 = args.ffn_version in ("v1", "both")
        do_fused = args.ffn_version in ("v2", "both")
    else:
        tokens = [t.strip() for t in args.kernels.split(",")]
        do_fused = ("fused" in tokens) or ("v2" in tokens)
        do_v1 = ("v1" in tokens)

    print(f"GPU: {torch.cuda.get_device_name()} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"dtype={args.dtype} device={device} H(d_model)={H} D(ffn)={D}")
    chosen = ("v1" if do_v1 else "") + ("," if do_v1 and do_fused else "") + ("v2" if do_fused else "")
    print(f"ranks={ranks} batch_sizes={batch_sizes} seq_lens={seq_lens} versions={chosen or 'none'}")
    print("")

    # Optional CSV
    rows: List[str] = []
    if args.csv:
        header = [
            "method", "B", "L", "H", "D", "R1", "R2", "dtype",
            "mean_ms", "std_ms", "p50_ms", "p95_ms",
            "peak_mem_bytes", "theoretical_peak_bytes", "est_traffic_bytes", "flops", "tflops_per_s",
        ]
        rows.append(",".join(header))

    for B in batch_sizes:
        for L in seq_lens:
            print(f"=== Long-context FFN: B={B} L={L} H={H} D={D} ===")
            # Create random weights once per (B,L) group
            U1 = torch.randn(H, max(ranks), device=device, dtype=dtype)
            V1_big = torch.randn(max(ranks), D, device=device, dtype=dtype)
            U2_big = torch.randn(D, max(ranks), device=device, dtype=dtype)
            V2 = torch.randn(max(ranks), H, device=device, dtype=dtype)
            b1 = torch.zeros(D, device=device, dtype=dtype)
            b2 = torch.zeros(H, device=device, dtype=dtype)
            X = torch.randn(B, L, H, device=device, dtype=dtype)

            # Dense baseline weights (match FlashSVD factorization at max rank)
            # W1 = U1[:,:R] @ V1[:R]; W2 = U2[:,:R] @ V2[:R], but to keep one baseline across ranks,
            # we build with max-rank factors and rely on sub-ranks using slices.
            W1_full = (U1 @ V1_big).contiguous()
            W2_full = (U2_big @ V2).contiguous()

            # Baseline runner
            fn_dense = lambda: dense_ffn_forward(X, W1_full, b1, W2_full, b2)
            peak_dense = run_once_peakmem(fn_dense)
            stats_dense = bench_cuda(fn_dense, n_warmup=args.warmup, n_runs=args.iters)
            flops_d = flops_ffn_dense(B, L, H, D)
            traff_d = traffic_ffn_dense_bytes(B, L, H, D, dtype)
            theo_peak_d = peak_ffn_dense_bytes(B, L, H, D, dtype)
            tflops_d = flops_d / (stats_dense["mean_ms"] / 1e3) / 1e12
            print(
                f"Dense FFN:      mean {stats_dense['mean_ms']:.2f} ms ±{stats_dense['std_ms']:.2f} | "
                f"p50 {stats_dense['p50_ms']:.2f} p95 {stats_dense['p95_ms']:.2f} | "
                f"peak {peak_dense/(1024**2):.2f} MiB (theor {theo_peak_d/(1024**2):.2f}) | est_traffic {traff_d/(1024**2):.2f} MiB | "
                f"FLOPs {flops_d/1e9:.2f} GF | {tflops_d:.2f} TF/s"
            )
            if args.csv:
                rows.append(
                    ",".join(map(str, [
                        "dense_ffn", B, L, H, D, 0, 0, args.dtype,
                        f"{stats_dense['mean_ms']:.4f}", f"{stats_dense['std_ms']:.4f}",
                        f"{stats_dense['p50_ms']:.4f}", f"{stats_dense['p95_ms']:.4f}",
                        peak_dense, theo_peak_d, traff_d, flops_d, f"{tflops_d:.6f}",
                    ]))
                )

            # FlashSVD variants for each rank
            for R in ranks:
                R1 = R2 = R
                # Slice factor matrices per rank
                U1_r = U1[:, :R1]
                V1_r = V1_big[:R1, :]
                U2_r = U2_big[:, :R2]
                V2_r = V2[:R2, :]

                # P = X @ U1_r
                P = X @ U1_r

                if do_fused:
                    fn_fused = lambda: flashsvd_ffn_fused(P, V1_r, U2_r, V2_r, b1, b2, args.bl, args.bd, args.bh, args.br1, args.br2)
                    peak_fused = run_once_peakmem(fn_fused)
                    stats_fused = bench_cuda(fn_fused, n_warmup=args.warmup, n_runs=args.iters)
                    flops_f = flops_ffn_flashsvd(B, L, H, D, R1, R2)
                    traff_f = traffic_ffn_flashsvd_fused_bytes(B, L, H, R1, dtype)
                    theo_peak_f = peak_ffn_flashsvd_fused_bytes(B, L, H, dtype)
                    tflops_f = flops_f / (stats_fused["mean_ms"] / 1e3) / 1e12
                    print(
                        f"FlashSVD FFN (v2,   R={R:>3}): mean {stats_fused['mean_ms']:.2f} ms ±{stats_fused['std_ms']:.2f} | "
                        f"p50 {stats_fused['p50_ms']:.2f} p95 {stats_fused['p95_ms']:.2f} | "
                        f"peak {peak_fused/(1024**2):.2f} MiB (theor {theo_peak_f/(1024**2):.2f}) | est_traffic {traff_f/(1024**2):.2f} MiB | "
                        f"FLOPs {flops_f/1e9:.2f} GF | {tflops_f:.2f} TF/s"
                    )
                    if args.csv:
                        rows.append(
                            ",".join(map(str, [
                                f"flashsvd_ffn_v2_r{R}", B, L, H, D, R1, R2, args.dtype,
                                f"{stats_fused['mean_ms']:.4f}", f"{stats_fused['std_ms']:.4f}",
                                f"{stats_fused['p50_ms']:.4f}", f"{stats_fused['p95_ms']:.4f}",
                                peak_fused, theo_peak_f, traff_f, flops_f, f"{tflops_f:.6f}",
                            ]))
                        )

                if do_v1:
                    fn_v1 = lambda: flashsvd_ffn_v1(P, V1_r, U2_r, V2_r, b1, b2, args.bl, args.bd, args.br1, args.br2)
                    peak_v1 = run_once_peakmem(fn_v1)
                    stats_v1 = bench_cuda(fn_v1, n_warmup=args.warmup, n_runs=args.iters)
                    flops_v = flops_ffn_flashsvd(B, L, H, D, R1, R2)
                    traff_v = traffic_ffn_flashsvd_v1_bytes(B, L, H, R1, R2, dtype)
                    theo_peak_v = peak_ffn_flashsvd_v1_bytes(B, L, H, R, dtype)
                    tflops_v = flops_v / (stats_v1["mean_ms"] / 1e3) / 1e12
                    print(
                        f"FlashSVD FFN (v1,   R={R:>3}): mean {stats_v1['mean_ms']:.2f} ms ±{stats_v1['std_ms']:.2f} | "
                        f"p50 {stats_v1['p50_ms']:.2f} p95 {stats_v1['p95_ms']:.2f} | "
                        f"peak {peak_v1/(1024**2):.2f} MiB (theor {theo_peak_v/(1024**2):.2f}) | est_traffic {traff_v/(1024**2):.2f} MiB | "
                        f"FLOPs {flops_v/1e9:.2f} GF | {tflops_v:.2f} TF/s"
                    )
                    if args.csv:
                        rows.append(
                            ",".join(map(str, [
                                f"flashsvd_ffn_v1_r{R}", B, L, H, D, R1, R2, args.dtype,
                                f"{stats_v1['mean_ms']:.4f}", f"{stats_v1['std_ms']:.4f}",
                                f"{stats_v1['p50_ms']:.4f}", f"{stats_v1['p95_ms']:.4f}",
                                peak_v1, theo_peak_v, traff_v, flops_v, f"{tflops_v:.6f}",
                            ]))
                        )

            print("")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True) if os.path.dirname(args.csv) else None
        with open(args.csv, "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
