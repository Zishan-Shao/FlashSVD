#!/usr/bin/env python3
"""
benchmark_long_context_attn.py — Long-context attention benchmark (FlashAttention vs PyTorch vs FlashSVD)

This script profiles attention kernels on long sequence lengths (e.g., 1024, 2048, 4096):
- PyTorch SDPA (math kernel forced)
- Triton FlashAttention (repo kernel)
- FlashSVD attention (repo kernel)

It records for each configuration/method:
- Latency: mean, std, p50, p95 (with warmup and repeated runs)
- Peak memory (measured)
- Estimated memory traffic (bytes moved; theoretical, model-level)
- Theoretical FLOPs (algorithm-level)

CLI flags let you control batch size(s), sequence length(s), heads, head dim or model dim, rank(s), dtype,
warmup/iters, and result CSV output.

Notes:
- Memory traffic and FLOPs are theoretical estimates per algorithm and do not capture kernel tiling/recompute.
- Peak memory is measured using CUDA allocator statistics.


python benchmark_long_context_attn.py --batch-sizes 8 --heads 12 --head-dim 64 --attn-ranks 32 --seq-lens 2048

"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Prefer encoder_kernel kernels shipped with benchmark; fallback to src if needed
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
ENC_DIR = os.path.join(BENCH_DIR, "encoder_kernel")
DEC_DIR = os.path.join(BENCH_DIR, "decoder_kernel")
for p in (DEC_DIR, ENC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    # Prefer decoder_kernel causal FlashAttention
    from flash_attn_causal import flash_attn_triton  # type: ignore
except Exception as e:
    # Fallback to src
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from src.kernels.flash_attn_triton import flash_attn_triton  # type: ignore

try:
    # Try FlashSVD attention from encoder_kernel
    from flashsvdattn import flash_svd_attention  # type: ignore
except Exception:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    # Fallback to src version
    from src.kernels.flashsvdattn import flash_svd_attention  # type: ignore


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
        # touch result to discourage DCE
        _ = out.view(-1)[0].item()

    times.sort()
    mean_ms = float(sum(times) / len(times))
    p50_ms = float(times[len(times) // 2])
    p95_ms = float(times[int(0.95 * len(times)) - 1])
    # sample std
    if len(times) > 1:
        mean = mean_ms
        var = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
        std_ms = var ** 0.5
    else:
        std_ms = 0.0
    return {"mean_ms": mean_ms, "std_ms": std_ms, "p50_ms": p50_ms, "p95_ms": p95_ms, "all_ms": times}


def elem_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


# ============================
# Theoretical FLOPs estimates
# ============================

def flops_sdpa(B: int, H: int, S: int, D: int) -> int:
    """Theoretical FLOPs for standard SDPA (QK^T and AV) + softmax overhead.
    - QK^T: 2*B*H*S*S*D (mul+add)
    - AV:   2*B*H*S*S*D
    - Softmax (per row): ~5*B*H*S*S (max, sub, exp, sum, div)
    """
    return 2 * B * H * S * S * D + 2 * B * H * S * S * D + 5 * B * H * S * S


def flops_flashattn(B: int, H: int, S: int, D: int) -> int:
    """Algorithmic FLOPs comparable to SDPA (no attn matrix materialization)."""
    return flops_sdpa(B, H, S, D)


def flops_flashsvd_ideal(B: int, H: int, S: int, D: int, R: int) -> int:
    """Idealized FlashSVD attention FLOPs (rank-space formulation).
    This ignores kernel tile recompute. Model-level:
      - Build Q/K/V from rank factors: ~6 * B * H * S * R * D
      - QK^T in rank space:            ~2 * B * H * S * S * R
      - AV in rank space:              ~2 * B * H * S * R * D
      - Softmax overhead:              ~5 * B * H * S * S
    Total = 6*BH S R D + 2*BH S S R + 2*BH S R D + 5*BH S S
          = 8*BH S R D + 2*BH S S R + 5*BH S S
    """
    return 8 * B * H * S * R * D + 2 * B * H * S * S * R + 5 * B * H * S * S


# ============================
# Estimated memory traffic
# ============================

def traffic_sdpa_bytes(B: int, H: int, S: int, D: int, dtype: torch.dtype) -> int:
    """Approximate bytes moved for SDPA math kernel.
    Reads Q,K,V and writes Out once; materializes attention matrix (read+write).
    """
    e = elem_size(dtype)
    qkv_out = e * (3 * B * H * S * D + B * H * S * D)
    attn_rw = e * (2 * B * H * S * S)  # read+write attn matrix
    return qkv_out + attn_rw


def traffic_flashattn_bytes(B: int, H: int, S: int, D: int, dtype: torch.dtype) -> int:
    """Approximate bytes moved for FlashAttention (streaming, no attn matrix)."""
    e = elem_size(dtype)
    return e * (4 * B * H * S * D)  # Q,K,V read + Out write


def traffic_flashsvd_bytes(B: int, H: int, S: int, D: int, R: int, dtype: torch.dtype) -> int:
    """Approximate bytes moved for FlashSVD (rank factors, minimal model-level traffic).
    Does not account for tile recompute; counts reading P/V/b for q,k,v and writing Out.
    """
    e = elem_size(dtype)
    per_stream = (B * H * S * R) + (B * H * R * D) + (B * H * D)  # P + V + b
    total_streams = 3 * per_stream  # q,k,v
    out = B * H * S * D
    return e * (total_streams + out)


# ============================
# Theoretical peak memory (activations only)
# ============================

def peak_sdpa_bytes(B: int, H: int, S: int, D: int, dtype: torch.dtype) -> int:
    """Theory peak for SDPA-math: attn matrix + output (excludes inputs/weights)."""
    e = elem_size(dtype)
    attn = B * H * S * S
    out = B * H * S * D
    return e * (attn + out)


def peak_flashattn_bytes(B: int, H: int, S: int, D: int, dtype: torch.dtype) -> int:
    """Theory peak for FlashAttention (streaming): output + small row buffers."""
    e = elem_size(dtype)
    out = B * H * S * D
    row = B * H * S  # l_i/m_i rows (approx)
    return e * (out + row)


def peak_flashsvd_bytes(B: int, H: int, S: int, D: int, R: int, dtype: torch.dtype) -> int:
    """Theory peak for FlashSVD attention: output + optional rank-space buffer."""
    e = elem_size(dtype)
    out = B * H * S * D
    rank_buf = B * H * S * R
    return e * (out + rank_buf)


# ============================
# Data generation
# ============================

@dataclass
class Problem:
    B: int
    H: int
    S: int
    D: int
    R: int
    dtype: torch.dtype
    device: torch.device


def make_inputs(problem: Problem):
    B, H, S, D, R = problem.B, problem.H, problem.S, problem.D, problem.R
    dtype, device = problem.dtype, problem.device

    # Q/K/V for SDPA + FlashAttention: [B,H,S,D]
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # Mask [B,1,1,S] — all valid by default (long-context fully utilized)
    attn_mask = torch.ones(B, 1, 1, S, device=device, dtype=torch.bool)

    # FlashSVD factors
    Pq = torch.randn(B, H, S, R, device=device, dtype=dtype)
    Vq = torch.randn(B, H, R, D, device=device, dtype=dtype)
    bq = torch.zeros(B, H, D, device=device, dtype=dtype)
    Pk = torch.randn(B, H, S, R, device=device, dtype=dtype)
    Vk = torch.randn(B, H, R, D, device=device, dtype=dtype)
    bk = torch.zeros(B, H, D, device=device, dtype=dtype)
    Pv = torch.randn(B, H, S, R, device=device, dtype=dtype)
    Vv = torch.randn(B, H, R, D, device=device, dtype=dtype)
    bv = torch.zeros(B, H, D, device=device, dtype=dtype)

    return {
        "Q": Q, "K": K, "V": V,
        "mask": attn_mask,
        "Pq": Pq, "Vq": Vq, "bq": bq,
        "Pk": Pk, "Vk": Vk, "bk": bk,
        "Pv": Pv, "Vv": Vv, "bv": bv,
    }


# ============================
# Runners for each method
# ============================

def run_torch_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    B, H, S, D = Q.shape
    # SDPA expects [B*H, S, D]
    Q_ = Q.reshape(B * H, S, D)
    K_ = K.reshape(B * H, S, D)
    V_ = V.reshape(B * H, S, D)
    # Build attn mask [B*H, S, S]: allow all keys; no causal mask
    if mask is not None:
        # mask: [B,1,1,S] -> [B,H,1,S] -> broadcast to [B,H,S,S] via key dimension
        mk = mask.expand(B, H, 1, S).reshape(B * H, 1, S)
        attn_mask = mk.expand(B * H, S, S)
    else:
        attn_mask = None

    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        out = F.scaled_dot_product_attention(Q_, K_, V_, attn_mask=attn_mask, is_causal=False)
    return out.reshape(B, H, S, D)


def run_flashattn_triton(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    return flash_attn_triton(Q, K, V, mask)


def run_flashsvd(Pq, Vq, bq, Pk, Vk, bk, Pv, Vv, bv, mask):
    return flash_svd_attention(Pq, Vq, bq, Pk, Vk, bk, Pv, Vv, bv, mask)


# ============================
# Main benchmark
# ============================

def format_bytes(n: int) -> str:
    return f"{n / (1024**2):.2f} MiB"


def run_once_peakmem(fn) -> int:
    torch.cuda.reset_peak_memory_stats()
    _ = fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def main():
    parser = argparse.ArgumentParser("Long-context attention benchmark: PyTorch vs FlashAttention vs FlashSVD")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension (overridden by --head-dim if given)")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=None, help="Per-head dimension; if set, d_model=heads*head_dim")
    # Separate integer ranks for attention and FFN; enforce caps later.
    parser.add_argument("--attn-ranks", type=int, nargs="+", default=[32], help="FlashSVD attention ranks (per-head)")
    parser.add_argument("--ffn-ranks", type=int, nargs="+", default=[384], help="FlashSVD FFN ranks (context only; capped <=768)")
    # Back-compat: --ranks merges into --attn-ranks if provided
    parser.add_argument("--ranks", type=int, nargs="+", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1], help="Batch sizes")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096], help="Sequence lengths")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Data type")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--flash-block-m", type=int, default=64, help="Triton FlashAttention block M (fallback if needed)")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("CUDA not available; please run on a CUDA-enabled machine.")
        sys.exit(1)

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    # Resolve dims
    if args.head_dim is not None:
        D = int(args.head_dim)
        d_model = args.heads * D
    else:
        d_model = int(args.d_model)
        D = d_model // int(args.heads)
    H = int(args.heads)

    # Merge back-compat --ranks into attn ranks if provided
    attn_ranks = args.attn_ranks if args.ranks is None else list(args.attn_ranks) + list(args.ranks)
    # Enforce integer caps: attn rank per-head <= min(D,64); FFN rank <= 768
    attn_ranks = sorted({max(1, min(int(r), D, 64)) for r in attn_ranks})
    ffn_ranks = sorted({max(1, min(int(r), 768)) for r in (args.ffn_ranks or [])})
    batch_sizes = [int(x) for x in args.batch_sizes]
    seq_lens = [int(x) for x in args.seq_lens]

    print(f"GPU: {torch.cuda.get_device_name()} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"dtype={args.dtype} device={device} heads={H} head_dim={D} (d_model={d_model})")
    print(f"attn_ranks={attn_ranks} ffn_ranks={ffn_ranks} batch_sizes={batch_sizes} seq_lens={seq_lens}")
    print("")

    # Optional CSV
    rows: List[str] = []
    if args.csv:
        header = [
            "method", "B", "H", "S", "D", "R_attn", "R_ffn", "dtype", "mean_ms", "std_ms", "p50_ms", "p95_ms",
            "peak_mem_bytes", "theoretical_peak_bytes", "est_traffic_bytes", "flops", "tflops_per_s",
        ]
        rows.append(",".join(header))

    for B in batch_sizes:
        for S in seq_lens:
            print(f"=== Long-context: B={B} S={S} H={H} D={D} ===")
            # Prepare rank-space factors once for the maximum attn rank of this sweep
            maxRa = max(attn_ranks) if attn_ranks else D
            problem_base = Problem(B=B, H=H, S=S, D=D, R=maxRa, dtype=dtype, device=device)
            base = make_inputs(problem_base)
            mask = base["mask"]

            # --- PyTorch SDPA (math) ---
            # Reconstruct Q,K,V locally from rank factors for fairness
            def _reconstruct_qkv(R:int):
                Pq, Vq, bq = base["Pq"][..., :R], base["Vq"][..., :R, :], base["bq"]
                Pk, Vk, bk = base["Pk"][..., :R], base["Vk"][..., :R, :], base["bk"]
                Pv, Vv, bv = base["Pv"][..., :R], base["Vv"][..., :R, :], base["bv"]
                Q = (Pq.float().reshape(B*H, S, R) @ Vq.float().reshape(B*H, R, D)).view(B,H,S,D) + bq.view(B,H,1,D).float()
                K = (Pk.float().reshape(B*H, S, R) @ Vk.float().reshape(B*H, R, D)).view(B,H,S,D) + bk.view(B,H,1,D).float()
                V = (Pv.float().reshape(B*H, S, R) @ Vv.float().reshape(B*H, R, D)).view(B,H,S,D) + bv.view(B,H,1,D).float()
                return Q.to(dtype), K.to(dtype), V.to(dtype)

            # Choose a representative attention rank for baseline SDPA/FlashAttention (use max rank)
            Ra_base = maxRa
            def fn_torch():
                Qr, Kr, Vr = _reconstruct_qkv(Ra_base)
                return run_torch_sdpa(Qr, Kr, Vr, mask)
            peak_torch = run_once_peakmem(fn_torch)
            stats_torch = bench_cuda(fn_torch, n_warmup=args.warmup, n_runs=args.iters)
            flops_t = flops_sdpa(B, H, S, D)
            # Estimated traffic includes local reconstruction of Q/K/V from rank factors
            def _traffic_reconstruct_bytes(R:int):
                e = elem_size(dtype)
                # Read P (3 streams), V (3 streams), biases (3 streams), write Q,K,V, read Q,K,V, write Out
                read_rank = 3 * (B*H*S*R + B*H*R*D + B*H*D)
                mat_qkv = 3 * B*H*S*D  # Q,K,V materialized
                out = B*H*S*D
                return e * (read_rank + mat_qkv + mat_qkv + out)
            traff_t = _traffic_reconstruct_bytes(Ra_base)
            tflops_t = flops_t / (stats_torch["mean_ms"] / 1e3) / 1e12
            theo_peak_t = peak_sdpa_bytes(B, H, S, D, dtype)
            print(
                f"PyTorch SDPA:   mean {stats_torch['mean_ms']:.2f} ms ±{stats_torch['std_ms']:.2f} | "
                f"p50 {stats_torch['p50_ms']:.2f} p95 {stats_torch['p95_ms']:.2f} | "
                f"peak {format_bytes(peak_torch)} (theor {format_bytes(theo_peak_t)}) | est_traffic {format_bytes(traff_t)} | "
                f"FLOPs {flops_t/1e9:.2f} GF | {tflops_t:.2f} TF/s"
            )
            if args.csv:
                rows.append(
                    ",".join(map(str, [
                        "torch_sdpa", B, H, S, D, Ra_base, 0, args.dtype,
                        f"{stats_torch['mean_ms']:.4f}", f"{stats_torch['std_ms']:.4f}",
                        f"{stats_torch['p50_ms']:.4f}", f"{stats_torch['p95_ms']:.4f}",
                        peak_torch, theo_peak_t, traff_t, flops_t, f"{tflops_t:.6f}",
                    ]))
                )

            # --- Triton FlashAttention ---
            def fn_flash():
                Qr, Kr, Vr = _reconstruct_qkv(Ra_base)
                return run_flashattn_triton(Qr, Kr, Vr, mask)
            # Try measuring; if an illegal memory access occurs, retry with larger BLOCK_M
            try:
                peak_flash = run_once_peakmem(fn_flash)
                stats_flash = bench_cuda(fn_flash, n_warmup=args.warmup, n_runs=args.iters)
            except RuntimeError:
                def fn_flash_blk():
                    Qr, Kr, Vr = _reconstruct_qkv(Ra_base)
                    # Reuse wrapper but pass block size via direct call to underlying function
                    return flash_attn_triton(Qr, Kr, Vr, mask, BLOCK_M=args.flash_block_m)
                peak_flash = run_once_peakmem(fn_flash_blk)
                stats_flash = bench_cuda(fn_flash_blk, n_warmup=args.warmup, n_runs=args.iters)
            flops_f = flops_flashattn(B, H, S, D)
            traff_f = _traffic_reconstruct_bytes(Ra_base)
            tflops_f = flops_f / (stats_flash["mean_ms"] / 1e3) / 1e12
            theo_peak_f = peak_flashattn_bytes(B, H, S, D, dtype)
            print(
                f"FlashAttention: mean {stats_flash['mean_ms']:.2f} ms ±{stats_flash['std_ms']:.2f} | "
                f"p50 {stats_flash['p50_ms']:.2f} p95 {stats_flash['p95_ms']:.2f} | "
                f"peak {format_bytes(peak_flash)} (theor {format_bytes(theo_peak_f)}) | est_traffic {format_bytes(traff_f)} | "
                f"FLOPs {flops_f/1e9:.2f} GF | {tflops_f:.2f} TF/s"
            )
            if args.csv:
                rows.append(
                    ",".join(map(str, [
                        "flashattention_triton", B, H, S, D, Ra_base, 0, args.dtype,
                        f"{stats_flash['mean_ms']:.4f}", f"{stats_flash['std_ms']:.4f}",
                        f"{stats_flash['p50_ms']:.4f}", f"{stats_flash['p95_ms']:.4f}",
                        peak_flash, theo_peak_f, traff_f, flops_f, f"{tflops_f:.6f}",
                    ]))
                )

            # --- FlashSVD (loop ranks) ---
            for R in attn_ranks:
                if R > D:
                    continue
                # Re-make inputs only for R-dependent factors, reuse Q/K/V
                # (slice preallocated tensors to avoid extra allocs)
                inp = {
                    "Pq": base["Pq"][..., :R], "Vq": base["Vq"][..., :R, :], "bq": base["bq"],
                    "Pk": base["Pk"][..., :R], "Vk": base["Vk"][..., :R, :], "bk": base["bk"],
                    "Pv": base["Pv"][..., :R], "Vv": base["Vv"][..., :R, :], "bv": base["bv"],
                    "mask": mask,
                }
                fn_svd = lambda: run_flashsvd(
                    inp["Pq"], inp["Vq"], inp["bq"], inp["Pk"], inp["Vk"], inp["bk"], inp["Pv"], inp["Vv"], inp["bv"], inp["mask"]
                )
                peak_svd = run_once_peakmem(fn_svd)
                stats_svd = bench_cuda(fn_svd, n_warmup=args.warmup, n_runs=args.iters)
                flops_s = flops_flashsvd_ideal(B, H, S, D, R)
                traff_s = traffic_flashsvd_bytes(B, H, S, D, R, dtype)
                tflops_s = flops_s / (stats_svd["mean_ms"] / 1e3) / 1e12
                theo_peak_s = peak_flashsvd_bytes(B, H, S, D, R, dtype)
                print(
                    f"FlashSVD (R={R:>3}): mean {stats_svd['mean_ms']:.2f} ms ±{stats_svd['std_ms']:.2f} | "
                    f"p50 {stats_svd['p50_ms']:.2f} p95 {stats_svd['p95_ms']:.2f} | "
                    f"peak {format_bytes(peak_svd)} (theor {format_bytes(theo_peak_s)}) | est_traffic {format_bytes(traff_s)} | "
                    f"FLOPs {flops_s/1e9:.2f} GF | {tflops_s:.2f} TF/s"
                )
                if args.csv:
                    rows.append(
                        ",".join(map(str, [
                            f"flashsvd_r{R}", B, H, S, D, R, 0, args.dtype,
                            f"{stats_svd['mean_ms']:.4f}", f"{stats_svd['std_ms']:.4f}",
                            f"{stats_svd['p50_ms']:.4f}", f"{stats_svd['p95_ms']:.4f}",
                            peak_svd, theo_peak_s, traff_s, flops_s, f"{tflops_s:.6f}",
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
