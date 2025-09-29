#!/usr/bin/env python3
import math
import statistics as stats
import argparse
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# ==============================================================
# Triton kernels (SwiGLU in rank space, P -> S in one shot)
# ==============================================================

@triton.jit
def _fused_ffn_phase1_swiglu_base(
    P_ptr, V1_ptr, U2_ptr, S_ptr, b1_ptr,
    B, L, D, R1, R2,
    sP_b, sP_l, sP_r1,
    sV1_r1, sV1_d,
    sU2_d, sU2_r2,
    sb1,
    sS_b, sS_l, sS_r2,
    BL: tl.constexpr, BD: tl.constexpr, BR1: tl.constexpr, BR2: tl.constexpr,
):
    pid_b  = tl.program_id(0)
    pid_l  = tl.program_id(1)
    pid_r2 = tl.program_id(2)

    offs_l  = pid_l * BL + tl.arange(0, BL)
    offs_r2 = pid_r2 * BR2 + tl.arange(0, BR2)

    Tu_acc = tl.zeros((BL, BD), dtype=tl.float32)
    Tv_acc = tl.zeros((BL, BD), dtype=tl.float32)
    acc    = tl.zeros((BL, BR2), dtype=tl.float32)

    for d0 in range(0, D, BD):
        d   = d0 + tl.arange(0, BD)
        m_d = d < D

        Tu_acc *= 0.0
        Tv_acc *= 0.0

        for r1_0 in range(0, R1, BR1):
            r1   = r1_0 + tl.arange(0, BR1)
            m_r1 = r1 < R1

            P_blk = tl.load(
                P_ptr + pid_b * sP_b + offs_l[:, None]*sP_l + r1[None, :]*sP_r1,
                mask=(offs_l[:, None] < L) & m_r1[None, :],
                other=0.0
            )

            V1u_blk = tl.load(
                V1_ptr + r1[:, None]*sV1_r1 + d[None, :]*sV1_d,
                mask=m_r1[:, None] & m_d[None, :],
                other=0.0
            )

            V1v_blk = tl.load(
                V1_ptr + r1[:, None]*sV1_r1 + (d[None, :] + D)*sV1_d,
                mask=m_r1[:, None] & m_d[None, :],
                other=0.0
            )

            # fp32 accumulate
            Tu_acc += tl.dot(P_blk, V1u_blk.to(P_blk.dtype)).to(tl.float32)
            Tv_acc += tl.dot(P_blk, V1v_blk.to(P_blk.dtype)).to(tl.float32)

        b1u = tl.load(b1_ptr + d*sb1,        mask=m_d, other=0.0).to(tl.float32)
        b1v = tl.load(b1_ptr + (d + D)*sb1,  mask=m_d, other=0.0).to(tl.float32)
        Tu  = Tu_acc + b1u[None, :]
        Tv  = Tv_acc + b1v[None, :]

        # SwiGLU: silu(Tu) * Tv, silu(z)=z*sigmoid(z); use stable branch
        sig = tl.where(Tu >= 0, 1.0 / (1.0 + tl.exp(-Tu)), tl.exp(Tu) / (1.0 + tl.exp(Tu)))
        H   = (Tu * sig) * Tv  # [BL, BD]

        U2_blk = tl.load(
            U2_ptr + d[:, None]*sU2_d + offs_r2[None, :]*sU2_r2,
            mask=m_d[:, None] & (offs_r2[None, :] < R2),
            other=0.0
        ).to(tl.float32)
        acc += tl.dot(H, U2_blk)

    mask = (offs_l[:, None] < L) & (offs_r2[None, :] < R2)
    tl.store(
        S_ptr + pid_b*sS_b + offs_l[:, None]*sS_l + offs_r2[None, :]*sS_r2,
        acc, mask=mask
    )


# ── Autotuning wrapper (configs control BL/BD/BR1/BR2 & warps/stages) ─────────
TUNE_FFN_SWIGLU = [
    triton.Config({'BL': 32,  'BD': 64,  'BR1': 64,  'BR2': 64},  num_warps=4, num_stages=2),
    triton.Config({'BL': 64,  'BD': 64,  'BR1': 64,  'BR2': 64},  num_warps=4, num_stages=2),
    triton.Config({'BL': 64,  'BD': 128, 'BR1': 64,  'BR2': 64},  num_warps=4, num_stages=2),
    triton.Config({'BL': 64,  'BD': 128, 'BR1': 64,  'BR2': 128}, num_warps=8, num_stages=2),
    triton.Config({'BL': 128, 'BD': 128, 'BR1': 64,  'BR2': 128}, num_warps=8, num_stages=2),
    # Larger tiles can OOR on some GPUs; add cautiously:
    # triton.Config({'BL': 128, 'BD': 256, 'BR1': 128, 'BR2': 128}, num_warps=8, num_stages=2),
]

@triton.autotune(configs=TUNE_FFN_SWIGLU, key=['D', 'R2'])
@triton.jit
def fused_ffn_phase1_swiglu_auto(
    P_ptr, V1_ptr, U2_ptr, S_ptr, b1_ptr,
    B, L, D, R1, R2,
    sP_b, sP_l, sP_r1,
    sV1_r1, sV1_d,
    sU2_d, sU2_r2,
    sb1,
    sS_b, sS_l, sS_r2,
    BL: tl.constexpr, BD: tl.constexpr, BR1: tl.constexpr, BR2: tl.constexpr,
):
    _fused_ffn_phase1_swiglu_base(
        P_ptr, V1_ptr, U2_ptr, S_ptr, b1_ptr,
        B, L, D, R1, R2,
        sP_b, sP_l, sP_r1,
        sV1_r1, sV1_d,
        sU2_d, sU2_r2,
        sb1,
        sS_b, sS_l, sS_r2,
        BL=BL, BD=BD, BR1=BR1, BR2=BR2,
    )


# ==============================================================
# Public wrapper
# ==============================================================

def flashsvd_ffn_swiglu(
    P, V1, U2, V2, b1, b2,
    BL=64, BD=64, BR1=64, BR2=64,
    *,
    store_s_fp32: bool = False,
    use_autotune: bool = True,
):
    """
    Computes:
      Z = P @ V1 + b1
      H = silu(Zu) * Zv
      S = H @ U2   (all inside kernel)
      Y = S @ V2 + b2
    """
    assert P.is_cuda and V1.is_cuda and U2.is_cuda and V2.is_cuda
    B, L, R1 = P.shape
    R1_v1, twoD = V1.shape
    D = twoD // 2
    assert R1_v1 == R1 and twoD == 2 * D
    D_u2, R2 = U2.shape
    assert D_u2 == D
    R2_v2, H = V2.shape
    assert R2_v2 == R2
    assert b1.shape[0] == 2*D and b2.shape[0] == H

    S_dtype = torch.float32 if store_s_fp32 else P.dtype
    S = torch.empty((B, L, R2), device=P.device, dtype=S_dtype)

    sP_b, sP_l, sP_r1 = P.stride()
    sV1_r1, sV1_d     = V1.stride()
    sU2_d, sU2_r2     = U2.stride()
    sb1               = b1.stride(0)
    sS_b, sS_l, sS_r2 = S.stride()

    grid = (B, triton.cdiv(L, BL), triton.cdiv(R2, BR2))

    if use_autotune:
        # IMPORTANT: do NOT pass BL/BD/BR*; autotuner provides them.
        fused_ffn_phase1_swiglu_auto[grid](
            P, V1, U2, S, b1,
            B, L, D, R1, R2,
            sP_b, sP_l, sP_r1,
            sV1_r1, sV1_d,
            sU2_d, sU2_r2,
            sb1,
            sS_b, sS_l, sS_r2,
        )
    else:
        _fused_ffn_phase1_swiglu_base[grid](
            P, V1, U2, S, b1,
            B, L, D, R1, R2,
            sP_b, sP_l, sP_r1,
            sV1_r1, sV1_d,
            sU2_d, sU2_r2,
            sb1,
            sS_b, sS_l, sS_r2,
            BL=BL, BD=BD, BR1=BR1, BR2=BR2,
        )

    Y = S.matmul(V2)
    Y = Y + b2.view(1, 1, -1)
    return Y


# ==============================================================
# PyTorch reference (SwiGLU in low rank)
# ==============================================================

def _pt_baseline_swiglu(P, V1, U2, V2, b1, b2):
    Z  = P.matmul(V1) + b1.view(1, 1, -1)
    Zu, Zv = Z.split(Z.shape[-1] // 2, dim=-1)
    H  = F.silu(Zu) * Zv
    S  = H.matmul(U2)
    Y  = S.matmul(V2) + b2.view(1, 1, -1)
    return Y


# ==============================================================
# Utilities: memory estimators, timing, profiler
# ==============================================================

def mib(nbytes): return nbytes / (1024**2)

def theoretical_peak_bytes_baseline(B, L, H, D, R2, dtype):
    """Activations-only theoretical peak (same as GEGLU script)."""
    bytes_e = torch.tensor([], dtype=dtype).element_size()
    Z = bytes_e * B * L * (2*D)
    Ht = bytes_e * B * L * D
    S = bytes_e * B * L * R2
    Y = bytes_e * B * L * H
    peak_H_stage = Z + Ht
    peak_Y_stage = S + Y
    peak_S_stage = Ht + S
    return max(peak_H_stage, peak_S_stage, peak_Y_stage, Z)

def theoretical_peak_bytes_triton(B, L, H, R2, dtype, store_s_fp32=False):
    el_S = torch.tensor([], dtype=torch.float32 if store_s_fp32 else dtype).element_size()
    el_Y = torch.tensor([], dtype=dtype).element_size()
    S = el_S * B * L * R2
    Y = el_Y * B * L * H
    return S + Y

@torch.no_grad()
def bench(fn, n_warmup=10, n_runs=50):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
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
        _ = out.view(-1)[0].item()

    times.sort()
    mean_ms = sum(times)/len(times)
    p50 = times[len(times)//2]
    p95 = times[int(0.95*len(times))-1]
    return {"mean_ms": mean_ms, "p50_ms": p50, "p95_ms": p95, "all_ms": times}

def _try_profile_config(P, V1, U2, V2, b1, b2, cfg):
    """Run one pinned config; catch OOR errors and report."""
    try:
        torch.cuda.synchronize()
        def run_once():
            return flashsvd_ffn_swiglu(P, V1, U2, V2, b1, b2,
                                       BL=cfg['BL'], BD=cfg['BD'], BR1=cfg['BR1'], BR2=cfg['BR2'],
                                       use_autotune=False)
        # warmup
        for _ in range(5):
            _ = run_once()
        torch.cuda.synchronize()
        # measure
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(20):
            start.record(); _ = run_once(); end.record()
            torch.cuda.synchronize()
            ts.append(start.elapsed_time(end))
        return sum(ts)/len(ts), None
    except Exception as e:
        msg = str(e)
        if "out of resource" in msg.lower() or "OutOfResources" in msg:
            return None, f"OutOfResources: {msg.splitlines()[-1]}"
        return None, f"Error: {msg}"

def manual_profile(P, V1, U2, V2, b1, b2):
    sweep = [
        {'BL': 32,  'BD': 64,  'BR1': 64,  'BR2': 64,  'warps': 4, 'stages': 2},
        {'BL': 64,  'BD': 64,  'BR1': 64,  'BR2': 64,  'warps': 4, 'stages': 2},
        {'BL': 64,  'BD': 128, 'BR1': 64,  'BR2': 64,  'warps': 4, 'stages': 2},
        {'BL': 64,  'BD': 128, 'BR1': 64,  'BR2': 128, 'warps': 8, 'stages': 2},
        {'BL': 128, 'BD': 128, 'BR1': 64,  'BR2': 128, 'warps': 8, 'stages': 2},
    ]
    print("\n=== Manual profile over configs (powers-of-two tiles) ===")
    results = []
    for cfg in sweep:
        # We set warps/stages via autotune normally; for pinned path the kernel body uses defaults.
        mean_ms, err = _try_profile_config(P, V1, U2, V2, b1, b2, cfg)
        if err is not None:
            print(f"Skip BL={cfg['BL']:>3} BD={cfg['BD']:>3} BR1={cfg['BR1']:>3} BR2={cfg['BR2']:>3} "
                  f"(warps={cfg['warps']}, stages={cfg['stages']}) -> {err}.")
        else:
            print(f"BL={cfg['BL']:>3} BD={cfg['BD']:>3} BR1={cfg['BR1']:>3} BR2={cfg['BR2']:>3} "
                  f"| warps={cfg['warps']} stages={cfg['stages']} | mean={mean_ms:.3f} ms")
            results.append((mean_ms, cfg))
    if results:
        results.sort(key=lambda x: x[0])
        best_ms, best = results[0]
        print(f"\nBest config: BL={best['BL']}, BD={best['BD']}, BR1={best['BR1']}, BR2={best['BR2']}, "
              f"warps={best['warps']}, stages={best['stages']} | mean={best_ms:.3f} ms")
    return results


# ==============================================================
# main(): parity, autotune timing, manual profiling, memory
# ==============================================================

def main():
    parser = argparse.ArgumentParser("FlashSVD SwiGLU (rank-space) kernel profile")
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--L", type=int, default=2048)
    parser.add_argument("--H", type=int, default=768, help="model hidden size")
    parser.add_argument("--D", type=int, default=3072, help="FFN inner size")
    parser.add_argument("--R1", type=int, default=384, help="rank for Wi")
    parser.add_argument("--R2", type=int, default=384, help="rank for Wo")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16","fp32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters",  type=int, default=50)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    torch.manual_seed(0)

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    B, L, H, D, R1, R2 = args.B, args.L, args.H, args.D, args.R1, args.R2

    # ---- random test tensors ----
    X  = torch.randn((B, L, H), device=device, dtype=dtype)
    U1 = torch.randn((H, R1), device=device, dtype=dtype) / math.sqrt(H)
    V1 = torch.randn((R1, 2*D), device=device, dtype=dtype) / math.sqrt(R1)
    U2 = torch.randn((D,  R2), device=device, dtype=dtype) / math.sqrt(D)
    V2 = torch.randn((R2, H),  device=device, dtype=dtype) / math.sqrt(R2)
    b1 = torch.zeros((2*D,), device=device, dtype=dtype)
    b2 = torch.zeros((H,),   device=device, dtype=dtype)

    # rank-space input (only low-rank P and S live large)
    P  = X.matmul(U1)

    # ---- warm compile (autotuned) ----
    _ = flashsvd_ffn_swiglu(P, V1, U2, V2, b1, b2, use_autotune=True)
    torch.cuda.synchronize()

    # ---- correctness vs. PyTorch baseline ----
    with torch.no_grad():
        Y_ref = _pt_baseline_swiglu(P, V1, U2, V2, b1, b2).float()
        Y_tr  = flashsvd_ffn_swiglu(P, V1, U2, V2, b1, b2, use_autotune=True).float()
    diff = Y_tr - Y_ref
    rel_f = diff.norm() / (Y_ref.norm() + 1e-12)
    print("\nCorrectness:")
    print("  finite(triton):", torch.isfinite(Y_tr).all().item())
    print("  finite(pt)    :", torch.isfinite(Y_ref).all().item())
    print("  finite(diff)  :", torch.isfinite(diff).all().item())
    print(f"  Max abs error : {diff.abs().max().item():.3e}")
    print(f"  Rel Fro error : {rel_f.item():.3e}")

    # ---- measured peak memory (single forward) ----
    torch.cuda.reset_peak_memory_stats()
    _ = flashsvd_ffn_swiglu(P, V1, U2, V2, b1, b2, use_autotune=True)
    torch.cuda.synchronize()
    peak_triton = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    _ = _pt_baseline_swiglu(P, V1, U2, V2, b1, b2)
    torch.cuda.synchronize()
    peak_baseline = torch.cuda.max_memory_allocated()

    # theoretical peaks (activations only)
    theo_base = theoretical_peak_bytes_baseline(B, L, H, D, R2, dtype)
    theo_trit = theoretical_peak_bytes_triton(B, L, H, R2, dtype, store_s_fp32=False)

    print("\n=== Peak Memory (measured) ===")
    print(f"Baseline (PyTorch): {mib(peak_baseline):,.2f} MiB")
    print(f"FlashSVD (Triton): {mib(peak_triton):,.2f} MiB")

    print("\n=== Peak Memory (theoretical activations) ===")
    print(f"Baseline (PyTorch): {mib(theo_base):,.2f} MiB")
    print(f"FlashSVD (Triton): {mib(theo_trit):,.2f} MiB")

    saved_meas = peak_baseline - peak_triton
    saved_theo = theo_base - theo_trit
    print("\n=== Savings ===")
    print(f"Measured savings: {mib(max(saved_meas, 0)):,.2f} MiB")
    print(f"Theoretical savings: {mib(max(saved_theo, 0)):,.2f} MiB")
    if theo_base > 0:
        print("Theoretical ratio (FlashSVD / Baseline):", f"{(theo_trit / theo_base):.3f}x")

    # ---- latency (warmups & averages) ----
    tokens = B * L
    def tps(ms): return tokens / (ms / 1e3)

    triton_stats = bench(lambda: flashsvd_ffn_swiglu(P, V1, U2, V2, b1, b2, use_autotune=True),
                         n_warmup=args.warmup, n_runs=args.iters)
    base_stats   = bench(lambda: _pt_baseline_swiglu(P, V1, U2, V2, b1, b2),
                         n_warmup=args.warmup, n_runs=args.iters)

    print("\n=== Inference Speed (latency per forward) ===")
    print("Baseline (PyTorch): "
          f"mean {base_stats['mean_ms']:.2f} ms | p50 {base_stats['p50_ms']:.2f} | p95 {base_stats['p95_ms']:.2f} "
          f"| {tps(base_stats['mean_ms']):,.0f} tok/s")
    print("FlashSVD (Triton): "
          f"mean {triton_stats['mean_ms']:.2f} ms | p50 {triton_stats['p50_ms']:.2f} | p95 {triton_stats['p95_ms']:.2f} "
          f"| {tps(triton_stats['mean_ms']):,.0f} tok/s")

    # ---- optional: manual config sweep (pinned) ----
    if args.profile:
        _ = manual_profile(P, V1, U2, V2, b1, b2)


if __name__ == "__main__":
    main()

































# #!/usr/bin/env python3
# import torch
# import triton
# import triton.language as tl
# import torch.nn.functional as F

# # -----------------------------
# # Triton kernel (SwiGLU in rank space)
# # -----------------------------
# @triton.jit
# def fused_ffn_phase1_swiglu(
#     P_ptr, V1_ptr, U2_ptr, S_ptr, b1_ptr,
#     B, L, D, R1, R2,
#     sP_b, sP_l, sP_r1,
#     sV1_r1, sV1_d,
#     sU2_d, sU2_r2,
#     sb1,
#     sS_b, sS_l, sS_r2,
#     BL: tl.constexpr, BD: tl.constexpr, BR1: tl.constexpr, BR2: tl.constexpr,
# ):
#     pid_b  = tl.program_id(0)
#     pid_l  = tl.program_id(1)
#     pid_r2 = tl.program_id(2)

#     offs_l  = pid_l * BL + tl.arange(0, BL)
#     offs_r2 = pid_r2 * BR2 + tl.arange(0, BR2)

#     Tu_acc = tl.zeros((BL, BD), dtype=tl.float32)
#     Tv_acc = tl.zeros((BL, BD), dtype=tl.float32)
#     acc    = tl.zeros((BL, BR2), dtype=tl.float32)

#     for d0 in range(0, D, BD):
#         d   = d0 + tl.arange(0, BD)
#         m_d = d < D

#         Tu_acc *= 0.0
#         Tv_acc *= 0.0

#         for r1_0 in range(0, R1, BR1):
#             r1   = r1_0 + tl.arange(0, BR1)
#             m_r1 = r1 < R1

#             P_blk = tl.load(
#                 P_ptr + pid_b * sP_b + offs_l[:, None]*sP_l + r1[None, :]*sP_r1,
#                 mask=(offs_l[:, None] < L) & m_r1[None, :],
#                 other=0.0
#             )

#             V1u_blk = tl.load(
#                 V1_ptr + r1[:, None]*sV1_r1 + d[None, :]*sV1_d,
#                 mask=m_r1[:, None] & m_d[None, :],
#                 other=0.0
#             )

#             V1v_blk = tl.load(
#                 V1_ptr + r1[:, None]*sV1_r1 + (d[None, :] + D)*sV1_d,
#                 mask=m_r1[:, None] & m_d[None, :],
#                 other=0.0
#             )

#             Tu_acc += tl.dot(P_blk, V1u_blk)
#             Tv_acc += tl.dot(P_blk, V1v_blk)

#         b1u = tl.load(b1_ptr + d*sb1,        mask=m_d, other=0.0).to(tl.float32)
#         b1v = tl.load(b1_ptr + (d + D)*sb1,  mask=m_d, other=0.0).to(tl.float32)
#         Tu  = Tu_acc + b1u[None, :]
#         Tv  = Tv_acc + b1v[None, :]

#         # SwiGLU: silu(Tu) * Tv, where silu(z) = z * sigmoid(z)
#         Hu = Tu * (1.0 / (1.0 + tl.exp(-Tu)))
#         H  = Hu * Tv

#         U2_blk = tl.load(
#             U2_ptr + d[:, None]*sU2_d + offs_r2[None, :]*sU2_r2,
#             mask=m_d[:, None] & (offs_r2[None, :] < R2),
#             other=0.0
#         ).to(tl.float32)
#         acc += tl.dot(H, U2_blk)

#     mask = (offs_l[:, None] < L) & (offs_r2[None, :] < R2)
#     tl.store(
#         S_ptr + pid_b*sS_b + offs_l[:, None]*sS_l + offs_r2[None, :]*sS_r2,
#         acc, mask=mask
#     )


# # -----------------------------
# # Wrapper
# # -----------------------------
# def flashsvd_ffn_swiglu(
#     P, V1, U2, V2, b1, b2,
#     BL=64, BD=64, BR1=64, BR2=64,
#     *, store_s_fp32: bool = False,
# ):
#     assert P.is_cuda and V1.is_cuda and U2.is_cuda and V2.is_cuda
#     B, L, R1 = P.shape
#     R1_v1, twoD = V1.shape
#     D = twoD // 2
#     assert R1_v1 == R1 and twoD == 2 * D
#     D_u2, R2 = U2.shape
#     assert D_u2 == D
#     R2_v2, H = V2.shape
#     assert R2_v2 == R2
#     assert b1.shape[0] == 2*D and b2.shape[0] == H

#     S_dtype = torch.float32 if store_s_fp32 else P.dtype
#     S = torch.empty((B, L, R2), device=P.device, dtype=S_dtype)

#     strides = dict(
#         sP_b=P.stride(0), sP_l=P.stride(1), sP_r1=P.stride(2),
#         sV1_r1=V1.stride(0), sV1_d=V1.stride(1),
#         sU2_d=U2.stride(0), sU2_r2=U2.stride(1),
#         sb1=b1.stride(0),
#         sS_b=S.stride(0), sS_l=S.stride(1), sS_r2=S.stride(2),
#     )

#     grid = (B, triton.cdiv(L, BL), triton.cdiv(R2, BR2))
#     fused_ffn_phase1_swiglu[grid](
#         P, V1, U2, S, b1,
#         B, L, D, R1, R2,
#         *strides.values(),
#         BL, BD, BR1, BR2,
#     )

#     Y = S.matmul(V2)
#     Y = Y + b2.view(1, 1, -1)
#     return Y


# # -----------------------------
# # PyTorch reference
# # -----------------------------
# def _pt_baseline_swiglu(P, V1, U2, V2, b1, b2):
#     Z  = P.matmul(V1) + b1.view(1, 1, -1)
#     Zu, Zv = Z.split(Z.shape[-1] // 2, dim=-1)
#     H  = F.silu(Zu) * Zv
#     S  = H.matmul(U2)
#     Y  = S.matmul(V2) + b2.view(1, 1, -1)
#     return Y


