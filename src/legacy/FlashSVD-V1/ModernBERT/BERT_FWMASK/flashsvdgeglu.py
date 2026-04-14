#!/usr/bin/env python3
import math
import statistics as stats
import numpy as np
import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# ============================================================
# Base Triton kernel (no autotune) for manual profiling/sweeps
# ============================================================
@triton.jit
def fused_ffn_phase1_geglu_kernel(
    P_ptr, V1_ptr, U2_ptr, S_ptr, b1_ptr,
    B, L, D, R1, R2,
    sP_b, sP_l, sP_r1,
    sV1_r1, sV1_d,
    sU2_d, sU2_r2,
    sb1,
    sS_b, sS_l, sS_r2,
    BL: tl.constexpr, BD: tl.constexpr, BR1: tl.constexpr, BR2: tl.constexpr,
    USE_TANH: tl.constexpr,
):
    pid_b  = tl.program_id(0)
    pid_l  = tl.program_id(1)
    pid_r2 = tl.program_id(2)

    offs_l  = pid_l * BL + tl.arange(0, BL)
    offs_r2 = pid_r2 * BR2 + tl.arange(0, BR2)

    Tu_acc = tl.zeros((BL, BD), dtype=tl.float32)
    Tv_acc = tl.zeros((BL, BD), dtype=tl.float32)
    acc    = tl.zeros((BL, BR2), dtype=tl.float32)

    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    inv_sqrt2 = 0.7071067811865476

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

            # upcast for accumulation
            P_blk_f = P_blk.to(tl.float32)
            V1u_f   = V1u_blk.to(tl.float32)
            V1v_f   = V1v_blk.to(tl.float32)

            Tu_acc += tl.dot(P_blk_f, V1u_f)
            Tv_acc += tl.dot(P_blk_f, V1v_f)

        b1u = tl.load(b1_ptr + d*sb1,        mask=m_d, other=0.0).to(tl.float32)
        b1v = tl.load(b1_ptr + (d + D)*sb1,  mask=m_d, other=0.0).to(tl.float32)
        Tu  = Tu_acc + b1u[None, :]
        Tv  = Tv_acc + b1v[None, :]

        if USE_TANH:
            z  = c0 * (Tu + c1 * Tu * Tu * Tu)
            z2 = 2.0 * z
            sig_2z = tl.where(
                z2 >= 0,
                1.0 / (1.0 + tl.exp(-z2)),
                tl.exp(z2) / (1.0 + tl.exp(z2))
            )
            tanh_z = 2.0 * sig_2z - 1.0
            Hu = 0.5 * Tu * (1.0 + tanh_z)
        else:
            Hu = 0.5 * Tu * (1.0 + tl.erf(Tu * inv_sqrt2))

        H = Hu * Tv

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


# ==================================================
# Autotuned version (grid depends on config's tiles)
# ==================================================
# Best config (float16): BL=64, BD=128, BR1=64, BR2=128, warps=8, stages=2 | mean=2.739 ms | peak=160.0 MiB
TUNE_CONFIGS = [
    # All tiles are powers-of-two -> safe with older Triton arange
    triton.Config({'BL': 32,  'BD': 64,  'BR1': 64,  'BR2': 64},  num_warps=4, num_stages=2),
    triton.Config({'BL': 64,  'BD': 64,  'BR1': 64,  'BR2': 64},  num_warps=4, num_stages=2),
    triton.Config({'BL': 64,  'BD': 128, 'BR1': 64,  'BR2': 64},  num_warps=4, num_stages=2),
    triton.Config({'BL': 64,  'BD': 128, 'BR1': 128, 'BR2': 64},  num_warps=8, num_stages=2),
    triton.Config({'BL': 64, 'BD': 128, 'BR1': 64,  'BR2': 128}, num_warps=8, num_stages=1),
    triton.Config({'BL': 128,'BD': 128, 'BR1': 64,  'BR2': 128}, num_warps=8, num_stages=1),
    triton.Config({'BL': 128,'BD': 256, 'BR1': 64,  'BR2': 128}, num_warps=8, num_stages=1),  # might fit with stages=1
    triton.Config({'BL': 64,  'BD': 128, 'BR1': 64,  'BR2': 128}, num_warps=8, num_stages=2),
    triton.Config({'BL': 128, 'BD': 128, 'BR1': 128, 'BR2': 64},  num_warps=8, num_stages=2),
    triton.Config({'BL': 128, 'BD': 128, 'BR1': 64,  'BR2': 128}, num_warps=8, num_stages=2),
    #triton.Config({'BL': 128, 'BD': 256, 'BR1': 128, 'BR2': 128}, num_warps=8, num_stages=2),
    #triton.Config({'BL': 128, 'BD': 256, 'BR1': 256, 'BR2': 64},  num_warps=8, num_stages=3),
    #triton.Config({'BL': 128, 'BD': 256, 'BR1': 128, 'BR2': 256}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=TUNE_CONFIGS, key=['D', 'R1', 'R2'])
@triton.jit
def fused_ffn_phase1_geglu_auto(
    P_ptr, V1_ptr, U2_ptr, S_ptr, b1_ptr,
    B, L, D, R1, R2,
    sP_b, sP_l, sP_r1,
    sV1_r1, sV1_d,
    sU2_d, sU2_r2,
    sb1,
    sS_b, sS_l, sS_r2,
    BL: tl.constexpr, BD: tl.constexpr, BR1: tl.constexpr, BR2: tl.constexpr,
    USE_TANH: tl.constexpr,
):
    # Identical body to the base kernel
    pid_b  = tl.program_id(0)
    pid_l  = tl.program_id(1)
    pid_r2 = tl.program_id(2)

    offs_l  = pid_l * BL + tl.arange(0, BL)
    offs_r2 = pid_r2 * BR2 + tl.arange(0, BR2)

    Tu_acc = tl.zeros((BL, BD), dtype=tl.float32)
    Tv_acc = tl.zeros((BL, BD), dtype=tl.float32)
    acc    = tl.zeros((BL, BR2), dtype=tl.float32)

    c0 = 0.7978845608028654
    c1 = 0.044715
    inv_sqrt2 = 0.7071067811865476

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

            P_blk_f = P_blk.to(tl.float32)
            V1u_f   = V1u_blk.to(tl.float32)
            V1v_f   = V1v_blk.to(tl.float32)

            Tu_acc += tl.dot(P_blk_f, V1u_f)
            Tv_acc += tl.dot(P_blk_f, V1v_f)

        b1u = tl.load(b1_ptr + d*sb1,        mask=m_d, other=0.0).to(tl.float32)
        b1v = tl.load(b1_ptr + (d + D)*sb1,  mask=m_d, other=0.0).to(tl.float32)
        Tu  = Tu_acc + b1u[None, :]
        Tv  = Tv_acc + b1v[None, :]

        if USE_TANH:
            z  = c0 * (Tu + c1 * Tu * Tu * Tu)
            z2 = 2.0 * z
            sig_2z = tl.where(
                z2 >= 0,
                1.0 / (1.0 + tl.exp(-z2)),
                tl.exp(z2) / (1.0 + tl.exp(z2))
            )
            tanh_z = 2.0 * sig_2z - 1.0
            Hu = 0.5 * Tu * (1.0 + tanh_z)
        else:
            Hu = 0.5 * Tu * (1.0 + tl.erf(Tu * inv_sqrt2))

        H = Hu * Tv

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


# -----------------------------
# Wrappers
# -----------------------------
def _grid_for(B, L, R2, BL, BR2):
    return (B, triton.cdiv(L, BL), triton.cdiv(R2, BR2))

def flashsvd_ffn_geglu_autotuned(
    P, V1, U2, V2, b1, b2,
    gelu_approx: str = "tanh",
    store_s_fp32: bool = False,
):
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

    strides = dict(
        sP_b=P.stride(0), sP_l=P.stride(1), sP_r1=P.stride(2),
        sV1_r1=V1.stride(0), sV1_d=V1.stride(1),
        sU2_d=U2.stride(0), sU2_r2=U2.stride(1),
        sb1=b1.stride(0),
        sS_b=S.stride(0), sS_l=S.stride(1), sS_r2=S.stride(2),
    )

    USE_TANH = 1 if gelu_approx == "tanh" else 0

    # Autotune grid depends on config tiles
    grid = lambda meta: _grid_for(B, L, R2, meta['BL'], meta['BR2'])

    fused_ffn_phase1_geglu_auto[grid](
        P, V1, U2, S, b1,
        B, L, D, R1, R2,
        *strides.values(),
        USE_TANH=USE_TANH,
    )

    Y = S.matmul(V2) + b2.view(1, 1, -1)
    return Y

def flashsvd_ffn_geglu_configured(
    P, V1, U2, V2, b1, b2,
    BL=64, BD=128, BR1=64, BR2=128,
    gelu_approx: str = "tanh",
    store_s_fp32: bool = False,
    num_warps: int = 8, num_stages: int = 2,
):
    # Manual, fixed-config call (for profiling each config)
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

    strides = dict(
        sP_b=P.stride(0), sP_l=P.stride(1), sP_r1=P.stride(2),
        sV1_r1=V1.stride(0), sV1_d=V1.stride(1),
        sU2_d=U2.stride(0), sU2_r2=U2.stride(1),
        sb1=b1.stride(0),
        sS_b=S.stride(0), sS_l=S.stride(1), sS_r2=S.stride(2),
    )

    grid = _grid_for(B, L, R2, BL, BR2)
    USE_TANH = 1 if gelu_approx == "tanh" else 0

    fused_ffn_phase1_geglu_kernel[grid](
        P, V1, U2, S, b1,
        B, L, D, R1, R2,
        *strides.values(),
        BL=BL, BD=BD, BR1=BR1, BR2=BR2,
        USE_TANH=USE_TANH,
        num_warps=num_warps, num_stages=num_stages,
    )

    Y = S.matmul(V2) + b2.view(1, 1, -1)
    return Y


def _pt_baseline(P, V1, U2, V2, b1, b2, gelu_approx="tanh"):
    Z  = P.matmul(V1) + b1.view(1, 1, -1)
    Zu, Zv = Z.split(Z.shape[-1] // 2, dim=-1)
    H  = F.gelu(Zu, approximate=gelu_approx) * Zv
    S  = H.matmul(U2)
    Y  = S.matmul(V2) + b2.view(1, 1, -1)
    return Y


# -----------------------------
# Memory estimators (activations only)
# -----------------------------
def mib(nbytes): return nbytes / (1024**2)

def theoretical_peak_bytes_baseline(B, L, H, D, R2, dtype):
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


# -----------------------------
# Timing helpers
# -----------------------------
@torch.no_grad()
def bench(fn, n_warmup=10, n_runs=50):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times_ms = []

    # warmup
    for _ in range(n_warmup):
        _ = fn()
    torch.cuda.synchronize()

    # timed runs
    for _ in range(n_runs):
        torch.cuda.reset_peak_memory_stats()
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
        _ = out.view(-1)[0].item()

    return {
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "all_ms": times_ms,
        "peak_mib": mib(torch.cuda.max_memory_allocated()),
    }


def profile_configs(P, V1, U2, V2, b1, b2, dtype_label="fp16"):
    print("\n=== Manual profile over configs (powers-of-two tiles) ===")
    rows = []
    for cfg in TUNE_CONFIGS:
        BL, BD, BR1, BR2 = cfg.kwargs['BL'], cfg.kwargs['BD'], cfg.kwargs['BR1'], cfg.kwargs['BR2']
        nw, ns = cfg.num_warps, cfg.num_stages
        try:
            stats_run = bench(lambda: flashsvd_ffn_geglu_configured(
                P, V1, U2, V2, b1, b2,
                BL=BL, BD=BD, BR1=BR1, BR2=BR2,
                gelu_approx="tanh", store_s_fp32=False,
                num_warps=nw, num_stages=ns
            ), n_warmup=5, n_runs=20)
            rows.append((stats_run['mean_ms'], BL, BD, BR1, BR2, nw, ns, stats_run['peak_mib']))
            print(f"BL={BL:>3} BD={BD:>3} BR1={BR1:>3} BR2={BR2:>3} | "
                  f"warps={nw} stages={ns} | mean={stats_run['mean_ms']:.3f} ms | peak={stats_run['peak_mib']:.1f} MiB")
        except Exception as e:
            print(f"Skip BL={BL} BD={BD} BR1={BR1} BR2={BR2} (warps={nw}, stages={ns}) -> {type(e).__name__}: {e}")
    if rows:
        rows.sort(key=lambda r: r[0])
        best = rows[0]
        print(f"\nBest config ({dtype_label}): "
              f"BL={best[1]}, BD={best[2]}, BR1={best[3]}, BR2={best[4]}, warps={best[5]}, stages={best[6]} "
              f"| mean={best[0]:.3f} ms | peak={best[7]:.1f} MiB")
    else:
        print("No valid configs ran.")


# -----------------------------
# Main
# -----------------------------
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")
    torch.manual_seed(0)

    # dims
    B, L, H = 8, 2048, 768
    D       = 3072
    R1, R2  = 384, 384

    # data
    dtype = torch.float16
    X  = torch.randn((B, L, H), device=device, dtype=dtype)

    U1 = torch.randn((H, R1), device=device, dtype=dtype) / math.sqrt(H)
    V1 = torch.randn((R1, 2*D), device=device, dtype=dtype) / math.sqrt(R1)
    U2 = torch.randn((D,  R2), device=device, dtype=dtype) / math.sqrt(D)
    V2 = torch.randn((R2, H),  device=device, dtype=dtype) / math.sqrt(R2)
    b1 = torch.zeros((2*D,), device=device, dtype=dtype)
    b2 = torch.zeros((H,),   device=device, dtype=dtype)

    # rank-space input
    P  = X.matmul(U1)

    # ------------------ Warmup compile ------------------
    _ = flashsvd_ffn_geglu_autotuned(P, V1, U2, V2, b1, b2, gelu_approx="tanh", store_s_fp32=False)
    torch.cuda.synchronize()

    # ------------------ Manual config profiling (optional but useful) ------------------
    profile_configs(P, V1, U2, V2, b1, b2, dtype_label=str(dtype).replace('torch.', ''))

    # ------------------ Peak memory (measured) ------------------
    torch.cuda.reset_peak_memory_stats()
    out_triton = flashsvd_ffn_geglu_autotuned(P, V1, U2, V2, b1, b2, gelu_approx="tanh", store_s_fp32=False)
    torch.cuda.synchronize()
    peak_triton = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    out_pt = _pt_baseline(P, V1, U2, V2, b1, b2, gelu_approx="tanh")
    torch.cuda.synchronize()
    peak_baseline = torch.cuda.max_memory_allocated()

    # ------------------ Correctness ------------------
    diff  = (out_triton - out_pt).to(torch.float32)
    rel_f = diff.norm() / (out_pt.to(torch.float32).norm() + 1e-12)
    print("\nCorrectness:")
    print("  finite(triton):", bool(torch.isfinite(out_triton).all()))
    print("  finite(pt)    :", bool(torch.isfinite(out_pt).all()))
    print("  finite(diff)  :", bool(torch.isfinite(diff).all()))
    print(f"  Max abs error : {torch.nan_to_num(diff.abs().max(), nan=0.0).item():.3e}")
    print(f"  Rel Fro error : {rel_f.item():.3e}")

    # ------------------ Theoretical activation peaks ------------------
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
        print("Theoretical ratio (FlashSVD / Baseline): "
              f"{(theo_trit / theo_base):.3f}x")

    # ------------------ SPEED: latency per forward ------------------
    tokens = B * L
    def tps(ms): return tokens / (ms / 1e3)

    triton_stats = bench(lambda: flashsvd_ffn_geglu_autotuned(
        P, V1, U2, V2, b1, b2, gelu_approx="tanh", store_s_fp32=False
    ), n_warmup=10, n_runs=50)
    base_stats   = bench(lambda: _pt_baseline(
        P, V1, U2, V2, b1, b2, gelu_approx="tanh"
    ), n_warmup=10, n_runs=50)

    print("\n=== Inference Speed (latency per forward) ===")
    print(f"Baseline (PyTorch): mean {base_stats['mean_ms']:.2f} ms | "
          f"median {base_stats['median_ms']:.2f} ms | p95 {base_stats['p95_ms']:.2f} ms | "
          f"{tps(base_stats['mean_ms']):,.0f} tok/s | peak {base_stats['peak_mib']:.1f} MiB")
    print(f"FlashSVD (Triton): mean {triton_stats['mean_ms']:.2f} ms | "
          f"median {triton_stats['median_ms']:.2f} ms | p95 {triton_stats['p95_ms']:.2f} ms | "
          f"{tps(triton_stats['mean_ms']):,.0f} tok/s | peak {triton_stats['peak_mib']:.1f} MiB")


if __name__ == "__main__":
    main()




# #!/usr/bin/env python3
# import math
# import statistics as stats
# import torch
# import triton
# import triton.language as tl
# import torch.nn.functional as F

# # -----------------------------
# # Triton kernel (GEGLU in rank space)
# # -----------------------------
# @triton.jit
# def fused_ffn_phase1_geglu(
#     P_ptr, V1_ptr, U2_ptr, S_ptr, b1_ptr,
#     B, L, D, R1, R2,
#     sP_b, sP_l, sP_r1,
#     sV1_r1, sV1_d,
#     sU2_d, sU2_r2,
#     sb1,
#     sS_b, sS_l, sS_r2,
#     BL: tl.constexpr, BD: tl.constexpr, BR1: tl.constexpr, BR2: tl.constexpr,
#     USE_TANH: tl.constexpr,
# ):
#     pid_b  = tl.program_id(0)
#     pid_l  = tl.program_id(1)
#     pid_r2 = tl.program_id(2)

#     offs_l  = pid_l * BL + tl.arange(0, BL)
#     offs_r2 = pid_r2 * BR2 + tl.arange(0, BR2)

#     Tu_acc = tl.zeros((BL, BD), dtype=tl.float32)
#     Tv_acc = tl.zeros((BL, BD), dtype=tl.float32)
#     acc    = tl.zeros((BL, BR2), dtype=tl.float32)

#     c0 = 0.7978845608028654  # sqrt(2/pi)
#     c1 = 0.044715
#     inv_sqrt2 = 0.7071067811865476

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

#         if USE_TANH:
#             z  = c0 * (Tu + c1 * Tu * Tu * Tu)
#             z2 = 2.0 * z
#             sig_2z = tl.where(
#                 z2 >= 0,
#                 1.0 / (1.0 + tl.exp(-z2)),
#                 tl.exp(z2) / (1.0 + tl.exp(z2))
#             )
#             tanh_z = 2.0 * sig_2z - 1.0
#             Hu = 0.5 * Tu * (1.0 + tanh_z)
#         else:
#             Hu = 0.5 * Tu * (1.0 + tl.erf(Tu * inv_sqrt2))

#         H = Hu * Tv

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
# # Wrappers
# # -----------------------------
# def flashsvd_ffn_geglu(
#     P, V1, U2, V2, b1, b2,
#     BL=64, BD=64, BR1=64, BR2=64,
#     gelu_approx: str = "tanh",
#     nan_scrub: bool = False,
#     store_s_fp32: bool = False,
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
#     USE_TANH = 1 if gelu_approx == "tanh" else 0

#     fused_ffn_phase1_geglu[grid](
#         P, V1, U2, S, b1,
#         B, L, D, R1, R2,
#         *strides.values(),
#         BL, BD, BR1, BR2,
#         USE_TANH=USE_TANH,
#     )

#     if nan_scrub:
#         S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

#     Y = S.matmul(V2)
#     Y = Y + b2.view(1, 1, -1)
#     return Y

# def _pt_baseline(P, V1, U2, V2, b1, b2, gelu_approx="tanh"):
#     Z  = P.matmul(V1) + b1.view(1, 1, -1)
#     Zu, Zv = Z.split(Z.shape[-1] // 2, dim=-1)
#     H  = F.gelu(Zu, approximate=gelu_approx) * Zv
#     S  = H.matmul(U2)
#     Y  = S.matmul(V2) + b2.view(1, 1, -1)
#     return Y

# # -----------------------------
# # Memory estimators (activations only)
# # -----------------------------
# def mib(nbytes): return nbytes / (1024**2)

# def theoretical_peak_bytes_baseline(B, L, H, D, R2, dtype):
#     bytes_e = torch.tensor([], dtype=dtype).element_size()
#     Z = bytes_e * B * L * (2*D)
#     Ht = bytes_e * B * L * D
#     S = bytes_e * B * L * R2
#     Y = bytes_e * B * L * H
#     peak_H_stage = Z + Ht
#     peak_Y_stage = S + Y
#     peak_S_stage = Ht + S
#     return max(peak_H_stage, peak_S_stage, peak_Y_stage, Z)

# def theoretical_peak_bytes_triton(B, L, H, R2, dtype, store_s_fp32=False):
#     el_S = torch.tensor([], dtype=torch.float32 if store_s_fp32 else dtype).element_size()
#     el_Y = torch.tensor([], dtype=dtype).element_size()
#     S = el_S * B * L * R2
#     Y = el_Y * B * L * H
#     return S + Y

# # -----------------------------
# # Timing helper
# # -----------------------------
# @torch.no_grad()
# def bench(fn, n_warmup=10, n_runs=50):
#     # CUDA event-based timing
#     start = torch.cuda.Event(enable_timing=True)
#     end   = torch.cuda.Event(enable_timing=True)
#     times_ms = []

#     # warmup
#     for _ in range(n_warmup):
#         _ = fn()
#     torch.cuda.synchronize()

#     # timed runs
#     for _ in range(n_runs):
#         start.record()
#         out = fn()
#         end.record()
#         torch.cuda.synchronize()
#         times_ms.append(start.elapsed_time(end))
#         # tiny usage to keep out alive
#         _ = out.view(-1)[0].item()

#     return {
#         "mean_ms": sum(times_ms)/len(times_ms),
#         "median_ms": stats.median(times_ms),
#         "p95_ms": stats.quantiles(times_ms, n=20)[-1] if len(times_ms) >= 20 else max(times_ms),
#         "all_ms": times_ms,
#     }

# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     torch.backends.cuda.matmul.allow_tf32 = True
#     device = torch.device("cuda")
#     torch.manual_seed(0)

#     # dims
#     B, L, H = 8, 2048, 768
#     D       = 3072
#     R1, R2  = 384, 384
#     BL, BD, BR1, BR2 = 64, 64, 64, 64

#     dtype = torch.float16
#     X  = torch.randn((B, L, H), device=device, dtype=dtype)

#     U1 = torch.randn((H, R1), device=device, dtype=dtype) / math.sqrt(H)
#     V1 = torch.randn((R1, 2*D), device=device, dtype=dtype) / math.sqrt(R1)
#     U2 = torch.randn((D,  R2), device=device, dtype=dtype) / math.sqrt(D)
#     V2 = torch.randn((R2, H),  device=device, dtype=dtype) / math.sqrt(R2)
#     b1 = torch.zeros((2*D,), device=device, dtype=dtype)
#     b2 = torch.zeros((H,),   device=device, dtype=dtype)

#     # rank-space input
#     P  = X.matmul(U1)

#     # warmup Triton once (compile)
#     _ = flashsvd_ffn_geglu(P, V1, U2, V2, b1, b2, BL, BD, BR1, BR2,
#                            gelu_approx="tanh", nan_scrub=False, store_s_fp32=False)
#     torch.cuda.synchronize()

#     # measured peak memory
#     torch.cuda.reset_peak_memory_stats(device)
#     out_triton = flashsvd_ffn_geglu(P, V1, U2, V2, b1, b2, BL, BD, BR1, BR2,
#                                     gelu_approx="tanh", nan_scrub=False, store_s_fp32=False)
#     torch.cuda.synchronize()
#     peak_triton = torch.cuda.max_memory_allocated(device)

#     torch.cuda.reset_peak_memory_stats(device)
#     out_pt = _pt_baseline(P, V1, U2, V2, b1, b2, gelu_approx="tanh")
#     torch.cuda.synchronize()
#     peak_baseline = torch.cuda.max_memory_allocated(device)

#     # correctness
#     diff  = (out_triton - out_pt).to(torch.float32)
#     rel_f = diff.norm() / (out_pt.to(torch.float32).norm() + 1e-12)
#     print("finite(triton):", torch.isfinite(out_triton).all().item(),
#           " finite(pt):", torch.isfinite(out_pt).all().item())
#     print("finite(diff):", torch.isfinite(diff).all().item())
#     print(f"Max abs error: {torch.nan_to_num(diff.abs().max(), nan=0.0).item():.3e}")
#     print(f"Rel Fro error: {rel_f.item():.3e}")

#     # theoretical activation peaks
#     theo_base = theoretical_peak_bytes_baseline(B, L, H, D, R2, dtype)
#     theo_trit = theoretical_peak_bytes_triton(B, L, H, R2, dtype, store_s_fp32=False)

#     print("\n=== Peak Memory (measured) ===")
#     print(f"Baseline (PyTorch): {mib(peak_baseline):,.2f} MiB")
#     print(f"FlashSVD (Triton): {mib(peak_triton):,.2f} MiB")

#     print("\n=== Peak Memory (theoretical activations) ===")
#     print(f"Baseline (PyTorch): {mib(theo_base):,.2f} MiB")
#     print(f"FlashSVD (Triton): {mib(theo_trit):,.2f} MiB")

#     saved_meas = peak_baseline - peak_triton
#     saved_theo = theo_base - theo_trit
#     print("\n=== Savings ===")
#     print(f"Measured savings: {mib(max(saved_meas, 0)):,.2f} MiB")
#     print(f"Theoretical savings: {mib(max(saved_theo, 0)):,.2f} MiB")
#     if theo_base > 0:
#         print("Theoretical ratio (FlashSVD / Baseline):",
#               f"{(theo_trit / theo_base):.3f}x")

#     # ------------------ SPEED: latency + tokens/sec ------------------
#     tokens = B * L
#     # Benchmark functions capture current tensors by closure
#     triton_stats = bench(lambda: flashsvd_ffn_geglu(P, V1, U2, V2, b1, b2,
#                                                     BL, BD, BR1, BR2,
#                                                     gelu_approx="tanh",
#                                                     nan_scrub=False, store_s_fp32=False),
#                          n_warmup=10, n_runs=50)
#     base_stats   = bench(lambda: _pt_baseline(P, V1, U2, V2, b1, b2, gelu_approx="tanh"),
#                          n_warmup=10, n_runs=50)

#     def tps(ms):  # tokens per second
#         return tokens / (ms / 1e3)

#     print("\n=== Inference Speed (latency per forward) ===")
#     print("Baseline (PyTorch): "
#           f"mean {base_stats['mean_ms']:.2f} ms | median {base_stats['median_ms']:.2f} ms | p95 {base_stats['p95_ms']:.2f} ms "
#           f"| {tps(base_stats['mean_ms']):,.0f} tok/s")
#     print("FlashSVD (Triton): "
#           f"mean {triton_stats['mean_ms']:.2f} ms | median {triton_stats['median_ms']:.2f} ms | p95 {triton_stats['p95_ms']:.2f} ms "
#           f"| {tps(triton_stats['mean_ms']):,.0f} tok/s")

# if __name__ == "__main__":
#     main()






























# # #!/usr/bin/env python3
# # import math
# # import torch
# # import triton
# # import triton.language as tl
# # import torch.nn.functional as F

# # # -----------------------------------------------------------------------------
# # # Optional NaN scrub (OFF by default). Not an approximation of GELU—just replaces
# # # NaNs that may come from earlier ops / overflow.
# # # -----------------------------------------------------------------------------
# # @triton.jit
# # def _nan_scrub(x, enable: tl.constexpr):
# #     return tl.where(x == x, x, 0.0) if enable else x  # NaN != NaN

# # # -----------------------------------------------------------------------------
# # # Kernel: S = (GEGLU(P @ V1 + b1)) @ U2
# # # - Exact(erf) GELU ONLY (no tanh path, no clamping)
# # # - All matmul math in fp32
# # # -----------------------------------------------------------------------------
# # @triton.jit
# # def fused_ffn_phase1_geglu_erf(
# #     P_ptr, V1_ptr, U2_ptr, S_ptr, b1_ptr,
# #     B, L, D, R1, R2,
# #     sP_b, sP_l, sP_r1,
# #     sV1_r1, sV1_d,
# #     sU2_d, sU2_r2,
# #     sb1,
# #     sS_b, sS_l, sS_r2,
# #     BL: tl.constexpr, BD: tl.constexpr, BR1: tl.constexpr, BR2: tl.constexpr,
# #     NAN_SCRUB: tl.constexpr,   # 0 or 1; purity -> set 0
# # ):
# #     pid_b  = tl.program_id(0)
# #     pid_l  = tl.program_id(1)
# #     pid_r2 = tl.program_id(2)

# #     offs_l  = pid_l * BL + tl.arange(0, BL)
# #     offs_r2 = pid_r2 * BR2 + tl.arange(0, BR2)

# #     # fp32 accumulators
# #     Tu_acc = tl.zeros((BL, BD), dtype=tl.float32)
# #     Tv_acc = tl.zeros((BL, BD), dtype=tl.float32)
# #     acc    = tl.zeros((BL, BR2), dtype=tl.float32)

# #     inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)

# #     # tile over D
# #     for d0 in range(0, D, BD):
# #         d   = d0 + tl.arange(0, BD)
# #         m_d = d < D

# #         # reset per-D tile
# #         Tu_acc = tl.zeros((BL, BD), dtype=tl.float32)
# #         Tv_acc = tl.zeros((BL, BD), dtype=tl.float32)

# #         # accumulate over R1
# #         for r1_0 in range(0, R1, BR1):
# #             r1   = r1_0 + tl.arange(0, BR1)
# #             m_r1 = r1 < R1

# #             P_blk = tl.load(
# #                 P_ptr + pid_b * sP_b + offs_l[:, None] * sP_l + r1[None, :] * sP_r1,
# #                 mask=(offs_l[:, None] < L) & m_r1[None, :],
# #                 other=0.0
# #             ).to(tl.float32)

# #             V1u_blk = tl.load(
# #                 V1_ptr + r1[:, None] * sV1_r1 + d[None, :] * sV1_d,
# #                 mask=m_r1[:, None] & m_d[None, :],
# #                 other=0.0
# #             ).to(tl.float32)

# #             V1v_blk = tl.load(
# #                 V1_ptr + r1[:, None] * sV1_r1 + (d[None, :] + D) * sV1_d,
# #                 mask=m_r1[:, None] & m_d[None, :],
# #                 other=0.0
# #             ).to(tl.float32)

# #             if NAN_SCRUB:
# #                 P_blk   = _nan_scrub(P_blk, 1)
# #                 V1u_blk = _nan_scrub(V1u_blk, 1)
# #                 V1v_blk = _nan_scrub(V1v_blk, 1)

# #             Tu_acc += tl.dot(P_blk, V1u_blk)  # [BL, BD]
# #             Tv_acc += tl.dot(P_blk, V1v_blk)  # [BL, BD]

# #         # bias add (fp32)
# #         b1u = tl.load(b1_ptr + d * sb1,       mask=m_d, other=0.0).to(tl.float32)
# #         b1v = tl.load(b1_ptr + (d + D) * sb1, mask=m_d, other=0.0).to(tl.float32)
# #         Tu  = Tu_acc + b1u[None, :]
# #         Tv  = Tv_acc + b1v[None, :]

# #         if NAN_SCRUB:
# #             Tu = _nan_scrub(Tu, 1)
# #             Tv = _nan_scrub(Tv, 1)

# #         # --- GELU ---
# #         Hu = 0.5 * Tu * (1.0 + tl.erf(Tu * inv_sqrt2))  # [BL, BD] fp32
# #         H  = Hu * Tv                                    # [BL, BD] fp32

# #         if NAN_SCRUB:
# #             H = _nan_scrub(H, 1)

# #         # multiply by U2 tile
# #         U2_blk = tl.load(
# #             U2_ptr + d[:, None] * sU2_d + offs_r2[None, :] * sU2_r2,
# #             mask=m_d[:, None] & (offs_r2[None, :] < R2),
# #             other=0.0
# #         ).to(tl.float32)
# #         if NAN_SCRUB:
# #             U2_blk = _nan_scrub(U2_blk, 1)

# #         acc += tl.dot(H, U2_blk)  # [BL, BR2]

# #     # store (Triton will cast to S dtype)
# #     mask = (offs_l[:, None] < L) & (offs_r2[None, :] < R2)
# #     tl.store(
# #         S_ptr + pid_b * sS_b + offs_l[:, None] * sS_l + offs_r2[None, :] * sS_r2,
# #         acc, mask=mask
# #     )

# # # -----------------------------------------------------------------------------
# # # Python wrapper
# # # -----------------------------------------------------------------------------
# # def flashsvd_ffn_geglu(
# #     P, V1, U2, V2, b1, b2,
# #     BL=64, BD=128, BR1=64, BR2=64,
# #     nan_scrub: bool = False,     # keep False for the "pure" run
# #     store_s_fp32: bool = False,  # try True once to rule out fp16 store overflow
# # ):
# #     assert P.is_cuda and V1.is_cuda and U2.is_cuda and V2.is_cuda
# #     B, L, R1 = P.shape
# #     R1_v1, twoD = V1.shape
# #     D = twoD // 2
# #     assert R1_v1 == R1 and twoD == 2 * D
# #     D_u2, R2 = U2.shape
# #     assert D_u2 == D
# #     R2_v2, H = V2.shape
# #     assert R2_v2 == R2
# #     assert b1.shape[0] == 2 * D and b2.shape[0] == H

# #     S_dtype = torch.float32 if store_s_fp32 else P.dtype
# #     S = torch.empty((B, L, R2), device=P.device, dtype=S_dtype)

# #     strides = dict(
# #         sP_b=P.stride(0), sP_l=P.stride(1), sP_r1=P.stride(2),
# #         sV1_r1=V1.stride(0), sV1_d=V1.stride(1),
# #         sU2_d=U2.stride(0), sU2_r2=U2.stride(1),
# #         sb1=b1.stride(0),
# #         sS_b=S.stride(0), sS_l=S.stride(1), sS_r2=S.stride(2),
# #     )

# #     grid = (B, triton.cdiv(L, BL), triton.cdiv(R2, BR2))
# #     fused_ffn_phase1_geglu_erf[grid](
# #         P, V1, U2, S, b1,
# #         B, L, D, R1, R2,
# #         *strides.values(),
# #         BL, BD, BR1, BR2,
# #         NAN_SCRUB=1 if nan_scrub else 0,
# #     )

# #     Y = S.matmul(V2) + b2.view(1, 1, -1)
# #     return Y

# # # -----------------------------------------------------------------------------
# # # PyTorch reference
# # # -----------------------------------------------------------------------------
# # def _pt_baseline(P, V1, U2, V2, b1, b2):
# #     Z = P.matmul(V1) + b1.view(1, 1, -1)
# #     Zu, Zv = Z.split(Z.shape[-1] // 2, dim=-1)
# #     H  = F.gelu(Zu, approximate="none") * Zv
# #     S  = H.matmul(U2)
# #     Y  = S.matmul(V2) + b2.view(1, 1, -1)
# #     return Y

# # def main():
# #     device = torch.device("cuda")
# #     torch.manual_seed(0)

# #     B, L, H = 8, 2048, 768
# #     D       = 3072
# #     R1, R2  = 512, 512
# #     BL, BD, BR1, BR2 = 64, 128, 64, 64

# #     dtype = torch.float16
# #     X  = torch.randn((B, L, H), device=device, dtype=dtype)

# #     U1 = torch.randn((H, R1), device=device, dtype=dtype) / math.sqrt(H)
# #     V1 = torch.randn((R1, 2*D), device=device, dtype=dtype) / math.sqrt(R1)
# #     U2 = torch.randn((D,  R2), device=device, dtype=dtype) / math.sqrt(D)
# #     V2 = torch.randn((R2, H),  device=device, dtype=dtype) / math.sqrt(R2)
# #     b1 = torch.zeros((2*D,), device=device, dtype=dtype)
# #     b2 = torch.zeros((H,),   device=device, dtype=dtype)

# #     P  = X.matmul(U1)

# #     # Warmup
# #     _ = flashsvd_ffn_geglu(P, V1, U2, V2, b1, b2,
# #                            BL, BD, BR1, BR2,
# #                            nan_scrub=False, store_s_fp32=False)
# #     torch.cuda.synchronize()

# #     out_triton = flashsvd_ffn_geglu(P, V1, U2, V2, b1, b2,
# #                                     BL, BD, BR1, BR2,
# #                                     nan_scrub=False, store_s_fp32=False)
# #     out_pt     = _pt_baseline(P, V1, U2, V2, b1, b2)

# #     print("finite(triton):", torch.isfinite(out_triton).all().item(),
# #           " finite(pt):", torch.isfinite(out_pt).all().item())

# #     diff  = (out_triton - out_pt).to(torch.float32)
# #     print("finite(diff):", torch.isfinite(diff).all().item())
# #     rel_f = diff.norm() / (out_pt.to(torch.float32).norm() + 1e-12)
# #     print(f"Max abs error: {torch.nan_to_num(diff.abs().max(), nan=0.0).item():.3e}")
# #     print(f"Rel Fro error: {rel_f.item():.3e}")

# # if __name__ == "__main__":
# #     main()
