#!/usr/bin/env python3

# NOTE: this one is not that desirable, we should use the normal version
import math
import statistics as stats
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# ============================================================
# Phase-1 kernel (kept for comparison): S = (GEGLU(PV1+b1)) @ U2
# ============================================================
@triton.jit
def fused_ffn_phase1_geglu(
    P_ptr, V1_ptr, U2_ptr, S_ptr, b1_ptr,
    B, L, D, R1, R2,
    sP_b, sP_l, sP_r1,
    sV1_r1, sV1_d,
    sU2_d, sU2_r2,
    sb1,
    sS_b, sS_l, sS_r2,
    BL: tl.constexpr, BD: tl.constexpr, BR1: tl.constexpr, BR2: tl.constexpr,
    USE_TANH: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
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

            P_blk = tl.load(P_ptr + pid_b*sP_b + offs_l[:,None]*sP_l + r1[None,:]*sP_r1,
                            mask=(offs_l[:,None]<L) & m_r1[None,:], other=0.0)

            V1u_blk = tl.load(V1_ptr + r1[:,None]*sV1_r1 + d[None,:]*sV1_d,
                              mask=m_r1[:,None] & m_d[None,:], other=0.0)
            V1v_blk = tl.load(V1_ptr + r1[:,None]*sV1_r1 + (d[None,:]+D)*sV1_d,
                              mask=m_r1[:,None] & m_d[None,:], other=0.0)

            Tu_acc += tl.dot(P_blk, V1u_blk)
            Tv_acc += tl.dot(P_blk, V1v_blk)

        b1u = tl.load(b1_ptr + d*sb1,        mask=m_d, other=0.0).to(tl.float32)
        b1v = tl.load(b1_ptr + (d + D)*sb1,  mask=m_d, other=0.0).to(tl.float32)
        Tu  = Tu_acc + b1u[None, :]
        Tv  = Tv_acc + b1v[None, :]

        if USE_TANH:
            z  = c0 * (Tu + c1 * Tu * Tu * Tu)
            z2 = 2.0 * z
            sig_2z = tl.where(z2 >= 0, 1.0/(1.0+tl.exp(-z2)), tl.exp(z2)/(1.0+tl.exp(z2)))
            tanh_z = 2.0 * sig_2z - 1.0
            Hu = 0.5 * Tu * (1.0 + tanh_z)
        else:
            Hu = 0.5 * Tu * (1.0 + tl.erf(Tu * inv_sqrt2))

        H = Hu * Tv

        U2_blk = tl.load(U2_ptr + d[:,None]*sU2_d + offs_r2[None,:]*sU2_r2,
                         mask=m_d[:,None] & (offs_r2[None,:] < R2), other=0.0).to(tl.float32)
        acc += tl.dot(H, U2_blk)

    mask = (offs_l[:, None] < L) & (offs_r2[None, :] < R2)
    tl.store(S_ptr + pid_b*sS_b + offs_l[:,None]*sS_l + offs_r2[None,:]*sS_r2, acc, mask=mask)

# ============================================================
# FUSED kernel (no S buffer): Y = GEGLU(PV1+b1) @ (U2 @ V2) + b2
#   Grid: (B, L // BL, H // BH)
#   Inner loops: d in D by BD, r1 in R1 by BR1, r2 in R2 by BR2
# ============================================================
@triton.jit
def fused_ffn_phase12_geglu(
    P_ptr, V1_ptr, U2_ptr, V2_ptr, Y_ptr, b1_ptr, b2_ptr,
    B, L, D, R1, R2, Hdim,
    sP_b, sP_l, sP_r1,
    sV1_r1, sV1_d,
    sU2_d, sU2_r2,
    sV2_r2, sV2_h,
    sY_b, sY_l, sY_h,
    sb1, sb2,
    BL: tl.constexpr, BD: tl.constexpr, BR1: tl.constexpr, BH: tl.constexpr, BR2: tl.constexpr,
    USE_TANH: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid_b  = tl.program_id(0)
    pid_l  = tl.program_id(1)
    pid_h  = tl.program_id(2)

    offs_l = pid_l * BL + tl.arange(0, BL)
    offs_h = pid_h * BH + tl.arange(0, BH)

    Tu_acc = tl.zeros((BL, BD), dtype=tl.float32)
    Tv_acc = tl.zeros((BL, BD), dtype=tl.float32)
    accY   = tl.zeros((BL, BH), dtype=tl.float32)

    c0 = 0.7978845608028654
    c1 = 0.044715
    inv_sqrt2 = 0.7071067811865476

    for d0 in range(0, D, BD):
        d   = d0 + tl.arange(0, BD)
        m_d = d < D

        # build Tu/Tv for this d-tile by accumulating over R1
        Tu_acc *= 0.0
        Tv_acc *= 0.0
        for r1_0 in range(0, R1, BR1):
            r1   = r1_0 + tl.arange(0, BR1)
            m_r1 = r1 < R1

            P_blk = tl.load(P_ptr + pid_b*sP_b + offs_l[:,None]*sP_l + r1[None,:]*sP_r1,
                            mask=(offs_l[:,None] < L) & m_r1[None,:], other=0.0)
            V1u_blk = tl.load(V1_ptr + r1[:,None]*sV1_r1 + d[None,:]*sV1_d,
                              mask=m_r1[:,None] & m_d[None,:], other=0.0)
            V1v_blk = tl.load(V1_ptr + r1[:,None]*sV1_r1 + (d[None,:]+D)*sV1_d,
                              mask=m_r1[:,None] & m_d[None,:], other=0.0)

            Tu_acc += tl.dot(P_blk, V1u_blk)
            Tv_acc += tl.dot(P_blk, V1v_blk)

        b1u = tl.load(b1_ptr + d*sb1,        mask=m_d, other=0.0).to(tl.float32)
        b1v = tl.load(b1_ptr + (d + D)*sb1,  mask=m_d, other=0.0).to(tl.float32)
        Tu  = Tu_acc + b1u[None, :]
        Tv  = Tv_acc + b1v[None, :]

        # GEGLU: H_blk = gelu(Tu) * Tv
        if USE_TANH:
            z  = c0 * (Tu + c1 * Tu * Tu * Tu)
            z2 = 2.0 * z
            sig_2z = tl.where(z2 >= 0, 1.0/(1.0+tl.exp(-z2)), tl.exp(z2)/(1.0+tl.exp(z2)))
            tanh_z = 2.0 * sig_2z - 1.0
            Hu = 0.5 * Tu * (1.0 + tanh_z)
        else:
            Hu = 0.5 * Tu * (1.0 + tl.erf(Tu * inv_sqrt2))
        H_blk = Hu * Tv  # [BL,BD]

        # G_acc = sum over r2 tiles of U2_d@V2_r2h   → [BD,BH]
        G_acc = tl.zeros((BD, BH), dtype=tl.float32)
        for r2_0 in range(0, R2, BR2):
            r2   = r2_0 + tl.arange(0, BR2)
            m_r2 = r2 < R2

            V2_blk = tl.load(V2_ptr + r2[:,None]*sV2_r2 + offs_h[None,:]*sV2_h,
                             mask=m_r2[:,None] & (offs_h[None,:] < Hdim), other=0.0)
            U2_blk = tl.load(U2_ptr + d[:,None]*sU2_d + r2[None,:]*sU2_r2,
                             mask=m_d[:,None] & m_r2[None,:], other=0.0)

            G_acc += tl.dot(U2_blk.to(tl.float32), V2_blk.to(tl.float32))  # [BD,BH]

        accY += tl.dot(H_blk, G_acc)  # [BL,BH]

    # write Y + b2
    bmask_h = offs_h < Hdim
    b2_blk = tl.load(b2_ptr + offs_h*sb2, mask=bmask_h, other=0.0).to(tl.float32)
    Y_base = Y_ptr + pid_b*sY_b
    mask = (offs_l[:, None] < L) & bmask_h[None, :]
    tl.store(Y_base + offs_l[:,None]*sY_l + offs_h[None,:]*sY_h,
             accY + b2_blk[None, :], mask=mask)

# ============================================================
# Python wrappers
# ============================================================
def flashsvd_ffn_geglu_two_stage(
    P, V1, U2, V2, b1, b2,
    BL=64, BD=128, BR1=64, BR2=64,
    gelu_approx: str = "tanh",
):
    B, L, R1 = P.shape
    R1_v1, twoD = V1.shape
    D = twoD // 2
    R2 = U2.shape[1]
    H  = V2.shape[1]

    S = torch.empty((B, L, R2), device=P.device, dtype=P.dtype)
    strides = dict(
        sP_b=P.stride(0), sP_l=P.stride(1), sP_r1=P.stride(2),
        sV1_r1=V1.stride(0), sV1_d=V1.stride(1),
        sU2_d=U2.stride(0), sU2_r2=U2.stride(1),
        sb1=b1.stride(0),
        sS_b=S.stride(0), sS_l=S.stride(1), sS_r2=S.stride(2),
    )
    grid = (B, triton.cdiv(L, BL), triton.cdiv(R2, BR2))
    USE_TANH = 1 if gelu_approx == "tanh" else 0
    fused_ffn_phase1_geglu[grid](
        P, V1, U2, S, b1,
        B, L, D, R1, R2,
        *strides.values(), BL, BD, BR1, BR2,
        USE_TANH=USE_TANH,
        num_warps=4, num_stages=2,
    )
    Y = S.matmul(V2)
    Y = Y + b2.view(1, 1, -1)
    return Y

def flashsvd_ffn_geglu_fused(
    P, V1, U2, V2, b1, b2,
    BL=64, BD=128, BR1=64, BH=128, BR2=64,   # <-- add BR2
    gelu_approx: str = "tanh",
):
    B, L, R1 = P.shape
    D = V1.shape[1] // 2
    R2, Hdim = V2.shape[0], V2.shape[1]
    Y = torch.empty((B, L, Hdim), device=P.device, dtype=P.dtype)

    strides = dict(
        sP_b=P.stride(0), sP_l=P.stride(1), sP_r1=P.stride(2),
        sV1_r1=V1.stride(0), sV1_d=V1.stride(1),
        sU2_d=U2.stride(0), sU2_r2=U2.stride(1),
        sV2_r2=V2.stride(0), sV2_h=V2.stride(1),
        sY_b=Y.stride(0), sY_l=Y.stride(1), sY_h=Y.stride(2),
        sb1=b1.stride(0), sb2=b2.stride(0),
    )
    grid = (B, triton.cdiv(L, BL), triton.cdiv(Hdim, BH))
    USE_TANH = 1 if gelu_approx == "tanh" else 0

    fused_ffn_phase12_geglu[grid](
        P, V1, U2, V2, Y, b1, b2,
        B, L, D, R1, R2, Hdim,
        *strides.values(),
        BL, BD, BR1, BH, BR2,      # <-- pass BR2 here
        USE_TANH=USE_TANH,
        num_warps=4, num_stages=2,
    )
    return Y

def _pt_baseline(P, V1, U2, V2, b1, b2, gelu_approx="tanh"):
    Z  = P.matmul(V1) + b1.view(1, 1, -1)
    Zu, Zv = Z.split(Z.shape[-1] // 2, dim=-1)
    H  = F.gelu(Zu, approximate=gelu_approx) * Zv
    S  = H.matmul(U2)
    Y  = S.matmul(V2) + b2.view(1, 1, -1)
    return Y

# ============================================================
# Bench helpers
# ============================================================
def mib(nbytes): return nbytes / (1024**2)

@torch.no_grad()
def bench(fn, n_warmup=10, n_runs=100):
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
        times.append(start.elapsed_time(end))
        _ = out.view(-1)[0].item()
    return {
        "mean_ms": sum(times)/len(times),
        "median_ms": stats.median(times),
        "p95_ms": stats.quantiles(times, n=20)[-1] if len(times) >= 20 else max(times),
    }

# ============================================================
# Main
# ============================================================
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Problem sizes (same as your test)
    B, L, H = 8, 2048, 768
    D       = 3072
    R1, R2  = 512, 512

    # Triton tiles (try these; tune later)
    BL, BD, BR1 = 64, 64, 64
    BR2, BH     = 64, 64

    dtype = torch.float16
    # Make inputs contiguous for better striding
    X  = torch.randn((B, L, H), device=device, dtype=dtype).contiguous()
    U1 = (torch.randn((H, R1),    device=device, dtype=dtype) / math.sqrt(H)).contiguous()
    V1 = (torch.randn((R1, 2*D),  device=device, dtype=dtype) / math.sqrt(R1)).contiguous()
    U2 = (torch.randn((D,  R2),   device=device, dtype=dtype) / math.sqrt(D)).contiguous()
    V2 = (torch.randn((R2, H),    device=device, dtype=dtype) / math.sqrt(R2)).contiguous()
    b1 = torch.zeros((2*D,), device=device, dtype=dtype).contiguous()
    b2 = torch.zeros((H,),   device=device, dtype=dtype).contiguous()

    P  = X.matmul(U1).contiguous()

    # Compile/fuse warmups
    _ = flashsvd_ffn_geglu_two_stage(P, V1, U2, V2, b1, b2, BL, BD, BR1, BR2)
    _ = flashsvd_ffn_geglu_fused(P, V1, U2, V2, b1, b2, BL, BD, BR1, BH)

    # Correctness (vs baseline)
    Y_base  = _pt_baseline(P, V1, U2, V2, b1, b2)
    Y_two   = flashsvd_ffn_geglu_two_stage(P, V1, U2, V2, b1, b2, BL, BD, BR1, BR2)
    Y_fused = flashsvd_ffn_geglu_fused(P, V1, U2, V2, b1, b2, BL, BD, BR1, BH)

    for name, y in [("two-stage", Y_two), ("fused", Y_fused)]:
        diff  = (y - Y_base).to(torch.float32)
        rel_f = diff.norm() / (Y_base.to(torch.float32).norm() + 1e-12)
        print(f"[check {name}] finite:", torch.isfinite(diff).all().item(),
              " max|err|:", torch.nan_to_num(diff.abs().max(), nan=0.0).item(),
              " relF:", rel_f.item())

    # Speed (tokens/sec)
    tokens = B * L
    base_t = bench(lambda: _pt_baseline(P, V1, U2, V2, b1, b2))
    two_t  = bench(lambda: flashsvd_ffn_geglu_two_stage(P, V1, U2, V2, b1, b2, BL, BD, BR1, BR2))
    fuse_t = bench(lambda: flashsvd_ffn_geglu_fused(P, V1, U2, V2, b1, b2, BL, BD, BR1, BH))

    def tps(ms): return tokens / (ms / 1e3)

    print("\n=== Inference Speed (latency per forward) ===")
    print("Baseline (PyTorch): "
          f"mean {base_t['mean_ms']:.2f} ms | median {base_t['median_ms']:.2f} ms | p95 {base_t['p95_ms']:.2f} ms "
          f"| {tps(base_t['mean_ms']):,.0f} tok/s")
    print("FlashSVD (Triton two-stage): "
          f"mean {two_t['mean_ms']:.2f} ms | median {two_t['median_ms']:.2f} ms | p95 {two_t['p95_ms']:.2f} ms "
          f"| {tps(two_t['mean_ms']):,.0f} tok/s")
    print("FlashSVD (Triton fused Phase1+2): "
          f"mean {fuse_t['mean_ms']:.2f} ms | median {fuse_t['median_ms']:.2f} ms | p95 {fuse_t['p95_ms']:.2f} ms "
          f"| {tps(fuse_t['mean_ms']):,.0f} tok/s")

if __name__ == "__main__":
    main()
