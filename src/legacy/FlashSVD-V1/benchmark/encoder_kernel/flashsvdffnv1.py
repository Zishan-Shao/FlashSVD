import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# -------------------------------------------------------------------
# Phase 1 Triton kernel: compute S = GELU(P @ V1 + b1) @ U2
# -------------------------------------------------------------------
@triton.jit
def fused_ffn_phase1(
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

    offs_b   = pid_b * sP_b
    offs_l   = pid_l * BL + tl.arange(0, BL)
    offs_r2  = pid_r2 * BR2 + tl.arange(0, BR2)

    # temp for hidden = P@V1 + b1, size [BL, BD]
    T_acc = tl.zeros((BL, BD), dtype=tl.float32)
    # accumulator for output S block, size [BL, BR2]
    acc   = tl.zeros((BL, BR2), dtype=tl.float32)

    for d0 in range(0, D, BD):
        d    = d0 + tl.arange(0, BD)
        m_d  = d < D

        # P @ V1
        T_acc *= 0.0
        for r1_0 in range(0, R1, BR1):
            r1   = r1_0 + tl.arange(0, BR1)
            m_r1 = r1 < R1
            P_sub = tl.load(
                P_ptr + offs_b + offs_l[:, None]*sP_l + r1[None, :]*sP_r1,
                mask=(offs_l[:, None] < L) & m_r1[None, :], other=0.0)
            V1_sub = tl.load(
                V1_ptr + r1[:, None]*sV1_r1 + d[None, :]*sV1_d,
                mask=m_r1[:, None] & m_d[None, :], other=0.0)
            T_acc += tl.dot(P_sub, V1_sub)

        # add bias b1 and GELU
        b1_d  = tl.load(b1_ptr + d*sb1, mask=m_d, other=0.0)
        T_acc += b1_d[None, :]
        T = T_acc * 0.5 * (1.0 + tl.erf(T_acc / tl.sqrt(2.0)))

        # T @ U2 for the assigned R2 block only
        U2_sub = tl.load(
            U2_ptr + d[:, None]*sU2_d + offs_r2[None, :]*sU2_r2,
            mask=m_d[:, None] & (offs_r2[None, :] < R2), other=0.0).to(tl.float32)
        acc += tl.dot(T, U2_sub)

    # store S block
    mask = (offs_l[:, None] < L) & (offs_r2[None, :] < R2)
    S_base = S_ptr + pid_b * sS_b
    tl.store(
        S_base + offs_l[:, None]*sS_l + offs_r2[None, :]*sS_r2,
        acc, mask=mask)

# -------------------------------------------------------------------
# Python wrapper: two-stage ffn under flashsvd_ffn_v1
# -------------------------------------------------------------------
def flashsvd_ffn_v1(
    P, V1, U2, V2, b1, b2,
    BL=64, BD=128, BR1=32, BR2=32,
):
    """
    Stage 1: Triton fused_ffn_phase1 -> S [B,L,R2]
    Stage 2: torch.matmul(S, V2) + b2
    """
    B, L, R1 = P.shape
    _, D      = V1.shape
    _, H      = V2.shape
    R2        = U2.shape[1]

    # allocate intermediate 
    S = torch.empty((B, L, R2), device=P.device, dtype=P.dtype)

    # strides dict
    strides = dict(
        sP_b=P.stride(0), sP_l=P.stride(1), sP_r1=P.stride(2),
        sV1_r1=V1.stride(0), sV1_d=V1.stride(1),
        sU2_d=U2.stride(0), sU2_r2=U2.stride(1),
        sb1=b1.stride(0),
        sS_b=S.stride(0), sS_l=S.stride(1), sS_r2=S.stride(2),
    )
    # print("DEBUG flashsvd_ffn_v1 strides")
    # print(f"P.shape={P.shape}, P.stride={P.stride()}")
    # print(f"V1.shape={V1.shape}, V1.stride={V1.stride()}")
    # print(f"U2.shape={U2.shape}, U2.stride={U2.stride()}")
    # print(f"S(shape is same as output buffer), strides dict:", strides)
    
    # launch Phase 1
    grid = (B, triton.cdiv(L, BL), triton.cdiv(R2, BR2))
    fused_ffn_phase1[grid](
        P, V1, U2, S, b1,
        B, L, D, R1, R2,
        *strides.values(), BL, BD, BR1, BR2,
    )

    # Phase 2: standard GEMM + bias
    # S: [B, L, R2], V2: [R2, H], b2: [H]
    C = torch.matmul(S, V2)
    C = C + b2.view(1,1,-1)
    return C

# -------------------------------------------------------------------
# main(): compare flashsvd_ffn_v1 vs PyTorch baseline
# -------------------------------------------------------------------
def main():
    device = torch.device('cuda')
    torch.manual_seed(0)

    # dims
    B, L, H, D, R1, R2 = 64, 512*4, 768, 3072, 512, 512
    BL, BD = 64, 64
    BR1, BR2 = 64, 64

    print("Batch Size:", B, "| Intermed D:", D, "| Rank R1:", R1)

    # random inputs
    X  = torch.randn((B, L, H), device=device)
    U1 = torch.randn((H, R1), device=device)
    P  = X.matmul(U1)             # [B, L, R1]
    V1 = torch.randn((R1, D), device=device)
    U2 = torch.randn((D,  R2), device=device)
    V2 = torch.randn((R2,  H), device=device)
    b1 = torch.randn((D,), device=device)
    b2 = torch.randn((H,), device=device)
    
    # Warm-up
    out_triton = flashsvd_ffn_v1(P, V1, U2, V2, b1, b2, BL, BD, BR1, BR2)
    torch.cuda.synchronize()

    # benchmarks
    triton_times, triton_mems = [], []
    pt_times, pt_mems         = [], []
    num_runs = 10

    for _ in range(num_runs):
        # Triton
        torch.cuda.reset_peak_memory_stats(device)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        out_t = flashsvd_ffn_v1(P, V1, U2, V2, b1, b2, BL, BD, BR1, BR2)
        torch.cuda.synchronize()
        end.record(); torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))
        triton_mems .append(torch.cuda.max_memory_allocated(device)/2**20)

        # PyTorch baseline (GELU+2 GEMMs)
        torch.cuda.reset_peak_memory_stats(device)
        start.record()
        Y1 = F.gelu(P.matmul(V1) + b1.view(1,1,-1))
        Y2 = Y1.matmul(U2)
        out_p = Y2.matmul(V2) + b2.view(1,1,-1)
        torch.cuda.synchronize()
        end.record(); torch.cuda.synchronize()
        pt_times.append(start.elapsed_time(end))
        pt_mems .append(torch.cuda.max_memory_allocated(device)/2**20)

    # accuracy
    diff    = out_t - out_p
    max_err = diff.abs().max().item()
    rel_f   = diff.norm().item()/out_p.norm().item()

    print(f"Max abs error:            {max_err:.3e}")
    print(f"Relative Frobenius error: {rel_f:.3e}\n")
    print(f"Triton v1 (avg {num_runs} runs): {sum(triton_times)/num_runs:.2f} ms, "
          f"{sum(triton_mems)/num_runs:.2f} MiB")
    print(f"PyTorch   (avg {num_runs} runs): {sum(pt_times)/num_runs:.2f} ms, "
          f"{sum(pt_mems)/num_runs:.2f} MiB")

if __name__ == "__main__":
    main()
