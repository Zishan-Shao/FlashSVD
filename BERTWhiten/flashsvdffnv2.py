# flashsvdffn_bias.py
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def fused_ffn_full_batched_bias(
    P_ptr, V1_ptr, U2_ptr, V2_ptr, C_ptr,
    b1_ptr, b2_ptr,    # we add the bias vector of Q,K,V here
    B, L, D, H, R1, R2,
    sP_b, sP_l, sP_r1,
    sV1_r1, sV1_d,
    sU2_d, sU2_r2, sV2_r2, sV2_h,
    sb1, sb2,                       # Q,K,V bias strides
    sC_b, sC_l, sC_h,
    BL: tl.constexpr, BD: tl.constexpr, BH: tl.constexpr,
    BR1: tl.constexpr, BR2: tl.constexpr,
):
    # batch, sequence, width tile IDs
    pid_b = tl.program_id(0) # contiguous
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # offsets
    offs_Pb = pid_b * sP_b
    offs_Cb = pid_b * sC_b
    offs_l  = pid_l * BL + tl.arange(0, BL)
    offs_h  = pid_h * BH + tl.arange(0, BH)

    # output accumulator and pre-ReLU accumulator
    acc   = tl.zeros((BL, BH), dtype=tl.float32)
    T_acc = tl.zeros((BL, BD), dtype=tl.float32)

    # loop over intermediate dim D in BD-sized blocks
    for d0 in range(0, D, BD):
        d    = d0 + tl.arange(0, BD)
        m_d  = d < D
        
        # reset and build P@V1 + bias_1
        T_acc *= 0.0
        for r1_0 in range(0, R1, BR1):
            r1   = r1_0 + tl.arange(0, BR1)
            m_r1 = r1 < R1
            P_sub = tl.load( # [BL, BR1]
                P_ptr + offs_Pb
                      + offs_l[:, None]*sP_l
                      + r1[None, :]*sP_r1,
                mask=(offs_l[:, None] < L) & m_r1[None, :],
                other=0.0
            )
            V1_sub = tl.load( # [BR1, BD]
                V1_ptr + r1[:, None]*sV1_r1
                       + d[None, :]*sV1_d,
                mask=m_r1[:, None] & m_d[None, :],
                other=0.0
            )
            T_acc += tl.dot(P_sub, V1_sub)
        b1_sl = tl.load(b1_ptr + d*sb1, mask=m_d, other=0.0)
        T_acc += b1_sl[None, :]

        # # ReLU
        # T = tl.where(T_acc > 0, T_acc, 0.0)#.to(tl.float32)
        # GELU
        # Constants for tanh approximation of GELU
        # a = 0.7978845608  # sqrt(2/pi)
        # b = 0.044715
        # # GELU using tanh approximation
        # T = 0.5 * T_acc * (1.0 + tl.tanh(a * (T_acc + b * T_acc**3)))
        # Exact GELU using erf
        T = T_acc * 0.5 * (1.0 + tl.erf(T_acc / tl.sqrt(2.0)))

        # 2) fuse T @ U2 @ V2 into acc
        for r2_0 in range(0, R2, BR2):
            r2   = r2_0 + tl.arange(0, BR2)
            m_r2 = r2 < R2
            U2_sub = tl.load( # [BD, BR2]
                U2_ptr + d[:, None]*sU2_d
                       + r2[None, :]*sU2_r2,
                mask=m_d[:, None] & m_r2[None, :],
                other=0.0
            ).to(tl.float32)
            TU2 = tl.dot(T, U2_sub).to(tl.float32)  # [BL, BR2]
            V2_sub = tl.load( # [BR2, BH]
                V2_ptr + r2[:, None]*sV2_r2
                       + offs_h[None, :]*sV2_h,
                mask=m_r2[:, None] & (offs_h[None, :] < H),
                other=0.0
            ).to(tl.float32)
            acc = tl.dot(TU2, V2_sub, acc=acc)

    b2_sl = tl.load(b2_ptr + offs_h*sb2, mask=(offs_h < H), other=0.0)
    acc  += b2_sl[None, :]

    mask = (offs_l[:, None] < L) & (offs_h[None, :] < H)
    base = C_ptr + offs_Cb
    tl.store(
      base + offs_l[:, None]*sC_l + offs_h[None, :]*sC_h,
      acc, mask=mask
    )



def flashsvd_ffn(
    P, V1, U2, V2, b1, b2,
    BL=64, BD=128, BH=64,
    BR1=32, BR2=32,
):
    B, L, R1 = P.shape
    _, D      = V1.shape
    _, H      = V2.shape
    R2        = U2.shape[1]
    # print("XU1 shape:", P.shape)
    # print("V2 shape:", V2.shape)
    
    #print("▶︎ using FLASH-SVD-FFN on", P.shape, V1.shape, U2.shape, V2.shape)
    
    C = torch.empty((B, L, H), device=P.device, dtype=P.dtype)
    strides = dict(
        sP_b=P.stride(0), sP_l=P.stride(1), sP_r1=P.stride(2),
        sV1_r1=V1.stride(0), sV1_d=V1.stride(1),
        sU2_d=U2.stride(0), sU2_r2=U2.stride(1),
        sV2_r2=V2.stride(0), sV2_h=V2.stride(1),
        sb1=b1.stride(0), sb2=b2.stride(0),
        sC_b=C.stride(0), sC_l=C.stride(1), sC_h=C.stride(2),
    )
    grid = (B, triton.cdiv(L, BL), triton.cdiv(H, BH))
    fused_ffn_full_batched_bias[grid](
        P, V1, U2, V2, C, b1, b2,
        B, L, D, H, R1, R2,
        *strides.values(), BL, BD, BH, BR1, BR2
    )
    return C





def main():
    device = torch.device('cuda')
    torch.manual_seed(0)
    
    # Dimensions
    B, L, H, D, R1, R2 = 64, 512*4, 768, 3072, 71, 48
    BL, BD, BH = 64, 64, 64
    BR1, BR2 = 64, 64
    
    print("Batch Size:", B, " | Intermediate:", D, " | Model Rank:", R1)
    

    # Prepare data
    X  = torch.randn((B, L, H), device=device)
    U1 = torch.randn((H, R1), device=device)
    P  = X.matmul(U1)                       # [B, L, R1]
    V1 = torch.randn((R1, D), device=device)
    U2 = torch.randn((D,  R2), device=device)
    V2 = torch.randn((R2,  H), device=device)
    b1 = torch.randn((D,), device=device)
    b2 = torch.randn((H,), device=device)
    
    num_runs = 10

    # Warm-up
    out_triton = flashsvd_ffn(P, V1, U2, V2, b1, b2, BL, BD, BH, BR1, BR2)
    torch.cuda.synchronize()

    # Storage for metrics
    triton_times = []
    triton_mems = []
    pt_times = []
    pt_mems = []

    for _ in range(num_runs):
        # Triton benchmark
        torch.cuda.reset_peak_memory_stats(device)
        start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_evt.record()
        out_triton = flashsvd_ffn(P, V1, U2, V2, b1, b2, BL, BD, BH, BR1, BR2)
        torch.cuda.synchronize()
        end_evt.record()
        torch.cuda.synchronize()
        triton_times.append(start_evt.elapsed_time(end_evt))
        triton_mems.append(torch.cuda.max_memory_allocated(device) / (1024**2))

        # PyTorch baseline
        torch.cuda.reset_peak_memory_stats(device)
        start_evt.record()
        Y1 = F.relu(P.matmul(V1) + b1.unsqueeze(0).unsqueeze(0))
        Y2 = Y1.matmul(U2)
        out_pt = Y2.matmul(V2) + b2.unsqueeze(0).unsqueeze(0)
        torch.cuda.synchronize()
        end_evt.record()
        torch.cuda.synchronize()
        pt_times.append(start_evt.elapsed_time(end_evt))
        pt_mems.append(torch.cuda.max_memory_allocated(device) / (1024**2))

    # Accuracy
    diff    = out_triton - out_pt
    max_err = diff.abs().max().item()
    rel_fro = diff.norm().item() / out_pt.norm().item()
    print(f"Max abs error:            {max_err:.3e}")
    print(f"Relative Frobenius error: {rel_fro:.3e}\n")
    print(f"Triton (avg over {num_runs} runs): time = {sum(triton_times)/num_runs:.2f} ms, peak mem = {sum(triton_mems)/num_runs:.2f} MiB")
    print(f"PyTorch (avg over {num_runs} runs): time = {sum(pt_times)/num_runs:.2f} ms, peak mem = {sum(pt_mems)/num_runs:.2f} MiB")


    # # Warm‐up Triton kernel
    # #out_triton = run_fused_ffn_batched_bias(P, V1, U2, V2, b1, b2, BL, BD, BH, BR1, BR2)
    # out_triton = flashsvd_ffn(P, V1, U2, V2, b1, b2, BL, BD, BH, BR1, BR2)
    # torch.cuda.synchronize()

    # # Benchmark Triton
    # torch.cuda.reset_peak_memory_stats(device)
    # start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # start_evt.record()
    # #out_triton = run_fused_ffn_batched_bias(P, V1, U2, V2, b1, b2, BL, BD, BH, BR1, BR2)
    # out_triton = flashsvd_ffn(P, V1, U2, V2, b1, b2, BL, BD, BH, BR1, BR2)
    # torch.cuda.synchronize()
    # end_evt.record()
    # torch.cuda.synchronize()
    # triton_time = start_evt.elapsed_time(end_evt)      # ms
    # triton_mem  = torch.cuda.max_memory_allocated(device) / (1024**2)

    # # PyTorch baseline
    # torch.cuda.reset_peak_memory_stats(device)
    # start_evt.record()
    # # Dense reference
    # Y1 = F.relu(P.matmul(V1) + b1.unsqueeze(0).unsqueeze(0))   # [B,L,D]
    # Y2 = Y1.matmul(U2)                                         # [B,L,R2]
    # out_pt = Y2.matmul(V2) + b2.unsqueeze(0).unsqueeze(0)      # [B,L,H]
    # torch.cuda.synchronize()
    # end_evt.record()
    # torch.cuda.synchronize()
    # pt_time = start_evt.elapsed_time(end_evt)
    # pt_mem  = torch.cuda.max_memory_allocated(device) / (1024**2)

    # # Accuracy check
    # diff    = out_triton - out_pt
    # max_err = diff.abs().max().item()
    # rel_fro = diff.norm().item() / out_pt.norm().item()

    # print(f"Max abs error:            {max_err:.3e}")
    # print(f"Relative Frobenius error: {rel_fro:.3e}\n")
    
    # print(f"Triton: time = {triton_time:.2f} ms, peak mem = {triton_mem:.2f} MiB")
    # print(f"PyTorch: time = {pt_time:.2f} ms, peak mem = {pt_mem:.2f} MiB")

if __name__ == "__main__":
    main()
