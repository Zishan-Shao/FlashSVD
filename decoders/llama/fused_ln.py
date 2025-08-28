import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def streaming_layernorm(
    x_ptr, w_ptr, b_ptr,
    B, M, D,
    s_xb, s_xm, s_xd,
    s_wd, s_bd,
    eps: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr
):  
    pid = tl.program_id(0)
    batch_idx = pid // M
    m_idx     = pid % M
    x_row = x_ptr + batch_idx * s_xb + m_idx * s_xm

    # first pass: sum & sumsq
    sum_val = tl.zeros([1], tl.float32)
    sumsq   = tl.zeros([1], tl.float32)
    offs   = tl.arange(0, BD)
    for d_start in range(0, D, BD):
        d = d_start + offs
        mask = d < D
        x = tl.load(x_row + d * s_xd, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sumsq   += tl.sum(x * x, axis=0)
    N = D
    mean = sum_val / N
    var  = sumsq / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize + affine
    for d_start in range(0, D, BD):
        d = d_start + offs
        mask = d < D
        x = tl.load(x_row + d * s_xd, mask=mask, other=0.0)
        w = tl.load(w_ptr + d * s_wd, mask=mask, other=1.0)
        b = tl.load(b_ptr + d * s_bd, mask=mask, other=0.0)
        y = (x - mean) * inv_std * w + b
        tl.store(x_row + d * s_xd, y, mask=mask)

def fused_layernorm(x, w, b, eps=1e-5, BM=1, BD=128):
    B, M, D = x.shape
    grid = lambda meta: (B * M, )
    
    print("Stride of x:", x.stride(0), x.stride(1), x.stride(2))
    print("Stride of w:", w.stride(0))
    print("Stride of b:", b.stride(0))
    
    streaming_layernorm[grid](
        x, w, b,
        B, M, D,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), b.stride(0),
        eps, BM, BD
    )

def main():
    torch.manual_seed(0)
    # dummy sizes (you can vary these)
    B, M, D = 2, 512, 384
    # random input
    x = torch.randn(B, M, D, device='cuda', dtype=torch.float32)
    # random weight & bias
    w = torch.randn(D, device='cuda', dtype=torch.float32)
    b = torch.randn(D, device='cuda', dtype=torch.float32)

    # reference with PyTorch
    x_ref = x.clone()
    y_ref = torch.layer_norm(x_ref, normalized_shape=(D,), weight=w, bias=b, eps=1e-5)

    # Triton in-place
    x_test = x.clone()
    fused_layernorm(x_test, w, b, eps=1e-5, BM=1, BD=128)

    # compare
    max_err = (x_test - y_ref).abs().max().item()
    print(f"Max absolute error: {max_err:.3e}")
    assert max_err < 1e-2, "Result mismatch!"

if __name__ == "__main__":
    main()
