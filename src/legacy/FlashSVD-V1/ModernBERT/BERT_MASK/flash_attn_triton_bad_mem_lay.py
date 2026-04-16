#!/usr/bin/env python3
import math, time
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ───────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────
def build_padding_mask_4d(attn_2d: torch.Tensor, B: int, H: int, M: int) -> torch.Tensor:
    """attn_2d: [B, M] with 1=valid, 0=pad  →  [B,H,1,M] bool"""
    assert attn_2d.dim() == 2 and attn_2d.shape == (B, M)
    mask_4d = attn_2d[:, None, None, :].to(torch.bool)  # [B,1,1,M]
    return mask_4d.expand(B, H, 1, M).contiguous()

def apply_rope_inplace(Q: torch.Tensor, K: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    RoPE in-place on Q/K. Shapes: Q/K [B,H,M,D], cos/sin [B,H,M,D].
    Uses local clones of halves to avoid aliasing bugs.
    """
    q1, q2 = Q.chunk(2, dim=-1)
    k1, k2 = K.chunk(2, dim=-1)
    q1o, q2o = q1.clone(), q2.clone()
    k1o, k2o = k1.clone(), k2.clone()
    Dh = q1.shape[-1]
    c, s = cos[..., :Dh], sin[..., :Dh]

    # Q
    q1.copy_(q1o * c - q2o * s)
    q2.copy_(q1o * s + q2o * c)
    # K
    k1.copy_(k1o * c - k2o * s)
    k2.copy_(k1o * s + k2o * c)


# ───────────────────────────────────────────────────────────────
# Triton kernels (Tri Dao style FlashAttention)
# ───────────────────────────────────────────────────────────────
@triton.jit
def _load_tile(
    ptr, off_b, off_h, row_off,
    sPb, sPh, sPm, sPd,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, seqlen
):
    offs_m = row_off + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    ptrs = (
        ptr
        + off_b * sPb
        + off_h * sPh
        + offs_m[:, None] * sPm
        + offs_d[None, :] * sPd
    )
    mask = offs_m < seqlen
    return tl.load(ptrs, mask=mask[:, None], other=0.).to(tl.float32)


@triton.jit
def flashattn_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, mask_ptr,
    sMb, sMh, sMq, sMk,    # mask strides [B,H,1,M]
    sQb, sQh, sQm, sQd,
    sKb, sKh, sKm, sKd,
    sVb, sVh, sVm, sVd,
    sOb, sOm, sOd,
    seqlen, nheads, softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # program ids
    bid = tl.program_id(0)         # along M-tiles
    bh  = tl.program_id(1)         # B*H
    off_b = bh // nheads
    off_h = bh %  nheads
    row_off = bid * BLOCK_M

    # load Q tile
    q = _load_tile(Q_ptr, off_b, off_h, row_off,
                   sQb, sQh, sQm, sQd,
                   BLOCK_M, BLOCK_D, seqlen)
    # online softmax state
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # iterate over K/V tiles along sequence
    for kb in range(0, seqlen, BLOCK_M):
        kb = tl.multiple_of(kb, BLOCK_M)
        offs_n = kb + tl.arange(0, BLOCK_M)

        # keys valid mask (keys-only semantics)
        mask_ptrs = (
            mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + offs_n * sMk
        )
        mask_i = tl.load(mask_ptrs, mask=offs_n < seqlen, other=0).to(tl.int32)
        mask_bool = mask_i > 0

        k = _load_tile(K_ptr, off_b, off_h, kb,
                       sKb, sKh, sKm, sKd,
                       BLOCK_M, BLOCK_D, seqlen)
        v = _load_tile(V_ptr, off_b, off_h, kb,
                       sVb, sVh, sVm, sVd,
                       BLOCK_M, BLOCK_D, seqlen)

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neginf = tl.full(qk.shape, -float("inf"), tl.float32)
        qk = tl.where(mask_bool[None, :], qk, neginf)

        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        p        = tl.exp(qk - m_new[:, None])
        l_new    = tl.sum(p, axis=1) + l_i * tl.exp(m_i - m_new)
        acc      = tl.dot(p, v) + acc * tl.exp(m_i - m_new)[:, None]
        m_i, l_i = m_new, l_new

    out = acc / tl.reshape(l_i, (BLOCK_M, 1))

    # store
    offs_m = row_off + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    out_ptrs = Out_ptr + bh * sOb + offs_m[:, None] * sOm + offs_d[None, :] * sOd
    tl.store(out_ptrs, out, mask=offs_m[:, None] < seqlen)


# ───────────────────────────────────────────────────────────────
# Autotuned kernel variant (tune BLOCK_M, warps, stages)
# ───────────────────────────────────────────────────────────────
TUNE_CONFIGS_FA: List[triton.Config] = [
    triton.Config({'BLOCK_M': 16},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 32},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64},  num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=1),  # fewer stages to ease smem
]

@triton.autotune(configs=TUNE_CONFIGS_FA, key=['seqlen'])
@triton.jit
def flashattn_kernel_auto(
    Q_ptr, K_ptr, V_ptr, Out_ptr, mask_ptr,
    sMb, sMh, sMq, sMk,
    sQb, sQh, sQm, sQd,
    sKb, sKh, sKm, sKd,
    sVb, sVh, sVm, sVd,
    sOb, sOm, sOd,
    seqlen, nheads, softmax_scale,
    BLOCK_M: tl.constexpr,         # <-- accept from autotuner
    BLOCK_D: tl.constexpr,         # head_dim constexpr
):
    flashattn_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr, mask_ptr,
        sMb, sMh, sMq, sMk,
        sQb, sQh, sQm, sQd,
        sKb, sKh, sKm, sKd,
        sVb, sVh, sVm, sVd,
        sOb, sOm, sOd,
        seqlen, nheads, softmax_scale,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
    )


# ───────────────────────────────────────────────────────────────
# Python wrappers
# ───────────────────────────────────────────────────────────────
def flash_attn_triton(Q: torch.Tensor,
                      K: torch.Tensor,
                      V: torch.Tensor,
                      mask_bh1m: torch.Tensor,
                      *,
                      use_autotune: bool = True,
                      BLOCK_M: int = 64) -> torch.Tensor:
    """
    Q,K,V:   [B, H, M, D]  (fp16/bf16/fp32 inputs OK; kernel does fp32 math)
    mask:    [B, H, 1, M] bool  (True = keep, False = mask out)
    returns: [B, H, M, D] (fp32)
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda and mask_bh1m.is_cuda
    assert Q.shape == K.shape == V.shape
    B, H, M, D = Q.shape
    softmax_scale = 1.0 / math.sqrt(D)

    Qc = Q.contiguous()
    Kc = K.contiguous()
    Vc = V.contiguous()
    Mc = mask_bh1m.contiguous().to(torch.int32)

    Out = torch.empty(B * H, M, D, device=Q.device, dtype=torch.float32)

    args = [
        Qc, Kc, Vc, Out,
        Mc, *Mc.stride(),      # sMb, sMh, sMq, sMk
        *Qc.stride(),          # sQb, sQh, sQm, sQd
        *Kc.stride(),          # sKb, sKh, sKm, sKd
        *Vc.stride(),          # sVb, sVh, sVm, sVd
        *Out.stride(),         # sOb, sOm, sOd
        M, H, softmax_scale,
    ]
    if use_autotune:
        grid = lambda META: ((M + META['BLOCK_M'] - 1) // META['BLOCK_M'], B * H)
        flashattn_kernel_auto[grid](
            *args,
            BLOCK_D=D,  # BLOCK_M is supplied via autotune config → kernel param
        )
    else:
        grid = ((M + BLOCK_M - 1) // BLOCK_M, B * H)
        flashattn_kernel[grid](
            *args,
            BLOCK_M=BLOCK_M,
            BLOCK_D=D
        )

    return Out.view(B, H, M, D)


def profile_flash_attn(Q, K, V, mask_bh1m,
                       sweep: Optional[List[Dict]] = None,
                       warmup=5, iters=20):
    """Manual sweep over BLOCK_M / warps / stages (pinned kernel path)."""
    if sweep is None:
        sweep = [
            {'BLOCK_M': 16,  'num_warps': 4, 'num_stages': 2},
            {'BLOCK_M': 32,  'num_warps': 4, 'num_stages': 2},
            {'BLOCK_M': 64,  'num_warps': 4, 'num_stages': 2},
            {'BLOCK_M': 64,  'num_warps': 8, 'num_stages': 2},
            {'BLOCK_M': 128, 'num_warps': 8, 'num_stages': 1},
        ]
    results = []

    # warm compile
    _run_pinned(Q, K, V, mask_bh1m, **sweep[0])

    for cfg in sweep:
        # warmup
        for _ in range(warmup):
            _ = _run_pinned(Q, K, V, mask_bh1m, **cfg)
        torch.cuda.synchronize()

        # time
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            start.record()
            _ = _run_pinned(Q, K, V, mask_bh1m, **cfg)
            end.record()
            torch.cuda.synchronize()
            ts.append(start.elapsed_time(end))
        mean_ms = float(sum(ts) / len(ts))
        results.append((mean_ms, cfg))

    results.sort(key=lambda x: x[0])
    print("\n=== Manual profile over FlashAttention configs ===")
    for mean_ms, cfg in results:
        print(f"BLOCK_M={cfg['BLOCK_M']:>3} | warps={cfg['num_warps']} stages={cfg['num_stages']} | mean={mean_ms:.3f} ms")
    best = results[0]
    print(f"\nBest: BLOCK_M={best[1]['BLOCK_M']}, warps={best[1]['num_warps']}, stages={best[1]['num_stages']} | mean={best[0]:.3f} ms")
    return best

def _run_pinned(Q, K, V, mask_bh1m, BLOCK_M=64, num_warps=4, num_stages=2):
    """Pinned launch helper for profiling."""
    B, H, M, D = Q.shape
    softmax_scale = 1.0 / math.sqrt(D)
    Qc, Kc, Vc = Q.contiguous(), K.contiguous(), V.contiguous()
    Mc = mask_bh1m.contiguous().to(torch.int32)
    Out = torch.empty(B * H, M, D, device=Q.device, dtype=torch.float32)

    args = [
        Qc, Kc, Vc, Out, Mc, *Mc.stride(),
        *Qc.stride(), *Kc.stride(), *Vc.stride(), *Out.stride(),
        M, H, softmax_scale
    ]
    grid = ((M + BLOCK_M - 1) // BLOCK_M, B * H)
    flashattn_kernel[grid](
        *args,
        BLOCK_M=BLOCK_M,
        BLOCK_D=D,
        num_warps=num_warps, num_stages=num_stages,
    )
    return Out.view(B, H, M, D)


# ───────────────────────────────────────────────────────────────
# Minimal ModernBERT attention wrapper using this kernel
#   - builds Q/K/V from Wqkv
#   - applies RoPE (torch)
#   - calls FlashAttention kernel (autotuned by default)
# ───────────────────────────────────────────────────────────────
@dataclass
class RoPEAdapter:
    """Thin adapter around HF rotary_emb: returns cos/sin for Q-like input."""
    rotary_emb: any
    def cos_sin(self, B: int, H: int, M: int, dh: int, device, dtype, position_ids: torch.Tensor):
        q_like = torch.empty((B*H, M, dh), device=device, dtype=dtype)
        posf = position_ids.unsqueeze(1).expand(B, H, M).reshape(B*H, M)
        cos, sin = self.rotary_emb(q_like, position_ids=posf)  # [(B*H), M, dh]
        return cos.view(B, H, M, dh).contiguous(), sin.view(B, H, M, dh).contiguous()

class ModernBERTFlashAttnBlock(nn.Module):
    def __init__(self, hf_layer, cfg, use_autotune: bool = True, pinned: Optional[Dict]=None):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hf_layer.attn.Wo.in_features
        self.num_heads   = cfg.num_attention_heads
        self.head_dim    = self.hidden_size // self.num_heads
        self.attn_norm   = nn.LayerNorm(self.hidden_size, eps=hf_layer.attn_norm.eps)
        self.rotary      = RoPEAdapter(hf_layer.attn.rotary_emb)
        # weight copies
        self.Wqkv = nn.Linear(self.hidden_size, 3*self.hidden_size, bias=True)
        self.Wqkv.load_state_dict(hf_layer.attn.Wqkv.state_dict())
        self.Wo   = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wo.load_state_dict(hf_layer.attn.Wo.state_dict())
        # keep HF MLP as-is to limit scope; swap only attention
        self.mlp_norm = hf_layer.mlp_norm
        self.mlp      = hf_layer.mlp
        # kernel knobs
        self.use_autotune = bool(use_autotune)
        self.pinned = pinned or {'BLOCK_M': 64, 'num_warps': 4, 'num_stages': 2}

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None):
        B, M, D = hidden_states.shape
        H, dh = self.num_heads, self.head_dim
        x = hidden_states
        xn = self.attn_norm(x)

        # QKV
        qkv = self.Wqkv(xn)                                      # [B,M,3D]
        q, k, v = qkv.view(B, M, 3, H, dh).permute(0, 3, 1, 2, 4).unbind(dim=3)  # each [B,H,M,dh]

        # RoPE
        if position_ids is None:
            position_ids = torch.arange(M, device=x.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary.cos_sin(B, H, M, dh, x.device, q.dtype, position_ids)
        apply_rope_inplace(q, k, cos, sin)

        # mask (keys-only semantics): prefer [B,H,1,M] bool
        if attention_mask is None:
            mask_bh1m = torch.ones(B, H, 1, M, device=x.device, dtype=torch.bool)
        elif attention_mask.dim() == 2:
            mask_bh1m = build_padding_mask_4d(attention_mask, B, H, M)
        elif attention_mask.dim() == 4:
            mask_bh1m = (attention_mask >= 0).to(torch.bool)
            assert mask_bh1m.shape == (B, H, 1, M)
        else:
            raise ValueError("Unsupported attention_mask shape")

        # kernel
        if self.use_autotune:
            o = flash_attn_triton(q, k, v, mask_bh1m, use_autotune=True)      # [B,H,M,dh] fp32
        else:
            o = _run_pinned(q, k, v, mask_bh1m, **self.pinned)                # [B,H,M,dh] fp32

        attn_out = o.transpose(1, 2).reshape(B, M, D).to(x.dtype)
        x = x + self.Wo(attn_out)
        # MLP unchanged
        x = x + self.mlp(self.mlp_norm(x))
        return (x,)


# ───────────────────────────────────────────────────────────────
# Quick self-test / latency + correctness demo
# ───────────────────────────────────────────────────────────────
def _demo_correctness_and_speed():
    torch.manual_seed(0)
    dev = "cuda"
    B, H, M, D = 32, 12, 1024, 64
    scale = 1.0 / math.sqrt(D)

    # random Q/K/V and mask
    Q = torch.randn(B, H, M, D, device=dev, dtype=torch.float16)
    K = torch.randn(B, H, M, D, device=dev, dtype=torch.float16)
    V = torch.randn(B, H, M, D, device=dev, dtype=torch.float16)
    mask = torch.zeros(B, H, 1, M, device=dev, dtype=torch.bool)
    valid = int(M * 0.9)
    mask[..., :valid] = True

    # reference (fp32)
    Q32, K32, V32 = Q.float(), K.float(), V.float()
    logits = torch.einsum("bhmd,bhnd->bhmn", Q32, K32) * scale
    logits = logits.masked_fill(~mask, float("-inf"))
    ref = torch.einsum("bhmn,bhnd->bhmd", torch.softmax(logits, dim=-1), V32)

    # autotuned kernel
    for _ in range(3):
        _ = flash_attn_triton(Q, K, V, mask, use_autotune=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        out = flash_attn_triton(Q, K, V, mask, use_autotune=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1e3 / 50

    err = (ref - out).abs()
    print("\n[FlashAttention Triton] autotune")
    print(f"Latency: {ms:.3f} ms | max-abs: {err.max().item():.3e} | rel-Fro: {(err.norm()/ref.norm()).item():.3e}")

    # manual profile (pinned)
    best = profile_flash_attn(Q, K, V, mask, warmup=5, iters=20)
    out2 = _run_pinned(Q, K, V, mask, **best[1])
    err2 = (ref - out2).abs()
    print(f"Best pinned => max-abs: {err2.max().item():.3e} | rel-Fro: {(err2.norm()/ref.norm()).item():.3e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    _demo_correctness_and_speed()
