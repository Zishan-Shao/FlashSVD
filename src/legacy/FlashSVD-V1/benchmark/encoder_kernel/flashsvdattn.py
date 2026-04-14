#!/usr/bin/env python3
# flashsvdattn4D.py  – rank-aware Flash-SVD attention (mask-friendly)
#
# Needs: utils_mask_4D.py  (Triton kernel)

import math, torch, torch.nn as nn
from utils_mask import _demo_attn_kernel

# ────────────────────────────────────────────────────────────────────
# 1. Triton wrapper
# ────────────────────────────────────────────────────────────────────
BLOCK_M = 32
BLOCK_R = 32                      # Triton tile, not the low rank R

def _contig(t): return t.contiguous() if not t.is_contiguous() else t

def flash_svd_attention(Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv, mask,
                        *, block_m=BLOCK_M, block_r=BLOCK_R):
    B,H,M,R = Pq.shape
    D       = Vq.shape[-1]
    scale   = 1.0/math.sqrt(D)

    Pq,Vq,bq = map(_contig,(Pq,Vq,bq))
    Pk,Vk,bk = map(_contig,(Pk,Vk,bk))
    Pv,Vv,bv = map(_contig,(Pv,Vv,bv))

    # HF mask [B,1,1,M] or per-head [B,H,M]
    base = mask if mask.ndim==4 else mask[:, :1, :].unsqueeze(2)
    m4   = base.to(torch.int32).expand(B,H,1,M)
    if m4.stride(1) or m4.stride(2):
        m4 = m4.as_strided(m4.size(), (m4.stride(0),0,0,m4.stride(3)))
    sMb,sMh,sMq,sMk = m4.stride()

    Out = torch.empty(B*H, M, D, device=Pq.device, dtype=torch.float32)
    args = [
        Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv,
        Out, m4, sMb,sMh,sMq,sMk,
        *Pq.stride(), *Vq.stride(), *bq.stride(),
        *Pk.stride(), *Vk.stride(), *bk.stride(),
        *Pv.stride(), *Vv.stride(), *bv.stride(),
        *Out.stride(), M, R, H, scale,
    ]
    grid = ((M + block_m - 1)//block_m, B*H)
    _demo_attn_kernel[grid](*args, BLOCK_M=BLOCK_M,
                            BLOCK_R=BLOCK_R, BLOCK_D=D)
    return Out.view(B,H,M,D).to(Pq.dtype)

# ────────────────────────────────────────────────────────────────────
# 2. Flash-SVD block
# ────────────────────────────────────────────────────────────────────
class FlashSVDBlock(nn.Module):
    def __init__(self, d_model:int, n_heads:int, rank:int, d_ff:int):
        super().__init__()
        assert d_model % n_heads == 0 and rank<=d_model//n_heads
        self.d_model,self.H,self.R = d_model,n_heads,rank
        self.dh = d_model//n_heads

        def mk():
            P = nn.Parameter(torch.randn(1,n_heads,d_model,rank)*0.02)
            V = nn.Parameter(torch.randn(1,n_heads,rank,self.dh)*0.02)
            b = nn.Parameter(torch.zeros (1,n_heads,       self.dh))
            return P,V,b
        self.Pq,self.Vq,self.bq = mk()
        self.Pk,self.Vk,self.bk = mk()
        self.Pv,self.Vv,self.bv = mk()

        self.proj_out = nn.Linear(d_model,d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff), nn.GELU(), nn.Linear(d_ff,d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        B,M,_ = x.shape
        H,R,dh = self.H,self.R,self.dh

        tmp_q = torch.einsum('bmd,hdr->bhmr', x, self.Pq[0])
        tmp_k = torch.einsum('bmd,hdr->bhmr', x, self.Pk[0])
        tmp_v = torch.einsum('bmd,hdr->bhmr', x, self.Pv[0])

        attn = flash_svd_attention(
            tmp_q, self.Vq.expand(B,H,R,dh), self.bq.expand(B,H,dh),
            tmp_k, self.Vk.expand(B,H,R,dh), self.bk.expand(B,H,dh),
            tmp_v, self.Vv.expand(B,H,R,dh), self.bv.expand(B,H,dh),
            mask=mask, block_r=R
        )                       # [B,H,M,dh]
        attn = attn.transpose(1,2).reshape(B,M,self.d_model)
        y = self.ln1(x + self.proj_out(attn))
        return self.ln2(y + self.ffn(y))

# ────────────────────────────────────────────────────────────────────
# 3. Dense baseline
# ────────────────────────────────────────────────────────────────────
class BaselineBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model,n_heads,batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff), nn.GELU(), nn.Linear(d_ff,d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        pad = ~mask.squeeze(1).squeeze(1)
        attn,_ = self.mha(x,x,x,key_padding_mask=pad)
        return self.ln2(self.ln1(x+attn) + self.ffn(self.ln1(x+attn)))

# ────────────────────────────────────────────────────────────────────
# 4. Rank-aware transplant  (slice **rows**, not columns!)
# ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def transplant_weights(dense: BaselineBlock, flash: FlashSVDBlock):
    Wqkv = dense.mha.in_proj_weight    # [3d_model, d_model]
    bqkv = dense.mha.in_proj_bias
    Wo,bo = dense.mha.out_proj.weight, dense.mha.out_proj.bias

    d_model, H = flash.d_model, flash.H
    dh,R = flash.dh, flash.R

    def W_head(p,h):  # rows for this head, then transpose → [d_model, dh]
        rows = slice(p*d_model + h*dh, p*d_model + (h+1)*dh)
        return Wqkv[rows, :].t().contiguous()

    def b_head(p,h):
        rows = slice(p*d_model + h*dh, p*d_model + (h+1)*dh)
        return bqkv[rows]

    for p,(P,V,b) in enumerate([(flash.Pq,flash.Vq,flash.bq),
                                (flash.Pk,flash.Vk,flash.bk),
                                (flash.Pv,flash.Vv,flash.bv)]):  # Q,K,V
        for h in range(H):
            W = W_head(p,h).float()               # [d_model, dh]
            U,S,Vt = torch.linalg.svd(W, full_matrices=False)
            P[0,h].copy_((U[:, :R] * S[:R]).to(P.dtype))  # U·Σ
            V[0,h].copy_(Vt[:R, :].to(V.dtype))           # [R, dh]
            b[0,h].copy_(b_head(p,h).to(b.dtype))

    flash.proj_out.weight.copy_(Wo); flash.proj_out.bias.copy_(bo)
    flash.ffn[0].weight.copy_(dense.ffn[0].weight)
    flash.ffn[0].bias  .copy_(dense.ffn[0].bias)
    flash.ffn[2].weight.copy_(dense.ffn[2].weight)
    flash.ffn[2].bias  .copy_(dense.ffn[2].bias)
    flash.ln1.load_state_dict(dense.ln1.state_dict())
    flash.ln2.load_state_dict(dense.ln2.state_dict())

# ────────────────────────────────────────────────────────────────────
# 5. Quick test
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    B,M = 8,128
    d_model,H,d_ff = 768,12,3072
    x = torch.randn(B,M,d_model, device=dev, dtype=torch.float16)

    mask4 = torch.zeros(B,1,1,M,device=dev,dtype=torch.bool)
    mask4[..., :96] = True           # any true length

    dense = BaselineBlock(d_model,H,d_ff).to(dev).half()

    for R in [64, 54, 48, 32, 28, 16]:
        flash = FlashSVDBlock(d_model,H,R,d_ff).to(dev).half()
        transplant_weights(dense, flash)

        with torch.no_grad():
            yd = dense (x, mask4).float()
            yf = flash (x, mask4).float()
        rel = (yf - yd).norm() / yd.norm()
        print(f"rank {R:2d}  → rel-err {rel:.4e}")
        
    # pick one projection (0=Q, 1=K, 2=V) and one head h
    proj, head = 0, 0               # change if you like

    Wqkv = dense.mha.in_proj_weight  # [3·d_model, d_model]
    d_model, H = flash.d_model, flash.H
    dh = d_model // H

    # rows that belong to this (proj, head) pair
    rows = slice(proj*d_model + head*dh, proj*d_model + (head+1)*dh)
    W = Wqkv[rows, :].t().contiguous().float()   # shape [d_model, dh]

    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    for R in [64, 32, 16, 1]:
        W_hat = (U[:, :R] @ torch.diag(S[:R]) @ Vt[:R]).to(W.dtype)
        rel_w = (W - W_hat).norm() / W.norm()
        print(f"W  rank {R:2d} → weight rel-err {rel_w:.2f}")