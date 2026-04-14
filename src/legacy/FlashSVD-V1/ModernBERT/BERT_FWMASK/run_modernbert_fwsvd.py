#!/usr/bin/env python3
import os
import copy
from typing import Optional, Tuple, Dict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from evaluate import load as load_metric
from transformers.modeling_outputs import SequenceClassifierOutput

# Import FlashAttention
from flash_attn_triton import flash_attn_triton_flex

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Path to your local ModernBERT checkpoint
MODEL_DIR = "../model/modernbert-base-sst2"


# ----------------------------
# GEGLU helpers (explicit)
# ----------------------------
class GEGLU(nn.Module):
    """Applies GEGLU to the last dimension: y = GELU(x1) * x2,
    where (x1, x2) = split(x, 2, dim=-1)."""
    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        self.approximate = approximate  # "none" or "tanh"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.gelu(x1, approximate=self.approximate) * x2

def geglu(x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return F.gelu(x1, approximate=approximate) * x2


# ----------------------------
# Utilities: RoPE
# ----------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def build_fisher_packs_for_modernbert(hf_model, dataloader, device, batches=8, eps=1e-6, alpha=1.0):
    """
    Returns: { layer_idx: {"q_in": D, "k_in": D, "v_in": D, "wi_in": D, "wo_in": D_ff} }
    Fisher proxy = sqrt(mean over (B,M) of x^2), with optional sharpening power alpha.
    """
    model = hf_model.to(device).train()
    L = len(model.model.layers)
    D = model.config.hidden_size

    # attn_norm output accumulators (length D)
    acc_attn = [torch.zeros(D, device=device) for _ in range(L)]
    cnt_attn = [0 for _ in range(L)]

    # mlp.Wo input (post-GEGLU h) accumulators; D_ff can vary per layer → init lazily
    acc_wo = [None for _ in range(L)]
    cnt_wo = [0 for _ in range(L)]

    hooks = []

    # Hook 1: capture pre-attention norm outputs (x_n) → length D
    def make_attn_norm_hook(li):
        def _hook(module, inp, out):
            x = out.detach()                      # [B,M,D]
            acc_attn[li] += x.float().pow(2).mean(dim=(0,1))
            cnt_attn[li] += 1
        return _hook

    # Hook 2: capture input to mlp.Wo (post-GEGLU h) via forward_pre_hook → length D_ff
    def make_mlp_wo_prehook(li):
        def _pre(module, inputs):
            # inputs is a tuple; inputs[0] = h = post-GEGLU activations [B,M,D_ff]
            h = inputs[0].detach()
            dff = h.shape[-1]
            if acc_wo[li] is None:
                acc_wo[li] = torch.zeros(dff, device=h.device)
            acc_wo[li] += h.float().pow(2).mean(dim=(0,1))
            cnt_wo[li] += 1
        return _pre

    # Register hooks
    for li, layer in enumerate(model.model.layers):
        hooks.append(layer.attn_norm.register_forward_hook(make_attn_norm_hook(li)))
        hooks.append(layer.mlp.Wo.register_forward_pre_hook(make_mlp_wo_prehook(li)))

    # Drive a few batches through the model (standard forward; position_ids handled internally)
    it = iter(dataloader)
    with torch.no_grad():
        for _ in range(batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    # Clean hooks
    for h in hooks:
        h.remove()

    # Build packs
    fisher_pack_per_layer = {}
    for li in range(L):
        # D-sized fisher (q/k/v/wi)
        if cnt_attn[li] == 0:
            fin_D = torch.ones(D, device=device)
        else:
            fin_D = (acc_attn[li] / cnt_attn[li]).sqrt().clamp_min(eps)

        # D_ff-sized fisher (wo)
        if cnt_wo[li] == 0 or acc_wo[li] is None:
            # if we didn't see mlp.Wo input, fallback to flat in Wo.in_features
            dff = model.model.layers[li].mlp.Wo.in_features
            fin_Dff = torch.ones(dff, device=device)
        else:
            fin_Dff = (acc_wo[li] / cnt_wo[li]).sqrt().clamp_min(eps)

        # normalize & (optionally) sharpen (helps deviate from vanilla SVD)
        fin_D   = (fin_D / fin_D.mean().clamp_min(eps)).pow(alpha)
        fin_Dff = (fin_Dff / fin_Dff.mean().clamp_min(eps)).pow(alpha)

        fisher_pack_per_layer[li] = {
            "q_in":  fin_D,
            "k_in":  fin_D,
            "v_in":  fin_D,
            "wi_in": fin_D,     # Wi input is D
            "wo_in": fin_Dff,   # Wo input is D_ff (post-GEGLU)
        }
    return fisher_pack_per_layer



# ----------------------------
# Head-wise Fisher-Weighted SVD Linear
# ----------------------------
class HeadwiseFWSVDLinear(nn.Module):
    """
    Head-wise Fisher-Weighted SVD: Combines head-wise decomposition with Fisher importance weighting.
    
    For attention matrices [hidden_size, hidden_size]:
    1. Split into num_heads matrices [hidden_size, head_dim] each
    2. Apply Fisher-weighted SVD per head to retain most essential components
    3. Fisher weights prioritize input dimensions that matter most for model behavior
    """
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], 
                 num_heads: int, rank: Optional[int] = None, fisher: Optional[torch.Tensor] = None, eps: float = 1e-6):
        super().__init__()
        out_dim, in_dim = weight.shape  # [hidden_size, hidden_size] for attention
        assert out_dim % num_heads == 0, f"out_dim {out_dim} not divisible by num_heads {num_heads}"
        
        head_dim = out_dim // num_heads  # 768 // 12 = 64
        dev, dt = weight.device, weight.dtype
        
        with torch.no_grad():
            W = weight.detach()  # [hidden_size, hidden_size]
            
            # Split into heads along output dimension 
            W_heads = W.view(num_heads, head_dim, in_dim)  # [num_heads, head_dim, hidden_size]
            
            # Apply Fisher-weighted SVD per head
            U_list, V_list = [], []
            r_full = min(head_dim, in_dim)  # min(64, 768) = 64 
            r = r_full if (rank is None or rank <= 0 or rank >= r_full) else int(rank)
            
            if fisher is not None:
                w = fisher.detach().to(W).float()
                if w.numel() != in_dim:
                    raise ValueError(f"fisher length {w.numel()} must equal input dim {in_dim}")
                w = torch.clamp(w, min=eps)
                sqrt_w = torch.sqrt(w)  # [hidden_size]
                print(f"[HeadwiseFWSVD] Applied Fisher weighting | mean={w.mean().item():.3e} | std={w.std().item():.3e}")
            
            for h in range(num_heads):
                W_head = W_heads[h]  # [head_dim, hidden_size]
                
                # SVD on W_head^T for consistency: [hidden_size, head_dim]
                WT_head = W_head.t().float()  # [hidden_size, head_dim]
                
                if fisher is not None:
                    # Apply Fisher-weighted SVD per head
                    # Weight the input dimensions by their importance: diag(sqrt_w) @ WT_head
                    A = sqrt_w[:, None] * WT_head  # [hidden_size, head_dim]
                    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
                    
                    # Unweight and incorporate singular values: U_fac = diag(1/sqrt_w) @ U @ S
                    U_r = (U[:, :r] * S[:r]) / sqrt_w[:, None]  # [hidden_size, r]
                    V_r = Vh[:r, :].to(dt)  # [r, head_dim]
                else:
                    # Vanilla SVD per head
                    U, S, Vh = torch.linalg.svd(WT_head, full_matrices=False)
                    U_r = (U[:, :r] * S[:r]).to(dt)   # [hidden_size, r]
                    V_r = Vh[:r, :].to(dt)            # [r, head_dim]
                
                U_list.append(U_r)
                V_list.append(V_r)
        
        # Stack all heads
        self.register_buffer("U", torch.stack(U_list, dim=0), persistent=False)  # [num_heads, hidden_size, r]
        self.register_buffer("V", torch.stack(V_list, dim=0), persistent=False)  # [num_heads, r, head_dim]
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rank = r
        
        if bias is not None:
            self.register_buffer("b", bias.detach().to(dt), persistent=False)
        else:
            self.b = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., hidden_size]
        batch_dims = x.shape[:-1]
        
        # Apply head-wise Fisher-weighted SVD: for each head h, compute x @ U_h @ V_h
        x_U = torch.einsum('...d,hdr->...hr', x, self.U)  # [..., num_heads, rank]
        y_heads = torch.einsum('...hr,hrd->...hd', x_U, self.V)  # [..., num_heads, head_dim]
        
        # Concatenate heads: [..., num_heads, head_dim] -> [..., num_heads * head_dim]
        y = y_heads.reshape(*batch_dims, self.num_heads * self.head_dim)
        
        if self.b is not None:
            y = y + self.b
            
        return y


# ----------------------------
# Fisher-Weighted SVD Linear (drop-in)
# ----------------------------
class ExplicitFWSVDLinear(nn.Module):
    """
    Fisher-Weighted low-rank affine: y = (x @ U_r) @ V_r + b
      - Factorization performed on W^T (shape [in, out]) for stability.
      - If `fisher` is provided (length = in), we solve:
          min || diag(sqrt(w)) (W^T - A_r) ||_F
        via SVD of A = diag(sqrt(w)) @ W^T, then
          W^T ≈ diag(1/sqrt(w)) @ U_r S_r V_r^T
        => store U_fac = diag(1/sqrt(w)) @ (U_r S_r),  V_fac = V_r^T
      - Falls back to vanilla SVD when fisher is None.
      - Stores factors/bias as buffers (frozen by default).
    """
    def __init__(
        self,
        weight: torch.Tensor,                 # [out, in]
        bias: Optional[torch.Tensor],         # [out] or None
        rank: Optional[int] = None,
        fisher: Optional[torch.Tensor] = None,# [in] (Fisher diag for inputs)
        eps: float = 1e-6
    ):
        super().__init__()
        assert weight.dim() == 2, "weight must be [out, in]"
        out_f, in_f = weight.shape
        dt = weight.dtype

        with torch.no_grad():
            WT = weight.detach().t().float()             # [in, out]
            r_full = min(WT.shape)
            r = r_full if (rank is None or rank <= 0 or rank >= r_full) else int(rank)

            if fisher is not None:
                w = fisher.detach().to(WT).float()
                if w.numel() != WT.shape[0]:
                    raise ValueError(
                        f"fisher length {w.numel()} must equal 'in'={WT.shape[0]}"
                    )
                w = torch.clamp(w, min=eps)
                Dsqrt = torch.sqrt(w)[:, None]           # [in, 1]
                A = Dsqrt * WT                           # [in, out]
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)  # U:[in,r], S:[r], Vh:[r,out]
                U_fac = (U[:, :r] * S[:r]) / Dsqrt       # [in, r]
                V_fac = Vh[:r, :].contiguous()           # [r, out]
            else:
                U, S, Vh = torch.linalg.svd(WT, full_matrices=False)
                U_fac = (U[:, :r] * S[:r])               # [in, r]
                V_fac = Vh[:r, :]                        # [r, out]
            
            if fisher is None:
                print("[FWSVD] fisher=None → falling back to vanilla SVD")
            else:
                w = fisher.detach().float().view(-1)
                w_rel_var = (w.std() / (w.mean().clamp_min(1e-8))).item()
                print(f"[FWSVD] fisher applied | len={w.numel()} | mean={w.mean().item():.3e} | rel_std={w_rel_var:.3e}")

            self.register_buffer("U", U_fac.to(dt), persistent=False)    # [in, r]
            self.register_buffer("V", V_fac.to(dt), persistent=False)    # [r, out]
            if bias is not None:
                self.register_buffer("b", bias.detach().to(dt), persistent=False)
            else:
                self.b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.matmul(self.U).matmul(self.V)  # [..., out]
        if self.b is not None:
            y = y + self.b
        return y


# ----------------------------
# QKV block using (Fisher-Weighted) SVD
# ----------------------------
class ExplicitSVDQKV(nn.Module):
    """
    Splits a fused Wqkv Linear ([3D, D]) into Q/K/V Fisher-weighted low-rank projections.
    If fisher_* are None, it falls back to vanilla SVD compression.
    """
    def __init__(
        self,
        Wqkv: nn.Linear,          # weight: [3D, D], bias: [3D] or None
        hidden_size: int,         # D
        rank_attn: Optional[int],
        fisher_q: Optional[torch.Tensor] = None,  # [D] fisher over inputs
        fisher_k: Optional[torch.Tensor] = None,  # [D]
        fisher_v: Optional[torch.Tensor] = None   # [D]
    ):
        super().__init__()
        D = hidden_size
        W = Wqkv.weight            # [3D, D]
        b = Wqkv.bias              # [3D] or None

        # Split weights (and bias if present) along out_features (=0 dim)
        Wq, Wk, Wv = torch.split(W, D, dim=0)    # each [D, D]
        if b is not None:
            bq, bk, bv = torch.split(b, D, dim=0)
        else:
            bq = bk = bv = None

        # Build three head-wise Fisher-weighted low-rank linears
        num_heads = 12  # ModernBERT default
        self.Wq = HeadwiseFWSVDLinear(Wq, bq, num_heads=num_heads, rank=rank_attn, fisher=fisher_q)
        self.Wk = HeadwiseFWSVDLinear(Wk, bk, num_heads=num_heads, rank=rank_attn, fisher=fisher_k)
        self.Wv = HeadwiseFWSVDLinear(Wv, bv, num_heads=num_heads, rank=rank_attn, fisher=fisher_v)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, M, D] → q/k/v each [B, M, D]
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        return q, k, v


# ----------------------------
# Explicit ModernBERT block with FWSVD (incl. explicit GEGLU MLP)
# ----------------------------
class ExplicitSVDBlock(nn.Module):
    """
    - Pre-norm residual wiring
    - Q/K/V via ExplicitFWSVDLinear (rank_attn) → reshape → RoPE → FlashAttention
    - FFN via ExplicitFWSVDLinear Wi & Wo (rank_ffn) with explicit GEGLU.
    - Attention output Wo kept dense (exact HF weight)
    """
    def __init__(
        self,
        hf_layer: nn.Module,
        cfg,
        *,
        rank_attn: Optional[int],
        rank_ffn: Optional[int],
        fisher_pack: Optional[Dict[str, torch.Tensor]] = None,  # optional fisher dict
        eps: float = 1e-6
    ):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hf_layer.attn.Wo.in_features
        self.num_heads = cfg.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Validate head dimension is power of 2 for Triton FlashAttention
        if not (self.head_dim > 0 and (self.head_dim & (self.head_dim - 1)) == 0):
            raise ValueError(f"Head dimension {self.head_dim} must be a power of 2 for FlashAttention Triton kernel")

        # Norms
        self.attn_norm = copy.deepcopy(hf_layer.attn_norm)
        self.mlp_norm  = copy.deepcopy(hf_layer.mlp_norm)

        # Rotary from HF layer
        self.rotary_emb = hf_layer.attn.rotary_emb

        # ---- Optional Fisher vectors (length = D) ----
        # Expect keys: "q_in", "k_in", "v_in", "wi_in", "wo_in"
        fisher_pack = fisher_pack or {}
        f_q  = fisher_pack.get("q_in", None)
        f_k  = fisher_pack.get("k_in", None)
        f_v  = fisher_pack.get("v_in", None)
        f_wi = fisher_pack.get("wi_in", None)   # for Wi input side
        f_wo = fisher_pack.get("wo_in", None)   # for Wo input side (i.e., dim D of h)

        # Q/K/V explicit Fisher-weighted SVD projections
        self.qkv = ExplicitSVDQKV(hf_layer.attn.Wqkv, self.hidden_size, rank_attn,
                                  fisher_q=f_q, fisher_k=f_k, fisher_v=f_v)

        # Attention output projection (keep dense for parity)
        self.Wo_attn = copy.deepcopy(hf_layer.attn.Wo)
        
        # FFN explicit FWSVD projections
        Wi = hf_layer.mlp.Wi   # [2*D_ff, D]
        Wo = hf_layer.mlp.Wo   # [D, D_ff]

        self.d_model = self.hidden_size          # = D
        self.d_ff    = Wo.in_features            # = D_ff (1152 in your run)

        # GEGLU detection: Wi.out_features should be 2 * D_ff
        self.ffn_is_geglu = (Wi.out_features == 2 * self.d_ff)

        # Fisher vectors
        fisher_pack = fisher_pack or {}
        f_q  = fisher_pack.get("q_in",  None)    # len D
        f_k  = fisher_pack.get("k_in",  None)    # len D
        f_v  = fisher_pack.get("v_in",  None)    # len D
        f_wi = fisher_pack.get("wi_in", None)    # len D
        f_wo = fisher_pack.get("wo_in", None)    # len D_ff  ← IMPORTANT

        # Projections
        self.qkv   = ExplicitSVDQKV(hf_layer.attn.Wqkv, self.hidden_size, rank_attn,
                                    fisher_q=f_q, fisher_k=f_k, fisher_v=f_v)
        self.Wo_attn = copy.deepcopy(hf_layer.attn.Wo)

        self.Wi_exp = ExplicitFWSVDLinear(Wi.weight, Wi.bias, rank=rank_ffn, fisher=f_wi, eps=eps)  # in = D
        self.Wo_exp = ExplicitFWSVDLinear(Wo.weight, Wo.bias, rank=rank_ffn, fisher=f_wo, eps=eps)  # in = D_ff

        # GEGLU module (ModernBERT typically uses GEGLU)
        gelu_approx = getattr(getattr(hf_layer.mlp, "act", nn.GELU()), "approximate", "tanh")
        self.geglu = GEGLU(approximate=gelu_approx)


        # # FFN explicit FWSVD projections
        # Wi = hf_layer.mlp.Wi   # [2D, D] for GEGLU
        # Wo = hf_layer.mlp.Wo   # [D, D]
        # self.Wi_exp = ExplicitFWSVDLinear(Wi.weight, Wi.bias, rank=rank_ffn, fisher=f_wi, eps=eps)
        # self.Wo_exp = ExplicitFWSVDLinear(Wo.weight, Wo.bias, rank=rank_ffn, fisher=f_wo, eps=eps)

        # # Detect GEGLU by shape, and get GELU approximate mode from HF if present
        # self.ffn_D = Wo.in_features
        # self.ffn_is_geglu = (Wi.out_features == 2 * self.ffn_D)
        # gelu_approx = getattr(getattr(hf_layer.mlp, "act", nn.GELU()), "approximate", "tanh")
        # self.geglu = GEGLU(approximate=gelu_approx)

    @staticmethod
    def _padding_mask_bool(attention_mask_2d: torch.Tensor) -> torch.Tensor:
        # [B,L] with 1=valid,0=pad -> [B,1,1,L] boolean; True = MASK
        return ~(attention_mask_2d.to(torch.bool))[:, None, None, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,      # 2D padding or 4D additive
        sliding_window_mask: Optional[torch.Tensor] = None, # 4D additive (local band)
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ):
        B, M, D = hidden_states.shape
        H, dh = self.num_heads, self.head_dim

        # === Attention (pre-norm) ===
        x = hidden_states
        xn = self.attn_norm(x)

        # Q/K/V (B,M,D) -> [B,H,M,dh]
        q, k, v = self.qkv(xn)
        def to_bhmd(t):
            return t.view(B, M, H, dh).transpose(1, 2).contiguous()
        q, k, v = to_bhmd(q), to_bhmd(k), to_bhmd(v)

        # RoPE on q,k
        qf = q.view(B * H, M, dh)
        kf = k.view(B * H, M, dh)
        if position_ids is None:
            position_ids = torch.arange(M, device=hidden_states.device).unsqueeze(0).expand(B, M)
        posf = position_ids.unsqueeze(1).expand(B, H, M).reshape(B * H, M)
        cos, sin = self.rotary_emb(qf, position_ids=posf)
        qf = apply_rotary(qf, cos, sin)
        kf = apply_rotary(kf, cos, sin)
        q = qf.view(B, H, M, dh)
        k = kf.view(B, H, M, dh)

        # ---- FlashAttention with BMHd layout ----
        q_bmhd = q.transpose(1, 2).contiguous()  # [B,M,H,dh]
        k_bmhd = k.transpose(1, 2).contiguous()  # [B,M,H,dh]
        v_bmhd = v.transpose(1, 2).contiguous()  # [B,M,H,dh]
        
        # Convert mask to FlashAttention format [B,H,1,M] int32 (1 = keep, 0 = mask)
        # Priority: sliding_window_mask > attention_mask > default (all ones)
        if sliding_window_mask is not None:
            sm = sliding_window_mask
            if sm.dim() == 4 and sm.shape[2] == M and sm.shape[3] == M:
                mask_bh1m = (sm[..., 0:1, :] >= 0).to(torch.int32)  # [B,H,1,M]
                if mask_bh1m.shape[1] == 1:  # Broadcast over heads if needed
                    mask_bh1m = mask_bh1m.expand(B, H, 1, M)
                mask_bh1m = mask_bh1m.contiguous()
            else:
                mask_bh1m = torch.ones(B, H, 1, M, device=q.device, dtype=torch.int32)
        elif attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bh1m = attention_mask.to(torch.int32)[:, None, None, :].expand(B, H, 1, M).contiguous()
            else:
                mask_bh1m = (attention_mask >= 0).to(torch.int32)
                if mask_bh1m.shape[2] != 1:
                    mask_bh1m = mask_bh1m[:, :, :1, :]
                mask_bh1m = mask_bh1m.contiguous()
        else:
            mask_bh1m = torch.ones(B, H, 1, M, device=q.device, dtype=torch.int32)

        # FlashAttention Triton kernel
        attn = flash_attn_triton_flex(
            q_bmhd, k_bmhd, v_bmhd, mask_bh1m,
            layout="BMHd", 
            use_autotune=True
        )  # Returns [B,H,M,dh]
        
        attn = attn.transpose(1, 2).reshape(B, M, D)  # [B,M,D]
        x = x + self.Wo_attn(attn)

        # === FFN (pre-norm) — explicit low-rank matmuls + GEGLU ===
        xn2 = self.mlp_norm(x)

        # # z = (xn2 @ U1) @ V1 + b1
        # z = self.Wi_exp(xn2)  # [B,M, 2D] (GEGLU) or [B,M, D'] (fallback)

        # if self.ffn_is_geglu:
        #     h = self.geglu(z)                        # GEGLU: gelu(u)*v → [B,M,D]
        # else:
        #     h = F.gelu(z, approximate=self.geglu.approximate)

        # # y = (h @ U2) @ V2 + b2
        # y = self.Wo_exp(h)                            # [B,M,D]
        # x = x + y
        
        # z = Wi(xn2) → [B,M, 2*D_ff]
        z = self.Wi_exp(xn2)

        if self.ffn_is_geglu:
            h = self.geglu(z)           # [B,M, D_ff]
        else:
            # If not GEGLU, fall back; but ModernBERT should be GEGLU
            h = F.gelu(z, approximate=self.geglu.approximate)

        y = self.Wo_exp(h)              # [B,M, D]
        x = x + y

        if output_attentions:
            return (x, None)
        return (x,)
    


# ----------------------------
# Full wrapper: swap layers (FWSVD)
# ----------------------------
class ModernBERT_SVD_Explicit(nn.Module):
    def __init__(
        self,
        hf_model: nn.Module,
        *,
        rank_attn: Optional[int],
        rank_ffn: Optional[int],
        fisher_pack_per_layer: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    ):
        super().__init__()
        self.config = hf_model.config

        # Deep-copy so we don't mutate hf_model.model in-place
        self.model = copy.deepcopy(hf_model.model)

        self.classifier = copy.deepcopy(hf_model.classifier)
        self.head = copy.deepcopy(getattr(hf_model, "head", None))
        self.drop = copy.deepcopy(getattr(hf_model, "drop", None))

        fisher_pack_per_layer = fisher_pack_per_layer or {}

        new_layers = []
        for i, layer in enumerate(self.model.layers):
            # Guard: if already wrapped, keep as-is
            if isinstance(layer, ExplicitSVDBlock) or (not hasattr(layer, "attn")):
                new_layers.append(layer)
            else:
                new_layers.append(
                    ExplicitSVDBlock(
                        layer,
                        self.config,
                        rank_attn=rank_attn,
                        rank_ffn=rank_ffn,
                        fisher_pack=fisher_pack_per_layer.get(i, None),
                    )
                )
        self.model.layers = nn.ModuleList(new_layers)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # run encoder
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{k: v for k, v in kwargs.items() if k != "labels"}
        )
        hidden_states = outputs[0]  # [B, M, D]

        # pooling (match HF behavior)
        if getattr(self.config, "classifier_pooling", "cls") == "cls":
            pooled = hidden_states[:, 0]
        else:
            if attention_mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        if self.head is not None:
            pooled = self.head(pooled)
        if self.drop is not None:
            pooled = self.drop(pooled)

        logits = self.classifier(pooled)

        return SequenceClassifierOutput(logits=logits)

# ----------------------------
# Quick sanity harness
# ----------------------------
def _build_loader(tokenizer, seq_len=128, batch_size=8):
    raw = load_dataset("glue", "sst2", split="validation")
    def tok(b): return tokenizer(b["sentence"], padding="max_length", truncation=True, max_length=seq_len)
    ds = raw.map(tok, batched=True, remove_columns=["sentence","idx"])
    ds.set_format("torch")
    return DataLoader(ds, batch_size, shuffle=False, collate_fn=lambda b: {
        "input_ids": torch.stack([x["input_ids"] for x in b]),
        "attention_mask": torch.stack([x["attention_mask"] for x in b]),
        "labels": torch.tensor([x["label"] for x in b]),
    })

@torch.no_grad()
def quick_check(model_svd, loader, device):
    metric = load_metric("accuracy")
    for i, batch in enumerate(loader):
        if i >= 3: break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model_svd(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        metric.add_batch(predictions=out.logits.argmax(-1).cpu(), references=batch["labels"].cpu())
    return metric.compute()["accuracy"]


@torch.no_grad()
def acc_peak_time(model_svd, loader, device, use_mask=True):
    metric = load_metric("accuracy")
    # Clean memory and reset peak tracking for accurate measurement
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    steps = 0
    start = time.perf_counter()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if use_mask:
            out = model_svd(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        else:
            out = model_svd(input_ids=batch["input_ids"]).logits
        metric.add_batch(predictions=out.argmax(-1).cpu(), references=batch["labels"].cpu())
        steps += 1

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_mib = 0.0

    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, steps)
    return metric.compute()["accuracy"], peak_mib, elapsed_ms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    RANK_ATTN = 64
    RANK_FFN  = 512

    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"

    dense = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, config=cfg, trust_remote_code=True
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    loader = _build_loader(tok, seq_len=128*4, batch_size=64)

    # Build Fisher from the dense model
    loader_fisher = _build_loader(tok, seq_len=512, batch_size=16)
    fisher_pack_per_layer = build_fisher_packs_for_modernbert(dense, loader_fisher, device)

    # Measure dense baseline BEFORE wrapping
    if device.startswith("cuda"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        dense_persistent = torch.cuda.max_memory_allocated() / 1024**2
    else:
        dense_persistent = 0.0
    print(f"Persistent dense model storage: {dense_persistent:6.1f} MiB")

    # Construct the FWSVD-wrapped model ONCE
    svd = ModernBERT_SVD_Explicit(
        dense, rank_attn=RANK_ATTN, rank_ffn=RANK_FFN,
        fisher_pack_per_layer=fisher_pack_per_layer
    ).to(device).eval()

    # Measure wrapper storage with GPU redundancy
    if device.startswith("cuda"):
        with_act = torch.cuda.max_memory_allocated() / 1024**2
    else:
        with_act = 0.0
    print(f"Flash low-rank model storage with GPU Redundancy: {with_act:.1f} MiB")

    # Clean up, then measure persistent SVD baseline
    import gc; gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        persistent_baseline = torch.cuda.max_memory_allocated() / 1024**2
    else:
        persistent_baseline = 0.0
    print(f"Persistent low-rank model storage (SVD): {persistent_baseline:.1f} MiB")

    # Now run eval
    acc = quick_check(svd, loader, device)
    print(f"[Sanity] SVD-explicit (FWSVD) model accuracy on 3 batches: {acc:.4f}")

    full_acc, peak_lr, latency_ms = acc_peak_time(svd, loader, device, use_mask=True)
    real_peak_mib = peak_lr - with_act + persistent_baseline
    transient_mib = peak_lr - persistent_baseline
    print(f"LowRank FWSVD | acc={full_acc:.4f} | peak ={peak_lr:6.1f} MiB | real peak ={real_peak_mib:6.1f} MiB | Transient={transient_mib:6.1f} MiB | {latency_ms:6.1f} ms/b")


if __name__ == "__main__":
    main()
