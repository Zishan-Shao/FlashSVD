#!/usr/bin/env python3
import os
import copy
from typing import Optional, Dict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# === NEW: import current FlashSVD GEGLU APIs ===
try:
    from flashsvdgeglu import (
        flashsvd_ffn_geglu_autotuned as flashsvd_ffn_geglu,           # default
        flashsvd_ffn_geglu_configured as flashsvd_ffn_geglu_pinned,   # optional pinned
    )
except Exception:
    flashsvd_ffn_geglu = None
    flashsvd_ffn_geglu_pinned = None  # will fail with an assert below if you force FlashSVD

try:
    from flashsvdropeattn import FlashSVDRoPEAttention, QKVFactors  # Attn kernel
except Exception:
    FlashSVDRoPEAttention, QKVFactors = None, None  # will fail with an assert below if you force FlashSVD

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from evaluate import load as load_metric

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
# Utilities: RoPE
# ----------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


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


# Smart wrapper that chooses head-wise or standard Fisher-weighted SVD based on matrix shape
class ExplicitSVDLinear(nn.Module):
    """Smart SVD wrapper: uses head-wise Fisher-weighted SVD for square attention matrices, standard Fisher-weighted SVD for others"""
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], 
                 rank: Optional[int] = None, num_heads: int = 12, fisher: Optional[torch.Tensor] = None):
        super().__init__()
        out_dim, in_dim = weight.shape
        
        # Use head-wise Fisher-weighted SVD for square matrices that are divisible by num_heads
        if (out_dim == in_dim and out_dim % num_heads == 0):
            self.svd_layer = HeadwiseFWSVDLinear(weight, bias, num_heads=num_heads, rank=rank, fisher=fisher)
        else:
            # Use standard Fisher-weighted SVD for non-square matrices (FFN layers)
            self.svd_layer = ExplicitFWSVDLinear(weight, bias, rank=rank, fisher=fisher)
    
    @property 
    def U(self):
        return self.svd_layer.U
    
    @property
    def V(self):
        return self.svd_layer.V
    
    @property
    def b(self):
        return self.svd_layer.b
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.svd_layer(x)


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
        self.q = HeadwiseFWSVDLinear(Wq, bq, num_heads=num_heads, rank=rank_attn, fisher=fisher_q)
        self.k = HeadwiseFWSVDLinear(Wk, bk, num_heads=num_heads, rank=rank_attn, fisher=fisher_k)
        self.v = HeadwiseFWSVDLinear(Wv, bv, num_heads=num_heads, rank=rank_attn, fisher=fisher_v)

    def forward(self, x: torch.Tensor):
        # x: [B, M, D] → q/k/v each [B, M, D]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q, k, v


# ----------------------------
# Explicit ModernBERT block with SVD (incl. explicit GEGLU MLP)
# ----------------------------
class ExplicitSVDBlock(nn.Module):
    """
    - Pre-norm residual wiring
    - Attention:
        * Uses FlashSVD+RoPE kernel.
    - FFN:
        * Uses FlashSVD GEGLU FFN kernel (autotuned by default, optional pinned config).
    - Attention output Wo kept dense (exact HF weight).
    """
    def __init__(
        self,
        hf_layer: nn.Module,
        cfg,
        *,
        rank_attn: Optional[int],
        rank_ffn: Optional[int],
        fisher_pack: Optional[Dict[str, torch.Tensor]] = None,  # optional fisher dict
        use_flash_attn: int = 1,
        use_flash_ffn: int = 1,
        attn_kernel_bm: int = 64,
        attn_kernel_bn: int = 64,
        attn_kernel_bdh: Optional[int] = None,
        attn_kernel_br: int = 64,
        attn_l2norm_qk: bool = False,  # off by default; parity-first
        ffn_use_pinned: bool = False,
        ffn_pinned_cfg: Optional[dict] = None,  # e.g., {'BL':64,'BD':128,'BR1':64,'BR2':128,'num_warps':8,'num_stages':2}
        eps: float = 1e-6
    ):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hf_layer.attn.Wo.in_features
        self.num_heads = cfg.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Norms and rotary
        self.attn_norm = copy.deepcopy(hf_layer.attn_norm)
        self.mlp_norm  = copy.deepcopy(hf_layer.mlp_norm)
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
        # Attention output projection (dense)
        self.Wo_attn = copy.deepcopy(hf_layer.attn.Wo)

        # FFN explicit Fisher-weighted SVD projections
        Wi = hf_layer.mlp.Wi   # [2D, D] for GEGLU
        Wo = hf_layer.mlp.Wo   # [D,  D]
        self.Wi_exp = ExplicitSVDLinear(Wi.weight, Wi.bias, rank=rank_ffn, fisher=f_wi)
        self.Wo_exp = ExplicitSVDLinear(Wo.weight, Wo.bias, rank=rank_ffn, fisher=f_wo)

        # Act / GEGLU
        self.ffn_D = Wo.in_features
        self.ffn_is_geglu = (Wi.out_features == 2 * self.ffn_D)
        gelu_approx = getattr(getattr(hf_layer.mlp, "act", nn.GELU()), "approximate", "tanh")
        self.geglu = GEGLU(approximate=gelu_approx)

        # ---- enforce FlashSVD paths ----
        assert (FlashSVDRoPEAttention is not None) and torch.cuda.is_available(), "FlashSVD attention kernel not available or CUDA not found."
        assert (flashsvd_ffn_geglu is not None) and torch.cuda.is_available(), "FlashSVD GEGLU FFN kernel not available or CUDA not found."
        self.attn_l2norm_qk = bool(attn_l2norm_qk)

        # Build the FlashSVD attention kernel wrapper
        self.flash_attn = FlashSVDRoPEAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rotary_emb=self.rotary_emb,
            bm=attn_kernel_bm,
            bn=attn_kernel_bn,
            bdh=self.head_dim if attn_kernel_bdh is None else attn_kernel_bdh,
            br=attn_kernel_br,
        )

        # FFN pinned config
        self.ffn_use_pinned = bool(ffn_use_pinned)
        self.ffn_pinned_cfg = ffn_pinned_cfg or {}

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

        if position_ids is None:
            position_ids = torch.arange(M, device=hidden_states.device).unsqueeze(0).expand(B, M)

        # Q/K/V (B,M,D) -> [B,H,M,dh] using head-wise SVD (same as working version)
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

        # ---- SDPA mask ----
        sdpa_mask = None
        if sliding_window_mask is not None:
            sm = sliding_window_mask
            if sm.dtype.is_floating_point and sm.dtype != q.dtype:
                sm = sm.to(q.dtype)
            sdpa_mask = sm  # additive 4D [B,1/H,M,M]
        elif attention_mask is not None:
            if attention_mask.dim() == 2:
                sdpa_mask = self._padding_mask_bool(attention_mask)  # boolean [B,1,1,M]
            elif attention_mask.dim() == 4:
                sm = attention_mask
                if sm.dtype.is_floating_point and sm.dtype != q.dtype:
                    sm = sm.to(q.dtype)
                sdpa_mask = sm

        # SDPA on [B,H,M,dh]
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, dropout_p=0.0)  # [B,H,M,dh]
        attn_out = attn.transpose(1, 2).reshape(B, M, D).contiguous()  # [B,M,D]
        x = x + self.Wo_attn(attn_out)

        # === FFN (pre-norm) ===
        xn2 = self.mlp_norm(x)
        if not self.ffn_is_geglu:
            raise NotImplementedError("Non-GEGLU FFN is not supported in FlashSVD-only script.")

        # Dimensions
        Hdim = self.hidden_size     # model hidden size (output of Wo)
        Dffn = self.ffn_D           # FFN expansion size (input of Wo)

        # Rank-space factors (ensure memory-friendly layout)
        U1 = self.Wi_exp.U.contiguous()              # [H, R1]
        V1 = self.Wi_exp.V.contiguous()              # [R1, 2D]
        U2 = self.Wo_exp.U.contiguous()              # [D,  R2]
        V2 = self.Wo_exp.V.contiguous()              # [R2, H]

        # Biases (fallbacks sized to the *correct* dims)
        b1 = (self.Wi_exp.b if self.Wi_exp.b is not None else xn2.new_zeros(2 * Dffn)).contiguous()
        b2 = (self.Wo_exp.b if self.Wo_exp.b is not None else xn2.new_zeros(Hdim)).contiguous()

        # Preflight checks (clear error messages if shapes drift)
        assert V1.shape[1] == 2 * Dffn, f"V1 must be [R1,2D], got {tuple(V1.shape)}; D={Dffn}"
        assert U2.shape[0] == Dffn,     f"U2 must be [D,R2], got {tuple(U2.shape)}; D={Dffn}"
        assert V2.shape[1] == Hdim,     f"V2 must be [R2,H], got {tuple(V2.shape)}; H={Hdim}"
        assert b1.numel() == 2 * Dffn,  f"b1 must be [2D], got {tuple(b1.shape)}; D={Dffn}"
        assert b2.numel() == Hdim,      f"b2 must be [H], got {tuple(b2.shape)}; H={Hdim}"

        # Rank-space input: P = X @ U1
        P = xn2.matmul(U1)  # [B, M, R1]

        # FlashSVD GEGLU FFN (autotuned by default; optionally pinned)
        approx = self.geglu.approximate  # "tanh" or "none"
        if self.ffn_use_pinned:
            assert flashsvd_ffn_geglu_pinned is not None, "Pinned FFN requested but pinned API not found."
            y = flashsvd_ffn_geglu_pinned(
                P, V1, U2, V2, b1, b2,
                BL=self.ffn_pinned_cfg.get('BL', 64),
                BD=self.ffn_pinned_cfg.get('BD', 128),
                BR1=self.ffn_pinned_cfg.get('BR1', 64),
                BR2=self.ffn_pinned_cfg.get('BR2', 128),
                gelu_approx=approx,
                store_s_fp32=False,
                num_warps=self.ffn_pinned_cfg.get('num_warps', 8),
                num_stages=self.ffn_pinned_cfg.get('num_stages', 2),
            )
        else:
            y = flashsvd_ffn_geglu(
                P, V1, U2, V2, b1, b2,
                gelu_approx=approx,
                store_s_fp32=False,
            )
        x = x + y

        if output_attentions:
            return (x, None)
        return (x,)


# ----------------------------
# Full wrapper: swap layers
# ----------------------------
class ModernBERT_SVD_Explicit(nn.Module):
    def __init__(
        self,
        hf_model: nn.Module,
        *,
        rank_attn: Optional[int],
        rank_ffn: Optional[int],
        fisher_pack_per_layer: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        use_flash_attn: int = 1,
        use_flash_ffn: int = 1,
        attn_kernel_bm: int = 64,
        attn_kernel_bn: int = 64,
        attn_kernel_bdh: Optional[int] = None,
        attn_kernel_br: int = 64,
        attn_l2norm_qk: bool = False,
        # NEW: allow FFN pinned config
        ffn_use_pinned: bool = False,
        ffn_pinned_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.config = hf_model.config
        self.model = hf_model.model
        self.classifier = hf_model.classifier
        self.head = getattr(hf_model, "head", None)
        self.drop = getattr(hf_model, "drop", None)

        fisher_pack_per_layer = fisher_pack_per_layer or {}

        new_layers = []
        for i, layer in enumerate(self.model.layers):
            new_layers.append(
                ExplicitSVDBlock(
                    layer, self.config,
                    rank_attn=rank_attn, rank_ffn=rank_ffn,
                    fisher_pack=fisher_pack_per_layer.get(i, None),
                    use_flash_attn=use_flash_attn,
                    use_flash_ffn=use_flash_ffn,
                    attn_kernel_bm=attn_kernel_bm,
                    attn_kernel_bn=attn_kernel_bn,
                    attn_kernel_bdh=attn_kernel_bdh,
                    attn_kernel_br=attn_kernel_br,
                    attn_l2norm_qk=attn_l2norm_qk,
                    ffn_use_pinned=ffn_use_pinned,
                    ffn_pinned_cfg=ffn_pinned_cfg,
                )
            )
        self.model.layers = nn.ModuleList(new_layers)

    def _update_attention_masks(self, attention_mask_2d, dtype: torch.dtype):
        """
        Exact replica of HF ModernBertModel._update_attention_mask for SDPA path:
          - Build 4D global attention additive mask
          - Build 4D sliding window additive mask with bandwidth = local_attention//2
        Returns (global_attention_mask, sliding_window_mask), either may be None.
        """
        if attention_mask_2d is None:
            return None, None

        # 4D additive mask: 0 for allowed, -inf for disallowed
        global_attention_mask = _prepare_4d_attention_mask(attention_mask_2d, dtype)

        # Build window band [ |i-j| <= local_attention//2 ]
        seq_len = global_attention_mask.shape[-1]
        rows = torch.arange(seq_len, device=attention_mask_2d.device).unsqueeze(0)
        distance = torch.abs(rows - rows.T)

        half_window = int(self.config.local_attention) // 2
        window_mask = (distance <= half_window).unsqueeze(0).unsqueeze(0).to(attention_mask_2d.device)

        # Apply window to global mask; outside window → add -inf
        neg_inf = torch.finfo(dtype).min
        sliding_window_mask = global_attention_mask.masked_fill(~window_mask, neg_inf)

        return global_attention_mask, sliding_window_mask

    @staticmethod
    def _default_position_ids(batch_size: int, seq_len: int, device):
        return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

    def forward(self, input_ids, attention_mask=None, position_ids=None, output_attentions=False, **kwargs):
        # Embeddings
        hidden_states = self.model.embeddings(input_ids)

        # Position ids (HF uses arange starting at 0)
        if position_ids is None:
            position_ids = self._default_position_ids(input_ids.shape[0], input_ids.shape[1], input_ids.device)

        # Masks (match HF exactly)
        attn_mask_4d, sliding_mask_4d = self._update_attention_masks(attention_mask, hidden_states.dtype)

        # Encoder with proper mask passing
        all_attn = [] if output_attentions else None
        for layer in self.model.layers:
            out = layer(
                hidden_states,
                attention_mask=attn_mask_4d,
                sliding_window_mask=sliding_mask_4d,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )
            hidden_states = out[0]
            if output_attentions:
                all_attn.append(out[1])

        # Final norm
        hidden_states = self.model.final_norm(hidden_states)

        # Pool & head identical to HF
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
        
        # Lightweight output object with .logits (like HF model output)
        out = type("Output", (), {})()
        out.logits = logits
        if output_attentions:
            out.attentions = all_attn
        return out


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
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, steps)

    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    return metric.compute()["accuracy"], peak_mib, elapsed_ms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Ranks for compression testing (now using correct head-wise approach)
    RANK_ATTN = 48#64   # Full rank per head
    RANK_FFN  = 400#512  # Keep full rank for FFN for now

    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"

    dense = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=cfg, trust_remote_code=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    loader = _build_loader(tok, seq_len=2048, batch_size=8)

    # Build Fisher from the dense model
    loader_fisher = _build_loader(tok, seq_len=512, batch_size=16)
    fisher_pack_per_layer = build_fisher_packs_for_modernbert(dense, loader_fisher, device)

    # Clean memory and measure baseline (original dense model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    CACHED_ORIG_MEM = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Persistent dense model storage: {CACHED_ORIG_MEM:6.1f} MiB")

    # === Pinned best FFN config from your profile ===
    PINNED_FFN_CFG = dict(BL=64, BD=128, BR1=64, BR2=128, num_warps=8, num_stages=2)
    USE_FFN_PINNED = True   # set False to use autotuned FFN

    svd = ModernBERT_SVD_Explicit(
        dense,
        rank_attn=RANK_ATTN,
        rank_ffn=RANK_FFN,
        fisher_pack_per_layer=fisher_pack_per_layer,
        use_flash_attn=1,
        use_flash_ffn=1,
        attn_kernel_bm=64, attn_kernel_bn=64, attn_kernel_bdh=None, attn_kernel_br=32,
        attn_l2norm_qk=False,
        ffn_use_pinned=USE_FFN_PINNED,
        ffn_pinned_cfg=PINNED_FFN_CFG,
    ).to(device).eval()

    # Enable performance optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Measure SVD model storage with GPU redundancy (before cleanup)
    with_act = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Flash low-rank model storage with GPU Redundancy: {with_act:.1f} MiB")

    # Clean up any construction artifacts and reset for inference measurements
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Measure the clean SVD model storage (persistent baseline for inference)
    persistent_baseline = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Persistent low-rank model storage (SVD): {persistent_baseline:.1f} MiB")
    CACHED_ORIG_MEM = persistent_baseline

    acc = quick_check(svd, loader, device)
    print(f"[Sanity] Model accuracy on 3 batches: {acc:.4f}")

    # Comprehensive memory and latency measurement (entire validation split)
    full_acc, peak_lr, latency_ms = acc_peak_time(svd, loader, device, use_mask=True)

    # Calculate real peak memory using the same formula as BERT profile
    real_peak_mib = peak_lr - with_act + CACHED_ORIG_MEM
    transient_mib = peak_lr - CACHED_ORIG_MEM

    print(f"FlashFWSVD   | acc={full_acc:.4f} | peak ={peak_lr:6.1f} MiB | real peak ={real_peak_mib:6.1f} MiB | Transient={transient_mib:6.1f} MiB | {latency_ms:6.1f} ms/b")


if __name__ == "__main__":
    main()
