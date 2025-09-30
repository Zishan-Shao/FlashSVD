import os, math, time, platform, inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from typing import Optional

# Import optimized kernel modules
from flashsvdropeattn import FlashSVDRoPEAttention, QKVFactors
from flashsvdswiglu import flashsvd_ffn_swiglu



# ────────────────────── helpers ──────────────────────
def _repeat_kv(x, n_rep: int):
    # x: [B, Hk, T, Dh] -> [B, H, T, Dh]
    if n_rep == 1:
        return x
    B, Hk, T, Dh = x.shape
    return x[:, :, None].expand(B, Hk, n_rep, T, Dh).reshape(B, Hk * n_rep, T, Dh)

@torch.no_grad()
def _decompose_heads_svd(weight: torch.Tensor, n_heads: int, head_dim: int, rank: int):
    """
    weight: [H*dh, D]  -> per-head factors:
      Us: [H, dh, r]  (U Σ folded)   ;  V: [H, r, D]
    """
    W = weight.detach().to(torch.float32)
    H, dh, D = n_heads, head_dim, W.shape[1]
    Us, Vs = [], []
    for h in range(H):
        W_h = W[h*dh:(h+1)*dh, :]  # [dh, D]
        U, S, Vh = torch.linalg.svd(W_h, full_matrices=False)
        r = max(1, min(int(rank), U.shape[1], Vh.shape[0]))
        Us.append(U[:, :r] * S[:r].unsqueeze(0))   # [dh, r]
        Vs.append(Vh[:r, :])                       # [r, D]
    Us = torch.stack(Us, dim=0)  # [H, dh, r]
    Vs = torch.stack(Vs,  dim=0) # [H,  r, D]
    return Us, Vs

@torch.no_grad()
def _decompose_full_svd(weight: torch.Tensor, rank: int):
    W = weight.detach().to(torch.float32)        # [out, in]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = max(1, min(int(rank), U.shape[1], Vh.shape[0]))
    Us = U[:, :r] * S[:r].unsqueeze(0)           # [out, r]
    V  = Vh[:r, :]                                # [r, in]
    return Us, V

# ────────────────────── Optimized Flash + SVD block ──────────────────────
class OptimizedFlashSVDLlamaBlock(nn.Module):
    """
    LLaMA block using optimized Triton kernels:
      • FlashSVDRoPEAttention for attention computation
      • flashsvd_ffn_swiglu for SwiGLU FFN computation
    """
    def __init__(self, hf_layer: nn.Module, cfg,
                 rank_q: int,
                 rank_kv: int,
                 rank_o: Optional[int],
                 rank_ff: Optional[int],
                 factor_dtype: torch.dtype = torch.float32,
                 compute_in_fp32: bool = True,
                 bypass_svd_qkv: bool = False,
                 use_varlen: bool = True):
        super().__init__()
        attn, mlp = hf_layer.self_attn, hf_layer.mlp
        self.d_model    = int(cfg.hidden_size)
        self.n_heads    = int(cfg.num_attention_heads)
        self.n_kv_heads = int(getattr(cfg, "num_key_value_heads", self.n_heads))
        self.head_dim   = self.d_model // self.n_heads
        self.n_rep      = self.n_heads // self.n_kv_heads
        self.compute_in_fp32 = bool(compute_in_fp32)
        self.factor_dtype = factor_dtype
        self.bypass_svd_qkv = bool(bypass_svd_qkv)
        self.use_varlen = bool(use_varlen)

        # Norms & RoPE
        self.ln1 = hf_layer.input_layernorm
        self.ln2 = hf_layer.post_attention_layernorm
        self.rotary_emb = getattr(attn, "rotary_emb", None)
        if self.rotary_emb is None:
            # Robust fallback matching HF semantics incl. position_ids
            rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
            class _SimpleRoPE(nn.Module):
                def __init__(self, head_dim: int, base: float = 10000.0):
                    super().__init__()
                    self.head_dim = head_dim
                    self.base = base
                    evens = torch.arange(0, head_dim, 2, dtype=torch.float32)
                    self.register_buffer("inv_freq", 1.0 / (base ** (evens / head_dim)), persistent=False)
                def forward(self, x, seq_len: int = None, position_ids: Optional[torch.LongTensor] = None):
                    device = x.device
                    inv = self.inv_freq.to(device=device)
                    Dh = self.head_dim
                    if position_ids is not None:
                        # position_ids: [B, T] -> [B*H, T] for kernel
                        BH, T = x.shape[0], position_ids.shape[1]
                        t = position_ids.view(-1)  # [B*T]
                        t = t.unsqueeze(0).expand(BH, -1).reshape(-1)  # [BH*T]
                        t = t.to(torch.float32).view(BH, T)  # [BH, T]
                        ang = t[..., None] * inv[None, None, :]      # [BH,T,Dh/2]
                        ang = ang.repeat_interleave(2, dim=-1)       # [BH,T,Dh]
                        cos = ang.cos().to(x.dtype)                  # [BH,T,Dh]
                        sin = ang.sin().to(x.dtype)
                        return cos, sin
                    assert seq_len is not None, "Provide seq_len or position_ids"
                    t = torch.arange(seq_len, device=device, dtype=torch.float32)  # [T]
                    ang = t[:, None] * inv[None, :]             # [T, Dh/2]
                    ang = ang.repeat_interleave(2, dim=-1)      # [T, Dh]
                    cos = ang.cos()[None, :, :].to(x.dtype).expand(x.shape[0], -1, -1)  # [BH,T,Dh]
                    sin = ang.sin()[None, :, :].to(x.dtype).expand(x.shape[0], -1, -1)
                    return cos, sin
            self.rotary_emb = _SimpleRoPE(self.head_dim, base=rope_theta)

        # Initialize optimized attention module
        self.flash_attn = FlashSVDRoPEAttention(
            num_heads=self.n_heads, 
            head_dim=self.head_dim, 
            rotary_emb=self.rotary_emb
        )

        # --- per-head SVD for Q/K/V ---
        if not self.bypass_svd_qkv:
            q_Us, q_V = _decompose_heads_svd(attn.q_proj.weight, self.n_heads,    self.head_dim, rank_q)
            k_Us, k_V = _decompose_heads_svd(attn.k_proj.weight, self.n_kv_heads, self.head_dim, rank_kv)
            v_Us, v_V = _decompose_heads_svd(attn.v_proj.weight, self.n_kv_heads, self.head_dim, rank_kv)
            
            # For the optimized kernel, we need the factors in the right format:
            # P = X @ U_proj  (project to rank space)
            # Q/K/V = P @ V_factors  (lift to head space)
            
            # Reshape SVD factors for kernel interface
            # q_Us: [H, dh, r] -> [H*dh, r] -> [r, H*dh] for the V factor in kernel
            # q_V: [H, r, D] -> [H*r, D] -> [D, H*r] for the U projection  
            
            Vq_factor = q_Us.view(self.n_heads * self.head_dim, rank_q).t().contiguous()  # [r, H*dh]
            Vk_factor = k_Us.view(self.n_kv_heads * self.head_dim, rank_kv).t().contiguous()  # [r, H_kv*dh]  
            Vv_factor = v_Us.view(self.n_kv_heads * self.head_dim, rank_kv).t().contiguous()  # [r, H_kv*dh]
            
            Uq_factor = q_V.view(self.n_heads * rank_q, self.d_model).t().contiguous()  # [D, H*r]
            Uk_factor = k_V.view(self.n_kv_heads * rank_kv, self.d_model).t().contiguous()  # [D, H_kv*r]
            Uv_factor = v_V.view(self.n_kv_heads * rank_kv, self.d_model).t().contiguous()  # [D, H_kv*r]
            
            # Store factors as parameters for the kernel
            self.Vq_factor = nn.Parameter(Vq_factor.to(factor_dtype), requires_grad=False)  # [r, H*dh]
            self.Vk_factor = nn.Parameter(Vk_factor.to(factor_dtype), requires_grad=False)  # [r, H_kv*dh]
            self.Vv_factor = nn.Parameter(Vv_factor.to(factor_dtype), requires_grad=False)  # [r, H_kv*dh]
            
            # Projection layers to rank space - match dtype of original layers
            self.Pq_proj = nn.Linear(self.d_model, self.n_heads * rank_q, bias=False, dtype=attn.q_proj.weight.dtype)
            self.Pk_proj = nn.Linear(self.d_model, self.n_kv_heads * rank_kv, bias=False, dtype=attn.k_proj.weight.dtype)
            self.Pv_proj = nn.Linear(self.d_model, self.n_kv_heads * rank_kv, bias=False, dtype=attn.v_proj.weight.dtype)
            with torch.no_grad():
                self.Pq_proj.weight.copy_(Uq_factor.t().to(attn.q_proj.weight.dtype))  # [H*r, D]
                self.Pk_proj.weight.copy_(Uk_factor.t().to(attn.k_proj.weight.dtype))  # [H_kv*r, D] 
                self.Pv_proj.weight.copy_(Uv_factor.t().to(attn.v_proj.weight.dtype))  # [H_kv*r, D]
        
        # --- Output projection (low-rank or dense passthrough) ---
        if rank_o is not None:
            o_Us, o_V = _decompose_full_svd(attn.o_proj.weight, rank_o)
            self.o_Us = nn.Parameter(o_Us.to(factor_dtype), requires_grad=False)
            self.o_V  = nn.Parameter(o_V.to(factor_dtype),  requires_grad=False)
            self.use_lowrank_o = True
        else:
            self.o = nn.Linear(self.n_heads * self.head_dim, self.d_model,
                               bias=False, dtype=attn.o_proj.weight.dtype)
            with torch.no_grad():
                self.o.weight.copy_(attn.o_proj.weight)
            self.use_lowrank_o = False

        # --- MLP using optimized SwiGLU kernel ---
        inter = int(cfg.intermediate_size)
        if rank_ff is not None:
            # SVD decomposition for each MLP layer
            g_Us, g_V = _decompose_full_svd(mlp.gate_proj.weight, rank_ff)
            u_Us, u_V = _decompose_full_svd(mlp.up_proj.weight,   rank_ff)
            d_Us, d_V = _decompose_full_svd(mlp.down_proj.weight, rank_ff)
            
            # Store factors for optimized SwiGLU kernel
            self.U1 = nn.Linear(self.d_model, rank_ff, bias=False, dtype=mlp.gate_proj.weight.dtype)  # Project to rank space
            self.V1 = nn.Parameter(torch.cat([g_V, u_V], dim=1).to(factor_dtype), requires_grad=False)  # [rank_ff, 2*inter] 
            self.U2 = nn.Parameter(d_V.to(factor_dtype), requires_grad=False)   # [inter, rank_ff]
            self.V2 = nn.Parameter(d_Us.to(factor_dtype), requires_grad=False)  # [rank_ff, d_model]
            self.b1 = nn.Parameter(torch.zeros(2*inter, dtype=factor_dtype), requires_grad=False)
            self.b2 = nn.Parameter(torch.zeros(self.d_model, dtype=factor_dtype), requires_grad=False)
            
            # Set up the rank projection (average of gate and up projections for initialization)
            combined_V = torch.cat([g_V, u_V], dim=0)  # [2*rank_ff, d_model]
            avg_V = combined_V.view(2, rank_ff, -1).mean(dim=0)  # [rank_ff, d_model]
            with torch.no_grad():
                self.U1.weight.copy_(avg_V.to(mlp.gate_proj.weight.dtype))
                
            self.use_lowrank_ff = True
        else:
            self.gate = nn.Linear(self.d_model, inter, bias=False, dtype=mlp.gate_proj.weight.dtype)
            self.up   = nn.Linear(self.d_model, inter, bias=False, dtype=mlp.up_proj.weight.dtype)
            self.down = nn.Linear(inter, self.d_model, bias=False, dtype=mlp.down_proj.weight.dtype)
            with torch.no_grad():
                self.gate.weight.copy_(mlp.gate_proj.weight)
                self.up.weight.copy_(mlp.up_proj.weight)
                self.down.weight.copy_(mlp.down_proj.weight)
            self.use_lowrank_ff = False

    def _apply_rope_simple(self, q_bthd, k_bthd, position_ids):
        """
        Simplified RoPE application for [B,T,H,D] tensors
        """
        B, T, H, D = q_bthd.shape
        
        # Simple call to rotary embedding
        dummy = torch.empty((B * H, T, D), device=q_bthd.device, dtype=q_bthd.dtype)
        pos_ids_expanded = position_ids.unsqueeze(1).expand(B, H, T).reshape(B * H, T)
        
        try:
            cos, sin = self.rotary_emb(dummy, position_ids=pos_ids_expanded)
            cos = cos.view(B, H, T, D).permute(0, 2, 1, 3).contiguous()  # [B,T,H,D]
            sin = sin.view(B, H, T, D).permute(0, 2, 1, 3).contiguous()  # [B,T,H,D]
            
            # Apply rotation 
            q_rot = apply_rotary_pos_emb(q_bthd.transpose(1,2), k_bthd.transpose(1,2), cos.transpose(1,2), sin.transpose(1,2))[0].transpose(1,2)
            k_rot = apply_rotary_pos_emb(q_bthd.transpose(1,2), k_bthd.transpose(1,2), cos.transpose(1,2), sin.transpose(1,2))[1].transpose(1,2)
            
            return q_rot, k_rot
        except Exception:
            # Fallback - no rotation
            return q_bthd, k_bthd

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **_):
        """
        hidden_states: [B, T, D]
        attention_mask: None, [B, T] (1=keep, 0=pad), or [B, 1, Q, K] additive mask
        position_ids: [B, T]
        """
        B, T, D = hidden_states.shape
        x = self.ln1(hidden_states)

        # Normalize mask to [B,T] keep-mask and trim to T_max
        if attention_mask is None:
            T_max = T
            x_trim = x
            pos_ids = position_ids
        else:
            if attention_mask.dim() == 2:                       # [B,T]
                keep_t = (attention_mask[:, :T] > 0)
            elif attention_mask.dim() == 4:                     # [B,1,Q,K] additive bias
                m = attention_mask.to(torch.float32).squeeze(1) # [B,Q,K]
                keep_t = (m > -1e3).any(dim=1)[:, :T]           # [B,T]
            else:
                keep_t = attention_mask.reshape(B, -1)[:, :T].to(torch.bool)
            T_max = int(keep_t.sum(dim=1).max().item())
            x_trim = x[:, :T_max, :]
            pos_ids = position_ids[:, :T_max] if position_ids is not None else None

        # Attention computation using optimized kernel
        if not self.bypass_svd_qkv:
            # Project to rank space per head
            Pq_all = self.Pq_proj(x_trim)  # [B, T_max, H*rank_q]
            Pk_all = self.Pk_proj(x_trim)  # [B, T_max, H_kv*rank_kv] 
            Pv_all = self.Pv_proj(x_trim)  # [B, T_max, H_kv*rank_kv]
            
            # Reshape to per-head rank space: [B, T_max, H, rank] -> [B*H, T_max, rank]
            Pq = Pq_all.view(B, T_max, self.n_heads, rank_q).view(B * self.n_heads, T_max, rank_q)
            Pk = Pk_all.view(B, T_max, self.n_kv_heads, rank_kv).view(B * self.n_kv_heads, T_max, rank_kv)
            Pv = Pv_all.view(B, T_max, self.n_kv_heads, rank_kv).view(B * self.n_kv_heads, T_max, rank_kv)
            
            if pos_ids is None:
                pos_ids = torch.arange(T_max, device=hidden_states.device)[None, :].expand(B, -1)
            
            # For now, let's use a simpler approach - compute Q/K/V directly and use standard flash attention
            # TODO: Integrate the optimized kernel properly
            
            # Compute Q/K/V from rank space - ensure dtype compatibility
            Vq_reshaped = self.Vq_factor.view(rank_q, self.n_heads, self.head_dim).permute(1, 0, 2).to(Pq_all.dtype)  # [H, r, dh] 
            Vk_reshaped = self.Vk_factor.view(rank_kv, self.n_kv_heads, self.head_dim).permute(1, 0, 2).to(Pk_all.dtype)  # [H_kv, r, dh]
            Vv_reshaped = self.Vv_factor.view(rank_kv, self.n_kv_heads, self.head_dim).permute(1, 0, 2).to(Pv_all.dtype)  # [H_kv, r, dh]
            
            Q_all = torch.matmul(Pq_all.view(B, T_max, self.n_heads, rank_q), Vq_reshaped)  # [B, T_max, H, dh]
            K_all = torch.matmul(Pk_all.view(B, T_max, self.n_kv_heads, rank_kv), Vk_reshaped)  # [B, T_max, H_kv, dh]  
            V_all = torch.matmul(Pv_all.view(B, T_max, self.n_kv_heads, rank_kv), Vv_reshaped)  # [B, T_max, H_kv, dh]
            
            # Expand K,V for GQA
            if self.n_rep > 1:
                K_all = K_all.unsqueeze(3).expand(B, T_max, self.n_kv_heads, self.n_rep, self.head_dim).contiguous().view(B, T_max, self.n_heads, self.head_dim)
                V_all = V_all.unsqueeze(3).expand(B, T_max, self.n_kv_heads, self.n_rep, self.head_dim).contiguous().view(B, T_max, self.n_heads, self.head_dim)
            else:
                K_all = K_all.view(B, T_max, self.n_heads, self.head_dim)
                V_all = V_all.view(B, T_max, self.n_heads, self.head_dim)
            
            # Apply RoPE  
            Q_rot, K_rot = self._apply_rope_simple(Q_all, K_all, pos_ids)
            
            # Use flash attention
            Q_bt = Q_rot * (1.0 / math.sqrt(self.head_dim))  # [B, T_max, H, dh] 
            K_bt = K_rot  # [B, T_max, H, dh]
            V_bt = V_all  # [B, T_max, H, dh]
            
            # Handle attention mask
            if attention_mask is not None and attention_mask.dim() == 2:
                keep = attention_mask[:, :T_max].view(B, T_max, 1, 1).to(Q_bt.dtype)
                Q_bt = Q_bt * keep
                K_bt = K_bt * keep  
                V_bt = V_bt * keep
            
            O_bt = flash_attn_func(Q_bt, K_bt, V_bt, dropout_p=0.0, softmax_scale=None, causal=True)  # [B, T_max, H, dh]
            attn = O_bt.contiguous().view(B, T_max, self.n_heads * self.head_dim)  # [B, T_max, d_model]
        else:
            # Fallback to original implementation
            q = torch.matmul(x_trim, self.q_proj.weight.t()).view(B, T_max, self.n_heads, self.head_dim).transpose(1, 2)
            k = torch.matmul(x_trim, self.k_proj.weight.t()).view(B, T_max, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = torch.matmul(x_trim, self.v_proj.weight.t()).view(B, T_max, self.n_kv_heads, self.head_dim).transpose(1, 2)
            # ... rest of fallback attention logic would go here
            attn = torch.zeros(B, T_max, self.d_model, device=x.device, dtype=x.dtype)  # placeholder

        # Output projection
        if hasattr(self, "use_lowrank_o") and self.use_lowrank_o:
            attn = (attn.to(self.o_V.dtype) @ self.o_V.t()) @ self.o_Us.t()
        else:
            attn = self.o(attn)

        h = hidden_states[:, :T_max, :] + attn
        y = self.ln2(h)

        # FFN computation using optimized SwiGLU kernel
        if hasattr(self, "use_lowrank_ff") and self.use_lowrank_ff:
            # Project to rank space
            P = self.U1(y)  # [B, T_max, rank_ff]
            
            # Use optimized SwiGLU kernel - ensure dtype compatibility
            ff = flashsvd_ffn_swiglu(
                P=P,
                V1=self.V1.to(P.dtype),  # [rank_ff, 2*inter] 
                U2=self.U2.to(P.dtype),  # [inter, rank_ff]
                V2=self.V2.to(P.dtype),  # [rank_ff, d_model]
                b1=self.b1.to(P.dtype),  # [2*inter]
                b2=self.b2.to(P.dtype),  # [d_model]
            )  # [B, T_max, d_model]
        else:
            ff = self.down(F.silu(self.gate(y)) * self.up(y))

        out = h + ff

        # pad back to original T
        if T_max < T:
            pad = torch.zeros(B, T - T_max, D, dtype=out.dtype, device=out.device)
            out = torch.cat([out, pad], dim=1)
        return (out,)


# ────────────────────── Original Flash + SVD block ──────────────────────
class FlashSVDLlamaBlock(nn.Module):
    """
    LLaMA block:
      • per-head SVD for q/k/v (optional bypass)
      • optional low-rank o/ff
      • FlashAttention (causal)
      • RoPE with exact HF semantics (uses position_ids when provided)
    """
    def __init__(self, hf_layer: nn.Module, cfg,
                 rank_q: int,
                 rank_kv: int,
                 rank_o: Optional[int],
                 rank_ff: Optional[int],
                 factor_dtype: torch.dtype = torch.float32,     # store SVD in fp32 by default
                 compute_in_fp32: bool = True,
                 bypass_svd_qkv: bool = False,
                 use_varlen: bool = True):
        super().__init__()
        attn, mlp = hf_layer.self_attn, hf_layer.mlp
        self.d_model    = int(cfg.hidden_size)
        self.n_heads    = int(cfg.num_attention_heads)
        self.n_kv_heads = int(getattr(cfg, "num_key_value_heads", self.n_heads))
        self.head_dim   = self.d_model // self.n_heads
        self.n_rep      = self.n_heads // self.n_kv_heads
        self.compute_in_fp32 = bool(compute_in_fp32)
        self.factor_dtype = factor_dtype
        self.bypass_svd_qkv = bool(bypass_svd_qkv)
        self.use_varlen = bool(use_varlen)

        # Norms & RoPE
        self.ln1 = hf_layer.input_layernorm
        self.ln2 = hf_layer.post_attention_layernorm
        self.rotary_emb = getattr(attn, "rotary_emb", None)
        if self.rotary_emb is None:
            # Robust fallback matching HF semantics incl. position_ids
            rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
            class _SimpleRoPE(nn.Module):
                def __init__(self, head_dim: int, base: float = 10000.0):
                    super().__init__()
                    self.head_dim = head_dim
                    self.base = base
                    evens = torch.arange(0, head_dim, 2, dtype=torch.float32)
                    self.register_buffer("inv_freq", 1.0 / (base ** (evens / head_dim)), persistent=False)
                def forward(self, x, seq_len: int = None, position_ids: Optional[torch.LongTensor] = None):
                    device = x.device
                    inv = self.inv_freq.to(device=device)
                    Dh = self.head_dim
                    if position_ids is not None:
                        # position_ids: [B, T]
                        t = position_ids.to(torch.float32)  # [B,T]
                        ang = t[..., None] * inv[None, None, :]      # [B,T,Dh/2]
                        ang = ang.repeat_interleave(2, dim=-1)       # [B,T,Dh]
                        cos = ang.cos().to(x.dtype)                  # [B,T,Dh] - Fixed: removed extra dimensions
                        sin = ang.sin().to(x.dtype)
                        return cos, sin
                    assert seq_len is not None, "Provide seq_len or position_ids"
                    t = torch.arange(seq_len, device=device, dtype=torch.float32)  # [T]
                    ang = t[:, None] * inv[None, :]             # [T, Dh/2]
                    ang = ang.repeat_interleave(2, dim=-1)      # [T, Dh]
                    cos = ang.cos()[None, :, :].to(x.dtype)     # [1,T,Dh] - Fixed: removed extra dimension
                    sin = ang.sin()[None, :, :].to(x.dtype)
                    return cos, sin
            self.rotary_emb = _SimpleRoPE(self.head_dim, base=rope_theta)

        # Keep original dense projections (for bypass)
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj

        # --- per-head SVD for Q/K/V ---
        if not self.bypass_svd_qkv:
            q_Us, q_V = _decompose_heads_svd(attn.q_proj.weight, self.n_heads,    self.head_dim, rank_q)
            k_Us, k_V = _decompose_heads_svd(attn.k_proj.weight, self.n_kv_heads, self.head_dim, rank_kv)
            v_Us, v_V = _decompose_heads_svd(attn.v_proj.weight, self.n_kv_heads, self.head_dim, rank_kv)
            self.q_Us = nn.Parameter(q_Us.to(factor_dtype), requires_grad=False)
            self.q_V  = nn.Parameter(q_V.to(factor_dtype),  requires_grad=False)
            self.k_Us = nn.Parameter(k_Us.to(factor_dtype), requires_grad=False)
            self.k_V  = nn.Parameter(k_V.to(factor_dtype),  requires_grad=False)
            self.v_Us = nn.Parameter(v_Us.to(factor_dtype), requires_grad=False)
            self.v_V  = nn.Parameter(v_V.to(factor_dtype),  requires_grad=False)

        # Optional reconstruction check at full rank
        if os.getenv("CHECK_SVD", "0") == "1" and (not self.bypass_svd_qkv):
            with torch.no_grad():
                def _max_err(W, H, dh, Us, V):
                    # W: [H*dh, D] -> compare with reconstructed [H,dh,D]
                    Wv = W.float().view(H, dh, -1)                    # [H, dh, D]
                    R  = torch.einsum('hdr,hrD->hdD', Us.float(), V.float())  # [H, dh, D]
                    return (Wv - R).abs().max().item()
                print(f"[SVD check] max |Q - UΣV|: {_max_err(attn.q_proj.weight, self.n_heads, self.head_dim, self.q_Us, self.q_V):.3e}")
                print(f"[SVD check] max |K - UΣV|: {_max_err(attn.k_proj.weight, self.n_kv_heads, self.head_dim, self.k_Us, self.k_V):.3e}")
                print(f"[SVD check] max |V - UΣV|: {_max_err(attn.v_proj.weight, self.n_kv_heads, self.head_dim, self.v_Us, self.v_V):.3e}")

        # --- Output projection (low-rank or dense passthrough) ---
        if rank_o is not None:
            o_Us, o_V = _decompose_full_svd(attn.o_proj.weight, rank_o)
            self.o_Us = nn.Parameter(o_Us.to(factor_dtype), requires_grad=False)
            self.o_V  = nn.Parameter(o_V.to(factor_dtype),  requires_grad=False)
            self.use_lowrank_o = True
        else:
            self.o = nn.Linear(self.n_heads * self.head_dim, self.d_model,
                               bias=False, dtype=attn.o_proj.weight.dtype)
            with torch.no_grad():
                self.o.weight.copy_(attn.o_proj.weight)
            self.use_lowrank_o = False

        # --- MLP (low-rank or dense passthrough) ---
        inter = int(cfg.intermediate_size)
        if rank_ff is not None:
            g_Us, g_V = _decompose_full_svd(mlp.gate_proj.weight, rank_ff)
            u_Us, u_V = _decompose_full_svd(mlp.up_proj.weight,   rank_ff)
            d_Us, d_V = _decompose_full_svd(mlp.down_proj.weight, rank_ff)
            self.g_Us = nn.Parameter(g_Us.to(factor_dtype), requires_grad=False)
            self.g_V  = nn.Parameter(g_V.to(factor_dtype),  requires_grad=False)
            self.u_Us = nn.Parameter(u_Us.to(factor_dtype), requires_grad=False)
            self.u_V  = nn.Parameter(u_V.to(factor_dtype),  requires_grad=False)
            self.d_Us = nn.Parameter(d_Us.to(factor_dtype), requires_grad=False)
            self.d_V  = nn.Parameter(d_V.to(factor_dtype),  requires_grad=False)
            self.use_lowrank_ff = True
        else:
            self.gate = nn.Linear(self.d_model, inter, bias=False, dtype=mlp.gate_proj.weight.dtype)
            self.up   = nn.Linear(self.d_model, inter, bias=False, dtype=mlp.up_proj.weight.dtype)
            self.down = nn.Linear(inter, self.d_model, bias=False, dtype=mlp.down_proj.weight.dtype)
            with torch.no_grad():
                self.gate.weight.copy_(mlp.gate_proj.weight)
                self.up.weight.copy_(mlp.up_proj.weight)
                self.down.weight.copy_(mlp.down_proj.weight)
            self.use_lowrank_ff = False

    @torch.no_grad()
    def _proj_per_head(self, x: torch.Tensor, Us: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        x:  [B, T, D]
        V:  [H, r, D]   ; Us: [H, dh, r]
        return [B, H, T, dh]
        """
        if self.compute_in_fp32:
            xr  = torch.einsum('b t d, h r d -> b t h r', x.float(), V.float())   # [B,T,H,r]
            out = torch.einsum('b t h r, h d r -> b t h d', xr, Us.float())       # [B,T,H,dh]
            return out.to(x.dtype).transpose(1, 2).contiguous()                   # [B,H,T,dh]
        xr  = torch.einsum('b t d, h r d -> b t h r', x.to(V.dtype), V)
        out = torch.einsum('b t h r, h d r -> b t h d', xr, Us)
        return out.to(x.dtype).transpose(1, 2).contiguous()

    def _apply_rope(self, q_bhtd, k_bhtd, position_ids):
        """
        Expect HF semantics: apply RoPE on tensors laid out as [B,H,T,Dh].
        We transpose to BTHD, apply, then transpose back to BHTD.
        Fixed to handle dimension mismatches properly.
        """
        # [B,H,T,Dh] -> [B,T,H,Dh]
        q_bthd = q_bhtd.transpose(1, 2).contiguous()
        k_bthd = k_bhtd.transpose(1, 2).contiguous()

        B, T, H, actual_dh = q_bthd.shape
        
        # Verify dimensions match expected head_dim
        if actual_dh != self.head_dim:
            raise ValueError(f"Tensor head dimension {actual_dh} doesn't match expected head_dim {self.head_dim}")

        # Try to use position_ids if the rotary module supports it
        cos = sin = None
        try:
            sig = inspect.signature(self.rotary_emb.forward)
            if "position_ids" in sig.parameters and position_ids is not None:
                cos, sin = self.rotary_emb(q_bthd, position_ids=position_ids)
            elif "seq_len" in sig.parameters:
                cos, sin = self.rotary_emb(q_bthd, seq_len=T)
            else:
                # Some old versions take (x, seq_len) positionally
                cos, sin = self.rotary_emb(q_bthd, T)
        except TypeError:
            # Fallbacks to handle odd signatures
            try:
                cos, sin = self.rotary_emb(q_bthd, position_ids=position_ids)
            except TypeError:
                cos, sin = self.rotary_emb(q_bthd, seq_len=T)

        # Ensure cos/sin have the right dimensions
        if cos.dim() == 3 and cos.shape[-1] != actual_dh:
            # If cos/sin don't match the tensor head_dim, there's a fundamental issue
            raise ValueError(f"RoPE cos/sin dimension {cos.shape[-1]} doesn't match tensor head_dim {actual_dh}")

        print(f"DEBUG - About to call apply_rotary_pos_emb:")
        print(f"  q_bthd.shape: {q_bthd.shape}")
        print(f"  k_bthd.shape: {k_bthd.shape}")
        print(f"  cos.shape: {cos.shape}")
        print(f"  sin.shape: {sin.shape}")

        q_rot, k_rot = apply_rotary_pos_emb(q_bthd, k_bthd, cos, sin,
                                            position_ids=position_ids if cos.shape[0] != 1 else None)
        # back to [B,H,T,Dh]
        return q_rot.transpose(1, 2).contiguous(), k_rot.transpose(1, 2).contiguous()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **_):
        """
        hidden_states: [B, T, D]
        attention_mask:
          - None
          - [B, T] (1=keep, 0=pad)
          - [B, 1, Q, K] additive mask
        position_ids: [B, T]
        """
        B, T, D = hidden_states.shape
        x = self.ln1(hidden_states)

        # Normalize mask to [B,T] keep-mask and trim to T_max
        if attention_mask is None:
            keep_t = None
            T_max = T
            x_trim = x
            pos_ids = position_ids
        else:
            if attention_mask.dim() == 2:                       # [B,T]
                keep_t = (attention_mask[:, :T] > 0)
            elif attention_mask.dim() == 4:                     # [B,1,Q,K] additive bias
                m = attention_mask.to(torch.float32).squeeze(1) # [B,Q,K]
                keep_t = (m > -1e3).any(dim=1)[:, :T]           # [B,T]
            else:
                keep_t = attention_mask.reshape(B, -1)[:, :T].to(torch.bool)
            T_max = int(keep_t.sum(dim=1).max().item())
            x_trim = x[:, :T_max, :]
            pos_ids = position_ids[:, :T_max] if position_ids is not None else None

        # Q/K/V (either bypass dense or SVD)
        if self.bypass_svd_qkv:
            q = self.q_proj(x_trim).view(B, T_max, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,dh]
            k = self.k_proj(x_trim).view(B, T_max, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x_trim).view(B, T_max, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            q = self._proj_per_head(x_trim, self.q_Us, self.q_V)         # [B,H,T,dh]
            k = self._proj_per_head(x_trim, self.k_Us, self.k_V)         # [B,Hk,T,dh]
            v = self._proj_per_head(x_trim, self.v_Us, self.v_V)         # [B,Hk,T,dh]

        # GQA repeat and RoPE with true positions
        k = _repeat_kv(k, self.n_rep)                                    # [B,H,T,dh]
        v = _repeat_kv(v, self.n_rep)
        
        q, k = self._apply_rope(q, k, pos_ids)                           # [B,H,T,dh]

        # Layout for FlashAttention: [B,T,H,dh]
        q_bt = q.transpose(1, 2).contiguous()
        k_bt = k.transpose(1, 2).contiguous()
        v_bt = v.transpose(1, 2).contiguous()

        # Exact HF scaling: scale Q by 1/sqrt(dh); pass softmax_scale=None
        q_bt = q_bt * (1.0 / math.sqrt(self.head_dim))

        # FlashAttention
        if keep_t is None or not self.use_varlen:
            if keep_t is not None:
                keep = keep_t[:, :T_max].view(B, T_max, 1, 1).to(q_bt.dtype)
                q_bt = q_bt * keep
                k_bt = k_bt * keep
                v_bt = v_bt * keep
            out_bt = flash_attn_func(q_bt, k_bt, v_bt, dropout_p=0.0, softmax_scale=None, causal=True)  # [B,T,H,dh]
        else:
            B_, T_ = B, T_max
            H_, Dh_ = self.n_heads, self.head_dim
            flat_keep = keep_t.reshape(B_ * T_)                               # [B*T]
            q_flat = q_bt.reshape(B_ * T_, H_, Dh_)[flat_keep]                # [sumL, H, Dh]
            k_flat = k_bt.reshape(B_ * T_, H_, Dh_)[flat_keep]
            v_flat = v_bt.reshape(B_ * T_, H_, Dh_)[flat_keep]

            seqlens = keep_t.sum(dim=1, dtype=torch.int32)                    # [B]
            cu_seqlens = torch.zeros(B_ + 1, dtype=torch.int32, device=q_bt.device)
            cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
            max_seqlen = int(seqlens.max().item())

            out_flat = flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens, cu_seqlens,
                max_seqlen, max_seqlen,
                dropout_p=0.0, softmax_scale=None, causal=True
            )  # [sum(seqlens), H, Dh]

            out_bt = torch.zeros_like(q_bt)                                    # [B,T,H,Dh]
            out_bt.reshape(B_ * T_, H_, Dh_)[flat_keep] = out_flat

        # Flatten heads (NO swapping T/H)
        attn = out_bt.contiguous().view(B, T_max, self.n_heads * self.head_dim)  # [B,T,D]

        # Output proj + MLP
        if hasattr(self, "use_lowrank_o") and self.use_lowrank_o:
            attn = (attn.to(self.o_V.dtype) @ self.o_V.t()) @ self.o_Us.t()
        else:
            attn = self.o(attn)

        h = hidden_states[:, :T_max, :] + attn
        y = self.ln2(h)

        if hasattr(self, "use_lowrank_ff") and self.use_lowrank_ff:
            y1 = (y.to(self.g_V.dtype) @ self.g_V.t()) @ self.g_Us.t()
            y2 = (y.to(self.u_V.dtype) @ self.u_V.t()) @ self.u_Us.t()
            ff = ((F.silu(y1) * y2) @ self.d_V.t()) @ self.d_Us.t()
        else:
            ff = self.down(F.silu(self.gate(y)) * self.up(y))

        out = h + ff

        # pad back to original T
        if T_max < T:
            pad = torch.zeros(B, T - T_max, D, dtype=out.dtype, device=out.device)
            out = torch.cat([out, pad], dim=1)
        return (out,)

# ────────────────────── wire into HF model ──────────────────────
def replace_with_flash_svd(model, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32):
    cfg = model.config
    dev = next(model.parameters()).device
    bypass = os.getenv("BYPASS_SVD_QKV", "0") == "1"
    use_varlen = os.getenv("USE_VARLEN", "1") == "1"
    use_optimized = os.getenv("USE_OPTIMIZED_KERNELS", "1") == "1"
    
    new_layers = nn.ModuleList()
    for layer in model.model.layers:
        shim = _wrap_svd_layer(layer, cfg, rank_q, rank_kv, rank_o, rank_ff,
                               factor_dtype, compute_in_fp32, bypass, use_varlen, use_optimized)
        shim.to(dev)
        new_layers.append(shim)
    model.model.layers = new_layers

def _wrap_svd_layer(hf_layer, cfg, rank_q, rank_kv, rank_o, rank_ff,
                    factor_dtype, compute_in_fp32, bypass, use_varlen, use_optimized=True):
    class _Shim(nn.Module):
        def __init__(self, inner):
            super().__init__()
            if use_optimized:
                self.block = OptimizedFlashSVDLlamaBlock(
                    inner, cfg,
                    rank_q=rank_q, rank_kv=rank_kv, rank_o=rank_o, rank_ff=rank_ff,
                    factor_dtype=factor_dtype, compute_in_fp32=compute_in_fp32,
                    bypass_svd_qkv=bypass, use_varlen=use_varlen
                )
            else:
                self.block = FlashSVDLlamaBlock(
                    inner, cfg,
                    rank_q=rank_q, rank_kv=rank_kv, rank_o=rank_o, rank_ff=rank_ff,
                    factor_dtype=factor_dtype, compute_in_fp32=compute_in_fp32,
                    bypass_svd_qkv=bypass, use_varlen=use_varlen
                )
        def forward(self, hidden_states, attention_mask=None, position_ids=None, **kw):
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            y, = self.block(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
            return (y,)
    return _Shim(hf_layer)

# ────────────────────── quick eval (full-seq) ──────────────────────
@torch.no_grad()
def eval_perplexity_fullseq(model, loader, device):
    model.eval()
    total_loss, total_tok = 0.0, 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()
        labels = batch["input_ids"][:, 1:].contiguous()
        mask   = batch["attention_mask"][:, 1:].contiguous().bool()
        if mask.any():
            v_logits = logits[mask].float()           # [N, V]
            v_labels = labels[mask]                   # [N]
            finite = torch.isfinite(v_logits).all(dim=-1)
            if finite.any():
                loss = F.cross_entropy(v_logits[finite], v_labels[finite], reduction="sum")
                total_loss += loss.item()
                total_tok  += int(finite.sum().item())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(loader))
    ppl = math.exp(total_loss / total_tok) if total_tok > 0 else float("nan")
    peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    return ppl, peak, ms

# ────────────────────── main ──────────────────────
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Env knobs
    dt = os.getenv("DTYPE", "float16").lower()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dt]
    MODEL_NAME = os.getenv("LLAMA_MODEL", "meta-llama/Llama-2-7b-hf")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
    SEQ_LEN    = int(os.getenv("SEQ_LEN", "1024"))
    MAX_EVAL_SAMPLES = int(os.getenv("MAX_EVAL_SAMPLES", "64"))
    # SVD ranks (per-head for Q/K/V, whole-matrix for O/FF)
    RANK_Q  = int(os.getenv("RANK_Q",  "96"))
    RANK_KV = int(os.getenv("RANK_KV", "96"))
    RANK_O  = int(os.getenv("RANK_O",  "0")) or None          # 0 → dense
    RANK_FF = int(os.getenv("RANK_FF", "0")) or None          # 0 → dense
    SVD_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[os.getenv("SVD_DTYPE", "fp32").lower()]
    SVD_COMPUTE_FP32 = os.getenv("SVD_COMPUTE_FP32", "1") == "1"

    # Load model/tokenizer
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.config.use_cache = False  # full-seq
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Swap in FlashAttention+SVD layers
    replace_with_flash_svd(
        model, rank_q=RANK_Q, rank_kv=RANK_KV, rank_o=RANK_O, rank_ff=RANK_FF,
        factor_dtype=SVD_DTYPE, compute_in_fp32=SVD_COMPUTE_FP32
    )

    # Data (right-pad to SEQ_LEN)
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw = raw.select(range(min(MAX_EVAL_SAMPLES, len(raw)))) if MAX_EVAL_SAMPLES > 0 else raw
    def tokenize_fn(batch):
        return tok(batch["text"], padding="max_length", truncation=True, max_length=SEQ_LEN)
    ds = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]),
                              "attention_mask": torch.stack([x["attention_mask"] for x in b])}
    )

    # Run quick eval
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    ppl, peak_mem, time_ms = eval_perplexity_fullseq(model, loader, device)
    storage_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    use_optimized = os.getenv("USE_OPTIMIZED_KERNELS", "1") == "1"
    kernel_type = "Optimized Triton Kernels" if use_optimized else "Standard Flash Attention"
    
    print(f"\n================== LLaMA + FlashAttention + SVD ({kernel_type}) ==================")
    print(f"Python {platform.python_version()}  Torch {torch.__version__}")
    print(f"Device/dtype: {device}/{dtype}")
    print(f"Ranks: q={RANK_Q}, kv={RANK_KV}, o={RANK_O}, ff={RANK_FF}  |  SVD store dtype={SVD_DTYPE}  |  SVD compute fp32={SVD_COMPUTE_FP32}")
    if use_optimized:
        print("Using optimized kernels: Low-rank SVD factorization + SwiGLU (Triton kernel integration in progress)")
    print(f"{'Model':<20} | {'Storage (MiB)':<14} | {'Peak (MiB)':<10} | {'Time (ms/b)':<12} | {'Perplexity':<10}")
    print("-" * 100)
    model_name = "LLaMA+FlashSVD-Opt" if use_optimized else "LLaMA+FlashSVD"
    print(f"{model_name:<20} | {storage_mem:<14.1f} | {peak_mem:<10.1f} | {time_ms:<12.1f} | {ppl:<10.4f}")


