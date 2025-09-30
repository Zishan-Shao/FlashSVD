import os
from contextlib import nullcontext
import sys
import time
import math
import platform
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# ─────────────────────────────────────────────────────────────────────────────
# Path bootstrap (same as your original)
THIS_FILE = os.path.abspath(__file__)
DECODERS_DIR = os.path.dirname(os.path.dirname(THIS_FILE))  # decoders/
if DECODERS_DIR not in sys.path:
    sys.path.insert(0, DECODERS_DIR)
PROJECT_ROOT = os.path.dirname(DECODERS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# FlashAttention detection (FA2 + varlen variants)
def _get_fa_variants():
    if not torch.cuda.is_available():
        return None, None, None
    _fa = _fa_varlen = _fa_varlen_kvpacked = None
    try:
        from flash_attn import flash_attn_func as _fa  # [B,S,H,D]
    except Exception:
        try:
            from flash_attn.flash_attn_interface import flash_attn_func as _fa
        except Exception:
            _fa = None
    try:
        from flash_attn import flash_attn_varlen_func as _fa_varlen  # [Tq,H,D] / [Tk,H,D]
    except Exception:
        try:
            from flash_attn.flash_attn_interface import flash_attn_varlen_func as _fa_varlen
        except Exception:
            _fa_varlen = None
    try:
        from flash_attn import flash_attn_varlen_kvpacked_func as _fa_varlen_kvpacked  # kv:[Tk,2,H,D]
    except Exception:
        try:
            from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func as _fa_varlen_kvpacked
        except Exception:
            _fa_varlen_kvpacked = None
    return _fa, _fa_varlen, _fa_varlen_kvpacked

_FA, _FA_VARLEN, _FA_VARLEN_KVPACKED = _get_fa_variants()

def _can_use_fa(x: torch.Tensor) -> bool:
    return (
        os.getenv("USE_FLASH_ATTENTION", "1") != "0"
        and _FA is not None
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
    )

def _want_varlen() -> bool:
    return os.getenv("FA_USE_VARLEN", "1") != "0"

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
def _repeat_kv(x, n_rep: int):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, :, None, :].expand(b, h, s, n_rep, d).reshape(b, h * n_rep, s, d)

def _build_full_bias(attention_mask, batch_size, q_len, k_len, device, dtype):
    # Causal (with KV offset)
    i = torch.arange(q_len, device=device).view(q_len, 1)
    j = torch.arange(k_len, device=device).view(1, k_len)
    past_len = k_len - q_len
    causal = (j <= (past_len + i))
    causal_bias = torch.zeros(q_len, k_len, device=device, dtype=torch.float32)
    causal_bias.masked_fill_(~causal, -1e4)
    causal_bias = causal_bias.view(1, 1, q_len, k_len)

    pad_bias = None
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            am = attention_mask
            if am.size(-1) < k_len:
                am = F.pad(am, (0, k_len - am.size(-1)), value=1)
            elif am.size(-1) > k_len:
                am = am[:, -k_len:]
            pad_bias = (1.0 - am.float()) * -1e4
            pad_bias = pad_bias.view(batch_size, 1, 1, k_len).to(dtype=torch.float32, device=device)
        elif attention_mask.dim() == 4:
            pad_bias = attention_mask.to(dtype=torch.float32, device=device)
            if pad_bias.size(-1) != k_len:
                if pad_bias.size(-1) < k_len:
                    pad_bias = F.pad(pad_bias, (0, k_len - pad_bias.size(-1)), value=0.0)
                else:
                    pad_bias = pad_bias[..., -k_len:]
    if pad_bias is None:
        return causal_bias
    if pad_bias.size(-2) == 1:
        pad_bias = pad_bias.expand(-1, -1, q_len, -1)
    return causal_bias + pad_bias

# ─────────────────────────────────────────────────────────────────────────────
# RoPE
class SimpleRoPE(torch.nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = int(head_dim)
        self.base = float(base)

    def forward(self, x: torch.Tensor, seq_len: int):
        device = x.device
        dtype = torch.float32
        evens = torch.arange(0, self.head_dim, 2, dtype=dtype, device=device)
        inv_freq = 1.0 / (self.base ** (evens / self.head_dim))
        t = torch.arange(seq_len, dtype=dtype, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = freqs.repeat_interleave(2, dim=-1)
        cos = emb.cos()[None, None, :, :].to(x.dtype)
        sin = emb.sin()[None, None, :, :].to(x.dtype)
        return cos, sin

# ─────────────────────────────────────────────────────────────────────────────
# Varlen packing / unpacking utilities
def _pack_varlen_BSHD(t: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    t: [B,S,H,D], m: [B,S] (bool/0-1)
    returns: t_packed [T,H,D], cu_seqlens [B+1]int32, max_S:int
    """
    B, S = m.shape
    m = m.to(dtype=torch.bool)
    lens = m.sum(dim=-1, dtype=torch.int32)        # [B]
    cu = torch.zeros(B + 1, dtype=torch.int32, device=t.device)
    cu[1:] = torch.cumsum(lens, dim=0)
    total = int(cu[-1].item())
    max_S = int(lens.max().item()) if B > 0 else 0
    if total == 0:
        empty = t.new_empty((0, t.size(2), t.size(3)))
        return empty, cu, max_S
    t_flat = t.reshape(B * S, t.size(2), t.size(3))
    m_flat = m.reshape(B * S)
    packed = t_flat[m_flat]                         # [T, H, D]
    return packed, cu, max_S

def _unpack_varlen_BSHD(packed: torch.Tensor, m: torch.Tensor, B: int, S: int, H: int, D: int) -> torch.Tensor:
    """
    inverse of _pack_varlen_BSHD: returns [B,S,H,D]
    """
    out = packed.new_zeros((B, S, H, D))
    if packed.numel() == 0:
        return out
    flat_out = out.view(B * S, H, D)
    flat_mask = m.to(dtype=torch.bool).view(B * S)
    flat_out[flat_mask] = packed
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Core FA varlen wrapper
def _flash_attn_varlen(
    q_bshd: torch.Tensor,       # [B,Q,H,D]
    k_bshd: torch.Tensor,       # [B,K,H,D]
    v_bshd: torch.Tensor,       # [B,K,H,D]
    q_mask: torch.Tensor,       # [B,Q] (1 for valid query positions)
    k_mask: torch.Tensor,       # [B,K] (1 for valid key positions)
    softmax_scale: float,
    causal: bool = True,
) -> Optional[torch.Tensor]:
    """
    Packs Q,K,V according to masks and runs FA2 varlen kernels when available.
    Returns attn_out [B,Q,H,D] or None if varlen kernels are unavailable.
    """
    if not _can_use_fa(q_bshd) or not _want_varlen():
        return None
    # Prefer kvpacked variant; else fall back to separate q/k/v varlen; else None
    have_kvpacked = _FA_VARLEN_KVPACKED is not None
    have_varlen = _FA_VARLEN is not None
    if not (have_kvpacked or have_varlen):
        return None

    # Pack Q/K/V
    q_packed, cu_q, max_q = _pack_varlen_BSHD(q_bshd, q_mask)   # [Tq,H,D]
    k_packed, cu_k, max_k = _pack_varlen_BSHD(k_bshd, k_mask)   # [Tk,H,D]
    v_packed, _,    _     = _pack_varlen_BSHD(v_bshd, k_mask)   # [Tk,H,D]
    if q_packed.numel() == 0:
        # No valid queries for the whole batch
        return q_bshd.new_zeros(q_bshd.shape)

    if have_kvpacked:
        kv_packed = torch.stack([k_packed, v_packed], dim=1).contiguous()  # [Tk, 2, H, D]
        out_packed = _FA_VARLEN_KVPACKED(
            q_packed, kv_packed,
            cu_q, cu_k,
            max_q, max_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal
        )  # [Tq, H, D]
    else:
        out_packed = _FA_VARLEN(
            q_packed, k_packed, v_packed,
            cu_q, cu_k,
            max_q, max_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal
        )  # [Tq, H, D]

    # Unpack back to [B,Q,H,D]
    B, Q = q_mask.shape
    H, D = q_bshd.size(-2), q_bshd.size(-1)
    out = _unpack_varlen_BSHD(out_packed, q_mask, B, Q, H, D)
    return out

# ─────────────────────────────────────────────────────────────────────────────
class LinearLlamaBlock(nn.Module):
    """Dense LLaMA block with clean Linear layers. Uses config sizes; handles RoPE + GQA + KV cache."""
    def __init__(self, hf_layer: nn.Module, config):
        super().__init__()
        attn = hf_layer.self_attn
        mlp  = hf_layer.mlp

        # sizes
        self.d_model    = int(getattr(config, "hidden_size"))
        self.n_heads    = int(getattr(config, "num_attention_heads"))
        self.n_kv_heads = int(getattr(config, "num_key_value_heads", self.n_heads))
        self.head_dim   = self.d_model // self.n_heads
        self.n_rep      = self.n_heads // self.n_kv_heads
        self.scale      = 1.0 / math.sqrt(self.head_dim)

        # dtype consistency
        w_dtype = attn.q_proj.weight.dtype
        def make_linear(in_f, out_f):
            return nn.Linear(in_f, out_f, bias=False, dtype=w_dtype)

        # projections
        self.q_proj = make_linear(self.d_model, self.n_heads    * self.head_dim)
        self.k_proj = make_linear(self.d_model, self.n_kv_heads * self.head_dim)
        self.v_proj = make_linear(self.d_model, self.n_kv_heads * self.head_dim)
        self.o_proj = make_linear(self.n_heads * self.head_dim, self.d_model)

        with torch.no_grad():
            self.q_proj.weight.copy_(attn.q_proj.weight.to(w_dtype))
            self.k_proj.weight.copy_(attn.k_proj.weight.to(w_dtype))
            self.v_proj.weight.copy_(attn.v_proj.weight.to(w_dtype))
            self.o_proj.weight.copy_(attn.o_proj.weight.to(w_dtype))

        # MLP (SwiGLU)
        inter = int(getattr(config, "intermediate_size"))
        self.gate_proj = make_linear(self.d_model, inter)
        self.up_proj   = make_linear(self.d_model, inter)
        self.down_proj = make_linear(inter, self.d_model)
        with torch.no_grad():
            self.gate_proj.weight.copy_(mlp.gate_proj.weight.to(w_dtype))
            self.up_proj.weight.copy_(mlp.up_proj.weight.to(w_dtype))
            self.down_proj.weight.copy_(mlp.down_proj.weight.to(w_dtype))

        # norms & RoPE
        self.ln1 = hf_layer.input_layernorm
        self.ln2 = hf_layer.post_attention_layernorm
        self.rotary_emb = getattr(attn, "rotary_emb", None)
        if self.rotary_emb is None:
            try:
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
                max_pos = int(getattr(config, "max_position_embeddings", 4096))
                rope_th = float(getattr(config, "rope_theta", 10000.0))
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=max_pos, base=rope_th)
            except Exception:
                self.rotary_emb = SimpleRoPE(self.head_dim, base=float(getattr(config, "rope_theta", 10000.0)))

    def _rope_qk(self, q, k, *, position_ids=None, past_len: int = 0):
        if position_ids is None:
            b, _, q_len, _ = q.shape
            position_ids = torch.arange(past_len, past_len + q_len, device=q.device).view(1, q_len).expand(b, q_len)
        try:
            cos, sin = self.rotary_emb(q, position_ids)
        except TypeError:
            seq_len = int(position_ids.max().item()) + 1
            cos, sin = self.rotary_emb(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)
        return q, k

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, position_ids=None, use_cache=False, **kwargs):
        bsz, q_len, _ = hidden_states.shape

        x = self.ln1(hidden_states)
        q = self.q_proj(x).view(bsz, q_len, self.n_heads,    self.head_dim).transpose(1, 2)  # [B,H,Q,D]
        k = self.k_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B,Hk,Q,D]
        v = self.v_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_key_value is not None and len(past_key_value) == 2:
            past_len = past_key_value[0].size(-2)

        q, k_now = self._rope_qk(q, k, position_ids=position_ids, past_len=past_len)

        if past_len > 0:
            k = torch.cat([past_key_value[0], k_now], dim=-2)  # [B,Hk,K,D]
            v = torch.cat([past_key_value[1], v],     dim=-2)  # [B,Hk,K,D]
        else:
            k = k_now
        present = (k, v) if use_cache else None

        # repeat K/V heads for GQA -> [B,H,K,D]
        k_rep = _repeat_kv(k, self.n_heads // self.n_kv_heads)
        v_rep = _repeat_kv(v, self.n_heads // self.n_kv_heads)

        # Prefer varlen FA if available (and enabled)
        if _can_use_fa(q) and _want_varlen():
            # Convert to [B,S,H,D]
            q_bshd = q.transpose(1, 2).contiguous()      # [B,Q,H,D]
            k_bshd = k_rep.transpose(1, 2).contiguous()  # [B,K,H,D]
            v_bshd = v_rep.transpose(1, 2).contiguous()  # [B,K,H,D]

            # Build masks for Q (just the last q_len) and K (all keys up to e)
            if attention_mask is None:
                q_mask = torch.ones(bsz, q_len, dtype=torch.bool, device=hidden_states.device)
                k_mask = torch.ones(bsz, k_bshd.size(1), dtype=torch.bool, device=hidden_states.device)
            else:
                km = attention_mask
                if km.size(-1) != k_bshd.size(1):
                    if km.size(-1) < k_bshd.size(1):
                        km = F.pad(km, (0, k_bshd.size(1) - km.size(-1)), value=0)
                    else:
                        km = km[:, -k_bshd.size(1):]
                km = km.to(dtype=torch.bool)
                k_mask = km
                q_mask = km[:, -q_len:]

            attn_v = _flash_attn_varlen(q_bshd, k_bshd, v_bshd, q_mask, k_mask, self.scale, causal=True)
            if attn_v is not None:
                attn = attn_v.view(bsz, q_len, self.n_heads * self.head_dim)
                attn = self.o_proj(attn)
                h = hidden_states + attn

                # MLP
                y = self.ln2(h)
                gate = F.silu(self.gate_proj(y))
                up   = self.up_proj(y)
                ff   = self.down_proj(gate * up)
                h = h + ff
                outputs = (h,)
                if use_cache:
                    outputs += (present,)
                return outputs
            # else fall through to FA non-varlen / SDPA

        # Non-varlen FA if available; otherwise SDPA fallback
        if _can_use_fa(q):
            q_bqhd = q.transpose(1, 2).contiguous()      # [B,Q,H,D]
            k_bkhd = k_rep.transpose(1, 2).contiguous()  # [B,K,H,D]
            v_bkhd = v_rep.transpose(1, 2).contiguous()  # [B,K,H,D]
            attn_out = _FA(
                q_bqhd, k_bkhd, v_bkhd,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=True
            )  # [B,Q,H,D]
            attn = attn_out.contiguous().view(bsz, q_len, self.d_model)
        else:
            k_len = k_rep.size(-2)
            bias = _build_full_bias(attention_mask, bsz, q_len, k_len, hidden_states.device, q.dtype)
            attn = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=bias, is_causal=False)
            attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)

        attn = self.o_proj(attn)
        h = hidden_states + attn

        y = self.ln2(h)
        gate = F.silu(self.gate_proj(y))
        up   = self.up_proj(y)
        ff   = self.down_proj(gate * up)
        h = h + ff

        outputs = (h,)
        if use_cache:
            outputs += (present,)
        return outputs

# ─────────────────────────────────────────────────────────────────────────────
class LinearPerHeadSVDLlamaBlock(nn.Module):
    """
    LLaMA block using per-head SVD for q/k/v and (optional) whole-matrix low-rank for o & MLP.
    Projects Q,K,V into shape [B, M, H, dh].
    """
    def __init__(self, dense_block: LinearLlamaBlock,
                 rank_q: Optional[int], rank_kv: Optional[int], rank_o: Optional[int], rank_ff: Optional[int],
                 compute_in_float32: bool = True):
        super().__init__()
        self.d_model    = dense_block.d_model
        self.n_heads    = dense_block.n_heads
        self.n_kv_heads = dense_block.n_kv_heads
        self.head_dim   = self.d_model // self.n_heads
        self.n_rep      = self.n_heads // self.n_kv_heads
        self.scale      = 1.0 / math.sqrt(self.head_dim)
        self.compute_in_float32 = bool(compute_in_float32)

        # ---- helpers ----
        def decompose_heads_svd(weight: torch.Tensor, n_heads: int, head_dim: int, rank: Optional[int]):
            if rank is None or rank >= head_dim:
                # fall back to full-row factors (no compression)
                rank = head_dim
            Usi, Vi = [], []
            Wf = weight.float()
            for h in range(n_heads):
                W_h = Wf[h*head_dim:(h+1)*head_dim, :]
                U, S, Vh = torch.linalg.svd(W_h, full_matrices=False)
                r = max(1, min(rank, U.shape[1], Vh.shape[0]))
                Usi.append((U[:, :r] * S[:r].unsqueeze(0)))  # [dh, r]
                Vi.append(Vh[:r, :])                         # [r,  D]
            Usi = torch.stack(Usi, dim=0).to(weight.dtype)  # [H, dh, r]
            Vi  = torch.stack(Vi,  dim=0).to(weight.dtype)  # [H, r,  D]
            return nn.Parameter(Usi, requires_grad=False), nn.Parameter(Vi, requires_grad=False)

        def decompose_full_svd(weight: torch.Tensor, rank: Optional[int]):
            if rank is None or rank >= min(weight.shape[0], weight.shape[1]):
                # No compression
                Usi = weight.detach().to(weight.dtype)
                Vi  = torch.eye(weight.shape[1], device=weight.device, dtype=weight.dtype)
                return nn.Parameter(Usi, requires_grad=False), nn.Parameter(Vi, requires_grad=False)
            Wf = weight.float()
            U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
            r = max(1, min(rank, U.shape[1], Vh.shape[0]))
            Usi = (U[:, :r] * S[:r].unsqueeze(0)).to(weight.dtype)  # [out, r]
            Vi  = Vh[:r, :].to(weight.dtype)                        # [r, in]
            return nn.Parameter(Usi, requires_grad=False), nn.Parameter(Vi, requires_grad=False)

        # ---- copy norms & rope ----
        self.ln1        = dense_block.ln1
        self.ln2        = dense_block.ln2
        self.rotary_emb = dense_block.rotary_emb

        # ---- per-head SVD for q/k/v ----
        attn = dense_block
        Wq = attn.q_proj.weight.data    # [H*dh, D]
        Wk = attn.k_proj.weight.data    # [Hk*dh, D]
        Wv = attn.v_proj.weight.data    # [Hk*dh, D]

        self.q_Us, self.q_V = decompose_heads_svd(Wq, self.n_heads,    self.head_dim, rank_q)
        self.k_Us, self.k_V = decompose_heads_svd(Wk, self.n_kv_heads, self.head_dim, rank_kv)
        self.v_Us, self.v_V = decompose_heads_svd(Wv, self.n_kv_heads, self.head_dim, rank_kv)

        # ---- output & MLP (whole-matrix low-rank; optional) ----
        Wo = attn.o_proj.weight.data  # [D, H*dh]
        self.o_Us, self.o_V = decompose_full_svd(Wo, rank_o)

        Wg = dense_block.gate_proj.weight.data
        Wu = dense_block.up_proj.weight.data
        Wd = dense_block.down_proj.weight.data
        self.g_Us, self.g_V = decompose_full_svd(Wg, rank_ff)
        self.u_Us, self.u_V = decompose_full_svd(Wu, rank_ff)
        self.d_Us, self.d_V = decompose_full_svd(Wd, rank_ff)

    @torch.no_grad()
    def _proj_per_head(self, x: torch.Tensor, Us: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        x:  [B, M, D]
        V:  [H, r, D]   (right factor)
        Us: [H, dh, r]  (left factor with singulars)
        return: [B, M, H, dh]
        """
        if self.compute_in_float32:
            x32 = x.to(torch.float32)
            V32 = V.to(torch.float32)
            Us32 = Us.to(torch.float32)
            xr = torch.einsum('b m d, h r d -> b m h r', x32, V32)         # [B,M,H,r]
            out = torch.einsum('b m h r, h d r -> b m h d', xr, Us32)      # [B,M,H,dh]
            return out.to(x.dtype)
        else:
            xr = torch.einsum('b m d, h r d -> b m h r', x.to(V.dtype), V) # [B,M,H,r]
            out = torch.einsum('b m h r, h d r -> b m h d', xr, Us)        # [B,M,H,dh]
            return out

    def _rope_qk(self, q_bmhd, k_bmhd, *, position_ids=None, past_len: int = 0):
        q = q_bmhd.permute(0, 2, 1, 3).contiguous()  # [B,H,M,dh]
        k = k_bmhd.permute(0, 2, 1, 3).contiguous()  # [B,Hk,M,dh]
        if position_ids is None:
            B, H, M, _ = q.shape
            position_ids = torch.arange(past_len, past_len + M, device=q.device).view(1, M).expand(B, M)
        try:
            cos, sin = self.rotary_emb(q, position_ids)
        except TypeError:
            seq_len = int(position_ids.max().item()) + 1
            cos, sin = self.rotary_emb(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)
        return q.permute(0, 2, 1, 3).contiguous(), k.permute(0, 2, 1, 3).contiguous()

    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                position_ids=None, use_cache=False, **kwargs):
        # hidden_states: [B, M, D]
        B, M, D = hidden_states.shape
        x = self.ln1(hidden_states)

        # ---- per-head projections to [B,M,H,dh] ----
        q = self._proj_per_head(x, self.q_Us, self.q_V)            # [B,M,H, dh]
        k = self._proj_per_head(x, self.k_Us, self.k_V)            # [B,M,Hk,dh]
        v = self._proj_per_head(x, self.v_Us, self.v_V)            # [B,M,Hk,dh]

        # KV cache
        past_len = 0
        if past_key_value is not None and len(past_key_value) == 2:
            past_len = past_key_value[0].size(1)  # [B,K,Hk,dh]

        # RoPE
        q, k_now = self._rope_qk(q, k, position_ids=position_ids, past_len=past_len)

        # Append cache along seq
        if past_len > 0:
            k = torch.cat([past_key_value[0], k_now], dim=1)  # [B,K,Hk,dh]
            v = torch.cat([past_key_value[1], v],     dim=1)  # [B,K,Hk,dh]
        else:
            k = k_now
        present = (k, v) if use_cache else None

        # Repeat K/V heads for GQA -> [B,K,H,dh]
        if self.n_rep > 1:
            k_rep = k.unsqueeze(3).expand(B, k.size(1), self.n_kv_heads, self.n_rep, self.head_dim).reshape(B, k.size(1), self.n_heads, self.head_dim)
            v_rep = v.unsqueeze(3).expand(B, v.size(1), self.n_kv_heads, self.n_rep, self.head_dim).reshape(B, v.size(1), self.n_heads, self.head_dim)
        else:
            k_rep, v_rep = k, v  # [B,K,H,dh]

        # Prefer varlen FA
        if _can_use_fa(q) and _want_varlen():
            q_bshd = q.contiguous()            # [B,M,H,dh]
            k_bshd = k_rep.contiguous()        # [B,K,H,dh]
            v_bshd = v_rep.contiguous()        # [B,K,H,dh]

            # masks
            if attention_mask is None:
                q_mask = torch.ones(B, M, dtype=torch.bool, device=hidden_states.device)
                k_mask = torch.ones(B, k_bshd.size(1), dtype=torch.bool, device=hidden_states.device)
            else:
                km = attention_mask
                if km.size(-1) != k_bshd.size(1):
                    if km.size(-1) < k_bshd.size(1):
                        km = F.pad(km, (0, k_bshd.size(1) - km.size(-1)), value=0)
                    else:
                        km = km[:, -k_bshd.size(1):]
                km = km.to(dtype=torch.bool)
                k_mask = km
                q_mask = km[:, -M:]

            attn_v = _flash_attn_varlen(q_bshd, k_bshd, v_bshd, q_mask, k_mask, self.scale, causal=True)
            if attn_v is not None:
                attn = attn_v.contiguous().view(B, M, self.d_model)
                # Output projection via low-rank (whole-matrix)
                attn = (attn @ self.o_V.t()) @ self.o_Us.t()
                h = hidden_states + attn

                # MLP (whole-matrix low-rank)
                y = self.ln2(h)
                gate = F.silu((y @ self.g_V.t()) @ self.g_Us.t())
                up   =        (y @ self.u_V.t()) @ self.u_Us.t()
                ff   = ((gate * up) @ self.d_V.t()) @ self.d_Us.t()
                h = h + ff

                outputs = (h,)
                if use_cache:
                    outputs += (present,)
                return outputs
            # else fall through

        # Non-varlen FA or SDPA fallback
        if _can_use_fa(q):
            # q: [B,M,H,dh]; k_rep/v_rep: [B,K,H,dh]
            attn_out = _FA(
                q, k_rep, v_rep,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=True
            )  # [B,M,H,dh]
            attn = attn_out.contiguous().view(B, M, self.d_model)
            attn = (attn @ self.o_V.t()) @ self.o_Us.t()
        else:
            # SDPA expects [B,H,S,D]
            q_sdpa = q.permute(0, 2, 1, 3).contiguous()     # [B,H,M,dh]
            k_sdpa = k_rep.permute(0, 2, 1, 3).contiguous() # [B,H,K,dh]
            v_sdpa = v_rep.permute(0, 2, 1, 3).contiguous() # [B,H,K,dh]
            k_len = k_sdpa.size(-2)
            bias = _build_full_bias(attention_mask, B, M, k_len, hidden_states.device, q_sdpa.dtype)
            attn = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=bias, is_causal=False)
            attn = attn.permute(0, 2, 1, 3).contiguous().view(B, M, self.d_model)
            attn = (attn @ self.o_V.t()) @ self.o_Us.t()

        h = hidden_states + attn

        y = self.ln2(h)
        gate = F.silu((y @ self.g_V.t()) @ self.g_Us.t())
        up   =        (y @ self.u_V.t()) @ self.u_Us.t()
        ff   = ((gate * up) @ self.d_V.t()) @ self.d_Us.t()
        h = h + ff

        outputs = (h,)
        if use_cache:
            outputs += (present,)
        return outputs

# ─────────────────────────────────────────────────────────────────────────────
class LayerShim(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, past_key_value=None, attention_mask=None,
                position_ids=None, use_cache=False, output_attentions=False, **kwargs):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return self.block(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            position_ids=position_ids,
            use_cache=use_cache
        )

# ─────────────────────────────────────────────────────────────────────────────
def replace_with_custom_blocks(
    model: LlamaForCausalLM,
    *,
    impl: str = None,            # "svd" or "dense"
    rank_q: Optional[int] = None,
    rank_kv: Optional[int] = None,
    rank_o: Optional[int] = None,
    rank_ff: Optional[int] = None,
    compute_in_float32: bool = True,
) -> None:
    """
    Replaces every HF decoder layer with LayerShim(custom_block).
    impl="svd" -> LinearPerHeadSVDLlamaBlock
    impl="dense" -> LinearLlamaBlock
    """
    impl = (impl or os.getenv("BLOCK_IMPL", "svd")).lower()
    new_layers = []

    for li, hf_layer in enumerate(model.model.layers):
        # The HF layer already resides on the model's device
        dev = next(hf_layer.parameters()).device

        # Build dense snapshot and then the chosen custom block
        dense_block = LinearLlamaBlock(hf_layer, model.config)
        dense_block.to(dev)  # <-- ensure the just-created module lives on the same device

        if impl == "svd":
            block = LinearPerHeadSVDLlamaBlock(
                dense_block,
                rank_q=rank_q, rank_kv=rank_kv, rank_o=rank_o, rank_ff=rank_ff,
                compute_in_float32=compute_in_float32
            )
        elif impl == "dense":
            block = dense_block
        else:
            raise ValueError(f"Unknown BLOCK_IMPL={impl} (use 'svd' or 'dense').")

        block.to(dev)  # <-- CRITICAL: move SVD factors/params to GPU along with everything else
        new_layers.append(LayerShim(block))

    model.model.layers = nn.ModuleList(new_layers)
    model.config.use_cache = True

    # Safety: if the model was on GPU and we inserted CPU modules by mistake, this second .to(dev) fixes it.
    model.to(next(model.parameters()).device)

    print(f"[custom] swapped {len(new_layers)} layers with impl='{impl}', "
          f"FA={'on' if (_FA is not None) else 'off'}, varlen={'on' if (_FA_VARLEN or _FA_VARLEN_KVPACKED) else 'off'}")

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def perplexity_peak_time_kv_cache(mdl, loader, device):
    """
    Stable perplexity with KV cache + timing/memory. Avoid sabotaging kernels.
    """
    mdl.eval()
    total_loss, total_tokens = 0.0, 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    chunk_size = int(os.getenv("CHUNK_SIZE", "256"))
    using_fa = (os.getenv("USE_FLASH_ATTENTION", "1") != "0")
    default_safe = "0" if using_fa else "1"
    use_safer_sdpa = os.getenv("PPL_SAFE_SDPA", default_safe) == "1"
    max_eval_batches = int(os.getenv("MAX_EVAL_BATCHES", "0"))
    ppl_debug = os.getenv("PPL_DEBUG", "0") == "1"
    dropped_rows = 0

    batch_idx = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B, L = batch["input_ids"].shape

        past_kv = None
        prev_last_logits = None
        prev_last_mask   = None  # [B]

        for s in range(0, L, chunk_size):
            e = min(s + chunk_size, L)
            ids = batch["input_ids"][:, s:e]
            am  = batch["attention_mask"][:, :e]  # keys visible up to e

            if device == "cuda":
                cm = torch.backends.cuda.sdp_kernel
                if use_safer_sdpa:
                    ctx = cm(enable_flash=False, enable_mem_efficient=True, enable_math=True)
                else:
                    ctx = cm(enable_flash=True, enable_mem_efficient=True, enable_math=True)
            else:
                ctx = nullcontext()

            with ctx:
                out = mdl(input_ids=ids, attention_mask=am, past_key_values=past_kv, use_cache=True)

            logits = out.logits                        # [B, cur_len, V]
            past_kv = out.past_key_values
            cur_len = logits.size(1)

            # boundary loss
            if prev_last_logits is not None and prev_last_mask is not None and cur_len > 0:
                cur_first_mask = batch["attention_mask"][:, s].bool()     # [B]
                both_valid = (prev_last_mask & cur_first_mask)
                if both_valid.any():
                    v_logits = prev_last_logits[both_valid].float()
                    v_labels = batch["input_ids"][both_valid, s]
                    finite = torch.isfinite(v_logits).all(dim=-1)
                    if finite.any():
                        loss = F.cross_entropy(v_logits[finite], v_labels[finite], reduction="sum")
                        total_loss += loss.item()
                        total_tokens += int(finite.sum().item())
                    if ppl_debug:
                        dropped_rows += int((~finite).sum().item())

            # intra-chunk loss
            if cur_len > 1:
                intra_logits = logits[:, :-1, :].contiguous()
                intra_labels = batch["input_ids"][:, s+1:e].contiguous()
                intra_mask   = batch["attention_mask"][:, s+1:e].contiguous().bool()
                if intra_mask.any():
                    v_logits = intra_logits[intra_mask].float()
                    v_labels = intra_labels[intra_mask]
                    finite = torch.isfinite(v_logits).all(dim=-1)
                    if finite.any():
                        loss = F.cross_entropy(v_logits[finite], v_labels[finite], reduction="sum")
                        total_loss += loss.item()
                        total_tokens += int(finite.sum().item())
                    if ppl_debug:
                        dropped_rows += int((~finite).sum().item())

            last_mask = batch["attention_mask"][:, e-1].bool() if cur_len > 0 else torch.zeros(B, dtype=torch.bool, device=device)
            prev_last_logits = logits[:, -1, :].contiguous() if cur_len > 0 else None
            prev_last_mask   = last_mask if cur_len > 0 else None

        del past_kv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        batch_idx += 1
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    processed_batches = max(1, batch_idx)
    t = (time.perf_counter() - start) * 1000.0 / processed_batches
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    if ppl_debug:
        print(f"[ppl_debug] dropped rows (non‑finite logits): {dropped_rows}")

    if total_tokens == 0:
        return float('nan'), peak, t

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if math.isfinite(avg_loss) else float('nan')
    return ppl, peak, t

# ─────────────────────────────────────────────────────────────────────────────
def print_llama_summary(cfg, model_name: str, *, device: str, dtype: torch.dtype,
                        batch_size: int, seq_len: int, chunk_size: int,
                        rank_q: Optional[int], rank_kv: Optional[int], rank_o: Optional[int], rank_ff: Optional[int],
                        for_layer=None):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    n_kv = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    n_rep = cfg.num_attention_heads // n_kv

    print("\n================== LLaMA / Run Configuration ==================")
    print(f"Model:              {model_name}")
    print(f"Transformers:       {transformers.__version__}")
    print(f"Torch:              {torch.__version__}")
    print(f"Python:             {platform.python_version()}")
    print(f"Device / dtype:     {device} / {dtype}")
    print(f"Batch / Seq / Chunk:{batch_size} / {seq_len} / {chunk_size}")
    print("---------------------------------------------------------------")
    print(f"hidden_size:        {cfg.hidden_size}")
    print(f"n_layers:           {cfg.num_hidden_layers}")
    print(f"n_heads:            {cfg.num_attention_heads}")
    print(f"n_kv_heads:         {n_kv}  (GQA n_rep={n_rep})")
    print(f"head_dim:           {head_dim}")
    print(f"intermediate_size:  {cfg.intermediate_size}")
    print(f"vocab_size:         {cfg.vocab_size}")
    print(f"rope_theta:         {getattr(cfg, 'rope_theta', 'N/A')}")
    print(f"max_pos_emb:        {getattr(cfg, 'max_position_embeddings', 'N/A')}")
    print("---------------------------------------------------------------")
    print(f"SVD ranks:          q={rank_q}, kv={rank_kv}, o={rank_o}, ff={rank_ff}")
    print(f"flash_attn:         {(_FA is not None)}  varlen: {(_FA_VARLEN is not None or _FA_VARLEN_KVPACKED is not None)}  "
          f"(USE_FLASH_ATTENTION={os.getenv('USE_FLASH_ATTENTION','1')}, FA_USE_VARLEN={os.getenv('FA_USE_VARLEN','1')})")
    if for_layer is not None:
        attn0 = for_layer.self_attn
        mlp0  = for_layer.mlp
        print("Layer[0] shapes:")
        print(f"  q_proj: {tuple(attn0.q_proj.weight.shape)}  "
              f"k_proj: {tuple(attn0.k_proj.weight.shape)}  "
              f"v_proj: {tuple(attn0.v_proj.weight.shape)}  "
              f"o_proj: {tuple(attn0.o_proj.weight.shape)}")
        print(f"  gate/up/down: {tuple(mlp0.gate_proj.weight.shape)} / "
              f"{tuple(mlp0.up_proj.weight.shape)} / {tuple(mlp0.down_proj.weight.shape)}")
    print("===============================================================\n")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = os.getenv("DTYPE", "float16").lower()
    if dt not in ("float16", "bfloat16", "float32"):
        dt = "float16"
    dtype  = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dt]

    MODEL_NAME = os.getenv("LLAMA_MODEL", "meta-llama/Llama-2-7b-hf")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
    SEQ_LEN    = int(os.getenv("SEQ_LEN", "1024"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))

    # Load model on device
    svd_model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    svd_model.to(device)
    for p in svd_model.parameters():
        p.requires_grad = False

    cfg = svd_model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    def _env_rank(name: str, default: Optional[int]):
        v = os.getenv(name, "")
        if v.strip() == "" or v.strip().lower() in {"none", "null"}:
            return default
        try:
            iv = int(v)
        except ValueError:
            return default
        return None if iv <= 0 else iv

    RANK_Q  = _env_rank("RANK_Q", 128)
    RANK_KV = _env_rank("RANK_KV", 128)
    RANK_O  = _env_rank("RANK_O",  None)  # 0 or negative -> None (no compression)
    RANK_FF = _env_rank("RANK_FF", None)

    # Summary before swap
    if os.getenv("PRINT_SUMMARY", "1") == "1":
        print_llama_summary(cfg, MODEL_NAME, device=device, dtype=dtype,
                            batch_size=BATCH_SIZE, seq_len=SEQ_LEN, chunk_size=CHUNK_SIZE,
                            rank_q=RANK_Q, rank_kv=RANK_KV, rank_o=RANK_O, rank_ff=RANK_FF,
                            for_layer=svd_model.model.layers[0])

    # Swap HF layers for our custom blocks (FA is inside these blocks)
    compute_in_fp32 = (os.getenv("SVD_COMPUTE_FP32", "1") == "1")
    replace_with_custom_blocks(
        svd_model,
        impl=os.getenv("BLOCK_IMPL", "svd"),
        rank_q=RANK_Q, rank_kv=RANK_KV, rank_o=RANK_O, rank_ff=RANK_FF,
        compute_in_float32=compute_in_fp32
    )

    # 4) Data
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    max_eval_samples = int(os.getenv("MAX_EVAL_SAMPLES", "0"))
    def tokenize_fn(batch):
        return tok(batch["text"], padding="max_length", truncation=True, max_length=SEQ_LEN)
    ds = raw.select(range(min(max_eval_samples, len(raw)))) if max_eval_samples > 0 else raw
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]),
                              "attention_mask": torch.stack([x["attention_mask"] for x in b])}
    )

    # quick sanity: types of first layer
    print(type(svd_model.model.layers[0]))
    print(type(svd_model.model.layers[0].block))

    # 5) Measure
    torch.cuda.reset_peak_memory_stats()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ppl, peak_mem, time_ms = perplexity_peak_time_kv_cache(svd_model, loader, device)
    storage_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    print(f"{'Model':<15} | {'Storage (MiB)':<12} | {'Peak (MiB)':<10} | {'Transient (MiB)':<14} | {'Time (ms/b)':<10} | {'Perplexity':<10}")
    print("-" * 90)
    print(f"{'LLaMA custom':<15} | {storage_mem:<12.1f} | {peak_mem:<10.1f} | {peak_mem - storage_mem:<14.1f} | {time_ms:<10.1f} | {ppl:<10.4f}")
