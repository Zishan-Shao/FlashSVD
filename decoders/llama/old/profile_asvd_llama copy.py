import os, math, time, platform, inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
# import present but unused to keep compatibility if needed
# from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from typing import Optional
from contextlib import nullcontext

"""
profile_asvd_llama.py: SVD-based LLaMA block using plain PyTorch attention (no FlashAttention).

Key fixes in this version:
  • Manual RoPE that never creates 5-D tensors (always [B,H,T,Dh]).
  • ASVD uses an internal, stateful (Pk,Pv) cache per block; we do NOT return tuple caches to HF.
  • We do not try to force legacy tuple past_key_values when ASVD is enabled.
  • Evaluation uses use_cache=False by default.
  • Guard cache_position kwarg (older HF may not accept it).
  • Tokenizer is right-padded to align trimming with position_ids.

ASVD additions (decode path only):
  • ASVD=1 enables low-rank KV caching: cache Pk, Pv of shape [B,Hk,T,rank_kv] in SVD_DTYPE.
  • Reconstruct K,V on-the-fly: K = Pk @ (UkΣk)^T, V = Pv @ (UvΣv)^T.
  • Apply RoPE after reconstruction; GQA repeat; SDPA as usual.
  • Profiling reports measured ASVD KV MiB and an estimate that scales with rank_kv.

Usage:
  Evaluation:
    MODE=eval BATCH_SIZE=1 SEQ_LEN=512 python profile_asvd_llama.py

  Decoding (ASVD on):
    MODE=decode PROMPT_BATCH=16 MAX_GEN_TOKENS=128 ASVD=1 RANK_KV=32 SVD_DTYPE=bf16 DTYPE=float16 python profile_asvd_llama.py

Env toggles:
  FORCE_LEGACY_KV=1   # ignored when ASVD=1
  ASVD=1              # === ASVD === enable low-rank (Pk,Pv) KV-cache in decode mode
  SVD_DTYPE=fp32|bf16|fp16
  SVD_COMPUTE_FP32=1|0
  DEBUG_CACHE=1
  DEBUG_EVAL=1
  CHECK_SVD=1
"""

# ────────────────────── helpers ──────────────────────
def _repeat_kv(x, n_rep: int):
    # x: [B, Hk, T, Dh] -> [B, H, T, Dh]
    if n_rep == 1:
        return x
    B, Hk, T, Dh = x.shape
    return x[:, :, None].expand(B, Hk, n_rep, T, Dh).reshape(B, Hk * n_rep, T, Dh)

def _build_full_bias(attention_mask, batch_size, q_len, k_len, device, dtype):
    # Causal (with KV offset)
    i = torch.arange(q_len, device=device).view(q_len, 1)
    j = torch.arange(k_len, device=device).view(1, k_len)
    past_len = k_len - q_len
    causal = (j <= (past_len + i))
    causal_bias = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    causal_bias.masked_fill_(~causal, torch.finfo(dtype).min)
    causal_bias = causal_bias.view(1, 1, q_len, k_len)

    pad_bias = None
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            am = attention_mask
            if am.size(-1) < k_len:
                am = F.pad(am, (0, k_len - am.size(-1)), value=1)
            elif am.size(-1) > k_len:
                am = am[:, -k_len:]
            pad_bias = (1.0 - am.to(dtype=dtype)) * torch.finfo(dtype).min
            pad_bias = pad_bias.view(batch_size, 1, 1, k_len)
        elif attention_mask.dim() == 4:
            pad_bias = attention_mask.to(dtype=dtype, device=device)
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
    V  = Vh[:r, :]                               # [r, in]
    return Us, V

# ────────────────────── KV cache estimators/measurements ──────────────────────
def _estimate_kv_cache_mib(cfg, batch_size: int, seq_len: int, dtype: torch.dtype) -> float:
    """
    Estimate full KV-cache memory in MiB for full context length `seq_len`.
    total_bytes = 2 (K&V) * L * B * S * Hk * Dh * bytes_per_elem
    with GQA: Hk = num_key_value_heads, Dh = hidden_size / num_attention_heads
    """
    num_layers = int(getattr(cfg, 'num_hidden_layers'))
    hidden_size = int(getattr(cfg, 'hidden_size'))
    n_heads = int(getattr(cfg, 'num_attention_heads'))
    n_kv_heads = int(getattr(cfg, 'num_key_value_heads', n_heads))
    head_dim = hidden_size // n_heads
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total_bytes = 2 * num_layers * batch_size * seq_len * n_kv_heads * head_dim * bytes_per_elem
    return total_bytes / (1024.0 ** 2)

def _estimate_asvd_kv_cache_mib(cfg, batch_size: int, seq_len: int,
                                factor_dtype: torch.dtype, rank_kv: int) -> float:
    """
    === ASVD ===
    Cache stores only Pk,Pv of shape [B, Hk, S, r]; K,V reconstructed on the fly.
    total_bytes = 2 * L * B * S * Hk * r * bytes_per_elem(factor_dtype)
    """
    num_layers = int(getattr(cfg, 'num_hidden_layers'))
    n_heads = int(getattr(cfg, 'num_attention_heads'))
    n_kv_heads = int(getattr(cfg, 'num_key_value_heads', n_heads))
    bytes_per_elem = 2 if factor_dtype in (torch.float16, torch.bfloat16) else 4
    total_bytes = 2 * num_layers * batch_size * seq_len * n_kv_heads * int(rank_kv) * bytes_per_elem
    return total_bytes / (1024.0 ** 2)

def _bytes_of_present(past_key_values) -> float:
    """Best-effort bytes of KV cache (MiB) across HF variants (and ASVD tuples)."""
    if past_key_values is None:
        return 0.0
    visited: set[int] = set()
    def rec(obj) -> int:
        oid = id(obj)
        if oid in visited:
            return 0
        visited.add(oid)
        if torch.is_tensor(obj):
            try:
                return obj.numel() * obj.element_size()
            except Exception:
                return 0
        if isinstance(obj, (list, tuple, set)):
            return sum(rec(x) for x in obj)
        if isinstance(obj, dict):
            return sum(rec(v) for v in obj.values())
        total = 0
        for name in ("key_cache", "value_cache", "k_cache", "v_cache", "caches", "keys", "values"):
            if hasattr(obj, name):
                try:
                    total += rec(getattr(obj, name))
                except Exception:
                    pass
        if total == 0:
            try:
                for name in dir(obj):
                    if name.startswith("_"):
                        continue
                    if not any(s in name.lower() for s in ("cache", "key", "value", "k", "v")):
                        continue
                    try:
                        val = getattr(obj, name)
                    except Exception:
                        continue
                    if callable(val):
                        continue
                    total += rec(val)
            except Exception:
                pass
        return total
    bytes_total = rec(past_key_values)
    return bytes_total / (1024.0 ** 2)

def _measure_asvd_cache_mib(model) -> float:
    """
    Sum actual bytes of Pk/Pv across all SVDLlamaBlock layers when ASVD is enabled.
    """
    total_bytes = 0
    try:
        for lyr in getattr(model.model, "layers", []):
            blk = getattr(lyr, "block", None)
            if blk is None or not getattr(blk, "asvd_enabled", False):
                continue
            for t in (getattr(blk, "_asvd_pk", None), getattr(blk, "_asvd_pv", None)):
                if torch.is_tensor(t):
                    total_bytes += t.numel() * t.element_size()
    except Exception:
        pass
    return total_bytes / (1024.0 ** 2)

# ────────────────────── SVD block ──────────────────────
class SVDLlamaBlock(nn.Module):
    """
    LLaMA block:
      • per-head SVD for q/k/v (optional bypass)
      • optional low-rank o/ff
      • PyTorch SDPA (causal)
      • Plan B: native support for HF Cache objects (DynamicCache / StaticCache)
      • === ASVD === optional low-rank KV caching (cache Pk,Pv only)
    """
    def __init__(self, hf_layer: nn.Module, cfg,
                 rank_q: int,
                 rank_kv: int,
                 rank_o: Optional[int],
                 rank_ff: Optional[int],
                 factor_dtype: torch.dtype = torch.float32,
                 compute_in_fp32: bool = True):
        super().__init__()
        attn, mlp = hf_layer.self_attn, hf_layer.mlp
        self.d_model    = int(cfg.hidden_size)
        self.n_heads    = int(cfg.num_attention_heads)
        self.n_kv_heads = int(getattr(cfg, "num_key_value_heads", self.n_heads))
        self.head_dim   = self.d_model // self.n_heads
        self.n_rep      = self.n_heads // self.n_kv_heads
        self.compute_in_fp32 = bool(compute_in_fp32)
        self.factor_dtype = factor_dtype
        self.layer_idx = getattr(getattr(hf_layer, "self_attn", None), "layer_idx", None)
        self.debug_cache = os.getenv("DEBUG_CACHE", "0") == "1"
        self.asvd_enabled = os.getenv("ASVD", "0") == "1"   # === ASVD ===

        # ASVD internal stateful cache (Pk,Pv sequences)
        self._asvd_pk = None  # [B,Hk,K,r]
        self._asvd_pv = None  # [B,Hk,K,r]

        # Norms & RoPE
        self.ln1 = hf_layer.input_layernorm
        self.ln2 = hf_layer.post_attention_layernorm
        self.rotary_emb = getattr(attn, "rotary_emb", None)
        if self.rotary_emb is None:
            rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
            class _SimpleRoPE(nn.Module):
                def __init__(self, head_dim: int, base: float = 10000.0):
                    super().__init__()
                    self.head_dim = head_dim
                    evens = torch.arange(0, head_dim, 2, dtype=torch.float32)
                    self.register_buffer("inv_freq", 1.0 / (base ** (evens / head_dim)), persistent=False)
                def forward(self, x, seq_len: int = None, position_ids: Optional[torch.LongTensor] = None):
                    device = x.device
                    Dh = self.head_dim
                    inv = self.inv_freq.to(device=device)
                    if seq_len is None:
                        if position_ids is not None:
                            seq_len = int(position_ids.max().item()) + 1
                        else:
                            seq_len = x.size(-2)
                    t = torch.arange(seq_len, device=device, dtype=torch.float32)        # [T]
                    ang = t[:, None] * inv[None, :]                                      # [T, Dh/2]
                    ang = ang.repeat_interleave(2, dim=-1)                               # [T, Dh]
                    cos = ang.cos()[None, None, :, :].to(x.dtype)                        # [1,1,T,Dh]
                    sin = ang.sin()[None, None, :, :].to(x.dtype)                        # [1,1,T,Dh]
                    return cos, sin
            self.rotary_emb = _SimpleRoPE(self.head_dim, base=rope_theta)

        # --- per-head SVD for Q/K/V ---
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
        if os.getenv("CHECK_SVD", "0") == "1":
            with torch.no_grad():
                def _max_err(W, H, dh, Us, V):
                    Wv = W.float().view(H, dh, -1)
                    R  = torch.einsum('hdr,hrD->hdD', Us.float(), V.float())
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

        # Explicitly clear heavy dense modules from the original HF layer to avoid GPU retention
        try:
            del attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
        except Exception:
            pass
        try:
            del mlp.gate_proj, mlp.up_proj, mlp.down_proj
        except Exception:
            pass

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

    @torch.no_grad()
    def _proj_per_head_lowrank(self, x: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        === ASVD ===
        Return only low-rank projections P = X·V : [B, H, T, r]
        Stored in factor_dtype to minimize cache bytes.
        """
        if self.compute_in_fp32:
            P = torch.einsum('b t d, h r d -> b t h r', x.float(), V.float())  # [B,T,H,r]
        else:
            P = torch.einsum('b t d, h r d -> b t h r', x.to(V.dtype), V)      # [B,T,H,r]
        return P.transpose(1, 2).to(self.factor_dtype).contiguous()            # [B,H,T,r]

    @torch.no_grad()
    def _reconstruct_from_P(self, P: torch.Tensor, Us: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """
        === ASVD ===
        P:  [B, H, T, r]; Us: [H, dh, r]
        return [B, H, T, dh] in out_dtype
        """
        if self.compute_in_fp32:
            X = torch.einsum('b h t r, h d r -> b h t d', P.float(), Us.float())
            return X.to(out_dtype).contiguous()
        X = torch.einsum('b h t r, h d r -> b h t d', P.to(Us.dtype), Us)
        return X.to(out_dtype).contiguous()

    # ─────────────── Manual RoPE (shape-safe: keep [B,H,T,Dh]) ───────────────
    def _manual_rope(self, x_bhtd: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        x_bhtd: [B,H,T,Dh]
        cos/sin: [B,1,T,Dh] (broadcast over H)
        """
        B, H, T, Dh = x_bhtd.shape
        x_even = x_bhtd[..., 0::2]   # [B,H,T,Dh/2]
        x_odd  = x_bhtd[..., 1::2]   # [B,H,T,Dh/2]
        # rotate_half(x) = [-x_odd, x_even]
        x_rot  = torch.stack((-x_odd, x_even), dim=-1).reshape(B, H, T, Dh)
        return (x_bhtd * cos + x_rot * sin).contiguous()

    @torch.no_grad()
    def _apply_rope_single(self, x_bhtd, position_ids=None, seq_len=None):
        """
        Apply RoPE to a single tensor [B,H,T,Dh] with its own length/positions, shape-safe.
        We ensure cos/sin are created with enough rows to cover max(position_ids).
        """
        B, H, T, Dh = x_bhtd.shape

        # Determine how many positions we need cos/sin for
        target_len = T
        if seq_len is not None:
            target_len = max(target_len, int(seq_len))

        if position_ids is not None:
            if position_ids.dtype != torch.long:
                position_ids = position_ids.to(dtype=torch.long)
            # guard against negatives (shouldn't happen, but cheap)
            if (position_ids < 0).any():
                position_ids = position_ids.clamp_min_(0)
            pos_max = int(position_ids.max().item())
            target_len = max(target_len, pos_max + 1)  # ensure gather is always in-bounds

        # Build cos/sin for target_len (never pass position_ids to rotary_emb)
        try:
            sig = inspect.signature(self.rotary_emb.forward)
            if "seq_len" in sig.parameters:
                cos, sin = self.rotary_emb(x_bhtd, seq_len=target_len)
            else:
                cos, sin = self.rotary_emb(x_bhtd, target_len)
        except TypeError:
            try:
                cos, sin = self.rotary_emb(x_bhtd, seq_len=target_len)
            except TypeError:
                cos, sin = self.rotary_emb(x_bhtd, target_len)

        # Select rows
        if position_ids is not None:
            # Expand to [B,1,target_len,Dh] and gather the exact positions in position_ids
            cos_b = cos.expand(B, 1, cos.size(-2), cos.size(-1))
            sin_b = sin.expand(B, 1, sin.size(-2), sin.size(-1))
            # Sanity: position_ids is guaranteed < target_len because of target_len logic above
            idx   = position_ids.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, Dh)  # [B,1,T,Dh]
            cos   = torch.gather(cos_b, -2, idx)
            sin   = torch.gather(sin_b, -2, idx)
        else:
            # Trim to current T and broadcast over heads
            cos = cos[..., :T, :].expand(B, 1, T, Dh)
            sin = sin[..., :T, :].expand(B, 1, T, Dh)

        cos = cos.to(dtype=x_bhtd.dtype, device=x_bhtd.device)
        sin = sin.to(dtype=x_bhtd.dtype, device=x_bhtd.device)

        # Manual rotate_half (keeps tensors 4-D)
        x_even = x_bhtd[..., 0::2]   # [B,H,T,Dh/2]
        x_odd  = x_bhtd[..., 1::2]   # [B,H,T,Dh/2]
        x_rot  = torch.stack((-x_odd, x_even), dim=-1).reshape(B, H, T, Dh)

        return (x_bhtd * cos + x_rot * sin).contiguous()


    def _apply_rope(self, q_bhtd, k_bhtd, position_ids, position_embeddings=None):
        """
        Apply RoPE on tensors laid out as [B,H,T,Dh] (shape-safe).
        """
        q_rot = self._apply_rope_single(q_bhtd, position_ids=position_ids, seq_len=q_bhtd.size(-2))
        k_rot = self._apply_rope_single(k_bhtd, position_ids=position_ids, seq_len=k_bhtd.size(-2))
        return q_rot, k_rot

    # ────────────────────────── forward ──────────────────────────
    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                position_ids=None, use_cache=False, position_embeddings=None, **kw):
        """
        hidden_states: [B, T, D]
        attention_mask:
          - None
          - [B, T] (1=keep, 0=pad)
          - [B, 1, Q, K] additive mask
        position_ids: [B, T]    (absolute positions)
        """
        B, T, D = hidden_states.shape
        x = self.ln1(hidden_states)

        # Normalize mask to [B,T] keep-mask and trim to T_max
        if attention_mask is None:
            T_max = T
            x_trim = x
            pos_ids = position_ids
        else:
            if use_cache:
                # Decode path: use last-T window semantics
                if attention_mask.dim() == 2:                       # [B,T_total]
                    keep_t = (attention_mask[:, -T:] > 0)
                elif attention_mask.dim() == 4:                     # [B,1,Q,K]
                    m = attention_mask.to(torch.float32).squeeze(1) # [B,Q,K]
                    keep_t = (m > -1e3).any(dim=1)[:, -T:]
                else:
                    keep_t = attention_mask.reshape(B, -1)[:, -T:].to(torch.bool)
            else:
                # Eval (no-cache): prefix semantics like profile_svd_llama.py
                if attention_mask.dim() == 2:                       # [B,T]
                    keep_t = (attention_mask[:, :T] > 0)
                elif attention_mask.dim() == 4:                     # [B,1,Q,K]
                    m = attention_mask.to(torch.float32).squeeze(1) # [B,Q,K]
                    keep_t = (m > -1e3).any(dim=1)[:, :T]
                else:
                    keep_t = attention_mask.reshape(B, -1)[:, :T].to(torch.bool)
            T_max = int(keep_t.sum(dim=1).max().item())
            x_trim = x[:, :T_max, :]
            pos_ids = position_ids[:, :T_max] if position_ids is not None else None

        # Q via SVD factors (full, not low-rank)
        q = self._proj_per_head(x_trim, self.q_Us, self.q_V)         # [B,H,T,dh] (T=Q)

        if not use_cache:
            # ---------- Evaluation path ----------
            k_raw = self._proj_per_head(x_trim, self.k_Us, self.k_V)     # [B,Hk,T,dh]
            v_raw = self._proj_per_head(x_trim, self.v_Us, self.v_V)
            k = _repeat_kv(k_raw, self.n_rep)                             # [B,H,T,dh]
            v = _repeat_kv(v_raw, self.n_rep)
            q, k = self._apply_rope(q, k, pos_ids, position_embeddings=position_embeddings)  # [B,H,T,dh]

            q_bhsd = q.contiguous()
            k_bhsd = k.contiguous()
            v_bhsd = v.contiguous()

            bias = _build_full_bias(attention_mask, B, T_max, k_bhsd.size(-2), q_bhsd.device, q_bhsd.dtype)
            attn_sdpa = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, attn_mask=bias, is_causal=False)

            present_out = None  # use_cache=False path

        else:
            # ---------- Decode path ----------
            if self.asvd_enabled:
                # === ASVD === cache low-rank Pk,Pv; reconstruct K,V on the fly (stateful)
                # Heuristic: reset internal ASVD cache at start of a new sequence (prefill or explicit reset).
                is_first_chunk = (self._asvd_pk is None or (position_ids is not None and int(position_ids.min().item()) == 0 and x_trim.size(1) > 1))
                if is_first_chunk:
                    self._asvd_pk = None
                    self._asvd_pv = None

                # 1) Low-rank projections for the current chunk
                Pk_new = self._proj_per_head_lowrank(x_trim, self.k_V)   # [B,Hk,T, r]
                Pv_new = self._proj_per_head_lowrank(x_trim, self.v_V)   # [B,Hk,T, r]

                # 2) Accumulate into the internal ASVD cache
                if self._asvd_pk is None:
                    self._asvd_pk, self._asvd_pv = Pk_new, Pv_new
                else:
                    self._asvd_pk = torch.cat([self._asvd_pk, Pk_new], dim=2)
                    self._asvd_pv = torch.cat([self._asvd_pv, Pv_new], dim=2)

                pk_seq, pv_seq = self._asvd_pk, self._asvd_pv  # [B,Hk,K,r]

                # 3) Reconstruct K,V for the full sequence so far
                k_seq = self._reconstruct_from_P(pk_seq, self.k_Us, out_dtype=q.dtype)  # [B,Hk,K,dh]
                v_seq = self._reconstruct_from_P(pv_seq, self.v_Us, out_dtype=q.dtype)

                # 4) RoPE: q length = T_max; k length = total K
                q_bhtd = self._apply_rope_single(q, position_ids=pos_ids)              # [B,H,T,dh]
                k_len = k_seq.size(-2)
                k_bhtd = self._apply_rope_single(k_seq, position_ids=None, seq_len=k_len)

                # 5) GQA repeat and SDPA
                k_bhsd = _repeat_kv(k_bhtd, self.n_rep)         # [B,H,K,dh]
                v_bhsd = _repeat_kv(v_seq,  self.n_rep)         # [B,H,K,dh]
                q_bhsd = q_bhtd.contiguous()                    # [B,H,T,dh]

                # Mask for (Q=T_max, K=k_len)
                if attention_mask is not None and attention_mask.dim() == 4:
                    bias = attention_mask.to(dtype=q_bhsd.dtype, device=q_bhsd.device)
                    bias = bias[..., -T_max:, -k_len:]
                    if (not bias.is_contiguous()) or (getattr(bias, "storage_offset", lambda: 0)() != 0) or ((bias.data_ptr() % 16) != 0):
                        bias = bias.contiguous().clone()
                else:
                    bias = _build_full_bias(attention_mask, B, T_max, k_len, q_bhsd.device, q_bhsd.dtype)

                if self.debug_cache and (self.layer_idx in (None, 0)):
                    try:
                        print(f"[ASVD] layer_idx={self.layer_idx} Pk={tuple(pk_seq.shape)} Pv={tuple(pv_seq.shape)} K={k_len}")
                        print(f"[ASVD] q={tuple(q_bhtd.shape)} k={tuple(k_bhtd.shape)} v={tuple(v_seq.shape)} T_max={T_max}")
                    except Exception:
                        pass

                # Manual SDPA (stable, explicit shapes): [B,H,T,dh]
                scale = 1.0 / math.sqrt(self.head_dim)
                qf = q_bhsd.float()
                kf = k_bhsd.float()
                vf = v_bhsd.float()
                logits = torch.einsum('b h t d, b h k d -> b h t k', qf, kf) * scale
                if bias is not None:
                    logits = logits + bias  # bias broadcast: [B,1,T,K] -> [B,H,T,K]
                weights = torch.softmax(logits, dim=-1)
                attn_sdpa = torch.einsum('b h t k, b h k d -> b h t d', weights, vf).to(q_bhsd.dtype)
                if self.debug_cache and (self.layer_idx in (None, 0)):
                    try:
                        print(f"[ASVD] attn_out={tuple(attn_sdpa.shape)}")
                    except Exception:
                        pass

                # IMPORTANT: return a HF-compatible "present" (Cache or None), never (Pk,Pv) tuples
                present_out = past_key_value if hasattr(past_key_value, "update") else None

            else:
                # ---------- Original (non-ASVD) KV cache path ----------
                k_raw = self._proj_per_head(x_trim, self.k_Us, self.k_V)     # [B,Hk,T,dh]
                v_raw = self._proj_per_head(x_trim, self.v_Us, self.v_V)     # [B,Hk,T,dh]

                # RoPE on Q and K (before GQA repeat)
                q, k_rot = self._apply_rope(q, k_raw, pos_ids, position_embeddings=position_embeddings)  # q:[B,H,T,dh], k_rot:[B,Hk,T,dh]

                present_out = None
                cache_position = kw.get("cache_position", None)

                if hasattr(past_key_value, "update"):
                    li = self.layer_idx if self.layer_idx is not None else kw.get("layer_idx", None)
                    try:
                        k_seq, v_seq = past_key_value.update(
                            k_rot, v_raw, layer_idx=li, cache_position=cache_position
                        )
                    except TypeError:
                        try:
                            k_seq, v_seq = past_key_value.update(k_rot, v_raw, li, cache_position)
                        except TypeError:
                            k_seq, v_seq = past_key_value.update(k_rot, v_raw, li)
                    present_out = past_key_value
                else:
                    if isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                        k_seq = torch.cat([past_key_value[0], k_rot], dim=-2)  # [B,Hk,K,dh]
                        v_seq = torch.cat([past_key_value[1], v_raw], dim=-2)
                        present_out = (k_seq, v_seq)
                    else:
                        k_seq, v_seq = k_rot, v_raw
                        present_out = (k_seq, v_seq) if not hasattr(past_key_value, "update") else None

                if self.debug_cache and (self.layer_idx in (None, 0)):
                    k_total = k_seq.size(-2)
                    pkv_t = type(past_key_value).__name__ if past_key_value is not None else "None"
                    print(f"[DEBUG] layer_idx={self.layer_idx} pkv={pkv_t} q_len={q.size(-2)} k_total={k_total}")

                k_bhsd = _repeat_kv(k_seq, self.n_rep)
                v_bhsd = _repeat_kv(v_seq, self.n_rep)
                q_bhsd = q.contiguous()

                k_len = k_bhsd.size(-2)
                if attention_mask is not None and attention_mask.dim() == 4:
                    bias = attention_mask.to(dtype=q_bhsd.dtype, device=q_bhsd.device)
                    bias = bias[..., -T_max:, -k_len:]
                    if (not bias.is_contiguous()) or (getattr(bias, "storage_offset", lambda: 0)() != 0) or ((bias.data_ptr() % 16) != 0):
                        bias = bias.contiguous().clone()
                else:
                    bias = _build_full_bias(attention_mask, B, T_max, k_len, q_bhsd.device, q_bhsd.dtype)

                if os.getenv("DEBUG_CACHE", "0") == "1" and (self.layer_idx in (None, 0)):
                    try:
                        print(f"[DEBUG] mask.shape={tuple(bias.shape)} contig={bias.is_contiguous()} ptr%16={bias.data_ptr()%16}")
                    except Exception:
                        pass

                use_bias = bias is not None
                ctx = (torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
                       if (use_bias and q_bhsd.is_cuda) else nullcontext())
                with ctx:
                    attn_sdpa = F.scaled_dot_product_attention(
                        q_bhsd, k_bhsd, v_bhsd, attn_mask=bias, is_causal=False
                    )

        # Flatten heads to [B,T,D]
        attn = attn_sdpa.transpose(1, 2).contiguous().view(B, T_max, self.n_heads * self.head_dim)

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

        # pad back to original T for API compatibility
        if T_max < T:
            pad = torch.zeros(B, T - T_max, D, dtype=out.dtype, device=out.device)
            out = torch.cat([out, pad], dim=1)
        if use_cache:
            return (out, present_out)
        return (out,)

# ────────────────────── wire into HF model ──────────────────────
def replace_with_svd(model, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32):
    cfg = model.config
    dev = next(model.parameters()).device
    new_layers = nn.ModuleList()
    for layer in model.model.layers:
        shim = _wrap_svd_layer(layer, cfg, rank_q, rank_kv, rank_o, rank_ff,
                               factor_dtype, compute_in_fp32)
        shim.to(dev)
        new_layers.append(shim)
    model.model.layers = new_layers

def _wrap_svd_layer(hf_layer, cfg, rank_q, rank_kv, rank_o, rank_ff,
                    factor_dtype, compute_in_fp32):
    class _Shim(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.layer_idx = getattr(getattr(inner, "self_attn", None), "layer_idx", None)
            self.block = SVDLlamaBlock(
                inner, cfg,
                rank_q=rank_q, rank_kv=rank_kv, rank_o=rank_o, rank_ff=rank_ff,
                factor_dtype=factor_dtype, compute_in_fp32=compute_in_fp32
            )
        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, use_cache=False, **kw):
            # Ensure layer_idx is available inside the block (for cache.update)
            if "layer_idx" not in kw and self.layer_idx is not None:
                kw = dict(kw)
                kw["layer_idx"] = self.layer_idx
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            outputs = self.block(hidden_states,
                                 attention_mask=attention_mask,
                                 past_key_value=past_key_value,
                                 position_ids=position_ids,
                                 use_cache=use_cache,
                                 **kw)
            if use_cache:
                y, present = outputs
                return (y, present)
            else:
                y, = outputs
                return (y,)
    return _Shim(hf_layer)

# ────────────────────── quick eval (full-seq) ──────────────────────
@torch.no_grad()
def eval_perplexity_fullseq(model, loader, device):
    model.eval()
    total_loss, total_tok = 0.0, 0
    debug_eval = os.getenv("DEBUG_EVAL", "0") == "1"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        B, T = batch["input_ids"].shape
        pos = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)

        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=pos,
                    use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()
        labels = batch["input_ids"][:, 1:].contiguous()
        mask   = batch["attention_mask"][:, 1:].contiguous().bool()

        if debug_eval and i == 0:
            tot = int(mask.sum().item())
            finite_token_mask = torch.isfinite(logits).all(dim=-1)
            finite_tok = int((finite_token_mask & mask).sum().item())
            any_nan = (~torch.isfinite(logits)).any().item()
            max_logit = float(torch.nan_to_num(logits).max().item())
            min_logit = float(torch.nan_to_num(logits).min().item())
            print(f"[DEBUG_EVAL] tokens={tot} finite_tokens={finite_tok} any_nan={any_nan} logits_range=[{min_logit:.2f},{max_logit:.2f}]")

        if mask.any():
            v_logits = logits[mask].float()
            v_labels = labels[mask]
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

# ────────────────────── small utils ──────────────────────
def _supports_kwarg(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False

# ────────────────────── generation (prefill + decode) ──────────────────────
@torch.no_grad()
def profile_prefill_and_decode(model, tok: AutoTokenizer, device: str,
                               prompt_path: str,
                               prompt_len: int = 256,
                               decode_tokens: int = 64,
                               batch_size: int = 1,
                               rank_kv: int = 128):
    asvd_enable = os.getenv("ASVD", "0") == "1"                  # === ASVD ===
    force_legacy_kv = False if asvd_enable else (os.getenv("FORCE_LEGACY_KV", "1") == "1")
    # dtype used to store Pk,Pv in ASVD mode
    svd_dtype_env = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[os.getenv("SVD_DTYPE", "fp32").lower()]

    # Generate or load prompt
    if not os.path.exists(prompt_path):
        text = ("LLaMA is a family of large language models developed by Meta. "
                "This script profiles SVD blocks using a synthetic prompt. ") * 8
        text = " ".join(text.split())
        with open(prompt_path, 'w') as f:
            f.write(text)
    else:
        with open(prompt_path, 'r') as f:
            text = f.read()

    enc = tok(text, return_tensors='pt', padding='max_length', truncation=True, max_length=prompt_len)
    input_ids = enc['input_ids'].to(device).repeat(batch_size, 1)
    attn_mask = enc['attention_mask'].to(device).repeat(batch_size, 1)

    # Position ids for prefill: 0..T-1 (trim to valid tokens inside the block)
    pos_ids_full = torch.arange(input_ids.size(1), device=device).unsqueeze(0).repeat(batch_size, 1)

    # Prefill: run once over full prompt, build KV cache (ASVD internal or HF)
    model.config.use_cache = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    storage_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    # For ASVD, do NOT pass legacy tuples; let blocks accumulate internally
    init_past = None if asvd_enable else ([None] * model.config.num_hidden_layers if force_legacy_kv else None)

    # Guard cache_position kwarg (older HF forward doesn't accept it)
    supports_cache_pos = _supports_kwarg(model.forward, "cache_position")

    t0 = time.perf_counter()
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
        position_ids=pos_ids_full,
        use_cache=True,
    )
    if init_past is not None:
        kwargs["past_key_values"] = init_past
    if supports_cache_pos:
        kwargs["cache_position"] = pos_ids_full[:, -1]

    out = model(**kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    prefill_peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    prefill_current = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    present = out.past_key_values
    # HF measured KV (may be 0 with ASVD)
    prefill_kv_bytes_hf = _bytes_of_present(present)
    # ASVD measured KV (Pk,Pv inside blocks)
    prefill_kv_bytes_asvd = _measure_asvd_cache_mib(model) if asvd_enable else prefill_kv_bytes_hf

    # Decode: token-by-token for decode_tokens
    running_attn = attn_mask.clone()
    next_ids = input_ids[:, -1:]                 # start from the last prompt token
    decode_start_alloc = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_dec = 0.0
    decode_kv_bytes_asvd = prefill_kv_bytes_asvd

    for step in range(decode_tokens):
        t1 = time.perf_counter()
        running_attn = torch.cat([running_attn, torch.ones(running_attn.size(0), 1, dtype=running_attn.dtype, device=running_attn.device)], dim=1)
        pos_next = (running_attn.long().sum(dim=1, keepdim=True) - 1)  # [B,1]

        kwargs = dict(
            input_ids=next_ids,
            attention_mask=running_attn,
            position_ids=pos_next,
            past_key_values=present,     # remains None for ASVD; Cache for non-ASVD
            use_cache=True,
        )
        if supports_cache_pos:
            kwargs["cache_position"] = pos_next.squeeze(1)

        out = model(**kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_dec += (time.perf_counter() - t1)
        present = out.past_key_values  # will be None in ASVD; Cache for non-ASVD
        # Measure ASVD cache directly
        decode_kv_bytes_asvd = _measure_asvd_cache_mib(model) if asvd_enable else _bytes_of_present(present)
        next_ids = out.logits[:, -1:, :].argmax(dim=-1)

    decode_ms = t_dec * 1000.0 / max(1, decode_tokens)
    decode_peak_abs = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    decode_peak_delta = max(0.0, decode_peak_abs - decode_start_alloc)

    storage_after = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    # Choose estimate based on ASVD or full KV. For ASVD, we count bytes using SVD_DTYPE and rank_kv.
    if asvd_enable:
        final_kv_est = _estimate_asvd_kv_cache_mib(model.config, batch_size=batch_size,
                                                   seq_len=prompt_len + decode_tokens,
                                                   factor_dtype=svd_dtype_env, rank_kv=rank_kv)
    else:
        final_kv_est = _estimate_kv_cache_mib(model.config, batch_size=batch_size,
                                              seq_len=prompt_len + decode_tokens,
                                              dtype=next(model.parameters()).dtype)

    return {
        'storage_mib': storage_before,
        'prefill_ms': prefill_ms,
        'prefill_peak_mib': prefill_peak,
        'prefill_current_mib': prefill_current,
        'decode_avg_ms_per_tok': decode_ms,
        'decode_peak_abs_mib': decode_peak_abs,
        'decode_peak_delta_mib': decode_peak_delta,
        'storage_after_mib': storage_after,
        'final_kv_est_mib': final_kv_est,
        'prefill_kv_measured_mib': prefill_kv_bytes_asvd,  # ASVD-aware measure
        'decode_kv_measured_mib': decode_kv_bytes_asvd,    # ASVD-aware measure
        'asvd_enabled': asvd_enable,
    }

# ────────────────────── main ──────────────────────
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Env knobs
    dt = os.getenv("DTYPE", "float16").lower()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dt]
    MODEL_NAME = os.getenv("LLAMA_MODEL", "meta-llama/Llama-2-7b-hf")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
    MODE = os.getenv("MODE", "eval").lower()  # "eval" or "decode"
    MAX_GEN_TOKENS = int(os.getenv("MAX_GEN_TOKENS", "64"))
    PROMPT_BATCH = int(os.getenv("PROMPT_BATCH", str(BATCH_SIZE)))
    SEQ_LEN    = int(os.getenv("SEQ_LEN", "512"))
    MAX_EVAL_SAMPLES = int(os.getenv("MAX_EVAL_SAMPLES", "64"))
    # SVD ranks (per-head for Q/K/V, whole-matrix for O/FF)
    RANK_Q  = int(os.getenv("RANK_Q",  "128"))
    RANK_KV = int(os.getenv("RANK_KV", "128"))
    RANK_O  = int(os.getenv("RANK_O",  "0")) or None          # 0 → dense
    RANK_FF = int(os.getenv("RANK_FF", "0")) or None          # 0 → dense
    SVD_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[os.getenv("SVD_DTYPE", "fp32").lower()]
    SVD_COMPUTE_FP32 = os.getenv("SVD_COMPUTE_FP32", "1") == "1"
    ASVD_ENABLE = os.getenv("ASVD", "0") == "1"               # === ASVD ===

    # Load model/tokenizer
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.config.use_cache = False  # full-seq for eval mode unless we override in decode

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Swap in SVD layers (plain PyTorch SDPA) and free any dense artifacts
    replace_with_svd(
        model, rank_q=RANK_Q, rank_kv=RANK_KV, rank_o=RANK_O, rank_ff=RANK_FF,
        factor_dtype=SVD_DTYPE, compute_in_fp32=SVD_COMPUTE_FP32
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optional debug: verify SVD block is active
    if os.getenv("DEBUG_CACHE", "0") == "1":
        try:
            print(type(model.model.layers[0]))
            blk = getattr(model.model.layers[0], 'block', None)
            if blk is not None:
                print(type(blk))
                print(f"[DEBUG] ASVD enabled: {blk.asvd_enabled}")
        except Exception as _e:
            print(f"[debug] layer type check failed: {_e}")

    if MODE == "eval":
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

        # Run quick eval with memory/reset & KV estimate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        storage_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        kv_cache_est_mib = _estimate_kv_cache_mib(model.config, BATCH_SIZE, SEQ_LEN, dtype)
        ppl, peak_mem, time_ms = eval_perplexity_fullseq(model, loader, device)
        transient_mem = max(0.0, peak_mem - storage_mem)

        print("\n================== LLaMA + SVD (SDPA / Eval) ==================")
        print(f"Python {platform.python_version()}  Torch {torch.__version__}")
        print(f"Device/dtype: {device}/{dtype}")
        print(f"Ranks: q={RANK_Q}, kv={RANK_KV}, o={RANK_O}, ff={RANK_FF}  |  SVD store dtype={SVD_DTYPE}  |  SVD compute fp32={SVD_COMPUTE_FP32}")
        print(f"{'Model':<20} | {'Storage (MiB)':<14} | {'Peak (MiB)':<10} | {'Transient (MiB)':<14} | {'KV est (MiB)':<12} | {'Time (ms/b)':<12} | {'Perplexity':<10}")
        print("-" * 108)
        print(f"{'LLaMA+SVD':<20} | {storage_mem:<14.1f} | {peak_mem:<10.1f} | {transient_mem:<14.1f} | {kv_cache_est_mib:<12.1f} | {time_ms:<12.1f} | {ppl:<10.4f}")

    elif MODE == "decode":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        prompt_file = os.path.join(os.path.dirname(__file__) if '__file__' in globals() else '.', 'synthetic_prompt_256.txt')
        gen_stats = profile_prefill_and_decode(
            model, tok, device,
            prompt_path=prompt_file,
            prompt_len=256,
            decode_tokens=MAX_GEN_TOKENS,
            batch_size=PROMPT_BATCH,
            rank_kv=RANK_KV,  # === ASVD === for estimator
        )
        kv_label = "KV est (MiB)"  # label is the same; value reflects ASVD or full depending on toggle
        print("\n================== LLaMA + SVD (Decode Mode) ==================")
        print(f"Python {platform.python_version()}  Torch {torch.__version__}")
        print(f"Device/dtype: {device}/{dtype}")
        print(f"ASVD={ASVD_ENABLE} | Ranks: q={RANK_Q}, kv={RANK_KV}, o={RANK_O}, ff={RANK_FF}  |  SVD store dtype={SVD_DTYPE}  |  SVD compute fp32={SVD_COMPUTE_FP32}")
        print(f"Prompt batch: {PROMPT_BATCH} | Prompt len: 256 | Max gen: {MAX_GEN_TOKENS}")
        print(f"{'Storage (MiB)':<16} | {'Prefill (ms)':<12} | {'Prefill Peak (MiB)':<18} | {'Prefill Current (MiB)':<20} | {'Prefill KV (MiB)':<16} | {'Decode ms/tok':<14} | {'Decode Peak Abs (MiB)':<20} | {'Decode Peak Δ (MiB)':<18} | {'Decode KV (MiB)':<16} | {kv_label:<18}")
        print("-" * 200)
        print(f"{gen_stats['storage_mib']:<16.1f} | {gen_stats['prefill_ms']:<12.1f} | {gen_stats['prefill_peak_mib']:<18.1f} | {gen_stats['prefill_current_mib']:<20.1f} | {gen_stats['prefill_kv_measured_mib']:<16.1f} | {gen_stats['decode_avg_ms_per_tok']:<14.2f} | {gen_stats['decode_peak_abs_mib']:<20.1f} | {gen_stats['decode_peak_delta_mib']:<18.1f} | {gen_stats['decode_kv_measured_mib']:<16.1f} | {gen_stats['final_kv_est_mib']:<18.1f}")
    else:
        print(f"Unknown MODE={MODE}. Use MODE=eval or MODE=decode.")


"""
# Eval Mode:
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
LLAMA_MODEL=meta-llama/Llama-2-7b-hf \
DTYPE=float16 \
MODE=eval \
BATCH_SIZE=1 \
SEQ_LEN=512 \
MAX_EVAL_SAMPLES=64 \
python profile_asvd_llama.py

# Decode Mode (ASVD off):
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
LLAMA_MODEL=meta-llama/Llama-2-7b-hf \
DTYPE=float16 \
MODE=decode \
PROMPT_BATCH=16 \
MAX_GEN_TOKENS=128 \
FORCE_LEGACY_KV=1 \
DEBUG_CACHE=1 \
python profile_asvd_llama.py

# Decode Mode (ASVD on; cache Pk,Pv only):
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
LLAMA_MODEL=meta-llama/Llama-2-7b-hf \
DTYPE=float16 \
MODE=decode \
PROMPT_BATCH=16 \
MAX_GEN_TOKENS=128 \
ASVD=1 \
RANK_KV=32 \
SVD_DTYPE=bf16 \
DEBUG_CACHE=1 \
python profile_asvd_llama.py
"""
