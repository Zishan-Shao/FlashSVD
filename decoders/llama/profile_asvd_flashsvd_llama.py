import os, math, time, platform, inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaForCausalLM

from typing import Optional

"""
profile_asvd_llama.py: SVD-based LLaMA block using plain PyTorch attention (no FlashAttention).

Highlights in this version:
  • Correct LLaMA RoPE (half-split rotation), implemented manually in [B,H,T,Dh].
  • ASVD decode: internal (Pk,Pv) cache per block; never return tuple caches to HF.
  • Eval is robust: trim to max non-pad length and pass a 2D mask (or use contiguous eval).
  • Works with right-padding; no 5-D shapes; no index OOB in RoPE.

Env toggles:
  MODE=eval|decode
  LLAMA_MODEL=meta-llama/Llama-2-7b-hf
  DTYPE=float16|bfloat16|float32
  BATCH_SIZE, PROMPT_BATCH, MAX_GEN_TOKENS, SEQ_LEN, MAX_EVAL_SAMPLES
  RANK_Q, RANK_KV, RANK_O(0=dense), RANK_FF(0=dense)
  SVD_DTYPE=fp16|bf16|fp32    # dtype to store (Pk,Pv)
  SVD_COMPUTE_FP32=1|0
  EVAL_CONTIGUOUS=1|0         # 1: concat & chunk dataset for standard PPL
  DEBUG_CACHE=1, DEBUG_EVAL=1, CHECK_SVD=1
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
    num_layers = int(getattr(cfg, 'num_hidden_layers'))
    n_heads = int(getattr(cfg, 'num_attention_heads'))
    n_kv_heads = int(getattr(cfg, 'num_key_value_heads', n_heads))
    bytes_per_elem = 2 if factor_dtype in (torch.float16, torch.bfloat16) else 4
    total_bytes = 2 * num_layers * batch_size * seq_len * n_kv_heads * int(rank_kv) * bytes_per_elem
    return total_bytes / (1024.0 ** 2)

    

def _measure_asvd_cache_mib(model) -> float:
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
    LLaMA block with:
      • per-head SVD for q/k/v (optional bypass)
      • optional low-rank o/ff
      • PyTorch SDPA (causal)
      • Optional ASVD cache: store only (Pk,Pv) and reconstruct K,V on the fly
      • Correct manual RoPE (half-split, LLaMA-compatible) in [B,H,T,Dh]
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
        # ASVD is always enabled by default in this simplified version
        self.asvd_enabled = True

        # ASVD internal stateful cache
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
                    self.half = head_dim // 2
                    evens = torch.arange(0, self.half, dtype=torch.float32)
                    self.register_buffer("inv_freq", 1.0 / (base ** (evens / self.half)), persistent=False)
                def forward(self, x, seq_len: int = None, position_ids: Optional[torch.LongTensor] = None):
                    device = x.device
                    if seq_len is None:
                        if position_ids is not None:
                            seq_len = int(position_ids.max().item()) + 1
                        else:
                            seq_len = x.size(-2)
                    t = torch.arange(seq_len, device=device, dtype=torch.float32)        # [T]
                    ang = t[:, None] * self.inv_freq[None, :]                             # [T, half]
                    cos = ang.cos()[None, None, :, :]                                     # [1,1,T,half]
                    sin = ang.sin()[None, None, :, :]                                     # [1,1,T,half]
                    return cos, sin
            self.rotary_emb = _SimpleRoPE(self.head_dim)

        # SVD factors for Q/K/V
        q_Us, q_V = _decompose_heads_svd(attn.q_proj.weight, self.n_heads,    self.head_dim, rank_q)
        k_Us, k_V = _decompose_heads_svd(attn.k_proj.weight, self.n_kv_heads, self.head_dim, rank_kv)
        v_Us, v_V = _decompose_heads_svd(attn.v_proj.weight, self.n_kv_heads, self.head_dim, rank_kv)
        self.q_Us = nn.Parameter(q_Us.to(factor_dtype), requires_grad=False)
        self.q_V  = nn.Parameter(q_V.to(factor_dtype),  requires_grad=False)
        self.k_Us = nn.Parameter(k_Us.to(factor_dtype), requires_grad=False)
        self.k_V  = nn.Parameter(k_V.to(factor_dtype),  requires_grad=False)
        self.v_Us = nn.Parameter(v_Us.to(factor_dtype), requires_grad=False)
        self.v_V  = nn.Parameter(v_V.to(factor_dtype),  requires_grad=False)

        # Optional reconstruction check
        if os.getenv("CHECK_SVD", "0") == "1":
            with torch.no_grad():
                def _max_err(W, H, dh, Us, V):
                    Wv = W.float().view(H, dh, -1)
                    R  = torch.einsum('hdr,hrD->hdD', Us.float(), V.float())
                    return (Wv - R).abs().max().item()
                print(f"[SVD check] max |Q - UΣV|: {_max_err(attn.q_proj.weight, self.n_heads, self.head_dim, self.q_Us, self.q_V):.3e}")
                print(f"[SVD check] max |K - UΣV|: {_max_err(attn.k_proj.weight, self.n_kv_heads, self.head_dim, self.k_Us, self.k_V):.3e}")
                print(f"[SVD check] max |V - UΣV|: {_max_err(attn.v_proj.weight, self.n_kv_heads, self.head_dim, self.v_Us, self.v_V):.3e}")

        # Output/MLP
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

        # Free dense modules from the original layer
        for obj in (attn, mlp):
            for name in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"):
                if hasattr(obj, name):
                    try: delattr(obj, name)
                    except Exception: pass

    # ─────────── RoPE helpers (half-split, shape-safe) ───────────
    @torch.no_grad()
    def _rope_get_cos_sin(self, x_bhtd, position_ids=None, seq_len=None):
        """
        Return cos,sin as [B,1,T,half] matching LLaMA half-split rotation.
        """
        B, H, T, Dh = x_bhtd.shape
        half = Dh // 2

        # Decide how many rows we need
        target_len = T
        if seq_len is not None:
            target_len = max(target_len, int(seq_len))
        if position_ids is not None:
            if position_ids.dtype != torch.long:
                position_ids = position_ids.to(torch.long)
            if (position_ids < 0).any():
                position_ids = position_ids.clamp_min_(0)
            pos_max = int(position_ids.max().item())
            target_len = max(target_len, pos_max + 1)

        # Call rotary_emb forward without passing position_ids
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

        # Normalize to [B,1,target_len,half]
        def _norm(z):
            if z.dim() == 2:       # [T, D]
                z = z.unsqueeze(0).unsqueeze(0)      # [1,1,T,D]
            elif z.dim() == 3:     # [1,T,D] or [B,T,D]
                z = z.unsqueeze(1)                   # [1,1,T,D] or [B,1,T,D]
            # now z is [*,1,T,D]
            if z.size(0) == 1 and B > 1:
                z = z.expand(B, -1, -1, -1)         # [B,1,T,D]
            return z
        cos = _norm(cos)
        sin = _norm(sin)

        # If some rotary variants return D=Dh, slice to half
        if cos.size(-1) != half:
            cos = cos[..., :half]
            sin = sin[..., :half]

        # If specific positions requested, gather
        if position_ids is not None:
            idx = position_ids.unsqueeze(1).unsqueeze(-1).expand(B, 1, T, half)  # [B,1,T,half]
            cos = torch.gather(cos, -2, idx)
            sin = torch.gather(sin, -2, idx)
        else:
            cos = cos[..., :T, :]
            sin = sin[..., :T, :]

        # Cast & device
        dtype = x_bhtd.dtype
        dev   = x_bhtd.device
        return cos.to(dtype=dtype, device=dev), sin.to(dtype=dtype, device=dev)

    def _manual_rope_half_split(self, x_bhtd, cos, sin):
        """
        LLaMA rotate_half: x=[x1|x2], rotate_half(x)=[-x2|x1]
        x_bhtd: [B,H,T,Dh], cos/sin: [B,1,T,half]
        """
        B, H, T, Dh = x_bhtd.shape
        half = Dh // 2
        x1, x2 = x_bhtd[..., :half], x_bhtd[..., half:]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.cat([out1, out2], dim=-1).contiguous()

    @torch.no_grad()
    def _apply_rope_single(self, x_bhtd, position_ids=None, seq_len=None):
        cos, sin = self._rope_get_cos_sin(x_bhtd, position_ids=position_ids, seq_len=seq_len)
        # broadcast cos/sin across heads (cos/sin are [B,1,T,half])
        return self._manual_rope_half_split(x_bhtd, cos, sin)

    def _apply_rope(self, q_bhtd, k_bhtd, position_ids, position_embeddings=None):
        q_rot = self._apply_rope_single(q_bhtd, position_ids=position_ids, seq_len=q_bhtd.size(-2))
        k_rot = self._apply_rope_single(k_bhtd, position_ids=position_ids, seq_len=k_bhtd.size(-2))
        return q_rot, k_rot

    # ────────────────────────── forward ──────────────────────────
    def _proj_per_head(self, x: torch.Tensor, Us: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        x:  [B, T, D],   V: [H, r, D],   Us: [H, dh, r]  ->  [B, H, T, dh]
        """
        if self.compute_in_fp32:
            xr  = torch.einsum('b t d, h r d -> b t h r', x.float(), V.float())   # [B,T,H,r]
            out = torch.einsum('b t h r, h d r -> b t h d', xr, Us.float())       # [B,T,H,dh]
            return out.to(x.dtype).transpose(1, 2).contiguous()
        xr  = torch.einsum('b t d, h r d -> b t h r', x.to(V.dtype), V)
        out = torch.einsum('b t h r, h d r -> b t h d', xr, Us)
        return out.to(x.dtype).transpose(1, 2).contiguous()

    def _proj_per_head_lowrank(self, x: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Return low-rank projections P = X·V : [B, H, T, r] (stored in factor_dtype)."""
        if self.compute_in_fp32:
            P = torch.einsum('b t d, h r d -> b t h r', x.float(), V.float())
        else:
            P = torch.einsum('b t d, h r d -> b t h r', x.to(V.dtype), V)
        return P.transpose(1, 2).to(self.factor_dtype).contiguous()

    def _reconstruct_from_P(self, P: torch.Tensor, Us: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """Reconstruct [B, H, T, dh] from P: [B,H,T,r] and Us: [H,dh,r]."""
        if self.compute_in_fp32:
            X = torch.einsum('b h t r, h d r -> b h t d', P.float(), Us.float())
            return X.to(out_dtype).contiguous()
        X = torch.einsum('b h t r, h d r -> b h t d', P.to(Us.dtype), Us)
        return X.to(out_dtype).contiguous()

    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                position_ids=None, use_cache=False, position_embeddings=None, **kw):
        """
        hidden_states: [B, T, D]
        attention_mask: None | [B,T] | [B,1,Q,K]
        position_ids: [B,T]
        """
        B, T, D = hidden_states.shape
        x = self.ln1(hidden_states)

        # Determine T_max and trimmed inputs
        if attention_mask is None:
            T_max = T
            x_trim = x
            pos_ids = position_ids
        else:
            if use_cache:
                if attention_mask.dim() == 2:
                    keep_t = (attention_mask[:, -T:] > 0)
                elif attention_mask.dim() == 4:
                    m = attention_mask.to(torch.float32).squeeze(1) # [B,Q,K]
                    keep_t = (m > -1e3).any(dim=1)[:, -T:]
                else:
                    keep_t = attention_mask.reshape(B, -1)[:, -T:].to(torch.bool)
            else:
                if attention_mask.dim() == 2:
                    keep_t = (attention_mask[:, :T] > 0)
                elif attention_mask.dim() == 4:
                    m = attention_mask.to(torch.float32).squeeze(1)
                    keep_t = (m > -1e3).any(dim=1)[:, :T]
                else:
                    keep_t = attention_mask.reshape(B, -1)[:, :T].to(torch.bool)
            T_max = int(keep_t.sum(dim=1).max().item())
            x_trim = x[:, :T_max, :]
            pos_ids = position_ids[:, :T_max] if position_ids is not None else None

        # Q per-head
        q = self._proj_per_head(x_trim, self.q_Us, self.q_V)         # [B,H,T,dh]

        if not use_cache:
            # ---------- Evaluation path ----------
            k_raw = self._proj_per_head(x_trim, self.k_Us, self.k_V)     # [B,Hk,T,dh]
            v_raw = self._proj_per_head(x_trim, self.v_Us, self.v_V)
            k = _repeat_kv(k_raw, self.n_rep)                             # [B,H,T,dh]
            v = _repeat_kv(v_raw, self.n_rep)
            q, k = self._apply_rope(q, k, pos_ids, position_embeddings=position_embeddings)

            q_bhsd = q.contiguous()
            k_bhsd = k.contiguous()
            v_bhsd = v.contiguous()

            bias = _build_full_bias(attention_mask, B, T_max, k_bhsd.size(-2), q_bhsd.device, q_bhsd.dtype)
            attn_sdpa = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, attn_mask=bias, is_causal=False)
            present_out = None

        else:
            # ---------- Decode path (ASVD only) ----------
            # Reset ASVD cache at the start of a sequence
            if past_key_value is None:
                self._asvd_pk = None
                self._asvd_pv = None

            # Low-rank projections for current chunk
            Pk_new = self._proj_per_head_lowrank(x_trim, self.k_V)   # [B,Hk,T,r]
            Pv_new = self._proj_per_head_lowrank(x_trim, self.v_V)

            if self._asvd_pk is None:
                self._asvd_pk, self._asvd_pv = Pk_new, Pv_new
            else:
                self._asvd_pk = torch.cat([self._asvd_pk, Pk_new], dim=2)
                self._asvd_pv = torch.cat([self._asvd_pv, Pv_new], dim=2)

            pk_seq, pv_seq = self._asvd_pk, self._asvd_pv  # [B,Hk,K,r]

            # Reconstruct K,V and apply RoPE
            k_seq = self._reconstruct_from_P(pk_seq, self.k_Us, out_dtype=q.dtype)  # [B,Hk,K,dh]
            v_seq = self._reconstruct_from_P(pv_seq, self.v_Us, out_dtype=q.dtype)

            q_bhtd = self._apply_rope_single(q, position_ids=pos_ids)              # [B,H,T,dh]
            k_len = k_seq.size(-2)
            k_bhtd = self._apply_rope_single(k_seq, position_ids=None, seq_len=k_len)

            k_bhsd = _repeat_kv(k_bhtd, self.n_rep)         # [B,H,K,dh]
            v_bhsd = _repeat_kv(v_seq,  self.n_rep)
            q_bhsd = q_bhtd.contiguous()

            if attention_mask is not None and attention_mask.dim() == 4:
                bias = attention_mask.to(dtype=q_bhsd.dtype, device=q_bhsd.device)
                bias = bias[..., -T_max:, -k_len:]
                if (not bias.is_contiguous()):
                    bias = bias.contiguous().clone()
            else:
                bias = _build_full_bias(attention_mask, B, T_max, k_len, q_bhsd.device, q_bhsd.dtype)

            if self.debug_cache and (self.layer_idx in (None, 0)):
                print(f"[ASVD] layer_idx={self.layer_idx} Pk={tuple(pk_seq.shape)} Pv={tuple(pv_seq.shape)} K={k_len}")
                print(f"[ASVD] q={tuple(q_bhtd.shape)} k={tuple(k_bhtd.shape)} v={tuple(v_seq.shape)} T_max={T_max}")

            scale = 1.0 / math.sqrt(self.head_dim)
            qf = q_bhsd.float(); kf = k_bhsd.float(); vf = v_bhsd.float()
            logits = torch.einsum('b h t d, b h k d -> b h t k', qf, kf) * scale
            if bias is not None:
                logits = logits + bias
            weights = torch.softmax(logits, dim=-1)
            attn_sdpa = torch.einsum('b h t k, b h k d -> b h t d', weights, vf).to(q_bhsd.dtype)

            # In ASVD mode, we do not return legacy tuple caches to HF
            present_out = None

        # Merge heads
        attn = attn_sdpa.transpose(1, 2).contiguous().view(B, T_max, self.n_heads * self.head_dim)

        # Output + MLP
        if getattr(self, "use_lowrank_o", False):
            attn = (attn.to(self.o_V.dtype) @ self.o_V.t()) @ self.o_Us.t()
        else:
            attn = self.o(attn)

        h = hidden_states[:, :T_max, :] + attn
        y = self.ln2(h)

        if getattr(self, "use_lowrank_ff", False):
            y1 = (y.to(self.g_V.dtype) @ self.g_V.t()) @ self.g_Us.t()
            y2 = (y.to(self.u_V.dtype) @ self.u_V.t()) @ self.u_Us.t()
            ff = ((F.silu(y1) * y2) @ self.d_V.t()) @ self.d_Us.t()
        else:
            ff = self.down(F.silu(self.gate(y)) * self.up(y))

        out = h + ff

        # Pad back to original T
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

# ────────────────────── eval utilities ──────────────────────
class TensorPairDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.ids = input_ids
        self.mask = attention_mask
    def __len__(self): return self.ids.size(0)
    def __getitem__(self, idx): return {"input_ids": self.ids[idx], "attention_mask": self.mask[idx]}

def build_contiguous_eval_set(tok, seq_len: int, max_eval_samples: int):
    """
    Concatenate WikiText-2-raw test split and chunk into fixed windows (no overlap).
    This matches common PPL setups better than per-line padding.
    """
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in raw["text"] if t and t.strip() != ""])
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    total = ids.size(0)
    n_chunks = total // seq_len
    if max_eval_samples > 0:
        n_chunks = min(n_chunks, max_eval_samples)
    used = n_chunks * seq_len
    input_ids = ids[:used].view(n_chunks, seq_len).contiguous()
    attention_mask = torch.ones_like(input_ids)
    return TensorPairDataset(input_ids, attention_mask)

@torch.no_grad()
def eval_perplexity_fullseq(model, loader, device):
    """
    Evaluation with right padding: trim each batch to the max non-pad length
    and pass a 2D mask of that same length. (For contiguous eval, mask is all ones.)
    """
    model.eval()
    total_loss, total_tok = 0.0, 0
    debug_eval = os.getenv("DEBUG_EVAL", "0") == "1"

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)      # [B,T]
        am2d      = batch["attention_mask"].to(device) # [B,T]
        B, T_full = input_ids.shape

        lengths = am2d.sum(dim=-1)                     # [B]
        L = int(lengths.max().item())
        if L <= 1:  # nothing to score
            continue

        ids = input_ids[:, :L]
        am  = am2d[:, :L]
        pos = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)

        out = model(input_ids=ids,
                    attention_mask=am,   # 2D mask; block adds causal + handles shapes
                    position_ids=pos,
                    use_cache=False)

        logits = out.logits[:, :-1, :].contiguous()
        labels = ids[:, 1:].contiguous()
        mask   = am[:, 1:].contiguous().bool()

        if debug_eval and i == 0:
            kept = int(mask.sum().item())
            total = int(B * (L - 1))
            finite_token_mask = torch.isfinite(logits).all(dim=-1)
            finite_tok = int((finite_token_mask & mask).sum().item())
            any_nan = (~torch.isfinite(logits)).any().item()
            max_logit = float(torch.nan_to_num(logits).max().item())
            min_logit = float(torch.nan_to_num(logits).min().item())
            print(f"[DEBUG_EVAL] L={L} tokens={kept}/{total} finite_tokens={finite_tok} any_nan={any_nan} "
                  f"logits_range=[{min_logit:.2f},{max_logit:.2f}]")

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
    # ASVD is always enabled in this script
    asvd_enable = True
    svd_dtype_env = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[os.getenv("SVD_DTYPE", "fp32").lower()]

    # Load/generate prompt
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
    pos_ids_full = torch.arange(input_ids.size(1), device=device).unsqueeze(0).repeat(batch_size, 1)

    # Prefill (ASVD: no legacy tuples; HF cache remains None)
    model.config.use_cache = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    storage_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    supports_cache_pos = _supports_kwarg(model.forward, "cache_position")

    t0 = time.perf_counter()
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
        position_ids=pos_ids_full,
        use_cache=True,
    )
    if supports_cache_pos:
        kwargs["cache_position"] = pos_ids_full[:, -1]
    out = model(**kwargs)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    prefill_peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    prefill_current = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    present = out.past_key_values
    prefill_kv_bytes_asvd = _measure_asvd_cache_mib(model)

    # Decode loop
    running_attn = attn_mask.clone()
    next_ids = input_ids[:, -1:]
    decode_start_alloc = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t_dec = 0.0
    decode_kv_bytes_asvd = prefill_kv_bytes_asvd

    for step in range(decode_tokens):
        t1 = time.perf_counter()
        running_attn = torch.cat([running_attn, torch.ones(running_attn.size(0), 1, dtype=running_attn.dtype, device=running_attn.device)], dim=1)
        pos_next = (running_attn.long().sum(dim=1, keepdim=True) - 1)

        kwargs = dict(
            input_ids=next_ids,
            attention_mask=running_attn,
            position_ids=pos_next,
            past_key_values=present,  # remains None in ASVD
            use_cache=True,
        )
        if supports_cache_pos:
            kwargs["cache_position"] = pos_next.squeeze(1)

        out = model(**kwargs)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_dec += (time.perf_counter() - t1)
        present = out.past_key_values
        decode_kv_bytes_asvd = _measure_asvd_cache_mib(model)
        next_ids = out.logits[:, -1:, :].argmax(dim=-1)

    decode_ms = t_dec * 1000.0 / max(1, decode_tokens)
    decode_peak_abs = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    decode_peak_delta = max(0.0, decode_peak_abs - decode_start_alloc)
    storage_after = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    final_kv_est = _estimate_asvd_kv_cache_mib(model.config, batch_size=batch_size,
                                               seq_len=prompt_len + decode_tokens,
                                               factor_dtype=svd_dtype_env, rank_kv=rank_kv)

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
        'prefill_kv_measured_mib': prefill_kv_bytes_asvd,
        'decode_kv_measured_mib': decode_kv_bytes_asvd,
        'asvd_enabled': True,
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
    RANK_O  = int(os.getenv("RANK_O",  "0")) or None
    RANK_FF = int(os.getenv("RANK_FF", "0")) or None
    SVD_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[os.getenv("SVD_DTYPE", "fp32").lower()]
    SVD_COMPUTE_FP32 = os.getenv("SVD_COMPUTE_FP32", "1") == "1"
    EVAL_CONTIGUOUS = os.getenv("EVAL_CONTIGUOUS", "1") == "1"

    # Load model/tokenizer
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.config.use_cache = False  # eval default; decode sets True

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Swap in SVD layers
    replace_with_svd(
        model, rank_q=RANK_Q, rank_kv=RANK_KV, rank_o=RANK_O, rank_ff=RANK_FF,
        factor_dtype=SVD_DTYPE, compute_in_fp32=SVD_COMPUTE_FP32
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if MODE == "eval":
        # Build dataset/loader
        if EVAL_CONTIGUOUS:
            ds = build_contiguous_eval_set(tok, seq_len=SEQ_LEN, max_eval_samples=MAX_EVAL_SAMPLES)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        else:
            raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            if MAX_EVAL_SAMPLES > 0:
                raw = raw.select(range(min(MAX_EVAL_SAMPLES, len(raw))))
            def tokenize_fn(batch):
                # Right-pad to SEQ_LEN (we will trim each batch to its max non-pad length)
                return tok(batch["text"], padding="max_length", truncation=True,
                           max_length=SEQ_LEN, add_special_tokens=False)
            ds = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
            ds.set_format("torch")
            loader = DataLoader(
                ds, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]),
                                      "attention_mask": torch.stack([x["attention_mask"] for x in b])}
            )

        # Run eval
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
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
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        prompt_file = os.path.join(os.path.dirname(__file__) if '__file__' in globals() else '.', 'synthetic_prompt_256.txt')
        gen_stats = profile_prefill_and_decode(
            model, tok, device,
            prompt_path=prompt_file,
            prompt_len=256,
            decode_tokens=MAX_GEN_TOKENS,
            batch_size=PROMPT_BATCH,
            rank_kv=RANK_KV,
        )
        kv_label = "KV est (MiB)"
        print("\n================== LLaMA + SVD (Decode Mode) ==================")
        print(f"Python {platform.python_version()}  Torch {torch.__version__}")
        print(f"Device/dtype: {device}/{dtype}")
        print(f"ASVD=on | Ranks: q={RANK_Q}, kv={RANK_KV}, o={RANK_O}, ff={RANK_FF}  |  SVD store dtype={SVD_DTYPE}  |  SVD compute fp32={SVD_COMPUTE_FP32}")
        print(f"Prompt batch: {PROMPT_BATCH} | Prompt len: 256 | Max gen: {MAX_GEN_TOKENS}")
        print(f"{'Storage (MiB)':<16} | {'Prefill (ms)':<12} | {'Prefill Peak (MiB)':<18} | {'Prefill Current (MiB)':<20} | {'Prefill KV (MiB)':<16} | {'Decode ms/tok':<14} | {'Decode Peak Abs (MiB)':<20} | {'Decode Peak Δ (MiB)':<18} | {'Decode KV (MiB)':<16} | {kv_label:<18}")
        print("-" * 200)
        print(f"{gen_stats['storage_mib']:<16.1f} | {gen_stats['prefill_ms']:<12.1f} | {gen_stats['prefill_peak_mib']:<18.1f} | {gen_stats['prefill_current_mib']:<20.1f} | {gen_stats['prefill_kv_measured_mib']:<16.1f} | {gen_stats['decode_avg_ms_per_tok']:<14.2f} | {gen_stats['decode_peak_abs_mib']:<20.1f} | {gen_stats['decode_peak_delta_mib']:<18.1f} | {gen_stats['decode_kv_measured_mib']:<16.1f} | {gen_stats['final_kv_est_mib']:<18.1f}")
    else:
        print(f"Unknown MODE={MODE}. Use MODE=eval or MODE=decode.")



"""
# Eval Mode:
CUDA_VISIBLE_DEVICES=6 CUDA_LAUNCH_BLOCKING=1 \
LLAMA_MODEL=meta-llama/Llama-2-7b-hf \
DTYPE=float16 \
MODE=eval \
BATCH_SIZE=1 \
SEQ_LEN=512 \
MAX_EVAL_SAMPLES=64 \
python profile_asvd_flashsvd_llama.py

# Decode Mode (ASVD on; cache Pk,Pv only):
CUDA_VISIBLE_DEVICES=6 CUDA_LAUNCH_BLOCKING=1 \
LLAMA_MODEL=meta-llama/Llama-2-7b-hf \
DTYPE=float16 \
MODE=decode \
PROMPT_BATCH=16 \
MAX_GEN_TOKENS=128 \
RANK_KV=128 \
SVD_DTYPE=bf16 \
DEBUG_CACHE=1 \
python profile_asvd_flashsvd_llama.py
"""

