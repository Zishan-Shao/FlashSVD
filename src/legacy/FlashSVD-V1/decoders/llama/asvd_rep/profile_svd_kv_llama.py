#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_svd_kv_llama_decode.py — SVD on weights + dense KV cache (no ASVD), with eval-only profiling

Key behaviors
- DynamicCache.update(..) is called through a robust helper that works across HF versions
  (with/without cache_position; positional/keyword layer_idx; or returns None).
- RoPE uses absolute position_ids for both Q and K during evaluation.
- Bias mask is always additive [B,H,Q,K] and contiguous for SDPA.

Env example:
CUDA_VISIBLE_DEVICES=0 \
LLAMA_MODEL=meta-llama/Llama-2-7b-hf \
DTYPE=float16 MODE=eval BATCH_SIZE=1 SEQ_LEN=512 MAX_EVAL_SAMPLES=64 \
python3 profile_svd_kv_llama_decode.py
"""

import os, math, time, platform
from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

MiB = float(1024**2)

# -------------------- utils --------------------
def _repeat_kv(x, n_rep: int):
    # x: [B, Hk, T, Dh] -> [B, H, T, Dh]
    if n_rep == 1:
        return x
    B, Hk, T, Dh = x.shape
    return x[:, :, None].expand(B, Hk, n_rep, T, Dh).reshape(B, Hk * n_rep, T, Dh)

def _ensure_bhtd(x: torch.Tensor, Hk: int):
    assert x.dim() == 4, f"expected 4D, got {tuple(x.shape)}"
    if x.size(1) == Hk:
        return x
    if x.size(2) == Hk:
        return x.permute(0, 2, 1, 3).contiguous()
    raise RuntimeError(f"Unrecognized KV layout {tuple(x.shape)} for Hk={Hk}")

def _build_full_bias(attention_mask, batch_size, q_len, k_len, device, dtype):
    # causal (with KV offset)
    i = torch.arange(q_len, device=device).view(q_len, 1)
    j = torch.arange(k_len, device=device).view(1, k_len)
    past_len = k_len - q_len
    causal = (j <= (past_len + i))
    causal_bias = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    causal_bias.masked_fill_(~causal, torch.finfo(dtype).min)      # -inf for masked
    causal_bias = causal_bias.view(1, 1, q_len, k_len)              # [1,1,Q,K]

    pad_bias = None
    if attention_mask is not None:
        if attention_mask.dim() == 2:           # [B,K]
            am = attention_mask
            if am.size(-1) < k_len:
                am = F.pad(am, (0, k_len - am.size(-1)), value=1)
            elif am.size(-1) > k_len:
                am = am[:, -k_len:]
            pad_bias = (1.0 - am.to(dtype=dtype)) * torch.finfo(dtype).min
            pad_bias = pad_bias.view(batch_size, 1, 1, k_len)  # [B,1,1,K]
        elif attention_mask.dim() == 4:
            pad_bias = attention_mask.to(dtype=dtype, device=device)
            if pad_bias.size(-1) != k_len:
                if pad_bias.size(-1) < k_len:
                    pad_bias = F.pad(pad_bias, (0, k_len - pad_bias.size(-1)), value=0.0)
                else:
                    pad_bias = pad_bias[..., -k_len:]
    if pad_bias is None:
        return causal_bias                                # [1,1,Q,K]
    if pad_bias.size(-2) == 1:
        pad_bias = pad_bias.expand(-1, -1, q_len, -1)     # [B,1,Q,K]
    return causal_bias + pad_bias                         # [B,1,Q,K] (broadcastable)

# ---------- SVD helpers ----------
@torch.no_grad()
def _decompose_heads_svd(weight: torch.Tensor, n_heads: int, head_dim: int, rank: int):
    # weight: [H*Dh, D_in] -> Us: [H, Dh, r], V: [H, r, D_in]
    W = weight.detach().to(torch.float32)
    H, dh, D = n_heads, head_dim, W.shape[1]
    Us, Vs = [], []
    for h in range(H):
        Wh = W[h*dh:(h+1)*dh, :]
        U, S, Vh = torch.linalg.svd(Wh, full_matrices=False)
        r = max(1, min(int(rank), U.shape[1], Vh.shape[0]))
        Us.append(U[:, :r] * S[:r].unsqueeze(0))
        Vs.append(Vh[:r, :])
    return torch.stack(Us, 0), torch.stack(Vs, 0)

@torch.no_grad()
def _decompose_full_svd(weight: torch.Tensor, rank: int):
    W = weight.detach().to(torch.float32)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = max(1, min(int(rank), U.shape[1], Vh.shape[0]))
    return U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :]

# ---------- KV size ----------
def _estimate_kv_cache_mib(cfg, batch_size: int, seq_len: int, dtype: torch.dtype) -> float:
    L = int(getattr(cfg, 'num_hidden_layers'))
    H = int(getattr(cfg, 'num_attention_heads'))
    Hk = int(getattr(cfg, 'num_key_value_heads', H))
    Dh = int(getattr(cfg, 'hidden_size')) // H
    bpe = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total = 2 * L * batch_size * seq_len * Hk * Dh * bpe
    return total / MiB

def _bytes_of_present(past_key_values) -> float:
    if past_key_values is None:
        return 0.0
    seen, total = set(), 0
    def rec(x):
        nonlocal total
        if torch.is_tensor(x):
            try:
                s = x.untyped_storage(); key = (s.data_ptr(), int(s.nbytes()))
            except Exception:
                s = x.storage(); key = (s.data_ptr() if hasattr(s,"data_ptr") else x.data_ptr(),
                                        int(s.nbytes() if hasattr(s,"nbytes") else s.size()*x.element_size()))
            if key not in seen:
                seen.add(key); total += key[1]
        elif isinstance(x, (list, tuple, set)):
            for y in x: rec(y)
        elif isinstance(x, dict):
            for y in x.values(): rec(y)
        else:
            for name in ("layers","keys","values","key_cache","value_cache","k_cache","v_cache"):
                if hasattr(x, name): rec(getattr(x, name))
    rec(past_key_values)
    return total / MiB

def _sdpa_ctx():
    try:
        from torch.nn.attention import sdpa_kernel
        return sdpa_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    except Exception:
        try:
            from torch.backends.cuda import sdp_kernel
            return sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        except Exception:
            return nullcontext()

# -------------------- SVD LLaMA block --------------------
class SVDLlamaBlock(nn.Module):
    """
    SVD for q/k/v (+ optional o/ff), SDPA attention, **dense** KV via HF cache.
    Uses absolute position_ids for RoPE (works for prefill/decode).
    """
    def __init__(self, hf_layer: nn.Module, cfg,
                 rank_q: int, rank_kv: int, rank_o: Optional[int], rank_ff: Optional[int],
                 factor_dtype: torch.dtype = torch.float32, compute_in_fp32: bool = True):
        super().__init__()
        attn, mlp = hf_layer.self_attn, hf_layer.mlp
        self.D  = int(cfg.hidden_size)
        self.H  = int(cfg.num_attention_heads)
        self.Hk = int(getattr(cfg, "num_key_value_heads", self.H))
        self.Dh = self.D // self.H
        self.rep = self.H // self.Hk

        self.compute_in_fp32 = bool(compute_in_fp32)
        self.factor_dtype = factor_dtype
        self.layer_idx = getattr(getattr(hf_layer, "self_attn", None), "layer_idx", None)
        self.debug_cache = os.getenv("DEBUG_CACHE", "0") == "1"

        # norms
        self.ln1 = hf_layer.input_layernorm
        self.ln2 = hf_layer.post_attention_layernorm

        # RoPE (absolute pos_ids): precompute inv_freq
        rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
        half = self.Dh // 2
        inv = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("rope_inv_freq", inv, persistent=False)

        # SVD factors
        q_Us, q_V = _decompose_heads_svd(attn.q_proj.weight, self.H,  self.Dh, rank_q)
        k_Us, k_V = _decompose_heads_svd(attn.k_proj.weight, self.Hk, self.Dh, rank_kv)
        v_Us, v_V = _decompose_heads_svd(attn.v_proj.weight, self.Hk, self.Dh, rank_kv)
        self.q_Us = nn.Parameter(q_Us.to(factor_dtype), requires_grad=False)
        self.q_V  = nn.Parameter(q_V.to(factor_dtype),  requires_grad=False)
        self.k_Us = nn.Parameter(k_Us.to(factor_dtype), requires_grad=False)
        self.k_V  = nn.Parameter(k_V.to(factor_dtype),  requires_grad=False)
        self.v_Us = nn.Parameter(v_Us.to(factor_dtype), requires_grad=False)
        self.v_V  = nn.Parameter(v_V.to(factor_dtype),  requires_grad=False)

        # Output / MLP (optional low-rank)
        if rank_o is not None and rank_o > 0:
            o_Us, o_V = _decompose_full_svd(attn.o_proj.weight, rank_o)
            self.o_Us = nn.Parameter(o_Us.to(factor_dtype), requires_grad=False)
            self.o_V  = nn.Parameter(o_V.to(factor_dtype),  requires_grad=False)
            self.use_lr_o = True
        else:
            self.o = nn.Linear(self.D, self.D, bias=False, dtype=attn.o_proj.weight.dtype)
            with torch.no_grad(): self.o.weight.copy_(attn.o_proj.weight)
            self.use_lr_o = False

        inter = int(cfg.intermediate_size)
        if rank_ff is not None and rank_ff > 0:
            g_Us, g_V = _decompose_full_svd(mlp.gate_proj.weight, rank_ff)
            u_Us, u_V = _decompose_full_svd(mlp.up_proj.weight,   rank_ff)
            d_Us, d_V = _decompose_full_svd(mlp.down_proj.weight, rank_ff)
            for name, val in dict(g_Us=g_Us, g_V=g_V, u_Us=u_Us, u_V=u_V, d_Us=d_Us, d_V=d_V).items():
                setattr(self, name, nn.Parameter(val.to(factor_dtype), requires_grad=False))
            self.use_lr_ff = True
        else:
            self.gate = nn.Linear(self.D, inter, bias=False, dtype=mlp.gate_proj.weight.dtype)
            self.up   = nn.Linear(self.D, inter, bias=False, dtype=mlp.up_proj.weight.dtype)
            self.down = nn.Linear(inter, self.D, bias=False, dtype=mlp.down_proj.weight.dtype)
            with torch.no_grad():
                self.gate.weight.copy_(mlp.gate_proj.weight)
                self.up.weight.copy_(mlp.up_proj.weight)
                self.down.weight.copy_(mlp.down_proj.weight)
            self.use_lr_ff = False

        # free dense modules kept on hf_layer to reduce retention
        for obj in (attn, mlp):
            for name in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"):
                if hasattr(obj, name):
                    try: delattr(obj, name)
                    except Exception: pass

    # ----- RoPE w/ absolute position_ids -----
    def _apply_rope(self, x_bhtd: torch.Tensor, position_ids: Optional[torch.Tensor]):
        # x: [B,H,T,Dh]; position_ids: [B,T] (absolute)
        B, H, T, Dh = x_bhtd.shape
        half = Dh // 2
        if position_ids is None:
            pos = torch.arange(T, device=x_bhtd.device).unsqueeze(0).expand(B, T)
        else:
            pos = position_ids.to(torch.long)
            pos = torch.clamp(pos, min=0)
        # compute trig in fp32 for accuracy, but return in x.dtype
        ang = (pos.to(torch.float32)[..., None] *
               self.rope_inv_freq[None, :].to(x_bhtd.device))  # [B,T,half]
        cos = ang.cos().to(x_bhtd.dtype).unsqueeze(1)  # [B,1,T,half]
        sin = ang.sin().to(x_bhtd.dtype).unsqueeze(1)  # [B,1,T,half]

        x1, x2 = x_bhtd[..., :half], x_bhtd[..., half:]
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return out.to(x_bhtd.dtype).contiguous()


    # ----- projections -----
    def _proj_per_head(self, x: torch.Tensor, Us: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D], V: [H|Hk, r, D], Us: [H|Hk, Dh, r] -> [B, H|Hk, T, Dh]
        if self.compute_in_fp32:
            xr  = torch.einsum('b t d, h r d -> b t h r', x.float(), V.float())
            out = torch.einsum('b t h r, h d r -> b t h d', xr, Us.float())
            return out.to(x.dtype).transpose(1, 2).contiguous()
        xr  = torch.einsum('b t d, h r d -> b t h r', x.to(V.dtype), V)
        out = torch.einsum('b t h r, h d r -> b t h d', xr, Us)
        return out.to(x.dtype).transpose(1, 2).contiguous()

    # ----- robust cache readback (when update returns None) -----
    @staticmethod
    def _read_layer_kv_from_cache(cache_obj, layer_idx: int):
        # Try dynamic cache layout first
        k = v = None
        if hasattr(cache_obj, "layers") and 0 <= layer_idx < len(cache_obj.layers):
            lyr = cache_obj.layers[layer_idx]
            k = getattr(lyr, "keys", None)
            v = getattr(lyr, "values", None)
        if (k is None or v is None) and hasattr(cache_obj, "key_cache"):
            try:
                k = cache_obj.key_cache[layer_idx]
                v = cache_obj.value_cache[layer_idx]
            except Exception:
                pass
        return k, v

    # ----- robust cache update across HF versions -----
    def _cache_update(self, past_key_value, k_bhtd, v_bhtd, layer_idx, cache_position):
        # HF cache expects [B,Hk,T,Dh]
        k_upd = _ensure_bhtd(k_bhtd.contiguous(), self.Hk)
        v_upd = _ensure_bhtd(v_bhtd.contiguous(), self.Hk)

        cache_kwargs = None
        if cache_position is not None:
            cache_kwargs = {"cache_position": cache_position}

        # Try signatures in decreasing specificity
        if cache_kwargs is not None:
            try:
                ret = past_key_value.update(k_upd, v_upd, layer_idx=layer_idx, cache_kwargs=cache_kwargs)
            except TypeError:
                cache_kwargs = None  # older HF versions do not accept cache_kwargs

        if cache_kwargs is None:
            try:
                ret = past_key_value.update(k_upd, v_upd, layer_idx=layer_idx)
            except TypeError:
                try:
                    ret = past_key_value.update(k_upd, v_upd, layer_idx)
                except TypeError:
                    ret = past_key_value.update(k_upd, v_upd)

        # Some HF versions return (k_seq, v_seq); others return None
        if isinstance(ret, tuple) and len(ret) == 2:
            k_seq, v_seq = ret
        else:
            k_seq, v_seq = self._read_layer_kv_from_cache(past_key_value, layer_idx)

        if k_seq is None or v_seq is None:
            raise RuntimeError("Failed to obtain updated K/V from cache after update().")

        # Normalize to [B,Hk,T,Dh]
        k_seq = _ensure_bhtd(k_seq, self.Hk)
        v_seq = _ensure_bhtd(v_seq, self.Hk)
        return k_seq, v_seq

    # ----- forward -----
    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                position_ids=None, use_cache: bool = False, **kw):
        B, T, _ = hidden_states.shape
        x = self.ln1(hidden_states)

        # normalize to max valid length T_max (right-pad)
        if attention_mask is None:
            T_max = T
            x_trim = x
            pos_ids = position_ids
        else:
            if attention_mask.dim() == 2:
                keep_t = (attention_mask[:, :T] > 0) if not use_cache else (attention_mask[:, -T:] > 0)
                T_max = int(keep_t.sum(dim=1).max().item())
                if T_max <= 0:
                    T_max = T
                x_trim = x[:, :T_max, :]
                pos_ids = position_ids[:, :T_max] if position_ids is not None else None
            else:
                # HF supplies 4D additive masks during eval; lengths already aligned with hidden_states
                T_max = T
                x_trim = x
                pos_ids = position_ids

        # SVD projections
        q = self._proj_per_head(x_trim, self.q_Us, self.q_V)     # [B,H,Tq,Dh]
        k_new = self._proj_per_head(x_trim, self.k_Us, self.k_V) # [B,Hk,Tq,Dh]
        v_new = self._proj_per_head(x_trim, self.v_Us, self.v_V)

        # RoPE with absolute pos_ids for new chunk
        q = self._apply_rope(q, pos_ids)
        k_rot = self._apply_rope(k_new, pos_ids)

        # integrate with HF cache (dense)
        present_out = None
        cache_position = kw.get("cache_position", None)
        if hasattr(past_key_value, "update"):
            li = self.layer_idx if self.layer_idx is not None else kw.get("layer_idx", None)
            k_seq, v_seq = self._cache_update(past_key_value, k_rot, v_new, li, cache_position)
            present_out = past_key_value
        else:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                k_seq = torch.cat([_ensure_bhtd(past_key_value[0], self.Hk), k_rot], dim=2)
                v_seq = torch.cat([_ensure_bhtd(past_key_value[1], self.Hk), v_new], dim=2)
                present_out = (k_seq, v_seq)
            else:
                k_seq, v_seq = k_rot, v_new
                present_out = (k_seq, v_seq)

        # GQA repeat to full heads
        k = _repeat_kv(k_seq, self.rep)   # [B,H,Tk,Dh]
        v = _repeat_kv(v_seq, self.rep)
        q_len, k_len = q.size(-2), k.size(-2)

        # additive mask [B,H,Q,K]
        bias = _build_full_bias(attention_mask, B, q_len, k_len, q.device, q.dtype)
        bias = bias.expand(B, self.H, q_len, k_len).contiguous()

        # Ensure q, k, v are the same dtype for SDPA (guard against any upstream casts)
        if not (q.dtype == k.dtype == v.dtype):
            target = hidden_states.dtype
            q = q.to(target); k = k.to(target); v = v.to(target)
        with _sdpa_ctx():
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False)  # [B,H,Q,Dh]

        y = y.transpose(1, 2).contiguous().view(B, T_max, self.D)

        # Out + MLP
        if getattr(self, "use_lr_o", False):
            y = (y.to(self.o_V.dtype) @ self.o_V.t()) @ self.o_Us.t()
        else:
            y = getattr(self, "o", None)(y) if hasattr(self, "o") else y

        h = hidden_states[:, :T_max, :] + y
        z = self.ln2(h)
        if getattr(self, "use_lr_ff", False):
            y1 = (z.to(self.g_V.dtype) @ self.g_V.t()) @ self.g_Us.t()
            y2 = (z.to(self.u_V.dtype) @ self.u_V.t()) @ self.u_Us.t()
            ff = ((F.silu(y1) * y2) @ self.d_V.t()) @ self.d_Us.t()
        else:
            ff = getattr(self, "down")(F.silu(getattr(self, "gate")(z)) * getattr(self, "up")(z))
        out = h + ff

        # pad back to original T for API compat
        if T_max < T:
            pad = torch.zeros(B, T - T_max, self.D, dtype=out.dtype, device=out.device)
            out = torch.cat([out, pad], dim=1)
        return (out, present_out) if use_cache else (out,)

# -------------------- wire into HF --------------------
def replace_with_svd(model, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32):
    cfg = model.config
    dev = next(model.parameters()).device
    new_layers = nn.ModuleList()
    for layer in model.model.layers:
        shim = _wrap_svd_layer(layer, cfg, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32)
        shim.to(dev); new_layers.append(shim)
    model.model.layers = new_layers

def _wrap_svd_layer(hf_layer, cfg, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32):
    class _Shim(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.layer_idx = getattr(getattr(inner, "self_attn", None), "layer_idx", None)
            self.block = SVDLlamaBlock(inner, cfg, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32)
        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, use_cache=False, **kw):
            if "layer_idx" not in kw and self.layer_idx is not None:
                kw = dict(kw); kw["layer_idx"] = self.layer_idx
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            return self.block(hidden_states, attention_mask=attention_mask,
                              past_key_value=past_key_value, position_ids=position_ids,
                              use_cache=use_cache, **kw)
    return _Shim(hf_layer)

# -------------------- eval (quick) --------------------
@torch.no_grad()
def eval_perplexity_fullseq(model, loader, device):
    model.eval(); total_loss, total_tok = 0.0, 0
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        am   = batch["attention_mask"].to(device)
        B, T = ids.shape
        pos  = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)
        out = model(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()
        labels = ids[:, 1:].contiguous()
        mask   = am[:, 1:].contiguous().bool()
        if mask.any():
            v_logits = logits[mask].float()
            v_labels = labels[mask]
            finite = torch.isfinite(v_logits).all(dim=-1)
            if finite.any():
                loss = F.cross_entropy(v_logits[finite], v_labels[finite], reduction="sum")
                total_loss += loss.item()
                total_tok  += int(finite.sum().item())
    if torch.cuda.is_available(): torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(loader))
    ppl = math.exp(total_loss / total_tok) if total_tok > 0 else float("nan")
    peak = torch.cuda.max_memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    return ppl, peak, ms

# -------------------- main --------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # env
    dt = os.getenv("DTYPE", "float16").lower()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dt]
    MODEL_NAME = os.getenv("LLAMA_MODEL", "meta-llama/Llama-2-7b-hf")
    MODE = os.getenv("MODE", "eval").lower()
    if MODE != "eval":
        raise ValueError(f"MODE={MODE} is unsupported. This script only profiles evaluation.")

    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
    SEQ_LEN = int(os.getenv("SEQ_LEN", "512"))
    MAX_EVAL_SAMPLES = int(os.getenv("MAX_EVAL_SAMPLES", "64"))

    # SVD ranks
    RANK_Q  = int(os.getenv("RANK_Q",  "128"))
    RANK_KV = int(os.getenv("RANK_KV", "128"))
    RANK_O  = int(os.getenv("RANK_O",  "0")) or None
    RANK_FF = int(os.getenv("RANK_FF", "0")) or None
    SVD_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[os.getenv("SVD_DTYPE", "fp32").lower()]
    SVD_COMPUTE_FP32 = os.getenv("SVD_COMPUTE_FP32", "1") == "1"

    # model
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # swap SVD blocks
    replace_with_svd(model, RANK_Q, RANK_KV, RANK_O, RANK_FF, factor_dtype=SVD_DTYPE, compute_in_fp32=SVD_COMPUTE_FP32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    if MAX_EVAL_SAMPLES > 0: raw = raw.select(range(min(MAX_EVAL_SAMPLES, len(raw))))
    def tok_fn(batch): return tok(batch["text"], padding="max_length", truncation=True, max_length=SEQ_LEN)
    ds = raw.map(tok_fn, batched=True, remove_columns=["text"]); ds.set_format("torch")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]),
                                              "attention_mask": torch.stack([x["attention_mask"] for x in b])})
    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    storage_mem = torch.cuda.memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    kv_est = _estimate_kv_cache_mib(model.config, BATCH_SIZE, SEQ_LEN, dtype)
    ppl, peak, ms = eval_perplexity_fullseq(model, loader, device)
    print("\n================== LLaMA + SVD (SDPA / Eval) ==================")
    print(f"Python {platform.python_version()}  Torch {torch.__version__}")
    print(f"Device/dtype: {device}/{dtype}")
    print(f"Ranks: q={RANK_Q}, kv={RANK_KV}, o={RANK_O}, ff={RANK_FF}  |  SVD store dtype={SVD_DTYPE}  |  SVD compute fp32={SVD_COMPUTE_FP32}")
    print(f"{'Storage (MiB)':<16} | {'Peak (MiB)':<10} | {'Transient (MiB)':<14} | {'KV est (MiB)':<12} | {'Time (ms/b)':<12} | {'Perplexity':<10}")
    print("-"*100)
    print(f"{storage_mem:<16.1f} | {peak:<10.1f} | {max(0.0, peak-storage_mem):<14.1f} | {kv_est:<12.1f} | {ms:<12.1f} | {ppl:<10.4f}")


'''

CUDA_VISIBLE_DEVICES=0 \
LLAMA_MODEL=meta-llama/Llama-2-7b-hf \
DTYPE=float16 MODE=eval BATCH_SIZE=1 SEQ_LEN=512 MAX_EVAL_SAMPLES=64 \
python3 profile_svd_kv_llama.py


'''
