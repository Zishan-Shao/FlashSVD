import os, math, time, platform, inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from typing import Optional

'''
profile_svd_kv_llama.py: SVD-based LLaMA block using plain PyTorch attention (no FlashAttention).
NOTE:
1. this script has no CHUNK_SIZE, because it is not a multihead compatible version.
2. this script has no FlashAttention, because it is not a multihead compatible version.
3. this script has no SVD_VERIFY, because it is not a multihead compatible version.
4. this script has no SVD_DTYPE, because it is not a multihead compatible version.
5. this script has no SVD_COMPUTE_FP32, because it is not a multihead compatible version.
6. this script has no SVD_VERIFY, because it is not a multihead compatible version.
7. this script has no SVD_DTYPE, because it is not a multihead compatible version.
8. Because no KV cache is created, the only extra memory during eval is transient 
   activations/workspace from SDPA. Modern SDPA kernels are memory-efficient (don’t 
   materialize the full Q×K matrix), so peak − storage can be much smaller than a 
   theoretical KV cache.
'''

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
    V  = Vh[:r, :]                                # [r, in]
    return Us, V

# ────────────────────── KV cache estimator ──────────────────────
def _estimate_kv_cache_mib(cfg, batch_size: int, seq_len: int, dtype: torch.dtype) -> float:
    """
    Estimate KV-cache memory in MiB for full context length `seq_len`.
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

# ────────────────────── Flash + SVD block ──────────────────────
class SVDLlamaBlock(nn.Module):
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
                    evens = torch.arange(0, head_dim, 2, dtype=torch.float32)
                    self.register_buffer("inv_freq", 1.0 / (base ** (evens / head_dim)), persistent=False)

                def forward(self, x, seq_len: int = None, position_ids: Optional[torch.LongTensor] = None):
                    # Return RoPE caches that HF will index via position_ids
                    device = x.device
                    Dh = self.head_dim
                    inv = self.inv_freq.to(device=device)
                    if seq_len is None:
                        # Fallback if only position_ids is given
                        seq_len = int(position_ids.max().item()) + 1
                    t = torch.arange(seq_len, device=device, dtype=torch.float32)        # [T]
                    ang = t[:, None] * inv[None, :]                                      # [T, Dh/2]
                    ang = ang.repeat_interleave(2, dim=-1)                               # [T, Dh]
                    cos = ang.cos()[None, None, :, :].to(x.dtype)                        # [1,1,T,Dh]
                    sin = ang.sin()[None, None, :, :].to(x.dtype)                        # [1,1,T,Dh]
                    return cos, sin

            self.rotary_emb = _SimpleRoPE(self.head_dim, base=rope_theta)

        # No dense projections retained; SVD-only path

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

        # Explicitly clear heavy dense modules from the original HF layer to avoid GPU retention
        try:
            del attn.q_proj
            del attn.k_proj
            del attn.v_proj
            del attn.o_proj
        except Exception:
            pass
        try:
            del mlp.gate_proj
            del mlp.up_proj
            del mlp.down_proj
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

    def _apply_rope(self, q_bhtd, k_bhtd, position_ids, position_embeddings=None):
        """
        Apply RoPE on tensors laid out as [B,H,T,Dh] to match HF semantics.
        We keep [B,H,T,Dh] ordering because `apply_rotary_pos_emb` expects it
        and internally unsqueezes cos/sin along dim=1 (heads) for broadcasting.
        """
        B, H, T, actual_dh = q_bhtd.shape

        if actual_dh != self.head_dim:
            raise ValueError(
                f"Tensor head dimension {actual_dh} doesn't match expected head_dim {self.head_dim}"
            )

        # Use precomputed position_embeddings from HF model if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            # Produce cos/sin with either true positions or just seq_len, depending on availability
            cos = sin = None
            try:
                sig = inspect.signature(self.rotary_emb.forward)
                if "position_ids" in sig.parameters and position_ids is not None:
                    cos, sin = self.rotary_emb(q_bhtd, position_ids=position_ids)
                elif "seq_len" in sig.parameters:
                    cos, sin = self.rotary_emb(q_bhtd, seq_len=T)
                else:
                    cos, sin = self.rotary_emb(q_bhtd, T)  # positional arg fallback
            except TypeError:
                # Fallback for older/newer variations
                try:
                    cos, sin = self.rotary_emb(q_bhtd, position_ids=position_ids)
                except TypeError:
                    cos, sin = self.rotary_emb(q_bhtd, seq_len=T)

        # Ensure cos/sin last-dim matches head_dim
        if cos.shape[-1] != actual_dh:
            raise ValueError(
                f"RoPE cos/sin dimension {cos.shape[-1]} doesn't match tensor head_dim {actual_dh}"
            )

        # HF helper handles the broadcasting: cos/sin [B|1, T, Dh] -> unsqueeze at dim=1
        q_rot, k_rot = apply_rotary_pos_emb(q_bhtd, k_bhtd, cos, sin)
        return q_rot.contiguous(), k_rot.contiguous()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, position_embeddings=None, **_):
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

        # Q/K/V via SVD factors
        q = self._proj_per_head(x_trim, self.q_Us, self.q_V)         # [B,H,T,dh]
        k = self._proj_per_head(x_trim, self.k_Us, self.k_V)         # [B,Hk,T,dh]
        v = self._proj_per_head(x_trim, self.v_Us, self.v_V)         # [B,Hk,T,dh]

        # GQA repeat and RoPE with true positions
        k = _repeat_kv(k, self.n_rep)                                    # [B,H,T,dh]
        v = _repeat_kv(v, self.n_rep)
        
        q, k = self._apply_rope(q, k, pos_ids, position_embeddings=position_embeddings)  # [B,H,T,dh]

        # SDPA expects [B,H,S,D]
        q_bhsd = q.contiguous()
        k_bhsd = k.contiguous()
        v_bhsd = v.contiguous()

        # Build additive mask with causal + padding using helper
        bias = _build_full_bias(attention_mask, B, T_max, k_bhsd.size(-2), q_bhsd.device, q_bhsd.dtype)

        attn_sdpa = F.scaled_dot_product_attention(
            q_bhsd, k_bhsd, v_bhsd, attn_mask=bias, is_causal=False
        )  # [B,H,T,dh]

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

        # pad back to original T
        if T_max < T:
            pad = torch.zeros(B, T - T_max, D, dtype=out.dtype, device=out.device)
            out = torch.cat([out, pad], dim=1)
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
            self.block = SVDLlamaBlock(
                inner, cfg,
                rank_q=rank_q, rank_kv=rank_kv, rank_o=rank_o, rank_ff=rank_ff,
                factor_dtype=factor_dtype, compute_in_fp32=compute_in_fp32
            )
        def forward(self, hidden_states, attention_mask=None, position_ids=None, **kw):
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            y, = self.block(hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            **kw)
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
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
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
    SEQ_LEN    = int(os.getenv("SEQ_LEN", "512"))
    MAX_EVAL_SAMPLES = int(os.getenv("MAX_EVAL_SAMPLES", "64"))
    # SVD ranks (per-head for Q/K/V, whole-matrix for O/FF)
    RANK_Q  = int(os.getenv("RANK_Q",  "128"))
    RANK_KV = int(os.getenv("RANK_KV", "128"))
    RANK_O  = int(os.getenv("RANK_O",  "0")) or None          # 0 → dense, 4096 is enough for 7b
    RANK_FF = int(os.getenv("RANK_FF", "0")) or None          # 0 → dense, 4096 is enough for 7b
    SVD_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[os.getenv("SVD_DTYPE", "fp32").lower()]
    SVD_COMPUTE_FP32 = os.getenv("SVD_COMPUTE_FP32", "1") == "1"

    # Load model/tokenizer
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.config.use_cache = False  # full-seq
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Swap in SVD layers (plain PyTorch SDPA) and free any dense artifacts
    replace_with_svd(
        model, rank_q=RANK_Q, rank_kv=RANK_KV, rank_o=RANK_O, rank_ff=RANK_FF,
        factor_dtype=SVD_DTYPE, compute_in_fp32=SVD_COMPUTE_FP32
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    # Storage prior to eval (parameters, buffers, SVD factors)
    storage_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    kv_cache_est_mib = _estimate_kv_cache_mib(model.config, BATCH_SIZE, SEQ_LEN, dtype)
    ppl, peak_mem, time_ms = eval_perplexity_fullseq(model, loader, device)
    transient_mem = max(0.0, peak_mem - storage_mem)

    print("\n================== LLaMA + SVD (SDPA) ==================")
    print(f"Python {platform.python_version()}  Torch {torch.__version__}")
    print(f"Device/dtype: {device}/{dtype}")
    print(f"Ranks: q={RANK_Q}, kv={RANK_KV}, o={RANK_O}, ff={RANK_FF}  |  SVD store dtype={SVD_DTYPE}  |  SVD compute fp32={SVD_COMPUTE_FP32}")
    print(f"{'Model':<20} | {'Storage (MiB)':<14} | {'Peak (MiB)':<10} | {'Transient (MiB)':<14} | {'KV (MiB)':<10} | {'Time (ms/b)':<12} | {'Perplexity':<10}")
    print("-" * 100)
    print(f"{'LLaMA+SVD':<20} | {storage_mem:<14.1f} | {peak_mem:<10.1f} | {transient_mem:<14.1f} | {kv_cache_est_mib:<10.1f} | {time_ms:<12.1f} | {ppl:<10.4f}")


# CUDA_VISIBLE_DEVICES=3,4,5,6,7 SVD_DTYPE=fp16 SVD_COMPUTE_FP32=1 SVD_VERIFY=1 PRINT_SUMMARY=1 MAX_EVAL_SAMPLES=10 MAX_EVAL_BATCHES=5 CHUNK_SIZE=256 python profile_svd_llama.py