#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_svd_kv_accum.py
Per-head low-rank SVD factorization for GPT-2 with correctness checks, profiling,
dense KV cache, and a decoding-time memory/throughput growth benchmark.

HF >= 4.55 compatibility notes (critical fixes):
- DynamicCache.update(...) must receive ONLY the new step's KV of shape [B,H,S_new,dh],
  not the concatenated full cache. Returning/feeding concatenated KV causes
  cache_position indexing to overflow and triggers CUDA indexSelect asserts.
- LayerShim now converts layouts both ways:
    * Input cache -> [B,H,T,dh] for our block compute.
    * Block's new-step KV -> back to DynamicCache's native layout on update.
"""

import os, math, time, itertools, argparse
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2LMHeadModel, AutoTokenizer


from kernels.flash_attn_causal import flash_attn_triton_kvcache

# =========================
# Utils
# =========================
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_persistent_memory(m: nn.Module) -> float:
    total = 0
    for p in itertools.chain(m.parameters(), m.buffers()):
        total += p.numel() * p.element_size()
    return total / (1024**2)

def svd_factor(W: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if W.dtype not in (torch.float32, torch.float64):
        W = W.float()
    W = W.contiguous()
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except TypeError:
        U_, S_, V_ = torch.svd(W)
        U, S, Vh = U_, S_, V_.t()
    r = min(rank, S.numel())
    U_r = U[:, :r].contiguous()
    V_r = (S[:r, None] * Vh[:r, :]).contiguous()
    return U_r, V_r

def make_causal_slice_mask(s_new: int, total_len: int, device, dtype=torch.bool) -> torch.Tensor:
    full = torch.ones(total_len, total_len, dtype=dtype, device=device).tril_()
    return full[-s_new:, :].contiguous()

def as_linear_weight(W_raw: torch.Tensor, in_dim: int, out_dim: int) -> torch.Tensor:
    if W_raw.shape == (in_dim, out_dim):
        return W_raw.contiguous()
    if W_raw.shape == (out_dim, in_dim):
        return W_raw.t().contiguous()
    raise ValueError(f"Unexpected weight shape {tuple(W_raw.shape)}; expected ({in_dim},{out_dim}) or ({out_dim},{in_dim}).")


# =========================
# Low-rank GPT-2 Block with dense KV cache
# =========================
class LowRankSVDBlock(nn.Module):
    def __init__(
        self,
        hf_layer: nn.Module,
        rank_ratio_attn: float = 1.0,
        rank_ratio_mlp: float = 1.0,
        preload_factors: Optional[Dict[str, torch.Tensor]] = None,
        save_factors_to: Optional[str] = None,
    ):
        super().__init__()
        attn = hf_layer.attn
        self.ln1 = hf_layer.ln_1
        self.ln2 = hf_layer.ln_2

        D = attn.embed_dim
        H = attn.num_heads
        if D % H != 0:
            raise ValueError(f"[LowRankSVDBlock] embed_dim={D} not divisible by heads={H}")
        dh = D // H

        self.D, self.H, self.dh = D, H, dh
        self.scale = 1.0 / math.sqrt(dh)

        dev = next(hf_layer.parameters()).device
        ptdtype = next(hf_layer.parameters()).dtype

        # ---------- ATTENTION (Q,K,V) ----------
        Wc_lin = as_linear_weight(hf_layer.attn.c_attn.weight.data, in_dim=D, out_dim=3 * D)  # [D,3D]
        bc = hf_layer.attn.c_attn.bias.data.clone().to(device=dev, dtype=ptdtype)

        q_w = Wc_lin[:, :D].contiguous().view(D, H, dh)
        k_w = Wc_lin[:, D:2*D].contiguous().view(D, H, dh)
        v_w = Wc_lin[:, 2*D:3*D].contiguous().view(D, H, dh)

        q_b = bc[:D].view(H, dh).contiguous()
        k_b = bc[D:2*D].view(H, dh).contiguous()
        v_b = bc[2*D:3*D].view(H, dh).contiguous()

        r_attn = max(1, int(rank_ratio_attn * min(D, dh)))

        def alloc_uv(name: str):
            U = nn.Parameter(torch.empty(D, H, r_attn, device=dev, dtype=ptdtype))
            V = nn.Parameter(torch.empty(H, r_attn, dh, device=dev, dtype=ptdtype))
            self.register_parameter(f"{name}_U", U)
            self.register_parameter(f"{name}_V", V)
            return U, V

        self.q_U, self.q_V = alloc_uv("q")
        self.k_U, self.k_V = alloc_uv("k")
        self.v_U, self.v_V = alloc_uv("v")

        self.q_b = nn.Parameter(q_b.to(device=dev, dtype=ptdtype))
        self.k_b = nn.Parameter(k_b.to(device=dev, dtype=ptdtype))
        self.v_b = nn.Parameter(v_b.to(device=dev, dtype=ptdtype))

        if preload_factors is None:
            with torch.no_grad():
                for name, W_h in (("q", q_w), ("k", k_w), ("v", v_w)):
                    U_param = getattr(self, f"{name}_U")
                    V_param = getattr(self, f"{name}_V")
                    Us, Vs = [], []
                    for h in range(H):
                        Wh = W_h[:, h, :]                # [D, dh]
                        Uh, Vh = svd_factor(Wh, r_attn)  # [D,r], [r,dh]
                        Us.append(Uh.to(device=dev, dtype=ptdtype))
                        Vs.append(Vh.to(device=dev, dtype=ptdtype))
                    U = torch.stack(Us, dim=1)           # [D,H,r]
                    V = torch.stack(Vs, dim=0)           # [H,r,dh]
                    U_param.copy_(U)
                    V_param.copy_(V)
        else:
            self.load_factors_(preload_factors)

        # ---------- OUT PROJ ----------
        W_out_lin = as_linear_weight(hf_layer.attn.c_proj.weight.data, in_dim=D, out_dim=D)
        b_out = hf_layer.attn.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_out = max(1, int(rank_ratio_attn * min(W_out_lin.shape)))
        Uo, Vo = svd_factor(W_out_lin, r_out)
        self.out_U = nn.Parameter(Uo.to(device=dev, dtype=ptdtype))
        self.out_V = nn.Parameter(Vo.to(device=dev, dtype=ptdtype))
        self.out_b = nn.Parameter(b_out)

        # ---------- MLP ----------
        I = hf_layer.mlp.c_fc.bias.data.numel()

        W1_lin = as_linear_weight(hf_layer.mlp.c_fc.weight.data, in_dim=D, out_dim=I)
        b_fc1 = hf_layer.mlp.c_fc.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_fc1 = max(1, int(rank_ratio_mlp * min(W1_lin.shape)))
        U1, V1 = svd_factor(W1_lin, r_fc1)
        self.fc1_U = nn.Parameter(U1.to(device=dev, dtype=ptdtype))
        self.fc1_V = nn.Parameter(V1.to(device=dev, dtype=ptdtype))
        self.fc1_b = nn.Parameter(b_fc1)

        W2_lin = as_linear_weight(hf_layer.mlp.c_proj.weight.data, in_dim=I, out_dim=D)
        b_fc2 = hf_layer.mlp.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_fc2 = max(1, int(rank_ratio_mlp * min(W2_lin.shape)))
        U2, V2 = svd_factor(W2_lin, r_fc2)
        self.fc2_U = nn.Parameter(U2.to(device=dev, dtype=ptdtype))
        self.fc2_V = nn.Parameter(V2.to(device=dev, dtype=ptdtype))
        self.fc2_b = nn.Parameter(b_fc2)

        # Keep ranks for logs
        self.r_attn = r_attn
        self.r_out  = self.out_V.shape[0]
        self.r_fc1  = self.fc1_V.shape[0]
        self.r_fc2  = self.fc2_V.shape[0]

    def factors_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "q_U": self.q_U, "q_V": self.q_V, "q_b": self.q_b,
            "k_U": self.k_U, "k_V": self.k_V, "k_b": self.k_b,
            "v_U": self.v_U, "v_V": self.v_V, "v_b": self.v_b,
            "out_U": self.out_U, "out_V": self.out_V, "out_b": self.out_b,
            "fc1_U": self.fc1_U, "fc1_V": self.fc1_V, "fc1_b": self.fc1_b,
            "fc2_U": self.fc2_U, "fc2_V": self.fc2_V, "fc2_b": self.fc2_b,
        }

    def load_factors_(self, tensors: Dict[str, torch.Tensor]):
        mine = self.factors_state_dict()
        for k, p in mine.items():
            if k not in tensors:
                raise KeyError(f"Missing factor '{k}' in preload_factors")
            with torch.no_grad():
                p.copy_(tensors[k].to(dtype=p.dtype, device=p.device))

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # [B,H,T_past,dh]
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        B, S, D = hidden_states.shape
        dev = hidden_states.device
        neg_inf = torch.finfo(hidden_states.dtype).min

        x = self.ln1(hidden_states)  # [B,S,D]

        Q = torch.einsum('bsd,dhr,hre->bhse', x, self.q_U, self.q_V) + self.q_b[None, :, None, :]
        K = torch.einsum('bsd,dhr,hre->bhse', x, self.k_U, self.k_V) + self.k_b[None, :, None, :]
        V = torch.einsum('bsd,dhr,hre->bhse', x, self.v_U, self.v_V) + self.v_b[None, :, None, :]

        # Concatenate with past if provided (expects [B,H,T_past,dh])
        past_len = 0
        if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2 and layer_past[0] is not None:
            past_k, past_v = layer_past
            if past_k.dtype != K.dtype: past_k = past_k.to(dtype=K.dtype)
            if past_v.dtype != V.dtype: past_v = past_v.to(dtype=V.dtype)
            if past_k.device != K.device: past_k = past_k.to(K.device)
            if past_v.device != V.device: past_v = past_v.to(V.device)

            # Sanity checks
            assert past_k.dim() == 4 and past_v.dim() == 4, "past K/V must be 4D"
            assert past_k.shape[:2] == (B, self.H) and past_v.shape[:2] == (B, self.H), \
                f"Expected past [B,H,*,dh], got K {tuple(past_k.shape)} V {tuple(past_v.shape)}"
            assert past_k.shape[-1] == self.dh and past_v.shape[-1] == self.dh, "Head dim mismatch in past cache"

            K_cat = torch.cat([past_k, K], dim=2)  # concat on time axis (for compute only)
            V_cat = torch.cat([past_v, V], dim=2)
            past_len = past_k.size(2)
        else:
            K_cat, V_cat = K, V

        total_len = past_len + S

        #attn_scores = torch.matmul(Q, K_cat.transpose(-2, -1)) * self.scale  # [B,H,S,total_len]

        causal = make_causal_slice_mask(S, total_len, device=dev, dtype=torch.bool)
        attn_scores = flash_attn_triton_kvcache(Q, K_cat, V_cat, causal) #attn_scores.masked_fill(~causal[None, None, :, :], neg_inf)

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                am = attention_mask[..., -total_len:]
                if am.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                    attn_scores = attn_scores + am.to(dtype=attn_scores.dtype)
                else:
                    key_keep = am.bool()
                    attn_scores = attn_scores.masked_fill(~key_keep, neg_inf)
            elif attention_mask.dim() == 2:
                if attention_mask.size(-1) == total_len:
                    key_keep = attention_mask[:, None, None, :].bool()
                elif attention_mask.size(-1) == S:
                    pad = torch.ones(B, past_len, dtype=attention_mask.dtype, device=dev)
                    key_keep = torch.cat([pad, attention_mask], dim=-1)[:, None, None, :].bool()
                else:
                    key_keep = torch.ones(B, 1, 1, total_len, dtype=torch.bool, device=dev)
                attn_scores = attn_scores.masked_fill(~key_keep, neg_inf)

        attn_probs = F.softmax(attn_scores, dim=-1)
        Y = torch.matmul(attn_probs, V_cat)          # [B,H,S,dh]
        Y = Y.transpose(1, 2).contiguous().view(B, S, self.D)  # [B,S,D]

        Y = torch.matmul(torch.matmul(Y, self.out_U), self.out_V) + self.out_b
        hidden_states = hidden_states + Y

        z = self.ln2(hidden_states)
        t1 = torch.matmul(z, self.fc1_U)
        h1 = torch.matmul(t1, self.fc1_V) + self.fc1_b
        h1 = F.gelu(h1)
        t2 = torch.matmul(h1, self.fc2_U)
        h2 = torch.matmul(t2, self.fc2_V) + self.fc2_b
        hidden_states = hidden_states + h2

        outputs = (hidden_states,)

        if use_cache:
            # IMPORTANT: return ONLY the *new* KV for DynamicCache.update(...)
            outputs = outputs + ((K, V),)

        if output_attentions:
            outputs = outputs + (attn_probs,)

        return outputs


# =========================
# Cache layout helpers & shim
# =========================
def _to_legacy_kv(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, (tuple, list)):
        return past_key_values
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            return past_key_values.to_legacy_cache()
        except Exception:
            return None
    return None

def _ensure_bhtd(k: torch.Tensor, v: torch.Tensor, H: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure K,V are [B, H, T, dh].
    Accepts [B, H, T, dh] (noop) or [B, T, H, dh] (permute).
    """
    assert k.dim() == 4 and v.dim() == 4, "Cache tensors must be 4D"
    if k.size(1) == H:  # [B,H,T,dh]
        return k, v
    if k.size(2) == H:  # [B,T,H,dh] -> [B,H,T,dh]
        return k.permute(0, 2, 1, 3).contiguous(), v.permute(0, 2, 1, 3).contiguous()
    raise RuntimeError(f"Unrecognized cache layout for shapes K={tuple(k.shape)} V={tuple(v.shape)} (H={H})")

def _from_bhtd_to_cache_layout(k_bhtd: torch.Tensor, v_bhtd: torch.Tensor, expect_bthd: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert [B,H,T,dh] (our internal) back to either:
      - [B,H,T,dh] if expect_bthd is False
      - [B,T,H,dh] if expect_bthd is True
    """
    if expect_bthd:
        return k_bhtd.permute(0, 2, 1, 3).contiguous(), v_bhtd.permute(0, 2, 1, 3).contiguous()
    return k_bhtd, v_bhtd

class LayerShim(nn.Module):
    def __init__(self, block: nn.Module, layer_idx: int = None):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx

    def forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, *args, **kwargs):
        """
        - Extract layer cache from DynamicCache if present.
        - Normalize to [B,H,T,dh] before calling the block.
        - After forward, convert ONLY the new-step KV back to the DynamicCache's expected
          layout and update it.
        """
        layer_past = None
        expect_bthd = False  # whether DynamicCache stores [B,T,H,dh]

        if past_key_value is not None and self.layer_idx is not None:
            # DynamicCache path
            if hasattr(past_key_value, "get_seq_length"):
                try:
                    seq_len = past_key_value.get_seq_length(self.layer_idx)
                except Exception:
                    seq_len = 0
                if seq_len and hasattr(past_key_value, "layers") and len(past_key_value.layers) > self.layer_idx:
                    layer_cache = past_key_value.layers[self.layer_idx]
                    k_cache = getattr(layer_cache, "keys", None)
                    v_cache = getattr(layer_cache, "values", None)
                    if k_cache is not None and v_cache is not None:
                        # Detect layout
                        if k_cache.dim() == 4:
                            expect_bthd = (k_cache.size(2) == self.block.H)  # True => [B,T,H,dh]
                            k_std, v_std = _ensure_bhtd(k_cache, v_cache, self.block.H)
                            layer_past = (k_std, v_std)
                # fallback for older cache attributes
                elif seq_len and hasattr(past_key_value, "key_cache"):
                    k_cache = past_key_value.key_cache[self.layer_idx]
                    v_cache = past_key_value.value_cache[self.layer_idx]
                    if k_cache is not None and v_cache is not None:
                        expect_bthd = (k_cache.size(2) == self.block.H)
                        k_std, v_std = _ensure_bhtd(k_cache, v_cache, self.block.H)
                        layer_past = (k_std, v_std)

            # Legacy tuple path (already [B,H,T,dh] in our code)
            elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                layer_past = past_key_value

        # Forward through our block (propagate use_cache flag in kwargs)
        result = self.block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            **kwargs,
        )

        # If DynamicCache present and we produced *new-step* present, update it
        if (past_key_value is not None and
            hasattr(past_key_value, "update") and
            self.layer_idx is not None and
            isinstance(result, tuple) and len(result) >= 2 and
            isinstance(result[1], tuple) and len(result[1]) == 2):

            k_new_bhtd, v_new_bhtd = result[1]  # [B,H,S_new,dh]  (NEW ONLY)
            # Convert to DynamicCache's layout
            k_upd, v_upd = _from_bhtd_to_cache_layout(k_new_bhtd, v_new_bhtd, expect_bthd)
            past_key_value.update(k_upd, v_upd, self.layer_idx)

        return result


# =========================
# Builders & Validators
# =========================
def build_svd_model(
    rank_ratio_attn: float,
    rank_ratio_mlp: float,
    save_factors_dir: Optional[str] = None,
    load_factors_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    if device:
        model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    for i, layer in enumerate(model.transformer.h):
        preload = None
        save_path = None
        if load_factors_dir is not None:
            fp = os.path.join(load_factors_dir, f"gpt2_block_{i}.pt")
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"Missing factors for block {i}: {fp}")
            preload = torch.load(fp, map_location="cpu")
        elif save_factors_dir is not None:
            save_path = os.path.join(save_factors_dir, f"gpt2_block_{i}.pt")

        blk = LowRankSVDBlock(
            layer,
            rank_ratio_attn=rank_ratio_attn,
            rank_ratio_mlp=rank_ratio_mlp,
            preload_factors=preload,
            save_factors_to=save_path,
        )
        shim = LayerShim(blk, layer_idx=i).to(device if device is not None else next(model.parameters()).device)
        model.transformer.h[i] = shim

    return model


@torch.no_grad()
def compare_qkv_intermediate(dense_block: nn.Module, svd_block: LowRankSVDBlock, x: torch.Tensor):
    D, H, dh = svd_block.D, svd_block.H, svd_block.dh
    Wc_lin = as_linear_weight(dense_block.attn.c_attn.weight.data, in_dim=D, out_dim=3 * D)
    bc = dense_block.attn.c_attn.bias.data.to(device=x.device, dtype=x.dtype)
    qkv = x @ Wc_lin + bc  # [B,S,3D]
    q, k, v = qkv.split(D, dim=-1)
    q = q.view(x.shape[0], x.shape[1], H, dh).permute(0, 2, 1, 3).contiguous()
    k = k.view(x.shape[0], x.shape[1], H, dh).permute(0, 2, 1, 3).contiguous()
    v = v.view(x.shape[0], x.shape[1], H, dh).permute(0, 2, 1, 3).contiguous()

    Q = torch.einsum('bsd,dhr,hre->bhse', x, svd_block.q_U, svd_block.q_V) + svd_block.q_b[None, :, None, :]
    K = torch.einsum('bsd,dhr,hre->bhse', x, svd_block.k_U, svd_block.k_V) + svd_block.k_b[None, :, None, :]
    V = torch.einsum('bsd,dhr,hre->bhse', x, svd_block.v_U, svd_block.v_V) + svd_block.v_b[None, :, None, :]

    def stats(name, A, B):
        md = (A - B).abs().max().item()
        rd = (A - B).norm() / (A.norm() + 1e-12)
        print(f"{name:>3}  max|Δ|={md:.8f}  rel={rd:.8f}")

    print("=== QKV intermediate comparison ===")
    stats("Q", Q, q)
    stats("K", K, k)
    stats("V", V, v)


@torch.no_grad()
def end_to_end_validation(dense: GPT2LMHeadModel, svd: GPT2LMHeadModel, device: str, use_cache: bool = False):
    test_input_ids = torch.randint(0, 1000, (2, 16), device=device)
    attn = torch.ones_like(test_input_ids, device=device)

    o1 = dense(input_ids=test_input_ids, attention_mask=attn, use_cache=use_cache).logits
    o2 = svd(input_ids=test_input_ids, attention_mask=attn, use_cache=use_cache).logits

    max_diff = (o1 - o2).abs().max().item()
    rel_diff = (o1 - o2).norm() / (o1.norm() + 1e-12)
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"Relative difference:     {rel_diff:.8f}")
    ok = max_diff < 1e-1
    print("✓ Validation PASSED" if ok else "✗ Validation FAILED")
    return max_diff, rel_diff


# =========================
# Perplexity + Memory + Time (evaluation mode)
# =========================
@torch.no_grad()
def perplexity_peak_time(mdl: GPT2LMHeadModel, loader, device: str, use_mask: bool = True):
    mdl.eval()
    total_loss, total_tokens = 0.0, 0
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        chunk = 4
        B = batch["input_ids"].size(0)
        for i in range(0, B, chunk):
            sl = slice(i, min(i + chunk, B))
            ids = batch["input_ids"][sl]
            mask = batch["attention_mask"][sl]

            if use_mask:
                out = mdl(input_ids=ids, attention_mask=mask, use_cache=False)
            else:
                out = mdl(input_ids=ids, use_cache=False)

            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = ids[..., 1:].contiguous()

            if use_mask:
                m = mask[..., 1:].contiguous().bool()
                if m.any():
                    valid_logits = shift_logits[m]
                    valid_labels = shift_labels[m]
                    loss = F.cross_entropy(valid_logits, valid_labels)
                    total_loss += loss.item() * m.sum().item()
                    total_tokens += m.sum().item()
                    del valid_logits, valid_labels, m
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                total_loss += loss.item() * shift_labels.numel()
                total_tokens += shift_labels.numel()

            del out, shift_logits, shift_labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per_batch = (time.perf_counter() - start) * 1000.0 / len(loader)
    peak = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if total_tokens > 0 else float("inf")
    return ppl, peak, ms_per_batch


# =========================
# Decoding-time KV-cache growth benchmark
# =========================
@torch.no_grad()
def estimate_kv_bytes(past_key_values) -> int:
    pkv = _to_legacy_kv(past_key_values)
    if pkv is None:
        return 0
    total = 0
    for layer_kv in pkv:
        if not isinstance(layer_kv, (tuple, list)) or len(layer_kv) != 2 or layer_kv[0] is None:
            continue
        k, v = layer_kv
        total += k.numel() * k.element_size()
        total += v.numel() * v.element_size()
    return total

@torch.no_grad()
def decode_once_with_cache(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    new_tokens: int,
    device: str,
    greedy: bool = True,
) -> Dict[str, float]:
    model.eval()
    B = input_ids.size(0)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values  # DynamicCache
    logits = out.logits
    generated = input_ids

    for _ in range(new_tokens):
        next_id = logits[:, -1, :].argmax(-1, keepdim=True) if greedy else torch.multinomial(F.softmax(logits[:, -1, :], -1), 1)
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        logits = out.logits
        past = out.past_key_values
        generated = torch.cat([generated, next_id], dim=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elap = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    kv_bytes = estimate_kv_bytes(past)
    kv_mib = kv_bytes / (1024**2)
    toks_per_s = (B * max(new_tokens, 1)) / max(elap, 1e-6)

    return {
        "peak_alloc_MiB": peak_alloc,
        "est_KV_MiB": kv_mib,
        "toks_per_s": toks_per_s,
        "total_tokens": int(generated.size(1)),
    }

@torch.no_grad()
def decode_growth_curve(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int,
    prompt_len: int,
    curve_lens: List[int],
    label: str,
):
    print(f"\n=== Decoding-time KV-cache growth ({label}) ===")
    vocab = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 50257
    safe_hi = min(1000, vocab)
    prompt = torch.randint(0, safe_hi, (batch_size, prompt_len), device=device)

    header = f"{'new_T':>8} | {'peak_alloc(MiB)':>16} | {'est_KV(MiB)':>12} | {'toks/s':>10} | {'total_T':>8}"
    print(header)
    print("-" * len(header))
    for new_T in curve_lens:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        metrics = decode_once_with_cache(model, prompt, new_T, device)
        print(f"{new_T:8d} | {metrics['peak_alloc_MiB']:16.1f} | {metrics['est_KV_MiB']:12.1f} | {metrics['toks_per_s']:10.2f} | {metrics['total_tokens']:8d}")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-ratio-attn", type=float, default=1.0)
    parser.add_argument("--rank-ratio-mlp",  type=float, default=1.0)
    parser.add_argument("--save-factors-dir", type=str, default=None)
    parser.add_argument("--load-factors-dir", type=str, default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--debug-attn", action="store_true")
    parser.add_argument("--validate-cache", action="store_true", help="Also validate with use_cache=True")

    parser.add_argument("--decode-mem", action="store_true", help="Run decoding-time KV-cache growth benchmark")
    parser.add_argument("--decode-curve", type=str, default="64,128,256,512", help="Comma-separated new-token lengths")
    parser.add_argument("--decode-batch", type=int, default=1, help="Batch size for decoding benchmark")
    parser.add_argument("--prompt-len", type=int, default=32, help="Prompt length for decoding benchmark")
    parser.add_argument("--compare-dense", action="store_true", help="Run dense baseline in decoding benchmark too")

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dense = GPT2LMHeadModel.from_pretrained("gpt2")
    dense = dense.to(device).eval()
    for p in dense.parameters():
        p.requires_grad = False
    dense_mem = compute_persistent_memory(dense)
    print(f"Dense model storage: {dense_mem:6.1f} MiB")

    print("\n=== Building SVD Model ===")
    svd_model = build_svd_model(
        rank_ratio_attn=args.rank_ratio_attn,
        rank_ratio_mlp=args.rank_ratio_mlp,
        save_factors_dir=args.save_factors_dir,
        load_factors_dir=args.load_factors_dir,
        device=device,
    )
    for p in svd_model.parameters():
        p.requires_grad = False
    print(f"SVD model built with per-head rank≈{args.rank_ratio_attn}*min(D,dh) and MLP ranks≈{args.rank_ratio_mlp}*...")

    first_blk = svd_model.transformer.h[0].block
    print(f"QKV rank: {first_blk.r_attn}, Out rank: {first_blk.r_out}")
    print(f"FC1 rank: {first_blk.r_fc1}, FC2 rank: {first_blk.r_fc2}")

    if args.debug_attn:
        blk0 = svd_model.transformer.h[0].block
        print(f"[debug-attn] D={blk0.D}, H={blk0.H}, dh={blk0.dh}")

    layer0 = dense.transformer.h[0]
    Wc = layer0.attn.c_attn.weight
    bc = layer0.attn.c_attn.bias
    print("weight", tuple(Wc.shape), "bias", tuple(bc.shape),
          "embed_dim", layer0.attn.embed_dim, "heads", layer0.attn.num_heads)

    if args.validate:
        print("\n=== SVD Validation ===")
        end_to_end_validation(dense, svd_model, device=device, use_cache=False)
        if args.validate_cache:
            print("\n--- With KV Cache ---")
            end_to_end_validation(dense, svd_model, device=device, use_cache=True)

        print("\n--- Block-0 Q/K/V check ---")
        with torch.no_grad():
            test_ids = torch.randint(0, 1000, (2, 16), device=device)
            wte = dense.transformer.wte(test_ids)
            pos = torch.arange(test_ids.size(1), device=device)[None, :]
            wpe = dense.transformer.wpe(pos)
            h0_in = dense.transformer.drop(wte + wpe)
            x0 = dense.transformer.h[0].ln_1(h0_in)
            compare_qkv_intermediate(dense.transformer.h[0], svd_model.transformer.h[0].block, x0)
        print("=== End Validation ===\n")

    if not args.decode_mem:
        print("Preparing Wikitext-2 (test split)...")
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token

        def tok_fn(batch):
            return tok(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
            )
        ds = raw.map(tok_fn, batched=True, remove_columns=["text"])
        ds.set_format("torch")
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: {
                "input_ids": torch.stack([x["input_ids"] for x in b]),
                "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            },
        )

        print("\n=== Dense baseline ===")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ppl_m, peak_m, t_m = perplexity_peak_time(dense, loader, device, use_mask=True)
        print(f"Dense w/ mask   | ppl={ppl_m:.4f} | peak={peak_m:7.1f} MiB | {t_m:6.1f} ms/b")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ppl_nm, peak_nm, t_nm = perplexity_peak_time(dense, loader, device, use_mask=False)
        print(f"Dense w/o mask  | ppl={ppl_nm:.4f} | peak={peak_nm:7.1f} MiB | {t_nm:6.1f} ms/b")

        print("\n=== SVD model ===")
        svd_mem = compute_persistent_memory(svd_model)
        print(f"SVD model storage: {svd_mem:6.1f} MiB "
              f"(saving {dense_mem - svd_mem:+.1f} MiB, {100*(dense_mem - svd_mem)/max(dense_mem,1e-9):.1f}%)")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ppl_m_s, peak_m_s, t_m_s = perplexity_peak_time(svd_model, loader, device, use_mask=True)
        print(f"SVD   w/ mask   | ppl={ppl_m_s:.4f} | peak={peak_m_s:7.1f} MiB | {t_m_s:6.1f} ms/b")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ppl_nm_s, peak_nm_s, t_nm_s = perplexity_peak_time(svd_model, loader, device, use_mask=False)
        print(f"SVD   w/o mask  | ppl={ppl_nm_s:.4f} | peak={peak_nm_s:7.1f} MiB | {t_nm_s:6.1f} ms/b")

        print("\n=== Performance Summary ===")
        print(f"Storage (MiB): Dense={dense_mem:.1f} | SVD={svd_mem:.1f} | Δ={dense_mem - svd_mem:+.1f} ({100*(dense_mem - svd_mem)/max(dense_mem,1e-9):.1f}%)")
        print(f"Perplexity: dense w/={ppl_m:.4f} w/o={ppl_nm:.4f} | svd w/={ppl_m_s:.4f} w/o={ppl_nm_s:.4f}")
        print(f"Peak (MiB): dense w/={peak_m:7.1f} w/o={peak_nm:7.1f} | svd w/={peak_m_s:7.1f} w/o={peak_nm_s:7.1f}")
        print(f"Latency (ms/batch): dense w/={t_m:6.1f} w/o={t_nm:6.1f} | svd w/={t_m_s:6.1f} w/o={t_nm_s:6.1f}")

    else:
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        curve = [int(x) for x in args.decode_curve.split(",") if x.strip()]
        bsz = args.decode_batch
        p_len = args.prompt_len

        decode_growth_curve(
            svd_model, tok, device=device,
            batch_size=bsz, prompt_len=p_len, curve_lens=curve, label="SVD"
        )

        if args.compare_dense:
            decode_growth_curve(
                dense, tok, device=device,
                batch_size=bsz, prompt_len=p_len, curve_lens=curve, label="Dense"
            )


if __name__ == "__main__":
    main()

# Example:
# CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python3 profile_svd_kv_accum.py --decode-mem --compare-dense --decode-batch 2 --prompt-len 64 --decode-curve 128,256 --rank-ratio-attn 0.5 --rank-ratio-mlp 0.5
