#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_asvd_accum_flashsvd.py — minimal FlashSVD+ASVD decode mem benchmark (clean)

Key changes vs your debug build:
- No profiler / debug toggles / local alloc taps.
- Prefill computes ONLY last-token logits (no [B,S,V] logits tensor).
- Careful freeing between prefill and decode (reset_peak_memory_stats after freeing).

CUDA_VISIBLE_DEVICES=5 python3 profile_asvd_accum_flashsvd.py   --decode-mem   --decode-batch 16   --prompt-len 256   --decode-curve 128   --rounds 1   --rank-ratio-attn 1   --rank-ratio-mlp 1

"""

import math, time, argparse, gc
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, AutoTokenizer

# Your kernels
from kernels.flash_attn_causal import flash_attn_triton_kvcache
from kernels.flashsvdattn import flash_svd_attention
from kernels.flashsvdffn import flashsvd_ffn

MiB = float(1024**2)

# -------------------- utils --------------------
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def svd_factor(W: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return U_r, V_r such that W ≈ U_r @ V_r."""
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

def as_linear_weight(W_raw: torch.Tensor, in_dim: int, out_dim: int) -> torch.Tensor:
    """Return weight in shape (in_dim, out_dim)."""
    if W_raw.shape == (in_dim, out_dim):
        return W_raw.contiguous()
    if W_raw.shape == (out_dim, in_dim):
        return W_raw.t().contiguous()
    raise ValueError(f"Unexpected weight shape {tuple(W_raw.shape)}; expected ({in_dim},{out_dim}) or ({out_dim},{in_dim}).")

# -------------------- ASVD cache --------------------
class ASVDCache:
    """Holds per-layer low-rank factors (Pk, Pv) with shape [B,H,T,r]."""
    def __init__(self, n_layers: int):
        self.layers = [None] * n_layers  # type: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
    def get_seq_length(self, layer_idx: int) -> int:
        e = self.layers[layer_idx]
        return 0 if e is None else e[0].size(2)
    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.layers[layer_idx]
    @torch.no_grad()
    def update(self, Pk_new: torch.Tensor, Pv_new: torch.Tensor, layer_idx: int):
        e = self.layers[layer_idx]
        if e is None:
            self.layers[layer_idx] = (Pk_new, Pv_new)
        else:
            Pk, Pv = e
            self.layers[layer_idx] = (torch.cat([Pk, Pk_new], dim=2),
                                      torch.cat([Pv, Pv_new], dim=2))

# -------------------- FlashSVD block --------------------
class LowRankSVDBlock(nn.Module):
    """
    Prefill: FlashSVD attention with rank r; low-rank out-proj; FlashSVD FFN.
    Decode:  Reconstruct dense K,V from ASVD cache + FlashAttention; out-proj; FFN.
    """
    def __init__(self, hf_layer: nn.Module, rank_ratio_attn: float = 1.0, rank_ratio_mlp: float = 1.0):
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
        dev = next(hf_layer.parameters()).device
        ptdtype = next(hf_layer.parameters()).dtype

        # --- decompose Q,K,V ---
        Wc = as_linear_weight(attn.c_attn.weight.data, in_dim=D, out_dim=3*D)
        bc = attn.c_attn.bias.data.clone().to(device=dev, dtype=ptdtype)
        q_w = Wc[:, :D].contiguous().view(D, H, dh)
        k_w = Wc[:, D:2*D].contiguous().view(D, H, dh)
        v_w = Wc[:, 2*D:3*D].contiguous().view(D, H, dh)
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

        with torch.no_grad():
            for name, W_h in (("q", q_w), ("k", k_w), ("v", v_w)):
                U_param = getattr(self, f"{name}_U")
                V_param = getattr(self, f"{name}_V")
                Us, Vs = [], []
                for h in range(H):
                    Uh, Vh = svd_factor(W_h[:, h, :], r_attn)
                    Us.append(Uh.to(device=dev, dtype=ptdtype))
                    Vs.append(Vh.to(device=dev, dtype=ptdtype))
                U = torch.stack(Us, dim=1)
                V = torch.stack(Vs, dim=0)
                U_param.copy_(U); V_param.copy_(V)

        # --- out-proj low-rank ---
        W_out = as_linear_weight(attn.c_proj.weight.data, in_dim=D, out_dim=D)
        b_out = attn.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_out = max(1, int(rank_ratio_attn * min(W_out.shape)))
        Uo, Vo = svd_factor(W_out, r_out)
        self.out_U = nn.Parameter(Uo.to(device=dev, dtype=ptdtype))
        self.out_V = nn.Parameter(Vo.to(device=dev, dtype=ptdtype))
        self.out_b = nn.Parameter(b_out)

        # --- FFN low-rank ---
        I = hf_layer.mlp.c_fc.bias.data.numel()
        W1 = as_linear_weight(hf_layer.mlp.c_fc.weight.data, in_dim=D, out_dim=I)
        b1 = hf_layer.mlp.c_fc.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_fc1 = max(1, int(rank_ratio_mlp * min(W1.shape)))
        U1, V1 = svd_factor(W1, r_fc1)
        self.fc1_U = nn.Parameter(U1.to(device=dev, dtype=ptdtype))
        self.fc1_V = nn.Parameter(V1.to(device=dev, dtype=ptdtype))
        self.fc1_b = nn.Parameter(b1)

        W2 = as_linear_weight(hf_layer.mlp.c_proj.weight.data, in_dim=I, out_dim=D)
        b2 = hf_layer.mlp.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_fc2 = max(1, int(rank_ratio_mlp * min(W2.shape)))
        U2, V2 = svd_factor(W2, r_fc2)
        self.fc2_U = nn.Parameter(U2.to(device=dev, dtype=ptdtype))
        self.fc2_V = nn.Parameter(V2.to(device=dev, dtype=ptdtype))
        self.fc2_b = nn.Parameter(b2)

        self.r_attn = r_attn

    def forward(self, hidden_states: torch.Tensor, layer_past=None, attention_mask=None,
                use_cache: bool = False, output_attentions: bool = False, **kwargs):
        B, S, D = hidden_states.size()
        dev = hidden_states.device
        H, dh, r = self.H, self.dh, self.r_attn

        x = self.ln1(hidden_states)

        # Pack Q,K,V in low rank
        Pq = torch.einsum('bsd,dhr->bhsr', x, self.q_U)
        Vq = self.q_V.unsqueeze(0).expand(B, H, r, dh)
        bq = self.q_b.unsqueeze(0).expand(B, H, dh)

        Pk_new = torch.einsum('bsd,dhr->bhsr', x, self.k_U)
        Pv_new = torch.einsum('bsd,dhr->bhsr', x, self.v_U)

        # Compact mask [B,H,1,S]
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                q_mask = attention_mask[..., -S:].to(dtype=torch.bool)
                if q_mask.size(2) != 1: q_mask = q_mask[..., :1, :]
            elif attention_mask.dim() == 2:
                q_mask = attention_mask[:, -S:].bool()[:, None, None, :]
            else:
                q_mask = torch.ones(B, 1, 1, S, dtype=torch.bool, device=dev)
        else:
            q_mask = torch.ones(B, 1, 1, S, dtype=torch.bool, device=dev)
        attn_mask_bh1s = q_mask.expand(B, H, 1, S).contiguous()
        del q_mask

        has_past = (isinstance(layer_past, (tuple, list)) and len(layer_past) == 2 and
                    layer_past[0] is not None and layer_past[0].dim() == 4 and layer_past[0].size(-1) == r)

        # Attention
        if not has_past:
            if False: pass  # (kept structure minimal)
            Vk = self.k_V.unsqueeze(0).expand(B, H, r, dh); bk = self.k_b.unsqueeze(0).expand(B, H, dh)
            Vv = self.v_V.unsqueeze(0).expand(B, H, r, dh); bv = self.v_b.unsqueeze(0).expand(B, H, dh)
            Y_heads = flash_svd_attention(Pq, Vq, bq, Pk_new, Vk, bk, Pv_new, Vv, bv,
                                          mask=attn_mask_bh1s, block_r=r)
        else:
            Pk_past, Pv_past = layer_past
            Pk_cat = torch.cat([Pk_past.to(Pk_new.dtype), Pk_new], dim=2)
            Pv_cat = torch.cat([Pv_past.to(Pv_new.dtype), Pv_new], dim=2)
            K_cat = torch.einsum('bhtR,hRd->bhtd', Pk_cat, self.k_V) + self.k_b[None, :, None, :]
            V_cat = torch.einsum('bhtR,hRd->bhtd', Pv_cat, self.v_V) + self.v_b[None, :, None, :]
            del Pk_cat, Pv_cat
            Q = torch.einsum('bsd,dhr,hre->bhse', x, self.q_U, self.q_V) + self.q_b[None, :, None, :]
            Y_heads = flash_attn_triton_kvcache(Q, K_cat, V_cat, attn_mask_bh1s)
            del Q, K_cat, V_cat

        del attn_mask_bh1s, Vq, bq

        # Merge heads
        Y = Y_heads.transpose(1, 2).contiguous().view(B, S, self.D); del Y_heads

        # Output projection (low-rank)
        Y = torch.matmul(torch.matmul(Y, self.out_U), self.out_V); Y.add_(self.out_b)
        hidden_states = hidden_states.add(Y); del Y

        # FFN
        z = self.ln2(hidden_states)
        t1 = torch.matmul(z, self.fc1_U); del z
        h2 = flashsvd_ffn(t1, self.fc1_V, self.fc2_U, self.fc2_V, self.fc1_b, self.fc2_b); del t1
        hidden_states.add_(h2); del h2

        outputs = (hidden_states,)
        if use_cache:
            outputs += ((Pk_new, Pv_new),)
        else:
            del Pk_new, Pv_new
        if output_attentions: outputs += (None,)
        return outputs

class LayerShim(nn.Module):
    """HuggingFace-compatible shim that updates our ASVD cache behind the scenes."""
    def __init__(self, block: LowRankSVDBlock, layer_idx: int):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self._asvd_cache: Optional[ASVDCache] = None

    def forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, *args, **kwargs):
        use_cache_flag = bool(kwargs.get("use_cache", False))
        layer_past = None
        if use_cache_flag and isinstance(self._asvd_cache, ASVDCache):
            entry = self._asvd_cache.get(self.layer_idx)
            if entry is not None and self._asvd_cache.get_seq_length(self.layer_idx) > 0:
                layer_past = entry

        result = self.block(hidden_states, layer_past=layer_past,
                            attention_mask=attention_mask,
                            use_cache=use_cache_flag,
                            output_attentions=kwargs.get("output_attentions", False))

        # Persist new low-rank factors
        if use_cache_flag and isinstance(self._asvd_cache, ASVDCache):
            if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], (tuple, list)) and len(result[1]) == 2:
                Pk_new, Pv_new = result[1]
                self._asvd_cache.update(Pk_new, Pv_new, self.layer_idx)

        return result

def _attach_asvd_cache_to_shims(model, asvd_cache: ASVDCache):
    for layer in model.transformer.h:
        if isinstance(layer, LayerShim):
            setattr(layer, "_asvd_cache", asvd_cache)

# -------------------- model builder --------------------
def build_flashsvd_model(rank_ratio_attn: float, rank_ratio_mlp: float, device: Optional[str] = None) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    if device: model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    for i, layer in enumerate(model.transformer.h):
        blk = LowRankSVDBlock(layer, rank_ratio_attn, rank_ratio_mlp)
        shim = LayerShim(blk, layer_idx=i).to(device if device is not None else next(model.parameters()).device)
        model.transformer.h[i] = shim
    model._uses_asvd_cache = True
    return model

# -------------------- KV bytes estimator --------------------
@torch.no_grad()
def estimate_kv_bytes_asvd(cache: ASVDCache) -> int:
    def storage_key_and_nbytes(t: torch.Tensor):
        try:
            s = t.untyped_storage()
            return (s.data_ptr(), int(s.nbytes()))
        except Exception:
            s = t.storage()
            nbytes = (s.nbytes() if hasattr(s, "nbytes") else s.size() * t.element_size())
            ptr = s.data_ptr() if hasattr(s, "data_ptr") else t.data_ptr()
            return (ptr, int(nbytes))
    seen = set(); total = 0
    for entry in cache.layers:
        if entry is None: continue
        Pk, Pv = entry
        for t in (Pk, Pv):
            if t is None or not t.is_cuda: continue
            key = storage_key_and_nbytes(t)
            if key in seen: continue
            seen.add(key); total += key[1]
    return total

# -------------------- benchmark helpers --------------------
@torch.no_grad()
def _last_token_argmax(model: GPT2LMHeadModel, last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Compute logits on only the last position and return argmax ids with shape [B,1]."""
    last = last_hidden_state[:, -1, :]              # [B, D]
    logits_last = model.lm_head(last)               # [B, V]
    next_id = logits_last.argmax(dim=-1, keepdim=True)  # [B, 1]
    return next_id

@torch.no_grad()
def decode_benchmark(model: GPT2LMHeadModel, prompt: torch.Tensor, new_tokens: int, device: str) -> Dict[str, float]:
    model.eval()
    B = prompt.size(0)

    # Fresh ASVD cache
    cache = ASVDCache(n_layers=len(model.transformer.h))
    _attach_asvd_cache_to_shims(model, cache)

    # Prefill (only last-token logits)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    out = model.transformer(input_ids=prompt, use_cache=True, return_dict=True)  # no full logits
    if torch.cuda.is_available(): torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    # Compute next token from last position only
    next_id = _last_token_argmax(model, out.last_hidden_state)

    # Measure & clean before decode
    prefill_peak = torch.cuda.max_memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    prefill_end_alloc = torch.cuda.memory_allocated() / MiB if torch.cuda.is_available() else 0.0

    del out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Decode loop
    decode_poststep_peak = 0.0
    t_dec = 0.0
    generated = prompt

    for _ in range(new_tokens):
        token_in = next_id  # [B,1]
        _attach_asvd_cache_to_shims(model, cache)
        t1 = time.perf_counter()
        step = model.transformer(input_ids=token_in, use_cache=True, return_dict=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_dec += (time.perf_counter() - t1)

        generated = torch.cat([generated, token_in], dim=1)
        next_id = _last_token_argmax(model, step.last_hidden_state)

        del step
        gc.collect()
        if torch.cuda.is_available():
            alloc_now = torch.cuda.memory_allocated() / MiB
            if alloc_now > decode_poststep_peak:
                decode_poststep_peak = alloc_now

    decode_ms = t_dec * 1000.0
    decode_peak = torch.cuda.max_memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    decode_end_alloc = torch.cuda.memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    kv_end_mib = estimate_kv_bytes_asvd(cache) / MiB
    toks_per_s = (B * max(new_tokens, 1)) / max(t_dec, 1e-6)

    return {
        "prefill_ms": prefill_ms,
        "prefill_peak_MiB": prefill_peak,
        "prefill_end_alloc_MiB": prefill_end_alloc,
        "decode_ms": decode_ms,
        "decode_peak_MiB": decode_peak,
        "decode_poststep_peak_MiB": decode_poststep_peak,
        "decode_end_alloc_MiB": decode_end_alloc,
        "kv_end_MiB": kv_end_mib,
        "toks_per_s": toks_per_s,
    }

def _fmt_mean_std(vals: List[float], width: int = None, prec: int = 2) -> str:
    if not vals:
        s = "nan"
    else:
        m = sum(vals) / len(vals)
        if len(vals) >= 2:
            var = sum((x - m) ** 2 for x in vals) / len(vals)
            sd = math.sqrt(var)
        else:
            sd = 0.0
        s = f"{m:.{prec}f}±{sd:.{prec}f}"
    return f"{s:>{width}}" if width else s

@torch.no_grad()
def decode_growth_curve(model: GPT2LMHeadModel, tokenizer: AutoTokenizer, device: str,
                        batch_size: int, prompt_len: int, curve_lens: List[int],
                        rounds: int = 5):
    print(f"\n=== Decoding-time KV-cache growth (FlashSVD+ASVD, last-token logits) — {rounds} rounds avg ===")
    vocab = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 50257
    prompt = torch.randint(0, min(1000, vocab), (batch_size, prompt_len), device=device)

    header = (f"{'new_T':>7} | {'t/s':>10} | {'prefill ms':>11} | {'decode ms':>10} | "
              f"{'prefill peak':>12} | {'dec peak':>9} | {'poststep':>9} | {'end_alloc':>9} | {'KV_end':>7}")
    print(header); print("-" * len(header))

    for new_T in curve_lens:
        tps, pre_ms, dec_ms, pre_peak, dec_peak, poststep, end_alloc, kv_end = [], [], [], [], [], [], [], []
        for _ in range(rounds):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            res = decode_benchmark(model, prompt, new_T, device)
            tps.append(res["toks_per_s"])
            pre_ms.append(res["prefill_ms"])
            dec_ms.append(res["decode_ms"])
            pre_peak.append(res["prefill_peak_MiB"])
            dec_peak.append(res["decode_peak_MiB"])
            poststep.append(res["decode_poststep_peak_MiB"])
            end_alloc.append(res["decode_end_alloc_MiB"])
            kv_end.append(res["kv_end_MiB"])

        print(f"{new_T:7d} | {_fmt_mean_std(tps,10,2)} | {_fmt_mean_std(pre_ms,11,1)} | {_fmt_mean_std(dec_ms,10,1)} | "
              f"{_fmt_mean_std(pre_peak,12,1)} | {_fmt_mean_std(dec_peak,9,1)} | "
              f"{_fmt_mean_std(poststep,9,1)} | {_fmt_mean_std(end_alloc,9,1)} | {_fmt_mean_std(kv_end,7,1)}")

# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-ratio-attn", type=float, default=1.0)
    parser.add_argument("--rank-ratio-mlp",  type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode-mem", action="store_true")
    parser.add_argument("--decode-curve", type=str, default="64,128,256,512")
    parser.add_argument("--decode-batch", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== Building FlashSVD Model (ASVD cache) ===")
    model = build_flashsvd_model(rank_ratio_attn=args.rank_ratio_attn,
                                 rank_ratio_mlp=args.rank_ratio_mlp,
                                 device=device)
    blk0 = model.transformer.h[0].block
    print(f"QKV rank: {blk0.r_attn}, embed_dim={blk0.D}, heads={blk0.H}, dh={blk0.dh}")

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    if args.decode_mem:  # keep CLI parity with your runs
        curve = [int(x) for x in args.decode_curve.split(",") if x.strip()]
        decode_growth_curve(model, tok, device=device,
                            batch_size=args.decode_batch,
                            prompt_len=args.prompt_len,
                            curve_lens=curve,
                            rounds=args.rounds)
    else:
        print("Nothing to do (pass --decode-mem).")

if __name__ == "__main__":
    main()
