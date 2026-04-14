#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_dense.py — GPT‑2 dense evaluation only (prefill and decode styles)

What this does
- Loads a vanilla GPT‑2 (`--model gpt2` by default).
- Prepares WikiText‑2 (raw, test split) with fixed `--max-length` and padding.
- Computes perplexity in two modes:
  • Prefill style: one forward pass over the whole sequence (optionally masked).
  • Decode style: step-by-step using the model cache (past_key_values), scoring
    only last-token logits, which mirrors generation.

Examples
  # Run both modes on 128 samples
  python3 profile_dense.py --mode both --max-eval-samples 128 --batch-size 2 --max-length 512

  # Decode-only (slower but uses KV cache)
  python3 profile_dense.py --mode decode --max-eval-samples 64 --batch-size 1 --max-length 512

"""

import argparse, math, time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch.nn as nn

# Optional FlashAttention kernel (for GPT-2 dense blocks)
try:
    from kernels.flash_attn_causal import flash_attn_triton_kvcache
    _HAS_FLASH = True
except Exception:
    _HAS_FLASH = False


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def perplexity_prefill(
    model: GPT2LMHeadModel,
    loader,
    device: str,
    *,
    use_mask: bool = True,
) -> Tuple[float, float, float]:
    """Compute PPL using a single forward pass over the sequence.
    Returns (ppl, peak_alloc_MiB, ms_per_batch)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        ids, mask = batch["input_ids"], batch["attention_mask"]

        out = model(input_ids=ids, attention_mask=mask if use_mask else None, use_cache=False)

        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()

        if use_mask:
            m = mask[..., 1:].contiguous().bool()
            if m.any():
                loss = F.cross_entropy(shift_logits[m], shift_labels[m])
                total_loss += loss.item() * int(m.sum().item())
                total_tokens += int(m.sum().item())
        else:
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_loss += loss.item() * shift_labels.numel()
            total_tokens += int(shift_labels.numel())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per_batch = (time.perf_counter() - t0) * 1000.0 / max(1, len(loader))
    peak = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if total_tokens > 0 else float("inf")
    return ppl, peak, ms_per_batch


@torch.no_grad()
def _measure_dynamic_cache_bytes(past_key_values) -> int:
    """Sum UNIQUE storage bytes for tensors reachable from HF DynamicCache-like objects."""
    if past_key_values is None:
        return 0
    seen, total = set(), 0
    def rec(x):
        nonlocal total
        if torch.is_tensor(x):
            try:
                s = x.untyped_storage(); key = (s.data_ptr(), int(s.nbytes()))
            except Exception:
                s = x.storage()
                nbytes = (s.nbytes() if hasattr(s, "nbytes") else s.size() * x.element_size())
                ptr = s.data_ptr() if hasattr(s, "data_ptr") else x.data_ptr()
                key = (ptr, int(nbytes))
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
    return total


@torch.no_grad()
def perplexity_decode(
    model: GPT2LMHeadModel,
    loader,
    device: str,
    *,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """Compute PPL by iterative decode using HF cache.
    Returns (ppl, avg_KV_MiB, peak_alloc_MiB, ms_per_batch)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    kv_bytes_total, kv_batches = 0.0, 0

    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    out = None
    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        S = ids.size(1)

        out = None
        # Absolute position ids for GPT-2: 0..S-1
        pos_ids_full = torch.arange(S, device=device).unsqueeze(0).expand(ids.size(0), S)
        for t in range(0, S - 1):
            inp = ids[:, t:t+1]
            pos_step = pos_ids_full[:, t:t+1]
            if out is None:
                out = model(input_ids=inp, position_ids=pos_step, use_cache=True)
            else:
                out = model(input_ids=inp, position_ids=pos_step, use_cache=True, past_key_values=out.past_key_values)

            logits_last = out.logits[:, -1, :]  # [B,V]
            target = ids[:, t + 1]
            m = mask[:, t + 1].bool()
            if m.any():
                loss = F.cross_entropy(logits_last[m], target[m])
                total_loss += loss.item() * int(m.sum().item())
                total_tokens += int(m.sum().item())

        # measure KV bytes at the end of this batch (allocator-accurate)
        if hasattr(out, "past_key_values"):
            kv_bytes_total += float(_measure_dynamic_cache_bytes(out.past_key_values))
            kv_batches += 1

        # small cleanup between batches
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per_batch = (time.perf_counter() - t0) * 1000.0 / max(1, (max_batches if max_batches is not None else len(loader)))
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if total_tokens > 0 else float("inf")
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    kv_mib = (kv_bytes_total / max(kv_batches, 1)) / (1024**2)
    return ppl, kv_mib, peak_mib, ms_per_batch


def build_loader(model_name: str, max_length: int, batch_size: int, max_eval_samples: Optional[int]):
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    def tok_fn(batch):
        return tok(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    ds = raw.map(tok_fn, batched=True, remove_columns=["text"])
    if max_eval_samples is not None:
        ds = ds.select(range(min(len(ds), max_eval_samples)))
    ds.set_format("torch")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
        },
    )
    return tok, loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--mode", type=str, choices=["prefill", "decode", "both"], default="both")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", type=str, default=None, choices=[None, "float16", "bfloat16", "float32"], help="Optional model dtype override")
    p.add_argument("--flash-attn", action="store_true", help="Use FlashAttention in GPT-2 blocks (dense)")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading model: {args.model}")
    model = GPT2LMHeadModel.from_pretrained(args.model)

    if args.dtype is not None:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
        model = model.to(dtype)
    model = model.to(device).eval()
    for p_ in model.parameters():
        p_.requires_grad = False

    # Optionally replace blocks to use FlashAttention kernel
    if args.flash_attn:
        if not _HAS_FLASH:
            print("[warn] FlashAttention kernel not available; ignoring --flash-attn")
        else:
            model = build_dense_flash_model(model, device)

    # Dataset / loader
    _, loader = build_loader(args.model, args.max_length, args.batch_size, args.max_eval_samples)

    # Prefill
    if args.mode in ("prefill", "both"):
        print("\n=== Prefill Perplexity ===")
        ppl_mask, peak_mask, t_mask = perplexity_prefill(model, loader, device, use_mask=True)
        print(f"Prefill w/ mask | ppl={ppl_mask:.4f} | peak={peak_mask:7.1f} MiB | {t_mask:6.1f} ms/b")

        ppl_nom, peak_nom, t_nom = perplexity_prefill(model, loader, device, use_mask=False)
        print(f"Prefill no mask | ppl={ppl_nom:.4f} | peak={peak_nom:7.1f} MiB | {t_nom:6.1f} ms/b")

    # Decode
    if args.mode in ("decode", "both"):
        print("\n=== Decode Perplexity (cache) ===")
        ppl_dec, kv_mib, peak_mib, t_dec = perplexity_decode(model, loader, device)
        print(f"Decode (HF cache) | ppl={ppl_dec:.4f} | KV≈{kv_mib:7.1f} MiB | peak={peak_mib:7.1f} MiB | {t_dec:6.1f} ms/b")


if __name__ == "__main__":
    main()


# -------------------- FlashAttention-backed dense model --------------------

class DenseFlashBlock(nn.Module):
    def __init__(self, hf_layer: nn.Module):
        super().__init__()
        attn = hf_layer.attn
        self.hf_attn = attn
        self.ln1 = hf_layer.ln_1
        self.ln2 = hf_layer.ln_2
        self.mlp = hf_layer.mlp

        D = attn.embed_dim
        H = attn.num_heads
        if D % H != 0:
            raise ValueError(f"[DenseFlashBlock] embed_dim={D} not divisible by heads={H}")
        self.D, self.H, self.dh = D, H, D // H

    def forward(self, hidden_states: torch.Tensor, layer_past=None, attention_mask=None,
                use_cache: bool = False, output_attentions: bool = False, **kwargs):
        B, S, D = hidden_states.shape
        dev = hidden_states.device

        x = self.ln1(hidden_states)
        qkv = self.hf_attn.c_attn(x)
        q, k, v = qkv.split(self.D, dim=-1)
        Q = q.view(B, S, self.H, self.dh).permute(0, 2, 1, 3).contiguous(); del q
        K = k.view(B, S, self.H, self.dh).permute(0, 2, 1, 3).contiguous(); del k
        V = v.view(B, S, self.H, self.dh).permute(0, 2, 1, 3).contiguous(); del v

        # Append past
        if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2 and layer_past[0] is not None:
            past_k, past_v = layer_past
            if past_k.dtype != K.dtype: past_k = past_k.to(K.dtype)
            if past_v.dtype != V.dtype: past_v = past_v.to(V.dtype)
            if past_k.device != K.device: past_k = past_k.to(K.device)
            if past_v.device != V.device: past_v = past_v.to(V.device)
            K_cat = torch.cat([past_k, K], dim=2)
            V_cat = torch.cat([past_v, V], dim=2)
        else:
            K_cat, V_cat = K, V

        # Build query padding mask [B,H,1,S]
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                q_mask = attention_mask[..., -S:].to(dtype=torch.bool)
                if q_mask.size(2) != 1:
                    q_mask = q_mask[..., :1, :]
            elif attention_mask.dim() == 2:
                q_mask = attention_mask[:, -S:].bool()[:, None, None, :]
            else:
                q_mask = torch.ones(B, 1, 1, S, dtype=torch.bool, device=dev)
        else:
            q_mask = torch.ones(B, 1, 1, S, dtype=torch.bool, device=dev)
        attn_mask_bh1s = q_mask.expand(B, self.H, 1, S).contiguous()

        Y_heads = flash_attn_triton_kvcache(Q, K_cat, V_cat, attn_mask_bh1s)
        del Q, K_cat, V_cat, attn_mask_bh1s

        Y = Y_heads.transpose(1, 2).contiguous().view(B, S, D)
        Y = self.hf_attn.c_proj(Y)
        hidden_states = hidden_states + Y

        z = self.ln2(hidden_states)
        h2 = self.mlp(z)
        hidden_states = hidden_states + h2

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + ((K, V),)
        else:
            del K, V
        if output_attentions:
            outputs = outputs + (None,)
        return outputs


def _ensure_bhtd(k: torch.Tensor, v: torch.Tensor, H: int):
    assert k.dim() == 4 and v.dim() == 4
    if k.size(1) == H:
        return k, v
    if k.size(2) == H:
        return k.permute(0, 2, 1, 3).contiguous(), v.permute(0, 2, 1, 3).contiguous()
    raise RuntimeError(f"Unexpected KV layout {tuple(k.shape)}")

def _from_bhtd_to_cache_layout(k_bhtd: torch.Tensor, v_bhtd: torch.Tensor, expect_bthd: bool):
    if expect_bthd:
        return k_bhtd.permute(0, 2, 1, 3).contiguous(), v_bhtd.permute(0, 2, 1, 3).contiguous()
    return k_bhtd, v_bhtd


class LayerShim(nn.Module):
    def __init__(self, block: DenseFlashBlock, layer_idx: int = None):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx

    def forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, *args, **kwargs):
        layer_past = None
        expect_bthd = False

        if past_key_value is not None and self.layer_idx is not None:
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
                        if k_cache.dim() == 4:
                            expect_bthd = (k_cache.size(2) == self.block.H)
                            k_std, v_std = _ensure_bhtd(k_cache, v_cache, self.block.H)
                            layer_past = (k_std, v_std)
                elif seq_len and hasattr(past_key_value, "key_cache"):
                    k_cache = past_key_value.key_cache[self.layer_idx]
                    v_cache = past_key_value.value_cache[self.layer_idx]
                    if k_cache is not None and v_cache is not None:
                        expect_bthd = (k_cache.size(2) == self.block.H)
                        k_std, v_std = _ensure_bhtd(k_cache, v_cache, self.block.H)
                        layer_past = (k_std, v_std)
            elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                layer_past = past_key_value

        result = self.block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=kwargs.get("output_attentions", False),
        )

        if (past_key_value is not None and hasattr(past_key_value, "update") and self.layer_idx is not None and
            isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], tuple) and len(result[1]) == 2):
            k_new_bhtd, v_new_bhtd = result[1]
            k_upd, v_upd = _from_bhtd_to_cache_layout(k_new_bhtd, v_new_bhtd, expect_bthd)
            past_key_value.update(k_upd, v_upd, self.layer_idx)
        return result


def build_dense_flash_model(model: GPT2LMHeadModel, device: Optional[str] = None) -> GPT2LMHeadModel:
    for i, layer in enumerate(model.transformer.h):
        shim = LayerShim(DenseFlashBlock(layer), layer_idx=i).to(device if device is not None else next(model.parameters()).device)
        model.transformer.h[i] = shim
    return model
