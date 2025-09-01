#!/usr/bin/env python3
import os
import copy
from typing import Optional
import time
import torch
import itertools

import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
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


# ----------------------------
# Utilities: RoPE
# ----------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


# ----------------------------
# Explicit low-rank affine via SVD (no nn.Linear inside)
# ----------------------------
class ExplicitSVDLinear(nn.Module):
    """
    Stores SVD factors of a dense Linear (weight shape [out, in]) and
    performs explicit matmul: y = (x @ U) @ V + b
      - We compute SVD on W^T (shape [in, out]) for stable factorization.
      - Full rank if rank is None or >= min(in, out).
    """
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], rank: Optional[int] = None):
        super().__init__()
        assert weight.dim() == 2, "weight must be [out, in]"
        out_f, in_f = weight.shape
        dev, dt = weight.device, weight.dtype

        # SVD on W^T
        with torch.no_grad():
            WT = weight.detach().t().float()              # [in, out]
            U, S, Vh = torch.linalg.svd(WT, full_matrices=False)  # U:[in,r], S:[r], Vh:[r,out]
        r_full = S.shape[0]
        r = r_full if (rank is None or rank <= 0 or rank >= r_full) else int(rank)

        U_r = (U[:, :r] * S[:r]).to(dt)   # [in, r]
        V_r = Vh[:r, :].to(dt)            # [r, out]

        # Register factors/bias as buffers (no gradients by default)
        self.register_buffer("U", U_r, persistent=False)           # [in, r]
        self.register_buffer("V", V_r, persistent=False)           # [r, out]
        if bias is not None:
            self.register_buffer("b", bias.detach().to(dt), persistent=False)  # [out]
        else:
            self.b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in]
        y = x.matmul(self.U).matmul(self.V)  # [..., out]
        if self.b is not None:
            y = y + self.b
        return y


# ----------------------------
# SVD Q/K/V as explicit low-rank matmul
# ----------------------------
class ExplicitSVDQKV(nn.Module):
    """
    Replace fused Wqkv ([3D, D]) with three ExplicitSVDLinear (q,k,v).
    """
    def __init__(self, wqkv: nn.Linear, hidden_size: int, rank_attn: Optional[int]):
        super().__init__()
        assert wqkv.out_features == 3 * hidden_size
        W = wqkv.weight          # [3D, D]
        b = wqkv.bias            # [3D] or None
        Wq, Wk, Wv = torch.chunk(W, 3, dim=0)
        bq, bk, bv = (None, None, None) if b is None else torch.chunk(b, 3, dim=0)

        self.q = ExplicitSVDLinear(Wq, bq, rank=rank_attn)
        self.k = ExplicitSVDLinear(Wk, bk, rank=rank_attn)
        self.v = ExplicitSVDLinear(Wv, bv, rank=rank_attn)

    def forward(self, x: torch.Tensor):
        return self.q(x), self.k(x), self.v(x)


# ----------------------------
# Explicit ModernBERT block with SVD (incl. explicit GEGLU MLP)
# ----------------------------
class ExplicitSVDBlock(nn.Module):
    """
    - Pre-norm residual wiring
    - Q/K/V via ExplicitSVDLinear (rank_attn) → reshape → RoPE → SDPA
    - FFN via ExplicitSVDLinear Wi & Wo (rank_ffn) with explicit GEGLU:
        z = (xn2 @ U1) @ V1 + b1
        h = GELU(z[..., :D]) * z[..., D:]
        y = (h  @ U2) @ V2 + b2
    - Attention output Wo kept dense (exact HF weight)
    """
    def __init__(self, hf_layer: nn.Module, cfg, *, rank_attn: Optional[int], rank_ffn: Optional[int]):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hf_layer.attn.Wo.in_features
        self.num_heads = cfg.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Norms
        self.attn_norm = copy.deepcopy(hf_layer.attn_norm)
        self.mlp_norm  = copy.deepcopy(hf_layer.mlp_norm)

        # Rotary from HF layer (critical)
        self.rotary_emb = hf_layer.attn.rotary_emb

        # Q/K/V explicit SVD projections
        self.qkv = ExplicitSVDQKV(hf_layer.attn.Wqkv, self.hidden_size, rank_attn)
        # Attention output projection (keep dense for parity)
        self.Wo_attn = copy.deepcopy(hf_layer.attn.Wo)

        # FFN explicit SVD projections
        Wi = hf_layer.mlp.Wi   # [2D, D] for GEGLU
        Wo = hf_layer.mlp.Wo   # [D, D]
        self.Wi_exp = ExplicitSVDLinear(Wi.weight, Wi.bias, rank=rank_ffn)   # explicit low-rank
        self.Wo_exp = ExplicitSVDLinear(Wo.weight, Wo.bias, rank=rank_ffn)   # explicit low-rank

        # Detect GEGLU by shape, and get GELU approximate mode from HF if present
        self.ffn_D = Wo.in_features
        self.ffn_is_geglu = (Wi.out_features == 2 * self.ffn_D)
        gelu_approx = getattr(getattr(hf_layer.mlp, "act", nn.GELU()), "approximate", "tanh")
        self.geglu = GEGLU(approximate=gelu_approx)

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

        # Q/K/V (B,M,D) -> [B,H,M,dh]
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
        #TODO: try use the flash_attn_trition.py implementation here, use the flashattention kernel for efficient inference
        # that avoids large intermediate tensors
        
        attn = attn.transpose(1, 2).reshape(B, M, D)  # [B,M,D]
        x = x + self.Wo_attn(attn)

        # === FFN (pre-norm) — explicit low-rank matmuls + GEGLU ===
        xn2 = self.mlp_norm(x)

        # z = (xn2 @ U1) @ V1 + b1
        z = self.Wi_exp(xn2)  # [B,M, 2D] (GEGLU) or [B,M, D'] (fallback)

        if self.ffn_is_geglu:
            h = self.geglu(z)                        # GEGLU: gelu(u)*v → [B,M,D]
        else:
            # Fallback (not expected for ModernBERT, but safe)
            h = F.gelu(z, approximate=self.geglu.approximate)

        # y = (h @ U2) @ V2 + b2
        y = self.Wo_exp(h)                            # [B,M,D]
        x = x + y
        
        if output_attentions:
            return (x, None)
        return (x,)


# ----------------------------
# Full wrapper: swap layers
# ----------------------------
class ModernBERT_SVD_Explicit(nn.Module):
    """
    Wraps an HF ModernBERTForSequenceClassification and replaces each encoder
    layer with ExplicitSVDBlock (SVD Q/K/V + explicit RoPE + SDPA + explicit SVD FFN with GEGLU).
    """
    def __init__(self, hf_model: nn.Module, *, rank_attn: Optional[int], rank_ffn: Optional[int]):
        super().__init__()
        self.config = hf_model.config
        self.model = hf_model.model
        self.classifier = hf_model.classifier
        self.head = getattr(hf_model, "head", None)
        self.drop = getattr(hf_model, "drop", None)

        new_layers = []
        for layer in self.model.layers:
            new_layers.append(ExplicitSVDBlock(layer, self.config, rank_attn=rank_attn, rank_ffn=rank_ffn))
        self.model.layers = nn.ModuleList(new_layers)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **{k: v for k, v in kwargs.items() if k != "labels"})
        hidden_states = outputs[0]  # [B,M,D]

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
        return type("Output", (), {"logits": logits})()


# ----------------------------
# Quick sanity harness
# ----------------------------
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
def compute_persistent_memory(m):
        total = 0
        for p in itertools.chain(m.parameters(), m.buffers()):
            total += p.numel() * p.element_size()
        return total / (1024**2)

def quick_check(model_svd, loader, device):
    metric = load_metric("accuracy")
# -------- Warm-up (no timing, no CPU syncs) --------
    it = iter(loader)
    for _ in range(10):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        _ = model_svd(
            input_ids=b["input_ids"].to(device, non_blocking=True),
            attention_mask=b["attention_mask"].to(device, non_blocking=True),
        )
    torch.cuda.synchronize()                 # ensure original pass fully finished
    torch.cuda.reset_peak_memory_stats()
    n = 0
    start = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 3: break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model_svd(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            metric.add_batch(predictions=out.logits.argmax(-1).cpu(), references=batch["labels"].cpu())
            n += 1
    torch.cuda.synchronize()
    t = (time.perf_counter() - start) * 1000 / n

    peak_m = torch.cuda.max_memory_allocated() / (1024**2)
    return t, peak_m, metric.compute()["accuracy"]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Full rank by default — set smaller ints later to compress
    RANK_ATTN = None
    RANK_FFN  = None

    cfg = AutoConfig.from_pretrained(MODE_DIR := MODEL_DIR, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"  # matches our explicit attention path

    dense = AutoModelForSequenceClassification.from_pretrained(MODE_DIR, config=cfg, trust_remote_code=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODE_DIR, trust_remote_code=True)
    loader = _build_loader(tok, seq_len=2048, batch_size=8)

    svd = ModernBERT_SVD_Explicit(dense, rank_attn=RANK_ATTN, rank_ffn=RANK_FFN).to(device).eval()
    CACHED_ORIG_MEM = compute_persistent_memory(svd)

    t, peak_m ,acc = quick_check(svd, loader, device)
    print(f"[Memory] Peak CUDA memory: {peak_m:.1f} MiB")
    print(f"[Model] Transient memory: {peak_m - CACHED_ORIG_MEM:.1f} MiB")
    print(f"[Latency] SVD-explicit forward pass time (no grad) over 3 batches: {t:.1f} ms")
    print(f"[Sanity] SVD-explicit model accuracy on 3 batches: {acc:.4f}")
    print("If parity looks good, lower RANK_ATTN / RANK_FFN to compress.")

if __name__ == "__main__":
    main()
