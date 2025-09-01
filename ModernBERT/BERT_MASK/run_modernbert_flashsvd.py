#!/usr/bin/env python3
import os
import copy
from typing import Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# NEW: import your kernels (assume they’re importable on PYTHONPATH)
try:
    from flashsvdgeglu import flashsvd_ffn_geglu  # FFN kernel
except Exception as e:
    flashsvd_ffn_geglu = None  # will gracefully fall back if not available

try:
    from flashsvdropeattn import FlashSVDRoPEAttention, QKVFactors  # Attn kernel
except Exception as e:
    FlashSVDRoPEAttention, QKVFactors = None, None  # will gracefully fall back


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
    - Attention:
        * If use_flash_attn==1: rank-space attention via FlashSVD+RoPE kernel.
        * Else: SVD Q/K/V → reshape → RoPE → PyTorch SDPA.
    - FFN:
        * If use_flash_ffn==1: rank-space GEGLU via FlashSVD FFN kernel.
        * Else: explicit SVD Wi/Wo + GEGLU (your current path).
    - Attention output Wo kept dense (exact HF weight).
    """
    def __init__(
        self,
        hf_layer: nn.Module,
        cfg,
        *,
        rank_attn: Optional[int],
        rank_ffn: Optional[int],
        use_flash_attn: int = 0,
        use_flash_ffn: int = 0,
        attn_kernel_bm: int = 64,
        attn_kernel_bn: int = 64,
        attn_kernel_bdh: Optional[int] = None,
        attn_kernel_br: int = 64,
        attn_l2norm_qk: bool = False,  # off by default; parity-first
    ):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hf_layer.attn.Wo.in_features
        self.num_heads = cfg.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Norms and rotary
        self.attn_norm = copy.deepcopy(hf_layer.attn_norm)
        self.mlp_norm  = copy.deepcopy(hf_layer.mlp_norm)
        self.rotary_emb = hf_layer.attn.rotary_emb

        # Q/K/V via explicit SVD factors (shared by both branches)
        self.qkv = ExplicitSVDQKV(hf_layer.attn.Wqkv, self.hidden_size, rank_attn)
        # Attention output projection (dense)
        self.Wo_attn = copy.deepcopy(hf_layer.attn.Wo)

        # FFN explicit SVD projections (shared by both branches)
        Wi = hf_layer.mlp.Wi   # [2D, D] for GEGLU
        Wo = hf_layer.mlp.Wo   # [D,  D]
        self.Wi_exp = ExplicitSVDLinear(Wi.weight, Wi.bias, rank=rank_ffn)
        self.Wo_exp = ExplicitSVDLinear(Wo.weight, Wo.bias, rank=rank_ffn)

        # Act / GEGLU
        self.ffn_D = Wo.in_features
        self.ffn_is_geglu = (Wi.out_features == 2 * self.ffn_D)
        gelu_approx = getattr(getattr(hf_layer.mlp, "act", nn.GELU()), "approximate", "tanh")
        self.geglu = GEGLU(approximate=gelu_approx)

        # ---- enforce FlashSVD paths ----
        assert (FlashSVDRoPEAttention is not None) and torch.cuda.is_available(), "FlashSVD attention kernel not available or CUDA not found."
        assert (flashsvd_ffn_geglu is not None) and torch.cuda.is_available(), "FlashSVD GEGLU FFN kernel not available or CUDA not found."
        self.attn_l2norm_qk = bool(attn_l2norm_qk)

        # Build the FlashSVD attention kernel wrapper
        self.flash_attn = FlashSVDRoPEAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rotary_emb=self.rotary_emb,
            bm=attn_kernel_bm,
            bn=attn_kernel_bn,
            bdh=self.head_dim if attn_kernel_bdh is None else attn_kernel_bdh,
            br=attn_kernel_br,
        )

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

        if position_ids is None:
            position_ids = torch.arange(M, device=hidden_states.device).unsqueeze(0).expand(B, M)

        # ---- FlashSVD attention path only ----
        # rank-space inputs: P = X @ U
        Uq, Uk, Uv = self.qkv.q.U, self.qkv.k.U, self.qkv.v.U            # [D, R]
        Vq, Vk, Vv = self.qkv.q.V, self.qkv.k.V, self.qkv.v.V            # [R, D]
        bq, bk, bv = self.qkv.q.b, self.qkv.k.b, self.qkv.v.b            # [D] or None

        Pq = xn.matmul(Uq)   # [B, M, R]
        Pk = xn.matmul(Uk)
        Pv = xn.matmul(Uv)

        # Optional: L2 normalize Q/K before RoPE
        if self.attn_l2norm_qk:
            Pq = F.normalize(Pq, dim=-1)
            Pk = F.normalize(Pk, dim=-1)

        # Call the FlashSVD RoPE+SDPA kernel (returns [B,H,M,dh])
        O = self.flash_attn(
            QKVFactors(
                Pq=Pq, Pk=Pk, Pv=Pv,
                Vq=Vq.contiguous(), Vk=Vk.contiguous(), Vv=Vv.contiguous(),
                bq=bq if bq is not None else None,
                bk=bk if bk is not None else None,
                bv=bv if bv is not None else None,
            ),
            attention_mask=attention_mask,           # 2D [B,L] or 4D additive
            position_ids=position_ids,               # [B,M]
            sliding_window_mask=sliding_window_mask, # optional 4D additive
        )  # [B,H,M,dh]

        attn_out = O.transpose(1, 2).reshape(B, M, D).contiguous()
        x = x + self.Wo_attn(attn_out)

        
        # === FFN (pre-norm) ===
        xn2 = self.mlp_norm(x)
        if not self.ffn_is_geglu:
            raise NotImplementedError("Non-GEGLU FFN is not supported in FlashSVD-only script.")

        # Dimensions
        H = self.hidden_size     # model hidden size (output of Wo)
        D = self.ffn_D           # FFN expansion size (input of Wo)

        # Rank-space inputs and factors
        U1 = self.Wi_exp.U.contiguous()              # [H, R1]
        V1 = self.Wi_exp.V.contiguous()              # [R1, 2D]
        U2 = self.Wo_exp.U.contiguous()              # [D,  R2]
        V2 = self.Wo_exp.V.contiguous()              # [R2, H]

        # Biases (fallbacks sized to the *correct* dims)
        b1 = self.Wi_exp.b if self.Wi_exp.b is not None else xn2.new_zeros(2 * D)
        b2 = self.Wo_exp.b if self.Wo_exp.b is not None else xn2.new_zeros(H)

        # Preflight checks (clear error messages if shapes drift)
        assert V1.shape[1] == 2 * D, f"V1 must be [R1,2D], got {tuple(V1.shape)}; D={D}"
        assert U2.shape[0] == D,     f"U2 must be [D,R2], got {tuple(U2.shape)}; D={D}"
        assert V2.shape[1] == H,     f"V2 must be [R2,H], got {tuple(V2.shape)}; H={H}"
        assert b1.numel() == 2 * D,  f"b1 must be [2D], got {tuple(b1.shape)}; D={D}"
        assert b2.numel() == H,      f"b2 must be [H], got {tuple(b2.shape)}; H={H}"

        # Rank-space input: P = X @ U1
        P = xn2.matmul(U1)  # [B, M, R1]

        # FlashSVD GEGLU FFN (tanh/erf picked by gelu_approx)
        y = flashsvd_ffn_geglu(
            P, V1, U2, V2, b1, b2,
            gelu_approx=self.geglu.approximate  # "tanh" or "none" (erf)
        )
        x = x + y

        if output_attentions:
            return (x, None)
        return (x,)


# ----------------------------
# Full wrapper: swap layers
# ----------------------------
class ModernBERT_SVD_Explicit(nn.Module):
    def __init__(
        self,
        hf_model: nn.Module,
        *,
        rank_attn: Optional[int],
        rank_ffn: Optional[int],
        use_flash_attn: int = 0,
        use_flash_ffn: int = 0,
        attn_kernel_bm: int = 64,
        attn_kernel_bn: int = 64,
        attn_kernel_bdh: Optional[int] = None,
        attn_kernel_br: int = 64,
        attn_l2norm_qk: bool = False,
    ):
        super().__init__()
        self.config = hf_model.config
        self.model = hf_model.model
        self.classifier = hf_model.classifier
        self.head = getattr(hf_model, "head", None)
        self.drop = getattr(hf_model, "drop", None)

        new_layers = []
        for layer in self.model.layers:
            new_layers.append(
                ExplicitSVDBlock(
                    layer, self.config,
                    rank_attn=rank_attn, rank_ffn=rank_ffn,
                    use_flash_attn=use_flash_attn,
                    use_flash_ffn=use_flash_ffn,
                    attn_kernel_bm=attn_kernel_bm,
                    attn_kernel_bn=attn_kernel_bn,
                    attn_kernel_bdh=attn_kernel_bdh,
                    attn_kernel_br=attn_kernel_br,
                    attn_l2norm_qk=attn_l2norm_qk,
                )
            )
        self.model.layers = nn.ModuleList(new_layers)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            **{k: v for k, v in kwargs.items() if k != "labels"}
        )
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
def quick_check(model_svd, loader, device):
    metric = load_metric("accuracy")
    for i, batch in enumerate(loader):
        if i >= 3: break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model_svd(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        metric.add_batch(predictions=out.logits.argmax(-1).cpu(), references=batch["labels"].cpu())
    return metric.compute()["accuracy"]



@torch.no_grad()
def acc_peak_time(model_svd, loader, device, use_mask=True):
    metric = load_metric("accuracy")
    # Record persistent baseline before resetting peak tracking
    baseline_mib = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    steps = 0
    start = time.perf_counter()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if use_mask:
            out = model_svd(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        else:
            out = model_svd(input_ids=batch["input_ids"]).logits
        metric.add_batch(predictions=out.argmax(-1).cpu(), references=batch["labels"].cpu())
        steps += 1
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, steps)
    delta_peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    abs_peak_mib = baseline_mib + delta_peak_mib
    return metric.compute()["accuracy"], delta_peak_mib, abs_peak_mib, elapsed_ms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Ranks (None = full rank)
    RANK_ATTN = 128 #// 4#None
    RANK_FFN  = 768 #// 4#None

    # # === runtime switches ===
    # # we are 100% sure the run_modernbert_svd.py code works exactly as the modernbert in full rank 
    # USE_FLASH_ATTN = 1  # 1: use your FlashSVD RoPE+SDPA kernel; 0: SVD+PyTorch SDPA
    # USE_FLASH_FFN  = 1  # 1: use your FlashSVD GEGLU FFN kernel;  0: explicit SVD FFN
    # # (If CUDA unavailable or the kernel modules cannot be imported, these silently fall back to 0.)
    
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"  

    dense = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=cfg, trust_remote_code=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    loader = _build_loader(tok, seq_len=128*4, batch_size=8)

    # Measure persistent memory for the SVD model construction on GPU
    torch.cuda.synchronize()

    svd = ModernBERT_SVD_Explicit(
        dense,
        rank_attn=RANK_ATTN,
        rank_ffn=RANK_FFN,
        use_flash_attn=1,
        use_flash_ffn=1,
        attn_kernel_bm=64, attn_kernel_bn=64, attn_kernel_bdh=None, attn_kernel_br=64,
        attn_l2norm_qk=False,  
    ).to(device).eval()

    # Report persistent model memory before any inference (resident allocations)
    persistent_mib = torch.cuda.memory_allocated() / (1024**2)
    print(f"[Model] persistent_mem_before_infer={persistent_mib:.1f} MiB")

    acc = quick_check(svd, loader, device)
    print(f"[Sanity] Model accuracy on 3 batches: {acc:.4f}")

    # Comprehensive memory and latency measurement (entire validation split)
    full_acc, delta_peak_mib, abs_peak_mib, latency_ms = acc_peak_time(svd, loader, device, use_mask=True)
    print(f"[Eval] acc={full_acc:.4f}  delta_peak={delta_peak_mib:.1f} MiB  abs_peak={abs_peak_mib:.1f} MiB  avg_latency={latency_ms:.2f} ms/batch")

if __name__ == "__main__":
    main()
