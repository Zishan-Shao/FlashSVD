# profile_flashsvd.py  — FlashSVD + DRONE-style data-aware low-rank (one-shot)
# -------------------------------------------------------------------------------
# How it works (high level):
# 1) Calibration (few batches): grab per-layer input covariances C = E[x x^T]
#    for (a) inputs to Q/K/VM (dm x dm), (b) inputs to attn.output.dense (dm x dm),
#    (c) inputs to FFN intermediate.dense (dm x dm), (d) inputs to FFN output.dense (d_ff x d_ff).
# 2) DRONE factorization for each Linear W with input covariance C:
#       S = chol(C),  A = W^T S,  SVD(A) = U Σ V^T (truncate to rank k)
#       U_data = S^{-T} V_k Σ_k^{1/2}   (shape: d_in x k)
#       V_data = Σ_k^{1/2} U_k^T        (shape: k x d_out)
#    This minimizes ||X^T(W - U_data V_data)||_F for the empirical inputs X.
# 3) Attention path stays Flash: we pass P=U_data (per-head), V=V_data, and bias to flash_svd_attention.
# 4) FFN stays Flash too: mid = x @ U1; then flashsvd_ffn_v1(mid, V1, U2, V2, b1, b2).
# -------------------------------------------------------------------------------

import os
import sys
import time
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig
from evaluate import load as load_metric

from flashsvdattn import flash_svd_attention
from flashsvdffnv2 import flashsvd_ffn  # this is actually V2
from flashsvdffnv1 import flashsvd_ffn_v1

# ─── locate repo & model ─────────────────────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Task / model ckpt
task_name = "sst2"
MODEL_DIR = os.path.join(REPO_ROOT, "models/BERT", f"bert-base-uncased-{task_name}")

# -----------------------------------------------------------------------------
# Numeric helpers for DRONE-style factorization
# -----------------------------------------------------------------------------
def _safe_cholesky(C: torch.Tensor, max_tries: int = 5, base_eps: float = 1e-6):
    """Robust Cholesky with adaptive jitter if C is near-singular."""
    D = C.shape[-1]
    eps = base_eps * float(C.diag().mean().item() + 1.0)
    I = torch.eye(D, dtype=C.dtype, device=C.device)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(C + eps * I)
        except RuntimeError:
            eps *= 10.0
    return torch.linalg.cholesky(C + (1e-2 * float(C.diag().mean().item() + 1.0)) * I)

@torch.no_grad()
def _data_aware_low_rank(W_in_out: torch.Tensor, rank: int, cov_in: torch.Tensor):
    """
    DRONE-style low-rank: given Linear weight W (shape [d_in, d_out]) and input covariance C,
    compute U:[d_in,k], V:[k,d_out] minimizing ||X^T(W - U V)||_F for empirical inputs X.

    Steps: S = chol(C), A = (W^T) @ S, SVD(A)=U Σ V^T. Then:
      U_data = S^{-T} V_k Σ_k^{1/2},   V_data = Σ_k^{1/2} U_k^T
    """
    d_in, d_out = W_in_out.shape
    assert rank > 0, "rank must be positive"
    Wf, Cf = W_in_out.float(), cov_in.float()
    S = _safe_cholesky(Cf)                            # [d_in, d_in], lower
    A = Wf.t().contiguous() @ S                       # [d_out, d_in]
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)  # U:[d_out,r], s:[r], Vh:[r,d_in]
    V = Vh.t()                                        # [d_in, r]
    k = min(rank, s.numel())
    U_k, s_k, V_k = U[:, :k], s[:k], V[:, :k]
    # Solve S^T X = V_k  ->  X = S^{-T} V_k
    X = torch.linalg.solve_triangular(S.t(), V_k, upper=True)  # [d_in, k]
    sqrt_s = torch.sqrt(torch.clamp(s_k, min=0))
    U_data = X * sqrt_s.unsqueeze(0)                 # [d_in, k]
    V_data = (sqrt_s.unsqueeze(1) * U_k.t())         # [k, d_out]
    return U_data.to(W_in_out.dtype), V_data.to(W_in_out.dtype)

@torch.no_grad()
def _data_aware_per_head(Wt_dm_dh_total: torch.Tensor, rank: int, cov_in: torch.Tensor, num_heads: int):
    """
    DRONE-style per-head factorization for attention Q/K/V.
    Wt_dm_dh_total is the weight^T reshaped to [d_model, d_model] originally.
    We view it as [d_model, H, dh] and factor each [d_model, dh] slice separately.
    Returns U:[H, d_model, k], V:[H, k, dh].
    """
    d_model = Wt_dm_dh_total.shape[0]
    dh = Wt_dm_dh_total.shape[1] // num_heads
    Wt3 = Wt_dm_dh_total.view(d_model, num_heads, dh)
    Us, Vs = [], []
    for h in range(num_heads):
        Wh = Wt3[:, h, :]  # [d_model, dh]
        Uh, Vh = _data_aware_low_rank(Wh, rank, cov_in)  # cov_in: [dm, dm]
        Us.append(Uh)        # [dm, k]
        Vs.append(Vh)        # [k, dh]
    U = torch.stack(Us, dim=0)  # [H, dm, k]
    V = torch.stack(Vs, dim=0)  # [H, k, dh]
    return U, V

# -----------------------------------------------------------------------------
# Layer shim to keep HF encoder interface
# -----------------------------------------------------------------------------
class LayerShim(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        raw_mask = attention_mask
        if attention_mask is not None and attention_mask.dim() == 4:
            raw_mask = (attention_mask[:, 0, 0, :] == 0)
        return (self.block(hidden_states, raw_mask),)

# -----------------------------------------------------------------------------
# Flash + Data-aware low-rank block
# -----------------------------------------------------------------------------
class FlashSVDBlockDA(nn.Module):
    """
    Flash-attention + Flash-FFN block using DRONE-style data-aware low-rank factors.
    Q/K/V and FFN are factorized with covariances measured during calibration.
    """
    def __init__(self, hf_layer, rank_attn, rank_ff,
                 cov_attn_in: torch.Tensor, cov_attn_out: torch.Tensor,
                 cov_ffn_in: torch.Tensor,  cov_ffn_out: torch.Tensor,
                 rank_wo: int):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        d_ff    = hf_layer.intermediate.dense.out_features

        # --- Attention Q/K/V (per-head data-aware) ---
        WqT, bq = cfg.query.weight.data.t(), cfg.query.bias.data.view(1, H, 1, dh)
        WkT, bk = cfg.key.weight.data.t(),   cfg.key.bias.data.view(1, H, 1, dh)
        WvT, bv = cfg.value.weight.data.t(), cfg.value.bias.data.view(1, H, 1, dh)

        Uq, Vq = _data_aware_per_head(WqT, rank_attn, cov_attn_in, H)
        Uk, Vk = _data_aware_per_head(WkT, rank_attn, cov_attn_in, H)
        Uv, Vv = _data_aware_per_head(WvT, rank_attn, cov_attn_in, H)

        self.Pq, self.Vq = nn.Parameter(Uq), nn.Parameter(Vq)  # P: [H, dm, R], V: [H, R, dh]
        self.Pk, self.Vk = nn.Parameter(Uk), nn.Parameter(Vk)
        self.Pv, self.Vv = nn.Parameter(Uv), nn.Parameter(Vv)
        self.bq, self.bk, self.bv = map(nn.Parameter, (bq, bk, bv))

        # --- FFN (data-aware) ---
        Wi, bi   = hf_layer.intermediate.dense.weight.data.t(), hf_layer.intermediate.dense.bias.data
        WoT, bo2 = hf_layer.output.dense.weight.data.t(),      hf_layer.output.dense.bias.data

        U1, V1 = _data_aware_low_rank(Wi,  rank_ff, cov_ffn_in)    # dm->d_ff
        U2, V2 = _data_aware_low_rank(WoT, rank_ff, cov_ffn_out)   # d_ff->dm

        self.U1, self.V1 = nn.Parameter(U1), nn.Parameter(V1)
        self.U2, self.V2 = nn.Parameter(U2), nn.Parameter(V2)
        self.b1, self.b2 = nn.Parameter(bi), nn.Parameter(bo2)

        # --- Attention output projection Wo (data-aware) ---
        Wo_full  = hf_layer.attention.output.dense.weight.data  # [dm, dm] (out, in)
        bo_attn  = hf_layer.attention.output.dense.bias.data
        Uo, Vo = _data_aware_low_rank(Wo_full.t(), rank_wo, cov_attn_out)  # we pass W^T so shape is [dm, dm]
        self.Uo, self.Vo = nn.Parameter(Uo), nn.Parameter(Vo)
        self.bo_attn     = nn.Parameter(bo_attn)

        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        H, R     = self.Pq.shape[0], self.Pq.shape[2]
        dh       = dm // H

        # Project to per-head rank-R space (do not multiply V yet; Flash kernel will)
        tmp_q = torch.einsum('bmd,hdr->bhmr', x, self.Pq)  # [B,H,M,R]
        tmp_k = torch.einsum('bmd,hdr->bhmr', x, self.Pk)  # [B,H,M,R]
        tmp_v = torch.einsum('bmd,hdr->bhmr', x, self.Pv)  # [B,H,M,R]

        # Expand V/bias for Flash SVD attention kernel
        Vq_full = self.Vq.expand(B, H, R, dh)
        Vk_full = self.Vk.expand(B, H, R, dh)
        Vv_full = self.Vv.expand(B, H, R, dh)
        bq_full = self.bq.expand(B, H, 1, dh).squeeze(2)
        bk_full = self.bk.expand(B, H, 1, dh).squeeze(2)
        bv_full = self.bv.expand(B, H, 1, dh).squeeze(2)

        mask4 = mask.view(B, 1, 1, M) if mask is not None else None
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4, block_m=32, block_r=R
        )
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()

        # back to [B,M,dm], then output projection (low-rank) + residual + LN
        attn = attn_out.view(B, H, M, dh).transpose(1, 2).reshape(B, M, dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)

        # FFN via Flash kernels:
        mid = x1 @ self.U1  # [B,M,R1]
        y = flashsvd_ffn_v1(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        out = self.ln2(x1 + y)
        return out

# -----------------------------------------------------------------------------
# One-shot calibration: collect per-layer input covariances
# -----------------------------------------------------------------------------
@torch.no_grad()
def calibrate_covariances(model: BertForSequenceClassification,
                          loader: DataLoader,
                          device: str,
                          max_batches: int = 4) -> Dict[str, List[torch.Tensor]]:
    """
    Collect (online) input covariances for each encoder layer:
      - cov_attn_in:  inputs to attention.self.query (shared for Q/K/V)  -> [dm, dm]
      - cov_attn_out: inputs to attention.output.dense                   -> [dm, dm]
      - cov_ffn_in:   inputs to intermediate.dense                       -> [dm, dm]
      - cov_ffn_out:  inputs to output.dense (post-GELU)                 -> [d_ff, d_ff]
    """
    model.eval()
    enc = model.bert.encoder
    num_layers = len(enc.layer)
    dm  = model.config.hidden_size
    dff = model.config.intermediate_size

    cov_attn_in  = [torch.zeros(dm,  dm,  dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_attn_in    = [0 for _ in range(num_layers)]
    cov_attn_out = [torch.zeros(dm,  dm,  dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_attn_out   = [0 for _ in range(num_layers)]
    cov_ffn_in   = [torch.zeros(dm,  dm,  dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_ffn_in     = [0 for _ in range(num_layers)]
    cov_ffn_out  = [torch.zeros(dff, dff, dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_ffn_out    = [0 for _ in range(num_layers)]
    handles = []

    def _upd(cov_mat, n_store, idx, x):
        if x is None:
            return
        x = x.detach()
        BMD = x.shape[0] * x.shape[1]
        X2d = x.reshape(BMD, x.shape[-1]).to(device=device, dtype=torch.float32)
        cov_mat[idx] += X2d.t() @ X2d
        n_store[idx] += BMD

    for i, layer in enumerate(enc.layer):
        # Inputs to Q/K/V: hook query pre-forward (shared input)
        def q_pre(mod, inp, idx=i): _upd(cov_attn_in, n_attn_in, idx, inp[0])
        handles.append(layer.attention.self.query.register_forward_pre_hook(q_pre))

        # Inputs to attention.output.dense
        def attn_out_pre(mod, inp, idx=i): _upd(cov_attn_out, n_attn_out, idx, inp[0])
        handles.append(layer.attention.output.dense.register_forward_pre_hook(attn_out_pre))

        # Inputs to FFN intermediate.dense
        def ffn_in_pre(mod, inp, idx=i): _upd(cov_ffn_in, n_ffn_in, idx, inp[0])
        handles.append(layer.intermediate.dense.register_forward_pre_hook(ffn_in_pre))

        # Inputs to FFN output.dense (post-GELU)
        def ffn_out_pre(mod, inp, idx=i): _upd(cov_ffn_out, n_ffn_out, idx, inp[0])
        handles.append(layer.output.dense.register_forward_pre_hook(ffn_out_pre))

    seen = 0
    for batch in loader:
        if seen >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        seen += 1

    for h in handles:
        h.remove()

    def _finalize(cov_list, n_list):
        out = []
        for C, n in zip(cov_list, n_list):
            if n == 0:
                D = C.shape[0]
                Cn = torch.eye(D, dtype=torch.float32, device=C.device)
            else:
                Cn = C / float(n)
                ridge = 1e-6 * float(Cn.diag().mean().item() + 1.0)
                Cn = Cn + ridge * torch.eye(Cn.shape[0], dtype=Cn.dtype, device=Cn.device)
            out.append(Cn.cpu())
        return out

    return {
        "cov_attn_in":  _finalize(cov_attn_in,  n_attn_in),
        "cov_attn_out": _finalize(cov_attn_out, n_attn_out),
        "cov_ffn_in":   _finalize(cov_ffn_in,   n_ffn_in),
        "cov_ffn_out":  _finalize(cov_ffn_out,  n_ffn_out),
    }

# -----------------------------------------------------------------------------
# Accuracy/latency/memory bench
# -----------------------------------------------------------------------------
@torch.no_grad()
def acc_peak_time(mdl, loader, device, task_name: str):
    mdl.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if task_name == "stsb":
        metric = load_metric("pearsonr")
        metric_key = "pearsonr"
    else:
        metric = load_metric("accuracy")
        metric_key = "accuracy"
    total, steps = 0.0, 0
    start = time.perf_counter()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = mdl(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        if task_name == "stsb":
            preds = logits.squeeze(-1)
        else:
            preds = torch.argmax(logits, -1)
        total += metric.compute(predictions=preds.cpu(), references=batch["labels"].cpu())[metric_key]
        steps += 1
    torch.cuda.synchronize()
    ms_per_batch = (time.perf_counter() - start) * 1000.0 / max(steps, 1)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return total / max(steps, 1), peak, ms_per_batch

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    BATCH_SIZE = 32
    SEQ_LEN    = 128 * 2
    device     = "cuda"
    RANK_ATTN  = 64
    RANK_FF    = 768
    RANK_WO    = 768
    CALIB_BATCHES = 4  # one-shot calibration window size

    # ─── Load & tokenize GLUE ────────────────────────────────────────────────
    if task_name == "mnli":
        val_split = "validation_matched"
    else:
        val_split = "validation"
    raw = load_dataset("glue", task_name, split=val_split)
    tokz = AutoTokenizer.from_pretrained(MODEL_DIR)

    single_sent_tasks = {"cola", "sst2"}
    pair_sent_tasks   = {"qqp", "mnli", "qnli", "stsb", "rte", "mrpc"}
    field_map = {
        "qqp":  ("question1", "question2"),
        "mnli": ("premise",   "hypothesis"),
        "qnli": ("question",  "sentence"),
        "stsb": ("sentence1", "sentence2"),
        "rte":  ("sentence1", "sentence2"),
        "mrpc": ("sentence1", "sentence2"),
    }

    def tokenize_fn(batch):
        if task_name in single_sent_tasks:
            return tokz(batch["sentence"], padding="max_length", truncation=True, max_length=SEQ_LEN)
        else:
            f1, f2 = field_map[task_name]
            return tokz(batch[f1], batch[f2], padding="max_length", truncation=True, max_length=SEQ_LEN)

    remove_cols = [c for c in raw.column_names if c != "label"]
    ds = raw.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    ds.set_format("torch")
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels":         torch.tensor([x["label"]         for x in b]),
        },
    )

    print(f"BATCH_SIZE: {BATCH_SIZE}  RANK_ATTN: {RANK_ATTN}  RANK_FF: {RANK_FF}  RANK_WO: {RANK_WO}")

    # ─── Build model ─────────────────────────────────────────────────────────
    if task_name == "mnli":
        num_labels, problem_type = 3, None
    elif task_name == "stsb":
        num_labels, problem_type = 1, "regression"
    else:
        num_labels, problem_type = 2, None

    cfg = AutoConfig.from_pretrained(MODEL_DIR, num_labels=num_labels, problem_type=problem_type)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR, config=cfg).to(device).eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    print(f"Persistent model storage (dense): {torch.cuda.max_memory_allocated()/1024**2:6.1f} MiB")

    # ─── Calibration: estimate per-layer input covariances ───────────────────
    print("Calibrating input covariances for DRONE (one-shot)…")
    covs = calibrate_covariances(model, loader, device, max_batches=CALIB_BATCHES)

    # ─── Replace each encoder layer with Flash + data-aware low-rank block ───
    for i, layer in enumerate(model.bert.encoder.layer):
        blk = FlashSVDBlockDA(
            hf_layer=layer,
            rank_attn=RANK_ATTN,
            rank_ff=RANK_FF,
            cov_attn_in=covs["cov_attn_in"][i].to(device),
            cov_attn_out=covs["cov_attn_out"][i].to(device),
            cov_ffn_in=covs["cov_ffn_in"][i].to(device),
            cov_ffn_out=covs["cov_ffn_out"][i].to(device),
            rank_wo=RANK_WO,
        )
        model.bert.encoder.layer[i] = LayerShim(blk).to(device).eval().float()

    # ─── Memory accounting (persistent params only) ──────────────────────────
    def summarize_dense_vs_lowrank(mdl):
        dense_bytes, lowrank_bytes = 0, 0
        for name, p in mdl.named_parameters():
            size = p.numel() * p.element_size()
            if ".block." in name or (
                name.startswith("bert.encoder.layer") and
                any(part in name for part in ("Pq","Vq","Pk","Vk","Pv","Vv","U1","V1","U2","V2","Uo","Vo"))
            ):
                lowrank_bytes += size
            else:
                dense_bytes += size
        print(f"{'Type':<12}{'MiB':>8}")
        print("----------------------")
        print(f"{'Dense':<12}{dense_bytes/1024**2:8.1f}")
        print(f"{'Low-rank':<12}{lowrank_bytes/1024**2:8.1f}")
        print("----------------------")
        print(f"{'TOTAL':<12}{(dense_bytes+lowrank_bytes)/1024**2:8.1f}")
        return dense_bytes + lowrank_bytes

    baseline_bytes = summarize_dense_vs_lowrank(model)
    with_act = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Flash low-rank model storage with GPU redundancy: {with_act:.1f} MiB")
    print(f"Persistent low-rank model storage (DRONE): {baseline_bytes/1024**2:6.1f} MiB")

    # ─── Evaluate ────────────────────────────────────────────────────────────
    metric_name = "pearson" if task_name == "stsb" else "acc"
    acc, peak_lr, t = acc_peak_time(model, loader, device, task_name)
    print(f"FlashSVD (DRONE) | {metric_name}={acc:.4f} | peak ={peak_lr:6.1f} MiB | {t:6.1f} ms/b")
