# profile_svd_full.py  — DRONE-style data-aware low-rank (one-shot)
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
import itertools
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig
from evaluate import load as load_metric

from flash_attn_triton import flash_attn_triton

# ─── locate repo & model ─────────────────────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

task_name = "sst2"
MODEL_DIR = os.path.join(REPO_ROOT, "models/BERT", f"bert-base-uncased-{task_name}")

# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------
def _safe_cholesky(C: torch.Tensor, max_tries: int = 5, base_eps: float = 1e-6):
    """
    Robust Cholesky: if C is near-singular (few calibration samples), add jitter.
    Returns lower-triangular S with C + eps*I = S S^T.
    """
    D = C.shape[-1]
    eps = base_eps * float(C.diag().mean().item() + 1.0)
    I = torch.eye(D, dtype=C.dtype, device=C.device)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(C + eps * I)
        except RuntimeError:
            eps *= 10.0
    # last resort: add bigger ridge
    return torch.linalg.cholesky(C + (1e-2 * float(C.diag().mean().item() + 1.0)) * I)

def _data_aware_low_rank(W_in_out: torch.Tensor, rank: int, cov_in: torch.Tensor):
    """
    DRONE-style factorization for a general Linear with weight shaped as we use it
    in this file: W has shape [d_in, d_out] (note: we pass .t() for some modules).
    Returns U:[d_in,k], V:[k,d_out] so that W' ≈ U @ V minimizing ||X^T(W-W')||_F.

    Steps: S = chol(C), A = (W^T) @ S;  A = U Σ V^T;  Udata = S^{-T} V Σ^{1/2},  Vdata = Σ^{1/2} U^T
    """
    d_in, d_out = W_in_out.shape
    if rank <= 0:
        raise ValueError("rank must be positive")

    # float32 for numerical stability
    Wf = W_in_out.float()
    Cf = cov_in.float()

    # Cholesky of input covariance C = S S^T
    S = _safe_cholesky(Cf)  # [d_in, d_in], lower

    # A = (W^T) @ S  -> [d_out, d_in]
    A = Wf.t().contiguous() @ S

    # SVD(A) = U [d_out,k] , s [k] , V [d_in,k]
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    V = Vh.t()

    k = min(rank, s.numel())
    U_k = U[:, :k]          # [d_out,k]
    s_k = s[:k]             # [k]
    V_k = V[:, :k]          # [d_in,k]

    # Compute S^{-T} V_k  by solving  S^T X = V_k
    X = torch.linalg.solve_triangular(S.t(), V_k, upper=True)  # [d_in,k]
    sqrt_s = torch.sqrt(torch.clamp(s_k, min=0))
    # U_data = (S^{-T} V_k) * sqrt(s)  (columnwise scaling)
    U_data = X * sqrt_s.unsqueeze(0)          # [d_in,k]
    # V_data = sqrt(s) * U_k^T
    V_data = (sqrt_s.unsqueeze(1) * U_k.t())  # [k,d_out]

    return U_data.to(W_in_out.dtype), V_data.to(W_in_out.dtype)

def _data_aware_per_head(Wt_dm_dh: torch.Tensor, rank: int, cov_in: torch.Tensor, num_heads: int):
    """
    DRONE-style factorization per-head for attention Q/K/V.
    Input:
      - Wt_dm_dh: weight^T with shape [d_model, d_head * H] but we reshape per-head
                  (we pass per-head as [d_model, dh] same as the original code did)
      - rank: target rank per head
      - cov_in: input covariance for this linear's input space (d_model x d_model)
      - num_heads: H
    Returns:
      stacked U:[H, d_model, rank], V:[H, rank, dh]
    """
    d_model = Wt_dm_dh.shape[0]
    dh = Wt_dm_dh.shape[1] // num_heads
    Wt3 = Wt_dm_dh.view(d_model, num_heads, dh)

    Us, Vs = [], []
    for h in range(num_heads):
        Wh = Wt3[:, h, :]  # [d_model, dh]
        Uh, Vh = _data_aware_low_rank(Wh, rank, cov_in)
        Us.append(Uh)                 # [d_model, rank]
        Vs.append(Vh)                 # [rank, dh]
    U = torch.stack(Us, dim=0)        # [H, d_model, rank]
    V = torch.stack(Vs, dim=0)        # [H, rank, dh]
    return U, V

# -----------------------------------------------------------------------------
# 1) LayerShim
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
# 2) Data-aware SVDBlock (DRONE)
# -----------------------------------------------------------------------------
class SVDBlock(nn.Module):
    def __init__(
        self,
        hf_layer,
        rank_attn: int,
        rank_ff: int,
        cov_attn_in: torch.Tensor,
        cov_attn_out: torch.Tensor,
        cov_ffn_in: torch.Tensor,
        cov_ffn_out: torch.Tensor,
        rank_wo: int = 768,
    ):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        d_ff    = hf_layer.intermediate.dense.out_features

        # 1) grab weights (transpose to [d_in, d_out] as in original code)
        WqT = hf_layer.attention.self.query.weight.data.t()   # [dm, dm] but we do per-head slicing
        WkT = hf_layer.attention.self.key.weight.data.t()
        WvT = hf_layer.attention.self.value.weight.data.t()
        bq  = hf_layer.attention.self.query.bias.data.view(1, H, 1, dh)
        bk  = hf_layer.attention.self.key.bias.data.view(1, H, 1, dh)
        bv  = hf_layer.attention.self.value.bias.data.view(1, H, 1, dh)

        # 2) DRONE factorization per head on Q/K/V using cov_attn_in (dm x dm)
        Uq, Vq = _data_aware_per_head(WqT, rank_attn, cov_attn_in, H)
        Uk, Vk = _data_aware_per_head(WkT, rank_attn, cov_attn_in, H)
        Uv, Vv = _data_aware_per_head(WvT, rank_attn, cov_attn_in, H)

        # 3) FFN factorization (data-aware)
        Wi   = hf_layer.intermediate.dense.weight.data.t()     # [dm, d_ff]
        bi   = hf_layer.intermediate.dense.bias.data
        WoT  = hf_layer.output.dense.weight.data.t()           # [d_ff, dm]
        bo2  = hf_layer.output.dense.bias.data

        U1, V1 = _data_aware_low_rank(Wi,  rank_ff, cov_ffn_in)   # input cov: dm x dm
        U2, V2 = _data_aware_low_rank(WoT, rank_ff, cov_ffn_out)  # input cov: d_ff x d_ff

        # 4) Attention output projection W_o (data-aware)
        Wo_full = hf_layer.attention.output.dense.weight.data    # [dm, dm] (out,in)
        bo_attn = hf_layer.attention.output.dense.bias.data
        # We followed the original code convention to pass .t() so shape is [dm, dm]
        Uo, Vo = _data_aware_low_rank(Wo_full.t(), rank_wo, cov_attn_out)  # input cov: dm x dm

        # stash everything as Parameters
        self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))

        self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)

        self.U1, self.V1, self.b1 = nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
        self.U2, self.V2, self.b2 = nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

        self.ln1, self.ln2 = hf_layer.attention.output.LayerNorm, hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        _, H, _, R = self.Pq.shape
        dh = dm // H

        # project into low-rank Q/K/V
        def project(x, P, V, b):
            # x:[B,M,dm], P:[H,dm,R], V:[H,R,dh], b:[1,H,1,dh]
            tmp = torch.einsum("bmd,hdr->bhmr", x, P)
            return torch.einsum("bhmr,hrd->bhmd", tmp, V) + b

        Q = project(x, self.Pq[0], self.Vq[0], self.bq).contiguous()
        K = project(x, self.Pk[0], self.Vk[0], self.bk).contiguous()
        V = project(x, self.Pv[0], self.Vv[0], self.bv).contiguous()

        # Attention mask
        if mask is not None:
            mask4d = mask.view(B, 1, 1, M).expand(B, H, 1, M).to(torch.bool)
        else:
            mask4d = torch.ones(B, H, 1, M, device=x.device, dtype=torch.bool)

        # Flash-attn returns [B, H, M, dh] float32
        attn = flash_attn_triton(Q, K, V, mask4d, BLOCK_M=32)

        del Q, K, V
        torch.cuda.empty_cache()

        # back to [B,M,dm]
        attn = attn.transpose(1, 2).reshape(B, M, dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)

        # FFN: (dm -> d_ff -> dm)
        mid  = x1 @ self.U1
        midV = mid @ self.V1
        midA = F.gelu(midV + self.b1)
        y    = (midA @ self.U2) @ self.V2 + self.b2
        out  = self.ln2(x1 + y)
        return out

# -----------------------------------------------------------------------------
# 3) Calibration: collect per-layer input covariances (one-shot)
# -----------------------------------------------------------------------------
@torch.no_grad()
def calibrate_covariances(model: BertForSequenceClassification,
                          loader: DataLoader,
                          device: str,
                          max_batches: int = 4) -> Dict[str, List[torch.Tensor]]:
    """
    Collects (online) covariance estimates for inputs of:
      - attention.self.query (shared for Q/K/V)  -> dm x dm
      - attention.output.dense                   -> dm x dm
      - intermediate.dense                       -> dm x dm
      - output.dense (FFN out, post-GELU input)  -> d_ff x d_ff
    Returns dict with lists over layers.
    """
    model.eval()
    enc = model.bert.encoder
    num_layers = len(enc.layer)
    dm = model.config.hidden_size
    d_ff = model.config.intermediate_size

    # Allocate accumulators on CUDA for speed; finalize to CPU later
    cov_attn_in  = [torch.zeros(dm, dm,  dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_attn_in    = [0 for _ in range(num_layers)]

    cov_attn_out = [torch.zeros(dm, dm,  dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_attn_out   = [0 for _ in range(num_layers)]

    cov_ffn_in   = [torch.zeros(dm, dm,  dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_ffn_in     = [0 for _ in range(num_layers)]

    cov_ffn_out  = [torch.zeros(d_ff, d_ff, dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_ffn_out    = [0 for _ in range(num_layers)]

    handles = []

    def _upd(cov_mat, n_store, idx, x):
        # x:[B,M,D] -> [N,D]
        if x is None:
            return
        x = x.detach()
        BMD = x.shape[0] * x.shape[1]
        X2d = x.reshape(BMD, x.shape[-1]).to(device=device, dtype=torch.float32)
        cov_mat[idx] += X2d.t() @ X2d
        n_store[idx] += BMD

    # Register hooks
    for i, layer in enumerate(enc.layer):
        # Inputs to Q/K/V (they share the same input): hook query pre-forward
        def q_pre_hook(mod, inp, idx=i):
            _upd(cov_attn_in, n_attn_in, idx, inp[0])
        handles.append(layer.attention.self.query.register_forward_pre_hook(q_pre_hook))

        # Inputs to attention.output.dense (post attention, before add&norm)
        def attn_out_pre_hook(mod, inp, idx=i):
            _upd(cov_attn_out, n_attn_out, idx, inp[0])
        handles.append(layer.attention.output.dense.register_forward_pre_hook(attn_out_pre_hook))

        # Inputs to intermediate.dense (after LN1)
        def ffn_in_pre_hook(mod, inp, idx=i):
            _upd(cov_ffn_in, n_ffn_in, idx, inp[0])
        handles.append(layer.intermediate.dense.register_forward_pre_hook(ffn_in_pre_hook))

        # Inputs to FFN output.dense (post-GELU)
        def ffn_out_pre_hook(mod, inp, idx=i):
            _upd(cov_ffn_out, n_ffn_out, idx, inp[0])
        handles.append(layer.output.dense.register_forward_pre_hook(ffn_out_pre_hook))

    # Run a few batches to collect stats
    seen = 0
    for batch in loader:
        if seen >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        seen += 1

    # Remove hooks
    for h in handles:
        h.remove()

    # Finalize covariance: normalize and add small ridge for stability; move to CPU
    def _finalize(cov_list, n_list):
        out = []
        for C, n in zip(cov_list, n_list):
            if n == 0:
                # fallback to identity if no samples (shouldn't happen)
                D = C.shape[0]
                Cn = torch.eye(D, dtype=torch.float32, device=C.device)
            else:
                Cn = C / float(n)
                # light ridge
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
# 4) Benchmark helper
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
        logits = mdl(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"]).logits
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
# 5) Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    BATCH_SIZE = 32
    SEQ_LEN    = 128 * 2
    device     = "cuda"
    RANK_ATTN  = 64 // 2
    RANK_FF    = 768 // 2
    RANK_WO    = 768 // 2

    # ─── GLUE load & tokenize ────────────────────────────────────────────────
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

    # ─── Build & move model ──────────────────────────────────────────────────
    if task_name == "mnli":
        num_labels, problem_type = 3, None
    elif task_name == "stsb":
        num_labels, problem_type = 1, "regression"
    else:
        num_labels, problem_type = 2, None

    cfg = AutoConfig.from_pretrained(MODEL_DIR, num_labels=num_labels, problem_type=problem_type)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR, config=cfg)
    model = model.to(device).eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # ─── Calibration pass: collect covariances ───────────────────────────────
    print("Calibrating input covariances (DRONE)…")
    covs = calibrate_covariances(model, loader, device, max_batches=4)

    # ─── Replace each encoder layer with data-aware low-rank block ───────────
    for i, layer in enumerate(model.bert.encoder.layer):
        blk = SVDBlock(
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
    def summarize_dense_vs_lowrank(model):
        dense_bytes, lowrank_bytes = 0, 0
        for name, p in model.named_parameters():
            size = p.numel() * p.element_size()
            if ".block." in name or (
                name.startswith("bert.encoder.layer")
                and any(part in name for part in ("Pq","Vq","Pk","Vk","Pv","Vv","U1","V1","U2","V2","Uo","Vo"))
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
        return (dense_bytes+lowrank_bytes)

    baseline_bytes = summarize_dense_vs_lowrank(model)
    with_act = torch.cuda.max_memory_allocated() / 1024**2
    print(f"low-rank model storage with GPU redundancy: {with_act:.1f} MiB")

    print(f"Persistent low-rank model storage (DRONE): {baseline_bytes/1024**2:6.1f} MiB")

    # ─── Evaluate ────────────────────────────────────────────────────────────
    metric_name = "pearson" if task_name == "stsb" else "acc"
    acc, peak_lr, t = acc_peak_time(model, loader, device, task_name)
    print(f"Data-aware (DRONE) | {metric_name}={acc:.4f} | peak ={peak_lr:6.1f} MiB | {t:6.1f} ms/b")
