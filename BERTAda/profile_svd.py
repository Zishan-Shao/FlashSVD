# profile_svd.py — Build low-rank model from ARS ranks (with argparse + budget report)
# -----------------------------------------------------------------------------
#   • Loads ars_out/ranks.json.
#   • Optional global rescaling via --ars_ratio_keep (no retraining needed).
#   • Prints a compact budget report (base ARS params, scaled params, ratio).
#   • Builds factorized blocks and runs evaluation.
# -----------------------------------------------------------------------------

import os  
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
import json
import math
import argparse
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig
from evaluate import load as load_metric

from flash_attn_triton import flash_attn_triton


'''

# Use ranks as-is
CUDA_VISIBLE_DEVICES=3,4,5,6,7 python profile_svd.py \
  --model_dir /home/zs89/FlashSVD/models/BERT/bert-base-uncased-stsb \
  --task_name stsb \
  --ranks_path BERTAda/ars_out/ranks.json

# Or shrink further at inference to 85% of ARS budget (no retrain)
CUDA_VISIBLE_DEVICES=3,4,5,6,7 python profile_svd.py \
  --model_dir /home/zs89/FlashSVD/models/BERT/bert-base-uncased-stsb \
  --task_name stsb \
  --ranks_path BERTAda/ars_out/ranks.json \
  --ars_ratio_keep 0.85

'''

# ─── SVD helpers ─────────────────────────────────────────────────────────────
def build_plain_svd_helpers(model):
    def svd_per_head(Wt: torch.Tensor, rank: int):
        d_model, _ = Wt.shape
        H          = model.config.num_attention_heads
        dh         = d_model // H
        Wt3        = Wt.view(d_model, H, dh)
        Us, Vs     = [], []
        for h in range(H):
            Wh = Wt3[:, h, :].float()
            U32, S32, Vh32 = torch.linalg.svd(Wh, full_matrices=False)
            k = max(1, min(rank, S32.numel()))
            U = (U32[:, :k] * S32[:k]).to(Wt.dtype)
            V = Vh32[:k, :].to(Wt.dtype)
            Us.append(U); Vs.append(V)
        return torch.stack(Us, 0), torch.stack(Vs, 0)

    def svd_low_rank(W: torch.Tensor, rank: int):
        Wf = W.float()
        U32, S32, Vh32 = torch.linalg.svd(Wf, full_matrices=False)
        k = max(1, min(rank, S32.numel()))
        U = (U32[:, :k] * S32[:k]).to(W.dtype)
        V = Vh32[:k, :].to(W.dtype)
        return U, V

    return svd_per_head, svd_low_rank

# ─── Names & accounting ──────────────────────────────────────────────────────
def attach_fullnames(mdl: nn.Module, prefix=""):
    for name, mod in mdl.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(mod, nn.Linear):
            setattr(mod, "_ars_fullname", full)
        attach_fullnames(mod, full)

def full_name(mod: nn.Module) -> str:
    return getattr(mod, "_ars_fullname", "")

def collect_linears(model: nn.Module) -> Dict[str, nn.Linear]:
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            out[name] = mod
    return out

def cap_full_rank(lin: nn.Linear) -> int:
    return min(lin.in_features, lin.out_features)

def op_cost_per_rank(lin: nn.Linear) -> int:
    return lin.in_features + lin.out_features

def build_scaled_ranks(model: nn.Module, ars_ranks: Dict[str, int], ratio_keep: float) -> Dict[str, int]:
    linears = collect_linears(model)
    caps    = {n: cap_full_rank(m) for n, m in linears.items()}
    costs   = {n: op_cost_per_rank(m) for n, m in linears.items()}

    base = {n: int(ars_ranks.get(n, caps[n])) for n in linears}  # ARS or full
    T_base = sum(costs[n] * base[n] for n in linears)
    target = int(math.floor(ratio_keep * T_base + 1e-6))

    k = {n: max(1, min(caps[n], int(round(base[n] * ratio_keep)))) for n in linears}
    T_now = sum(costs[n] * k[n] for n in linears)
    if T_now <= target:
        return k

    # greedy water-filling: decrement the op with largest cost per rank until target
    import heapq
    heap = [(-costs[n], n) for n in linears]; heapq.heapify(heap)
    while T_now > target and heap:
        negc, name = heapq.heappop(heap)
        c = -negc
        if k[name] > 1:
            k[name] -= 1
            T_now -= c
            heapq.heappush(heap, (-c, name))
    return k

# ─── Layer shim & low-rank block ─────────────────────────────────────────────
class LayerShim(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        raw_mask = attention_mask
        if attention_mask is not None and attention_mask.dim() == 4:
            raw_mask = (attention_mask[:,0,0,:] == 0)
        return (self.block(hidden_states, raw_mask),)

class FWSVDBlock(nn.Module):
    def __init__(self, hf_layer, rkdict: dict, svd_per_head: Callable, svd_low_rank: Callable):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H

        q_lin = hf_layer.attention.self.query
        k_lin = hf_layer.attention.self.key
        v_lin = hf_layer.attention.self.value
        o_lin = hf_layer.attention.output.dense
        i_lin = hf_layer.intermediate.dense
        o2_lin= hf_layer.output.dense

        def rk(m):
            name = full_name(m)
            return max(1, int(rkdict.get(name, 1)))

        WqT = q_lin.weight.data.t(); bq = q_lin.bias.data.view(1,H,1,dh)
        WkT = k_lin.weight.data.t(); bk = k_lin.bias.data.view(1,H,1,dh)
        WvT = v_lin.weight.data.t(); bv = v_lin.bias.data.view(1,H,1,dh)

        Uq,Vq = svd_per_head(WqT, rk(q_lin))
        Uk,Vk = svd_per_head(WkT, rk(k_lin))
        Uv,Vv = svd_per_head(WvT, rk(v_lin))

        Wi, bi   = i_lin.weight.data.t(),  i_lin.bias.data
        WoT, bo2 = o2_lin.weight.data.t(), o2_lin.bias.data
        U1,V1 = svd_low_rank(Wi,  rk(i_lin))
        U2,V2 = svd_low_rank(WoT, rk(o2_lin))

        Wo_full = o_lin.weight.data; bo_attn = o_lin.bias.data
        Uo, Vo  = svd_low_rank(Wo_full.t(), rk(o_lin))

        self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))
        self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
        self.U1, self.V1, self.b1 = nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
        self.U2, self.V2, self.b2 = nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)
        self.ln1, self.ln2 = hf_layer.attention.output.LayerNorm, hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        _, H, _, _ = self.Pq.shape

        def project(x, P, V, b):
            tmp = torch.einsum("bmd,hdr->bhmr", x, P)
            return torch.einsum("bhmr,hrd->bhmd", tmp, V) + b

        Q = project(x, self.Pq[0], self.Vq[0], self.bq).contiguous()
        K = project(x, self.Pk[0], self.Vk[0], self.bk).contiguous()
        Vv= project(x, self.Pv[0], self.Vv[0], self.bv).contiguous()

        if mask is not None:
            mask4d = mask.view(B, 1, 1, M).expand(B, H, 1, M).to(torch.bool)
        else:
            mask4d = torch.ones(B, H, 1, M, device=x.device, dtype=torch.bool)

        attn = flash_attn_triton(Q, K, Vv, mask4d, BLOCK_M=32)
        del Q, K, Vv; torch.cuda.empty_cache()

        attn = attn.transpose(1,2).reshape(B, M, dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)

        mid  = x1 @ self.U1
        midV = mid @ self.V1
        midA = F.gelu(midV + self.b1)
        y    = (midA @ self.U2) @ self.V2 + self.b2
        out  = self.ln2(x1 + y)
        return out

# ─── CLI + main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build/eval low-rank model from ARS ranks.")
    parser.add_argument("--task_name", type=str, default="stsb")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to HF checkpoint dir")
    parser.add_argument("--ranks_path", type=str, required=True, help="Path to ars_out/ranks.json")
    parser.add_argument("--ars_ratio_keep", type=float, default=1.0, help="Scale ARS ranks globally (1.0 = as-is)")
    parser.add_argument("--seq_len", type=int, default=256*2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if (args.device in ["cuda","cpu"]) else ("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    val_split = "validation_matched" if args.task_name == "mnli" else "validation"
    raw = load_dataset("glue", args.task_name, split=val_split)
    tokz = AutoTokenizer.from_pretrained(args.model_dir)

    single_sent = {"cola", "sst2"}
    field_map = {
        "qqp": ("question1", "question2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "stsb": ("sentence1", "sentence2"),
        "rte":  ("sentence1", "sentence2"),
        "mrpc": ("sentence1", "sentence2"),
    }

    def tokenize_fn(batch):
        if args.task_name in single_sent:
            return tokz(batch["sentence"], padding="max_length", truncation=True, max_length=args.seq_len)
        else:
            f1, f2 = field_map[args.task_name]
            return tokz(batch[f1], batch[f2], padding="max_length", truncation=True, max_length=args.seq_len)

    remove_cols = [c for c in raw.column_names if c != "label"]
    ds = raw.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: {
                            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
                            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                            "labels":         torch.tensor([x["label"]         for x in b]),
                        })

    # Model
    if args.task_name == "mnli":
        num_labels, problem_type = 3, None
    elif args.task_name == "stsb":
        num_labels, problem_type = 1, "regression"
    else:
        num_labels, problem_type = 2, None

    cfg = AutoConfig.from_pretrained(args.model_dir, num_labels=num_labels, problem_type=problem_type)
    model = BertForSequenceClassification.from_pretrained(args.model_dir, config=cfg).to(device).eval()

    # Names for rank lookup
    attach_fullnames(model)

    # Load ranks
    if not os.path.exists(args.ranks_path):
        raise FileNotFoundError(f"Could not find ranks file: {args.ranks_path}")
    with open(args.ranks_path, "r") as f:
        ars_ranks = json.load(f)

    # Build scaled ranks and budget report
    linears = collect_linears(model)
    caps    = {n: cap_full_rank(m) for n, m in linears.items()}
    costs   = {n: op_cost_per_rank(m) for n, m in linears.items()}

    base = {n: int(ars_ranks.get(n, caps[n])) for n in linears}  # ARS or full
    T_base = sum(costs[n] * base[n] for n in linears)

    rkdict = build_scaled_ranks(model, ars_ranks=ars_ranks, ratio_keep=args.ars_ratio_keep)
    T_scaled = sum(costs[n] * rkdict[n] for n in rkdict)
    achieved = (T_scaled / (T_base + 1e-12)) if T_base > 0 else 1.0

    print("\n=== Inference Budget Report ===")
    print(f"ops               : {len(linears)}")
    print(f"ARS base params   : {T_base:,}")
    print(f"scaled params     : {T_scaled:,}  (keep={args.ars_ratio_keep:.3f})")
    print(f"achieved ratio    : {achieved*100:.2f}%")
    print(f"saved vs ARS base : {100*(1-achieved):.2f}%\n")

    # Build SVD blocks
    svd_per_head, svd_low_rank = build_plain_svd_helpers(model)

    class FWSVDBlockShim(FWSVDBlock):
        pass

    # Patch encoder
    for i, layer in enumerate(model.bert.encoder.layer):
        blk = FWSVDBlockShim(layer, rkdict, svd_per_head, svd_low_rank)
        model.bert.encoder.layer[i] = LayerShim(blk).to(device).eval().float()

    # --- Eval ---
    @torch.no_grad()
    def acc_peak_time(mdl):
        mdl.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if args.task_name == "stsb":
            metric = load_metric("pearsonr"); key = "pearsonr"
        else:
            metric = load_metric("accuracy"); key = "accuracy"
        total, steps = 0.0, 0
        start = time.perf_counter()
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            logits = mdl(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            if args.task_name == "stsb":
                preds = logits.squeeze(-1)
            else:
                preds = torch.argmax(logits, -1)
            total += metric.compute(predictions=preds.cpu(), references=batch["labels"].cpu())[key]
            steps += 1
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start)*1000/steps
        peak = torch.cuda.max_memory_allocated()/1024**2
        return total/steps, peak, elapsed

    metric_name = "pearson" if args.task_name == "stsb" else "acc"
    score, peak_mem, ms = acc_peak_time(model)
    print(f"LowRank SVD (ARS) | {metric_name}={score:.4f} | peak ={peak_mem:6.1f} MiB | {ms:6.1f} ms/b")

if __name__ == "__main__":
    main()
