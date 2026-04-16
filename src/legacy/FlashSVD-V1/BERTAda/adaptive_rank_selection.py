# adaptive_rank_selection.py
# -----------------------------------------------------------------------------
# Adaptive Rank Selection (NAACL 2024) — with argparse + budget report
#   • Freezes a pretrained HF model (e.g., BERT).
#   • For each Linear, computes SVD of W^T and trains a hypernetwork that emits
#     binary masks (via straight-through Gumbel-Sigmoid) under a global budget.
#   • Converts masks → integer ranks and saves to ars_out/ranks.json.
#   • Prints a compact budget report (full params, kept params, achieved ratio).
# -----------------------------------------------------------------------------

import os, sys, json, math, time, random, argparse
from typing import List, Tuple
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig

'''

CUDA_VISIBLE_DEVICES=3,4,5,6,7 \
python adaptive_rank_selection.py \
  --model_dir /home/zs89/FlashSVD/models/BERT/bert-base-uncased-stsb \
  --task_name stsb \
  --p_keep 0.60 \
  --steps 800 \
  --out_dir ars_out
  
'''

# ------------------------------ Utils ----------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = True):
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y = torch.sigmoid((logits + g) / tau)
    if not hard:
        return y
    y_hard = (y > 0.5).float()
    return (y_hard - y).detach() + y  # straight-through

# ----------------------- SVD wrappers for Linear -----------------------------
def svd_of_linear_weight(linear: nn.Linear, device: str):
    W = linear.weight.data # [out, in]
    Win = W.t().float().to(device)  # [in, out]
    U, s, Vh = torch.linalg.svd(Win, full_matrices=False)  # U:[in,R], s:[R], Vh:[R,out]
    V = Vh.t().contiguous()
    return U, s, V  # FP32 on device

class MaskedSVDLinear(nn.Module):
    """
    Drop-in SVD wrapper. Forward expects mask set by context (self._current_mask).
    """
    def __init__(self, linear: nn.Linear, device: str, rank_cap: int = None):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

        U, s, V = svd_of_linear_weight(linear, device)
        Rfull = s.numel()
        R = Rfull if rank_cap is None else min(rank_cap, Rfull)
        self.U = nn.Parameter(U[:, :R], requires_grad=False)      # [in,R]
        self.s = nn.Parameter(s[:R],   requires_grad=False)       # [R]
        self.V = nn.Parameter(V[:, :R], requires_grad=False)      # [out,R]
        self.R = R
        self._current_mask = None

    def forward(self, x: torch.Tensor):
        m = self._current_mask
        if m is None:
            raise RuntimeError("MaskedSVDLinear: _current_mask is None (masking context not set).")
        ms  = (m * self.s)                       # [R]
        US  = self.U * ms.unsqueeze(0)           # [in,R]
        mid = torch.matmul(x, US)                # [...,R]
        y   = torch.matmul(mid, self.V.t())      # [...,out]
        if self.bias is not None: y = y + self.bias
        return y

    def param_count_given_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # (M + N) * sum(mask)
        return (self.in_features + self.out_features) * torch.sum(mask)

# --------------------- HyperNetwork over operations --------------------------
class SimpleHN(nn.Module):
    """
    Hypernetwork: Embedding + GRU → per-op hidden → per-op Linear heads → logits.
    """
    def __init__(self, op_sizes: List[int], feat_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.op_sizes = op_sizes
        self.L        = len(op_sizes)
        self.embed    = nn.Embedding(self.L, feat_dim)
        self.gru      = nn.GRU(input_size=feat_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.heads    = nn.ModuleList([nn.Linear(hidden, r) for r in op_sizes])

    def forward(self) -> List[torch.Tensor]:
        ids = torch.arange(self.L, device=self.embed.weight.device).long().unsqueeze(0)  # [1,L]
        z   = self.embed(ids)                                                            # [1,L,feat]
        h,_ = self.gru(z)                                                                # [1,L,H]
        h   = h.squeeze(0)                                                               # [L,H]
        return [ self.heads[i](h[i]) for i in range(self.L) ]                            # list of [R_i]

# -------------------- Model patcher: Linear -> MaskedSVDLinear ----------------
def collect_linear_modules(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    liners = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            liners.append((name, mod))
    return liners

def replace_with_masked(model: nn.Module, device: str, rank_cap_per_op: List[int], original_names: List[str]):
    ops = []
    for name, Rcap in zip(original_names, rank_cap_per_op):
        parent = model
        *parents, last = name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        lin = getattr(parent, last)
        wrapped = MaskedSVDLinear(lin, device=device, rank_cap=Rcap)
        setattr(parent, last, wrapped)
        ops.append((name, wrapped))
    return model, ops

# ---------------------------- Loss functions ---------------------------------
def topk_like(mask: torch.Tensor, s: torch.Tensor):
    with torch.no_grad():
        k = int(torch.clamp(torch.round(mask.sum()), 0, mask.numel()).item())
        idx = torch.arange(mask.numel(), device=mask.device)
        m_top = torch.zeros_like(mask); m_top[idx[:k]] = 1.0
    return m_top

def alignment_loss(mask: torch.Tensor, s: torch.Tensor):
    m_top = topk_like(mask, s)
    return torch.sum(((mask - m_top) * s) ** 2)

def parameter_budget(op_list: List[MaskedSVDLinear], masks: List[torch.Tensor], p: float):
    device = masks[0].device
    Tm, Tmax = torch.zeros((), device=device), torch.zeros((), device=device)
    for op, m in zip(op_list, masks):
        Tm   = Tm   + op.param_count_given_mask(m)
        Tmax = Tmax + (op.in_features + op.out_features) * op.R
    Tmax = p * Tmax
    # zero penalty if under budget; positive if exceed:
    return torch.log(torch.clamp(torch.maximum(Tm, Tmax) / (Tmax + 1e-12), min=1.0 + 1e-12))

# ------------------------------ Training loop --------------------------------
def train_ars(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    p_keep: float = 0.6,
    tau: float = 1.0,
    lr: float = 1e-3,
    steps: int = 1500,
    rank_cap: int = None,
    lambda_param: float = 16.0,
    gamma_align: float = 10.0,
    out_dir: str = "ars_out",
):
    model.eval().to(device)

    # 1) Snapshot original Linear modules BEFORE replacement
    orig_linear = collect_linear_modules(model)         # [(full_name, nn.Linear), ...]
    orig_names  = [n for n, _ in orig_linear]

    # 2) Replace with MaskedSVDLinear
    rank_caps = []
    full_ranks = {}   # for reporting
    costs = {}        # per-op cost per rank
    for name, lin in orig_linear:
        Rfull = min(lin.in_features, lin.out_features)
        full_ranks[name] = Rfull
        costs[name] = lin.in_features + lin.out_features
        rank_caps.append(Rfull if rank_cap is None else min(rank_cap, Rfull))

    model, ops = replace_with_masked(model, device=device, rank_cap_per_op=rank_caps, original_names=orig_names)
    masked_modules = [m for _, m in ops]

    # 3) Hypernetwork
    HN = SimpleHN(op_sizes=[m.R for m in masked_modules]).to(device)
    opt = torch.optim.Adam(HN.parameters(), lr=lr)

    # 4) Task loss (auto detect regression/classification)
    is_regression = getattr(model.config, "problem_type", None) == "regression"
    mse = nn.MSELoss()

    # 5) Mask context manager
    class MaskCtx:
        def __init__(self, mods, masks): self.mods, self.masks = mods, masks
        def __enter__(self):
            for mod, m in zip(self.mods, self.masks): mod._current_mask = m
        def __exit__(self, exc_type, exc, tb):
            for mod in self.mods: mod._current_mask = None

    def one_step(batch):
        opt.zero_grad(set_to_none=True)
        logits_list = HN()  # list of [R_l]
        masks = [gumbel_sigmoid(logits, tau=tau, hard=True) for logits in logits_list]
        with MaskCtx(masked_modules, masks):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits

        if is_regression:
            target = batch["labels"].float().view(-1, 1).to(device)
            task_loss = mse(logits, target)
        else:
            target = batch["labels"].to(device)
            task_loss = F.cross_entropy(logits, target)

        align = 0.0
        for m, op in zip(masks, masked_modules):
            align = align + alignment_loss(m, op.s)
        rparam = parameter_budget(masked_modules, masks, p_keep)

        loss = task_loss + lambda_param * rparam + gamma_align * align
        loss.backward()
        opt.step()

        return float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(rparam.detach().cpu()), float(align.detach().cpu())

    # 6) Train
    it = iter(dataloader)
    for step in range(steps):
        try: batch = next(it)
        except StopIteration:
            it = iter(dataloader); batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, tl, rp, al = one_step(batch)
        if (step + 1) % 50 == 0:
            print(f"[{step+1:5d}] loss={loss:.4f} task={tl:.4f} param={rp:.4f} align={al:.4f}")

    # 7) Extract integer ranks (SOFT masks from raw logits)
    with torch.no_grad():
        logits_list = HN()
        masks_soft = [torch.sigmoid(l) for l in logits_list]
        ks = [int(torch.clamp(torch.round(m.sum()), 0, m.numel()).item()) for m in masks_soft]

    ranks = {name: int(k) for name, k in zip(orig_names, ks)}

    # 8) Budget report
    T_full = sum(costs[n] * full_ranks[n] for n in orig_names)
    T_keep = sum(costs[n] * ranks[n]      for n in orig_names)
    achieved = T_keep / (T_full + 1e-12)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ranks.json"), "w") as f:
        json.dump(ranks, f, indent=2)
    with open(os.path.join(out_dir, "budget_report.json"), "w") as f:
        json.dump({
            "num_ops": len(orig_names),
            "total_params_full": int(T_full),
            "total_params_kept": int(T_keep),
            "achieved_ratio": achieved,
            "target_ratio_p_keep": p_keep,
        }, f, indent=2)

    print("\n=== ARS Budget Report ===")
    print(f"ops            : {len(orig_names)}")
    print(f"full params    : {T_full:,}")
    print(f"kept params    : {T_keep:,}")
    print(f"achieved ratio : {achieved*100:.2f}% (target ~ {p_keep*100:.2f}%)")
    print(f"saved          : {100*(1-achieved):.2f}%\n")
    print(f"Saved ranks to: {os.path.join(out_dir, 'ranks.json')}")
    print(f"Saved report to: {os.path.join(out_dir, 'budget_report.json')}")

    return ranks

# ---------------------------- CLI runner -------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Adaptive Rank Selection (learn per-op ranks).")
    parser.add_argument("--task_name", type=str, default="stsb", help="GLUE task name")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to HF checkpoint dir")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--p_keep", type=float, default=0.60, help="Global param budget (keep ratio)")
    parser.add_argument("--tau", type=float, default=1.0, help="Gumbel-Sigmoid temperature")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank_cap", type=int, default=None, help="Optional per-op rank cap")
    parser.add_argument("--lambda_param", type=float, default=16.0)
    parser.add_argument("--gamma_align", type=float, default=10.0)
    parser.add_argument("--out_dir", type=str, default="ars_out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=lambda b: {
                            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
                            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                            "labels":         torch.tensor([x["label"]         for x in b]),
                        })

    # Model (frozen)
    if args.task_name == "mnli":
        num_labels, problem_type = 3, None
    elif args.task_name == "stsb":
        num_labels, problem_type = 1, "regression"
    else:
        num_labels, problem_type = 2, None

    cfg = AutoConfig.from_pretrained(args.model_dir, num_labels=num_labels, problem_type=problem_type)
    model = BertForSequenceClassification.from_pretrained(args.model_dir, config=cfg).to(device)
    for p in model.parameters(): p.requires_grad_(False)

    _ = train_ars(
        model, loader, device=device,
        p_keep=args.p_keep, tau=args.tau, lr=args.lr, steps=args.steps,
        rank_cap=args.rank_cap, lambda_param=args.lambda_param, gamma_align=args.gamma_align,
        out_dir=args.out_dir,
    )

if __name__ == "__main__":
    main()
