import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from evaluate import load as load_metric

# ─── 0) Setup ─────────────────────────────────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.fwsvd import (
    compute_row_sum_svd_decomposition,
    estimate_fisher_weights_bert,
    estimate_fisher_weights_bert_with_attention,
)
MODEL_DIR = os.path.join(REPO_ROOT, "model", "bert-base-uncased-sst2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 1) Load model & tokenizer ───────────────────────────────────────────────
model     = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# ─── 2) Plain SVD helpers ─────────────────────────────────────────────────────
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
            Us.append((U32[:, :rank] * S32[:rank]).to(Wt.dtype))
            Vs.append(Vh32[:rank, :].to(Wt.dtype))
        return torch.stack(Us,0), torch.stack(Vs,0)

    def svd_low_rank(W: torch.Tensor, rank: int):
        Wf            = W.float()
        U32, S32, Vh32 = torch.linalg.svd(Wf, full_matrices=False)
        U = (U32[:, :rank] * S32[:rank]).to(W.dtype)
        V = Vh32[:rank, :].to(W.dtype)
        return U, V

    return svd_per_head, svd_low_rank

svd_per_head, svd_low_rank = build_plain_svd_helpers(model)

# ─── 3) In-place SVD decomposition ─────────────────────────────────────────────
def decompose_bert_layer(layer: nn.Module, rank_attn: int, rank_ffn: int):
    H = model.config.num_attention_heads

    # Attention projections
    for name in [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
    ]:
        lin: nn.Linear = dict(layer.named_modules())[name]
        Wt = lin.weight.data.t().contiguous()        # [d_model, d_model]
        Uh, Vh = svd_per_head(Wt, rank_attn)         # Uh:[H,d_model,r], Vh:[H,r,dh]
        d_model, _ = Wt.shape
        dh         = d_model // H
        r          = rank_attn

        # build U_big: [d_model, r*H]
        U_big = Uh.permute(1,0,2).reshape(d_model, r*H)

        # build V_big: [r*H, d_model]
        V_big = torch.zeros(r*H, d_model, dtype=Wt.dtype, device=Wt.device)
        for h in range(H):
            V_big[h*r:(h+1)*r, h*dh:(h+1)*dh] = Vh[h]

        seq = nn.Sequential(
            nn.Linear(lin.in_features,  r*H, bias=False),
            nn.Linear(r*H, lin.out_features, bias=True),
        ).to(DEVICE)
        seq[0].weight.data.copy_(V_big)
        seq[1].weight.data.copy_(U_big)
        seq[1].bias.data.copy_(lin.bias.data)

        parent, attr = name.rsplit(".",1)
        setattr(dict(layer.named_modules())[parent], attr, seq)

    # FFN projections
    for name, rank in [
        ("intermediate.dense",  rank_ffn),
        ("output.dense",        rank_ffn),
    ]:
        lin: nn.Linear = dict(layer.named_modules())[name]
        U, V = svd_low_rank(lin.weight.data, rank)
        seq = nn.Sequential(
            nn.Linear(lin.in_features, rank, bias=False),
            nn.Linear(rank, lin.out_features, bias=True),
        ).to(DEVICE)
        seq[0].weight.data.copy_(V)
        seq[1].weight.data.copy_(U)
        seq[1].bias.data.copy_(lin.bias.data)

        parent, attr = name.rsplit(".",1)
        setattr(dict(layer.named_modules())[parent], attr, seq)

# apply to all encoder layers
for layer in model.bert.encoder.layer:
    decompose_bert_layer(layer, rank_attn=64, rank_ffn=192 // 2)

# ─── 4) Prepare SST-2 DataLoaders ─────────────────────────────────────────────
ds = load_dataset("glue", "sst2")

# tokenize and remove the raw "sentence" column so DataCollator only sees tensors+labels
tokenized = ds.map(
    lambda ex: tokenizer(ex["sentence"], truncation=True),
    batched=True,
    remove_columns=["sentence"]
)
collator    = DataCollatorWithPadding(tokenizer)
train_loader = DataLoader(tokenized["train"],      batch_size=32, shuffle=True,  collate_fn=collator)
val_loader   = DataLoader(tokenized["validation"], batch_size=64, shuffle=False, collate_fn=collator)

# ─── 5) Fine-tuning ───────────────────────────────────────────────────────────
metric    = load_metric("accuracy")
optimizer = optim.AdamW(model.parameters(), lr=3e-5) # 2e-5
model.train()

for epoch in range(5):
    total_loss = 0.0
    for batch in train_loader:
        # split inputs vs labels
        inputs = {
            "input_ids":      batch["input_ids"].to(DEVICE),
            "attention_mask": batch["attention_mask"].to(DEVICE),
        }
        labels = batch["labels"].to(DEVICE)

        outputs = model(**inputs, labels=labels)
        loss    = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Train] Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

    # ── Validation ─────────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                "input_ids":      batch["input_ids"].to(DEVICE),
                "attention_mask": batch["attention_mask"].to(DEVICE),
            }
            labels = batch["labels"].to(DEVICE)

            logits = model(**inputs).logits
            preds  = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = metric.compute(predictions=all_preds, references=all_labels)["accuracy"]
    print(f"[Eval ] Epoch {epoch+1} accuracy: {acc:.4f}")
    model.train()
    
    
# ─── 6) Inference on examples ─────────────────────────────────────────────────
model.eval()

examples = [
    "A wonderfully executed performance!",
    "This was utterly dull and uninspired."
]
enc = tokenizer(examples, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    logits = model(**enc).logits
    preds  = torch.argmax(logits, dim=-1).cpu().tolist()

for sent, p in zip(examples, preds):
    print(f"\"{sent}\" --> {model.config.id2label[p]}")


