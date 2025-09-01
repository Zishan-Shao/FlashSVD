#!/usr/bin/env python3

# this one is fine

import os
import copy
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from evaluate import load as load_metric

# Optional: quiet the tokenizer warning on forked dataloaders
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Path to your local ModernBERT checkpoint
MODEL_DIR = "../model/modernbert-base-sst2"

# ----------------------------
# Low-rank building blocks
# ----------------------------

class SVDLinear(nn.Module):
    """
    Factorizes a Linear W (out x in) into two Linear layers via SVD:
      W = (U_r S_r) @ V_r^T
    Forward: x -> Linear(in->r, bias=False, weight=V_r^T) -> Linear(r->out, bias=orig_bias, weight=U_r S_r)
    If rank is None or <=0, uses full rank (min(in, out)).
    """

    def __init__(self, linear: nn.Linear, rank: int = None):
        super().__init__()
        in_f = linear.in_features
        out_f = linear.out_features
        device = linear.weight.device
        dtype = linear.weight.dtype

        # Extract weights/bias
        with torch.no_grad():
            W = linear.weight.detach()           # [out, in]
            b = linear.bias.detach() if linear.bias is not None else None

        # Compute SVD in float32 for numerical stability, then cast back
        # (if dtype is already float32 this is a no-op)
        W_f32 = W.float()
        U, S, Vh = torch.linalg.svd(W_f32, full_matrices=False)  # U:[out,r_full], S:[r_full], Vh:[r_full,in]
        r_full = S.shape[0]

        if rank is None or rank <= 0 or rank >= r_full:
            r = r_full
        else:
            r = int(rank)

        # Compose low-rank factors
        # A = U_r @ diag(S_r)  -> shape [out, r]
        # B = Vh_r             -> shape [r, in]
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]

        A = (U_r * S_r.unsqueeze(0)).to(dtype)     # [out, r]
        B = Vh_r.to(dtype)                         # [r, in]

        # Build modules
        self.proj_in = nn.Linear(in_f, r, bias=False, device=device, dtype=dtype)
        self.proj_out = nn.Linear(r, out_f, bias=(b is not None), device=device, dtype=dtype)

        # Load weights
        with torch.no_grad():
            self.proj_in.weight.copy_(B)          # [r, in]
            self.proj_out.weight.copy_(A)         # [out, r]
            if b is not None:
                self.proj_out.bias.copy_(b)

    def forward(self, x):
        return self.proj_out(self.proj_in(x))


class SVDQKV(nn.Module):
    """
    Replaces a fused Wqkv Linear (out = 3*hidden) with three SVDLinear
    modules applied separately to q/k/v, then concatenated on the last dim.
    rank_attn controls the SVD rank for each of Q/K/V (None/full for parity).
    """

    def __init__(self, wqkv_linear: nn.Linear, hidden_size: int, rank_attn: int = None):
        super().__init__()
        assert wqkv_linear.out_features == 3 * hidden_size, \
            f"Wqkv out_features={wqkv_linear.out_features}, expected 3*hidden={3*hidden_size}"

        # Split original fused params
        with torch.no_grad():
            W = wqkv_linear.weight.detach()  # [3*H, D]
            b = wqkv_linear.bias.detach() if wqkv_linear.bias is not None else None

        Wq, Wk, Wv = torch.chunk(W, 3, dim=0)
        bq, bk, bv = (None, None, None)
        if b is not None:
            bq, bk, bv = torch.chunk(b, 3, dim=0)

        # Build three SVDLinear modules with those weights/biases
        # We instantiate a dummy Linear for each then wrap with SVD.
        def make_svd_from_weight_bias(W_part, b_part):
            lin = nn.Linear(W_part.shape[1], W_part.shape[0], bias=(b_part is not None),
                            device=W_part.device, dtype=W_part.dtype)
            with torch.no_grad():
                lin.weight.copy_(W_part)
                if b_part is not None:
                    lin.bias.copy_(b_part)
            return SVDLinear(lin, rank=rank_attn)

        self.q = make_svd_from_weight_bias(Wq, bq)
        self.k = make_svd_from_weight_bias(Wk, bk)
        self.v = make_svd_from_weight_bias(Wv, bv)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return torch.cat([q, k, v], dim=-1)


def _maybe_get_local_window(config):
    """
    ModernBERT configs differ by name; grab any of these if present.
    """
    for name in ("local_attention", "local_window_size", "sliding_window"):
        if hasattr(config, name):
            return int(getattr(config, name))
    # Fallback: no local windowing
    return 0


# ----------------------------
# SVD-enabled ModernBERT block
# ----------------------------

class SVDBlock(nn.Module):
    """
    Deep-copied ModernBERT encoder block with SVD applied to:
      - Attention's fused Wqkv (split into Q/K/V and factorized separately).
      - All Linear layers inside the MLP subtree (recursively).
    Pre-norm residual wiring is preserved. Wo is left untouched by default.
    """

    def __init__(self, original_layer: nn.Module, rank_attn: int = None, rank_ffn: int = None):
        super().__init__()
        self.config = original_layer.config

        # Deep copy submodules to avoid mutating the original model
        self.attn_norm = copy.deepcopy(original_layer.attn_norm)
        self.attn = copy.deepcopy(original_layer.attn)
        self.mlp_norm = copy.deepcopy(original_layer.mlp_norm)
        self.mlp = copy.deepcopy(original_layer.mlp)

        # ---- Replace Wqkv with SVDQKV ----
        # Infer hidden size from Wqkv
        hidden_size = self.attn.Wqkv.out_features // 3
        self.attn.Wqkv = SVDQKV(self.attn.Wqkv, hidden_size, rank_attn)

        # ---- Replace all Linear layers inside MLP with SVDLinear ----
        def replace_mlps_inplace(module: nn.Module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(module, name, SVDLinear(child, rank_ffn))
                else:
                    replace_mlps_inplace(child)
        replace_mlps_inplace(self.mlp)

        # If HF compiled the MLP, ignore compiled path and use .mlp directly
        self._compiled_mlp = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,          # 4D additive mask [B,1,L,L] or None
        sliding_window_mask=None,     # 4D additive mask [B,1,L,L] or None
        position_ids=None,
        output_attentions=False,
    ):
        # === Attention (pre-norm) ===
        residual = hidden_states
        normed = self.attn_norm(hidden_states)
        attn_outputs = self.attn(
            normed,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        attn_out = attn_outputs[0]
        hidden_states = residual + attn_out

        # === MLP (pre-norm) ===
        residual = hidden_states
        normed = self.mlp_norm(hidden_states)
        mlp_out = self.mlp(normed)
        hidden_states = residual + mlp_out

        if output_attentions:
            return (hidden_states, attn_outputs[1])
        return (hidden_states,)


# ----------------------------
# SVD-enabled ModernBERT model
# ----------------------------

class ModernBERTWithSVD(nn.Module):
    """
    Wraps an HF ModernBERTForSequenceClassification:
      - Reuses embeddings, final norm, and classification head
      - Replaces each encoder layer with SVDBlock
      - Builds the same masks and position ids
    """

    def __init__(self, original_model: nn.Module, rank_attn: int = None, rank_ffn: int = None):
        super().__init__()
        self.config = original_model.config
        # Force SDPA for parity and simpler masking
        self.config._attn_implementation = "sdpa"

        # Base parts
        self.embeddings = original_model.model.embeddings
        self.final_norm = original_model.model.final_norm

        # SVD Blocks
        self.layers = nn.ModuleList(
            [SVDBlock(layer, rank_attn=rank_attn, rank_ffn=rank_ffn) for layer in original_model.model.layers]
        )

        # Classification head path (identical to HF)
        self.head = original_model.head
        self.drop = original_model.drop
        self.classifier = original_model.classifier

        # For pooling behavior ('cls' or 'mean')
        self.pooling = getattr(self.config, "classifier_pooling", "cls")

        # Cache local window size if present
        self._local_window = _maybe_get_local_window(self.config)

    @staticmethod
    def _default_position_ids(batch_size: int, seq_len: int, device):
        return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

    def _update_attention_masks(self, attention_mask_2d, dtype: torch.dtype):
        """
        Builds:
          - global 4D additive mask via HF util
          - sliding-window additive mask with bandwidth = local_window//2
        """
        if attention_mask_2d is None:
            return None, None

        global_attention_mask = _prepare_4d_attention_mask(attention_mask_2d, dtype)

        if self._local_window and self._local_window > 0:
            seq_len = global_attention_mask.shape[-1]
            rows = torch.arange(seq_len, device=attention_mask_2d.device).unsqueeze(0)
            distance = torch.abs(rows - rows.T)
            half_window = int(self._local_window) // 2
            window_ok = (distance <= half_window).unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
            neg_inf = torch.finfo(dtype).min
            sliding_window_mask = global_attention_mask.masked_fill(~window_ok, neg_inf)
        else:
            # No local window constraint; just use the global mask
            sliding_window_mask = global_attention_mask

        return global_attention_mask, sliding_window_mask

    def forward(self, input_ids, attention_mask=None, position_ids=None, output_attentions=False):
        hidden_states = self.embeddings(input_ids)

        if position_ids is None:
            position_ids = self._default_position_ids(input_ids.shape[0], input_ids.shape[1], input_ids.device)

        attn_mask_4d, sliding_mask_4d = self._update_attention_masks(attention_mask, hidden_states.dtype)

        all_attn = [] if output_attentions else None
        for layer in self.layers:
            out = layer(
                hidden_states,
                attention_mask=attn_mask_4d,
                sliding_window_mask=sliding_mask_4d,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )
            hidden_states = out[0]
            if output_attentions:
                all_attn.append(out[1])

        # Final norm
        hidden_states = self.final_norm(hidden_states)

        # Classification head (identical to HF)
        if self.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:  # mean pooling
            if attention_mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                    dim=1, keepdim=True
                )

        pooled = self.head(pooled)
        pooled = self.drop(pooled)
        logits = self.classifier(pooled)

        out = type("Output", (), {})()
        out.logits = logits
        if output_attentions:
            out.attentions = all_attn
        return out


# ----------------------------
# Quick validation harness
# ----------------------------

def test_model_replication(rank_attn=None, rank_ffn=None):
    """
    In full-rank mode (default), the SVD model should match the original.
    Set rank_attn/rank_ffn to smaller ints to test compression.
    """
    print("=== Testing ModernBERT SVD Replication ===")
    BATCH_SIZE, SEQ_LEN = 4, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load original model (force SDPA for parity)
    print("Loading original ModernBERT...")
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    original_model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, config=cfg, trust_remote_code=True
        )
        .to(device)
        .eval()
    )

    print("Building SVD ModernBERT...")
    svd_model = ModernBERTWithSVD(original_model, rank_attn=rank_attn, rank_ffn=rank_ffn).to(device).eval()

    # Data
    print("Loading data...")
    raw = load_dataset("glue", "sst2", split="validation")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    def tok_fn(b):
        return tok(b["sentence"], padding="max_length", truncation=True, max_length=SEQ_LEN)

    ds = raw.map(tok_fn, batched=True, remove_columns=["sentence", "idx"])
    ds.set_format("torch")
    loader = DataLoader(
        ds, BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels": torch.tensor([x["label"] for x in b]),
        },
    )

    # End-to-end parity check
    print("\n=== End-to-End Accuracy (first 10 batches) ===")
    metric = load_metric("accuracy")

    print("Original...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = original_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            pred = out.logits.argmax(dim=-1)
            metric.add_batch(predictions=pred.cpu(), references=batch["labels"].cpu())
    acc_orig = metric.compute()["accuracy"]
    print(f"Original accuracy: {acc_orig:.4f}")

    print("SVD model...")
    metric = load_metric("accuracy")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = svd_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            pred = out.logits.argmax(dim=-1)
            metric.add_batch(predictions=pred.cpu(), references=batch["labels"].cpu())
    acc_svd = metric.compute()["accuracy"]
    print(f"SVD  accuracy: {acc_svd:.4f}")
    print(f"Accuracy difference: {abs(acc_orig - acc_svd):.6f}")

    # Single-layer relative difference
    print("\n=== Single-Layer Output Difference ===")
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        x = original_model.model.embeddings(input_ids)

    # Build masks with the SVD model helper (same logic used in forward)
    gmask, smask = svd_model._update_attention_masks(attention_mask, x.dtype)
    pos_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)

    ori_layer = original_model.model.layers[0]
    svd_layer = svd_model.layers[0]

    with torch.no_grad():
        o = ori_layer(x, attention_mask=gmask, sliding_window_mask=smask, position_ids=pos_ids)[0]
        c = svd_layer(x, attention_mask=gmask, sliding_window_mask=smask, position_ids=pos_ids)[0]

    rel = (o - c).norm() / (o.norm().clamp_min(1e-12))
    print(f"First layer relative diff: {rel:.6e}")

    print("\n=== Done ===")
    return acc_orig, acc_svd, rel


if __name__ == "__main__":
    # Full-rank parity check by default
    # Set e.g. rank_attn=64, rank_ffn=256 to test compression.
    test_model_replication(rank_attn=None, rank_ffn=None)
