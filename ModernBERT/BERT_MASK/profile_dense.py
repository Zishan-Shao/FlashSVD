#!/usr/bin/env python3
# this will run modernbert with our defined code
import os
import sys
import itertools

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from evaluate import load as load_metric
import time


# Optional: quiet the tokenizer warning on forked dataloaders
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Path to your local ModernBERT checkpoint
MODEL_DIR = "../model/modernbert-base-sst2"


class CustomModernBERTBlock(nn.Module):
    """
    ModernBERT encoder block that matches HF behavior:
    - pre-norm residual wiring
    - calls original attention & MLP modules
    - passes BOTH attention_mask and sliding_window_mask to attention
    """

    def __init__(self, original_layer: nn.Module):
        super().__init__()
        # Keep references to original submodules (do not copy weights)
        self.config = original_layer.config
        self.attn_norm = original_layer.attn_norm
        self.attn = original_layer.attn
        self.mlp_norm = original_layer.mlp_norm
        self.mlp = original_layer.mlp

        # If HF compiled their MLP, we still call the plain path; numerically identical in eval
        self._compiled_mlp = getattr(original_layer, "compiled_mlp", None)

    def forward(
        self,
        hidden_states,
        attention_mask=None,            # 4D additive mask [B, 1, L, L] or None
        sliding_window_mask=None,       # 4D additive mask [B, 1, L, L] or None
        position_ids=None,
        output_attentions=False,
    ):
        # === Attention (pre-norm) ===
        residual = hidden_states
        normed = self.attn_norm(hidden_states)
        attn_outputs = self.attn(
            normed,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,  # <- REQUIRED by sdpa path
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        attn_out = attn_outputs[0]
        hidden_states = residual + attn_out

        # === MLP (pre-norm) ===
        residual = hidden_states
        normed = self.mlp_norm(hidden_states)
        if self._compiled_mlp is not None and self.config.reference_compile:
            mlp_out = self._compiled_mlp(hidden_states)
        else:
            mlp_out = self.mlp(normed)
        hidden_states = residual + mlp_out

        if output_attentions:
            return (hidden_states, attn_outputs[1])
        return (hidden_states,)


class CustomModernBERT(nn.Module):
    """
    Top-level wrapper that reproduces the HF forward pass:
    - embeddings → N * [block] → final_norm
    - builds attention masks exactly like HF
    - classification head: pooling → head → drop → classifier
    """

    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.config = original_model.config

        # Reuse embedding + encoder stack parts
        self.embeddings = original_model.model.embeddings
        self.layers = nn.ModuleList([CustomModernBERTBlock(lyr) for lyr in original_model.model.layers])
        self.final_norm = original_model.model.final_norm

        # Reuse HF sequence classification head
        self.head = original_model.head              # ModernBertPredictionHead
        self.drop = original_model.drop              # dropout
        self.classifier = original_model.classifier  # linear

        # Keep same attention backend as the base model; to avoid FA2 unpadding logic here,
        # we force SDPA for both models (matches HF outputs and avoids cu_seqlens plumbing).
        self.config._attn_implementation = "sdpa"

    @staticmethod
    def _default_position_ids(batch_size: int, seq_len: int, device):
        return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

    def _update_attention_masks(self, attention_mask_2d, dtype: torch.dtype):
        """
        Exact replica of HF ModernBertModel._update_attention_mask for SDPA path:
          - Build 4D global attention additive mask
          - Build 4D sliding window additive mask with bandwidth = local_attention//2
        Returns (global_attention_mask, sliding_window_mask), either may be None.
        """
        if attention_mask_2d is None:
            return None, None

        # 4D additive mask: 0 for allowed, -inf for disallowed
        global_attention_mask = _prepare_4d_attention_mask(attention_mask_2d, dtype)

        # Build window band [ |i-j| <= local_attention//2 ]
        seq_len = global_attention_mask.shape[-1]
        rows = torch.arange(seq_len, device=attention_mask_2d.device).unsqueeze(0)
        distance = torch.abs(rows - rows.T)

        half_window = int(self.config.local_attention) // 2
        window_mask = (distance <= half_window).unsqueeze(0).unsqueeze(0).to(attention_mask_2d.device)

        # Apply window to global mask; outside window → add -inf
        neg_inf = torch.finfo(dtype).min
        sliding_window_mask = global_attention_mask.masked_fill(~window_mask, neg_inf)

        return global_attention_mask, sliding_window_mask

    def forward(
        self,
        input_ids,
        attention_mask=None,   # 2D padding mask [B, L] as in HF
        position_ids=None,
        output_attentions=False,
    ):
        # Embeddings
        hidden_states = self.embeddings(input_ids)

        # Position ids (HF uses arange starting at 0)
        if position_ids is None:
            position_ids = self._default_position_ids(input_ids.shape[0], input_ids.shape[1], input_ids.device)

        # Masks (match HF exactly)
        attn_mask_4d, sliding_mask_4d = self._update_attention_masks(attention_mask, hidden_states.dtype)

        # Encoder
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

        # === Classification head (identical to HF) ===
        if getattr(self.config, "classifier_pooling", "cls") == "cls":
            pooled = hidden_states[:, 0]
        else:  # "mean"
            if attention_mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                    dim=1, keepdim=True
                )

        pooled = self.head(pooled)
        pooled = self.drop(pooled)
        logits = self.classifier(pooled)

        # Lightweight output object with .logits (like HF model output)
        out = type("Output", (), {})()
        out.logits = logits
        if output_attentions:
            out.attentions = all_attn
        return out
    
def compute_persistent_memory(m):
        total = 0
        for p in itertools.chain(m.parameters(), m.buffers()):
            total += p.numel() * p.element_size()
        return total / (1024**2)


def test_model_replication():
    """Test that our custom ModernBERT reproduces the HF model."""
    print("=== Testing ModernBERT Replication ===")
    BATCH_SIZE, SEQ_LEN = 8, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load original model (force SDPA so both paths are identical) ---
    print("Loading original ModernBERT...")
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"  # keep parity and avoid FA2 unpadding plumbing in the custom wrapper

    original_model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            config=cfg,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )

    # --- Custom model wrapping the original parts ---
    print("Creating custom ModernBERT...")
    custom_model = CustomModernBERT(original_model).to(device).eval()
    CACHED_ORIG_MEM = compute_persistent_memory(custom_model)
    # --- Data ---
    print("Loading test data...")
    raw = load_dataset("glue", "sst2", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=SEQ_LEN)

    ds = raw.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
    ds.set_format("torch")

    loader = DataLoader(
        ds,
        BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels": torch.tensor([x["label"] for x in b]),
        },
    )

    # --- End-to-end accuracy ---
    print("\n=== Testing End-to-End Accuracy ===")
    acc_metric = load_metric("accuracy")

    print("Testing original model...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = original_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            preds = out.logits.argmax(dim=-1)
            acc_metric.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())
    orig_acc = acc_metric.compute()["accuracy"]
    print(f"Original model accuracy: {orig_acc:.4f}")


    # --- Memory reset --- #
    torch.cuda.synchronize()                 # ensure original pass fully finished
    torch.cuda.empty_cache()                 # optional: free cached blocks
    torch.cuda.reset_peak_memory_stats()     # zero the peak counter

    print("Testing custom model...")
    acc_metric = load_metric("accuracy")
    start = time.perf_counter()
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = custom_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            preds = out.logits.argmax(dim=-1)
            acc_metric.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())
            n += 1
    custom_acc = acc_metric.compute()["accuracy"]
    torch.cuda.synchronize()


    t = (time.perf_counter() - start) * 1000 / n # latency 
   
    

    
    # --- Memory printout --- #
    peak = torch.cuda.max_memory_allocated()/(1024**2)

    print(f"Peak memory usage during custom model eval: {peak:6.1f} MiB")
    print(f"Transient memory (peak - persistent): {peak - CACHED_ORIG_MEM:6.1f} MiB")
    print(f"Custom model latency: {t:6.1f} ms")

    


    print(f"Custom model accuracy: {custom_acc:.4f}")

    acc_diff = abs(orig_acc - custom_acc)
    print(f"Accuracy difference: {acc_diff:.6f}")
    print("✓ Custom model exactly replicates original model accuracy!" if acc_diff < 1e-6 else "✗ Custom model differs from original model!")

    # --- Single-layer parity test (build masks like HF) ---
    print("\n=== Testing Single Layer Comparison ===")
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        x = original_model.model.embeddings(input_ids)

    # HF-style position ids
    position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)

    # Build the same 4D masks HF would (using our helper)
    global_mask, sliding_mask = custom_model._update_attention_masks(attention_mask, x.dtype)

    original_layer = original_model.model.layers[0]
    custom_layer = custom_model.layers[0]

    with torch.no_grad():
        o = original_layer(
            x,
            attention_mask=global_mask,
            sliding_window_mask=sliding_mask,
            position_ids=position_ids,
        )[0]
        c = custom_layer(
            x,
            attention_mask=global_mask,
            sliding_window_mask=sliding_mask,
            position_ids=position_ids,
        )[0]

    rel_diff = (o - c).norm() / (o.norm().clamp_min(1e-12))
    print(f"First layer output difference: {rel_diff:.6f}")
    print("✓ Custom layer exactly replicates original layer!" if rel_diff < 1e-6 else "✗ Custom layer differs from original layer!")

    print("\n=== Test completed! ===")
    return orig_acc, custom_acc, rel_diff


if __name__ == "__main__":
    test_model_replication()
