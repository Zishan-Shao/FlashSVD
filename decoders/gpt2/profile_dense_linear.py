# gpt2.py should load the pre-computed low-rank factors for computation
#         it should not redo the SVD, but load it directly

# NOTE: this one gives correct replication of forward process
#       we can use this to check the correctness of the SVD implementation

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2Model

import time
import pandas as pd

from transformers import GPT2LMHeadModel, AutoConfig
from typing import Callable



torch.manual_seed(0)


# we need to access this directory first
# use sys.path(), where the src.fwsvd is located at /home/zs89/FlashSVDFFN/src/fwsvd
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.fwsvd import (
    compute_row_sum_svd_decomposition,
    estimate_fisher_weights_bert,
    estimate_fisher_weights_bert_with_attention,
)
MODEL_DIR = os.path.join(REPO_ROOT, "model", "gpt2")


def compute_persistent_memory(m):
        total = 0
        for p in itertools.chain(m.parameters(), m.buffers()):
            total += p.numel() * p.element_size()
        return total / (1024**2)


# ─── Custom GPT-2 Block with Linear Weight Format ─────────────────────────────
class LinearGPT2Block(nn.Module):
    """
    Custom GPT-2 block that converts Conv1D weights to standard Linear format
    and implements a clean forward pass for validation purposes.
    """
    def __init__(self, hf_layer: nn.Module):
        super().__init__()
        
        # Get configuration
        self.config = hf_layer.attn
        d_model = self.config.embed_dim
        n_heads = self.config.num_heads
        head_dim = d_model // n_heads
        intermediate_size = hf_layer.mlp.c_fc.weight.shape[0]  # GPT-2 uses 4*d_model
        
        # Store dimensions
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        
        # ─── Convert Attention Weights to Linear Format ─────────────────────────
        # GPT-2 uses Conv1D: weight shape is [out_features, in_features]
        # Linear expects [out_features, in_features] but applies as input @ weight.T + bias
        # So we need to transpose Conv1D weights when copying to Linear
        
        # QKV projection: Conv1D weight [3*d_model, d_model] -> Linear [3*d_model, d_model]
        qkv_weight = hf_layer.attn.c_attn.weight.data.t()  # [d_model, 3*d_model] -> [3*d_model, d_model]
        qkv_bias = hf_layer.attn.c_attn.bias.data           # [3*d_model]
        
        # Create standard linear layers
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.qkv_proj.weight.data = qkv_weight
        self.qkv_proj.bias.data = qkv_bias
        
        # Output projection: Conv1D weight [d_model, d_model] -> Linear [d_model, d_model]
        out_weight = hf_layer.attn.c_proj.weight.data.t()  # [d_model, d_model] -> [d_model, d_model]
        out_bias = hf_layer.attn.c_proj.bias.data           # [d_model]
        
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj.weight.data = out_weight
        self.out_proj.bias.data = out_bias
        
        # ─── Convert FFN Weights to Linear Format ─────────────────────────────
        # First FFN layer: Conv1D weight [intermediate, d_model] -> Linear [intermediate, d_model]
        fc1_weight = hf_layer.mlp.c_fc.weight.data.t()   # [d_model, intermediate] -> [intermediate, d_model]
        fc1_bias = hf_layer.mlp.c_fc.bias.data            # [intermediate]
        
        self.fc1 = nn.Linear(d_model, intermediate_size, bias=True)
        self.fc1.weight.data = fc1_weight
        self.fc1.bias.data = fc1_bias
        
        # Second FFN layer: Conv1D weight [d_model, intermediate] -> Linear [d_model, intermediate]
        fc2_weight = hf_layer.mlp.c_proj.weight.data.t()  # [intermediate, d_model] -> [d_model, intermediate]
        fc2_bias = hf_layer.mlp.c_proj.bias.data           # [d_model]
        
        self.fc2 = nn.Linear(intermediate_size, d_model, bias=True)
        self.fc2.weight.data = fc2_weight
        self.fc2.bias.data = fc2_bias
        
        # ─── Copy Layer Norms ─────────────────────────────────────────────────
        self.ln1 = hf_layer.ln_1
        self.ln2 = hf_layer.ln_2
    
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """
        Clean forward pass using standard linear operations.
        
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len] or [batch_size, 1, 1, seq_len]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # ─── Self-Attention ─────────────────────────────────────────────────
        # Layer norm + QKV projection
        normed = self.ln1(hidden_states)  # [batch, seq, d_model]
        qkv = self.qkv_proj(normed)       # [batch, seq, 3*d_model]
        
        # Split and reshape QKV
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [batch, n_heads, seq, head_dim]
        
        # Attention computation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, n_heads, seq, seq]
        
        # Apply causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:  # [batch, seq]
                attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            elif attention_mask.dim() == 4:  # [batch, 1, 1, seq] or [batch, 1, seq, seq]
                if attention_mask.shape[-2] == 1:
                    attention_mask = attention_mask.expand(-1, -1, seq_len, -1)
            
            # Convert to boolean and apply
            mask_value = float('-inf')
            attn_scores = attn_scores.masked_fill(~attention_mask.bool(), mask_value)
        
        # Softmax and apply to values
        attn_probs = F.softmax(attn_scores, dim=-1)  # [batch, n_heads, seq, seq]
        attn_output = torch.matmul(attn_probs, v)    # [batch, n_heads, seq, head_dim]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq, n_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, d_model)  # [batch, seq, d_model]
        attn_output = self.out_proj(attn_output)  # [batch, seq, d_model]
        
        # Residual connection
        hidden_states = hidden_states + attn_output
        
        # ─── Feed-Forward Network ─────────────────────────────────────────────
        # Layer norm + FFN
        normed = self.ln2(hidden_states)         # [batch, seq, d_model]
        ff_output = self.fc1(normed)             # [batch, seq, intermediate]
        ff_output = F.gelu(ff_output)            # GELU activation
        ff_output = self.fc2(ff_output)          # [batch, seq, d_model]
        
        # Residual connection
        hidden_states = hidden_states + ff_output
        
        return (hidden_states,)


class LayerShim(nn.Module):
    """Wrapper to match the expected interface."""
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
    
    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        return self.block(hidden_states, attention_mask, **kwargs)


if __name__ == "__main__":
    import os, time, math, torch, pandas as pd
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from evaluate import load as load_metric
    import itertools
    
    # ─── 1. dataset & loader ────────────────────────────────────────────────────
    # For GPT-2, we'll use a text generation task instead of classification
    # Using a simple text dataset for language modeling
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokz = AutoTokenizer.from_pretrained("gpt2")
    tokz.pad_token = tokz.eos_token  # GPT-2 doesn't have a pad token by default
    
    def tokenize_fn(batch):
        return tokz(batch["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=256)  # Reduced from 512 to 256
    
    BATCH_SIZE = 8  # Reduced from 64 to 8
    ds = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: {
                            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
                            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                        })
    
    # ─── 2. constants ─────────────────────────────────────────────────────────────
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ─── 3. load & prep model ────────────────────────────────────────────────────
    # For GPT-2, we'll use perplexity as the metric instead of accuracy
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Enable memory optimizations
    model.config.use_cache = False  # Disable KV cache to save memory
    model = model.to(device).eval()
    
    # Ensure model is in inference mode and disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    CACHED_ORIG_MEM = compute_persistent_memory(model)
    print(f"GPT-2 model storage: {CACHED_ORIG_MEM:6.1f} MiB")

    # ─── Create Linear Format Model for Validation ─────────────────────────────
    print("\n=== Creating Linear Format Model ===")
    
    # Create a copy of the model with linear format blocks
    linear_model = GPT2LMHeadModel.from_pretrained("gpt2")
    linear_model.config.use_cache = False
    linear_model = linear_model.to(device).eval()
    
    # Disable gradients
    for param in linear_model.parameters():
        param.requires_grad = False
    
    # Replace transformer blocks with our custom linear format blocks
    for i, layer in enumerate(linear_model.transformer.h):
        linear_block = LinearGPT2Block(layer)
        linear_model.transformer.h[i] = LayerShim(linear_block).to(device).eval()
    
    print(f"Linear format model created with {len(linear_model.transformer.h)} custom blocks")
    
    # ─── Validation: Compare Original vs Linear Format ─────────────────────────
    print("=== Validation Check ===")
    
    # Create test input
    test_input_ids = torch.randint(0, 1000, (2, 16), device=device)
    test_attention_mask = torch.ones(2, 16, device=device)
    
    with torch.no_grad():
        # Original model output
        orig_output = model(input_ids=test_input_ids, 
                           attention_mask=test_attention_mask, 
                           use_cache=False)
        orig_logits = orig_output.logits
        
        # Linear format model output  
        linear_output = linear_model(input_ids=test_input_ids,
                                   attention_mask=test_attention_mask,
                                   use_cache=False)
        linear_logits = linear_output.logits
    
    # Compare outputs
    max_diff = (orig_logits - linear_logits).abs().max().item()
    rel_diff = (orig_logits - linear_logits).norm() / orig_logits.norm()
    
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"Relative difference: {rel_diff:.8f}")
    
    if max_diff < 1e-1:
        print("✓ Validation PASSED: Linear format model matches original")
    else:
        print("✗ Validation FAILED: Significant differences detected")
        print("Check the implementation for weight conversion issues")
    
    print("=== End Validation ===\n")
    
    # Clean up test tensors
    del test_input_ids, test_attention_mask, orig_output, linear_output
    del orig_logits, linear_logits
    torch.cuda.empty_cache()

    
    # ─── 6. perplexity + memory / time helper ────────────────────────────────────
    @torch.no_grad()
    def perplexity_peak_time(mdl, use_mask=True):
        mdl.eval()
        total_loss, total_tokens = 0.0, 0
        start = time.perf_counter()
        torch.cuda.synchronize()  # Make sure to synchronize before capturing the baseline memory
        
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            
            # Process in smaller chunks to reduce peak memory
            chunk_size = 4  # Process 4 samples at a time
            num_chunks = (batch["input_ids"].size(0) + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, batch["input_ids"].size(0))
                
                chunk_batch = {
                    "input_ids": batch["input_ids"][start_idx:end_idx],
                    "attention_mask": batch["attention_mask"][start_idx:end_idx]
                }
                
                if use_mask:
                    outputs = mdl(input_ids=chunk_batch["input_ids"],
                                 attention_mask=chunk_batch["attention_mask"],
                                 use_cache=False)  # Disable KV cache
                else:
                    outputs = mdl(input_ids=chunk_batch["input_ids"],
                                 use_cache=False)  # Disable KV cache

                # Calculate loss for language modeling - memory efficient approach
                shift_logits = outputs.logits[..., :-1, :].contiguous()  # (batch, seq-1, vocab)
                shift_labels = chunk_batch["input_ids"][..., 1:].contiguous()  # (batch, seq-1)
                
                # Only compute loss on non-padded tokens
                if use_mask:
                    mask = chunk_batch["attention_mask"][..., 1:].contiguous()  # (batch, seq-1)
                    
                    # Memory efficient loss computation using gather
                    # Only compute loss for positions where mask is True
                    valid_positions = mask.bool()
                    
                    if valid_positions.sum() > 0:
                        # Gather only the valid positions
                        valid_logits = shift_logits[valid_positions]  # (num_valid, vocab)
                        valid_labels = shift_labels[valid_positions]  # (num_valid,)
                        
                        # Compute loss only on valid positions
                        loss = F.cross_entropy(valid_logits, valid_labels)
                        total_loss += loss.item() * valid_positions.sum().item()
                        total_tokens += valid_positions.sum().item()
                        
                        # Clear intermediate tensors
                        del valid_logits, valid_labels, valid_positions
                else:
                    # For no mask case, we can use the standard approach since we don't need to filter
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                         shift_labels.view(-1))
                    total_loss += loss.item() * shift_labels.numel()
                    total_tokens += shift_labels.numel()
                
                # Clear intermediate activations
                del outputs, shift_logits, shift_labels
                if use_mask:
                    del mask
                torch.cuda.empty_cache()
            
        torch.cuda.synchronize()
        t = (time.perf_counter() - start)*1000.0/len(loader)
        peak = torch.cuda.max_memory_allocated()/(1024**2)
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity, peak, t

    # ─── 7. Dense baseline ─────────────────────────────────────────────────────
    # Reset memory stats before evaluation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    perplexity, peak_m, t = perplexity_peak_time(model, use_mask=True) 
    print(f"GPT-2 Dense w/ mask   | ppl={perplexity:.4f} | peak={peak_m:6.1f} MiB | transient={peak_m - CACHED_ORIG_MEM:6.1f} MiB | {t:6.1f} ms/b")
    
    # Reset memory stats before second evaluation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    perplexity, peak_nm, t = perplexity_peak_time(model, use_mask=False)
    print(f"GPT-2 Dense w/o mask  | ppl={perplexity:.4f} | peak={peak_nm:6.1f} MiB | transient={peak_nm - CACHED_ORIG_MEM:6.1f} MiB | {t:6.1f} ms/b")
    BASE_PEAK = max(peak_m, peak_nm)
    
    # ─── Linear Format Model Evaluation ─────────────────────────────────────────
    print(f"\n=== Linear Format Model Evaluation ===")
    
    # Calculate memory for linear format model
    LINEAR_CACHED_MEM = compute_persistent_memory(linear_model)
    print(f"Linear format model storage: {LINEAR_CACHED_MEM:6.1f} MiB")
    
    # Reset memory stats before evaluation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    perplexity, peak_m, t = perplexity_peak_time(linear_model, use_mask=True) 
    print(f"Linear format w/ mask | ppl={perplexity:.4f} | peak={peak_m:6.1f} MiB | transient={peak_m - LINEAR_CACHED_MEM:6.1f} MiB | {t:6.1f} ms/b")
    
    # Reset memory stats before second evaluation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    perplexity, peak_nm, t = perplexity_peak_time(linear_model, use_mask=False)
    print(f"Linear format w/o mask| ppl={perplexity:.4f} | peak={peak_nm:6.1f} MiB | transient={peak_nm - LINEAR_CACHED_MEM:6.1f} MiB | {t:6.1f} ms/b")
    
    # Summary comparison
    print(f"\n=== Performance Summary ===")
    print(f"Model storage: Original={CACHED_ORIG_MEM:.1f} MiB, Linear={LINEAR_CACHED_MEM:.1f} MiB")
    print(f"Memory overhead: {LINEAR_CACHED_MEM - CACHED_ORIG_MEM:+.1f} MiB")
    print("Linear format model provides a clean baseline for SVD implementation.")

