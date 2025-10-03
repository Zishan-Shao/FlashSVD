# gpt2.py should load the pre-computed low-rank factors for computation
#         it should not redo the SVD, but load it directly

# NOTE: this code do inference with built-in GPT2
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

