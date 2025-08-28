# profile_svd_full.py

import os
import sys
import time
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer
import math
import torch.nn.functional as F

# ─── locate repo & model ─────────────────────────────────────────────────────
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


# ─── LinearGPT2Block: Clean Linear Format for SVD Base ──────────────────────
class LinearGPT2Block(nn.Module):
    """
    Custom GPT-2 block that converts Conv1D weights to standard Linear format
    and implements a clean forward pass for easier SVD implementation.
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


# ─── LayerShim ────────────────────────────────────────────────────────────────
class LayerShim(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        return self.block(hidden_states, attention_mask, **kwargs)


# ─── LinearSVDBlock: SVD decomposition of LinearGPT2Block ───────────────────
class LinearSVDBlock(nn.Module):
    """
    SVD-decomposed version of LinearGPT2Block for memory-efficient inference.
    """
    def __init__(self, linear_block: LinearGPT2Block, rank_attn: int, rank_ff: int, rank_out: int):
        super().__init__()
        
        # Copy configuration from linear block
        self.d_model = linear_block.d_model
        self.n_heads = linear_block.n_heads
        self.head_dim = linear_block.head_dim
        self.scale = linear_block.scale
        
        # ─── SVD Decomposition of QKV Projection (Per-Head) ─────────────────────
        qkv_weight = linear_block.qkv_proj.weight.data  # [3*d_model, d_model]
        qkv_bias = linear_block.qkv_proj.bias.data       # [3*d_model]
        
        # Split QKV into separate Q, K, V matrices
        q_weight = qkv_weight[:self.d_model, :]          # [d_model, d_model]
        k_weight = qkv_weight[self.d_model:2*self.d_model, :]  # [d_model, d_model]  
        v_weight = qkv_weight[2*self.d_model:, :]        # [d_model, d_model]
        
        q_bias = qkv_bias[:self.d_model]                 # [d_model]
        k_bias = qkv_bias[self.d_model:2*self.d_model]   # [d_model]
        v_bias = qkv_bias[2*self.d_model:]               # [d_model]
        
        # Decompose each head separately for Q, K, V
        def decompose_per_head(weight, rank):
            """Decompose weight matrix per head: [d_model, d_model] -> per-head factors"""
            Us, Vs = [], []
            for h in range(self.n_heads):
                # Extract per-head weight: [head_dim, d_model]
                head_weight = weight[h*self.head_dim:(h+1)*self.head_dim, :].float()
                
                # Apply SVD to this head
                U, S, Vh = torch.linalg.svd(head_weight, full_matrices=False)
                r = min(rank, U.shape[0], Vh.shape[0])
                
                # Create low-rank factors
                U_r = (U[:, :r] * S[:r]).to(weight.dtype)  # [head_dim, r]
                V_r = Vh[:r, :].to(weight.dtype)           # [r, d_model]
                
                Us.append(U_r)
                Vs.append(V_r)
            
            return torch.stack(Us, dim=0), torch.stack(Vs, dim=0)  # [n_heads, head_dim, r], [n_heads, r, d_model]
        
        # Decompose Q, K, V per head
        self.q_U, self.q_V = decompose_per_head(q_weight, rank_attn)  # [n_heads, head_dim, r], [n_heads, r, d_model]
        self.k_U, self.k_V = decompose_per_head(k_weight, rank_attn)
        self.v_U, self.v_V = decompose_per_head(v_weight, rank_attn)
        
        # Store biases (reshaped per head)
        self.q_bias = nn.Parameter(q_bias.view(self.n_heads, self.head_dim))  # [n_heads, head_dim]
        self.k_bias = nn.Parameter(k_bias.view(self.n_heads, self.head_dim))
        self.v_bias = nn.Parameter(v_bias.view(self.n_heads, self.head_dim))
        
        # ─── SVD Decomposition of Output Projection ─────────────────────────────
        out_weight = linear_block.out_proj.weight.data  # [d_model, d_model]
        out_bias = linear_block.out_proj.bias.data       # [d_model]
        
        U, S, Vh = torch.linalg.svd(out_weight.float(), full_matrices=False)
        r_out = min(rank_out, U.shape[0], Vh.shape[0])
        
        self.out_U = nn.Parameter((U[:, :r_out] * S[:r_out]).to(out_weight.dtype))  # [d_model, r_out]
        self.out_V = nn.Parameter(Vh[:r_out, :].to(out_weight.dtype))               # [r_out, d_model]
        self.out_bias = nn.Parameter(out_bias)
        
        # ─── SVD Decomposition of FFN Layers ─────────────────────────────────────
        # First FFN layer
        fc1_weight = linear_block.fc1.weight.data  # [intermediate, d_model]
        fc1_bias = linear_block.fc1.bias.data      # [intermediate]
        
        U, S, Vh = torch.linalg.svd(fc1_weight.float(), full_matrices=False)
        r_fc1 = min(rank_ff, U.shape[0], Vh.shape[0])
        
        self.fc1_U = nn.Parameter((U[:, :r_fc1] * S[:r_fc1]).to(fc1_weight.dtype))  # [intermediate, r_fc1]
        self.fc1_V = nn.Parameter(Vh[:r_fc1, :].to(fc1_weight.dtype))               # [r_fc1, d_model]
        self.fc1_bias = nn.Parameter(fc1_bias)
        
        # Second FFN layer
        fc2_weight = linear_block.fc2.weight.data  # [d_model, intermediate]
        fc2_bias = linear_block.fc2.bias.data      # [d_model]
        
        U, S, Vh = torch.linalg.svd(fc2_weight.float(), full_matrices=False)
        r_fc2 = min(rank_ff, U.shape[0], Vh.shape[0])
        
        self.fc2_U = nn.Parameter((U[:, :r_fc2] * S[:r_fc2]).to(fc2_weight.dtype))  # [d_model, r_fc2]
        self.fc2_V = nn.Parameter(Vh[:r_fc2, :].to(fc2_weight.dtype))               # [r_fc2, intermediate]
        self.fc2_bias = nn.Parameter(fc2_bias)
        
        # ─── Copy Layer Norms (no decomposition needed) ─────────────────────────
        self.ln1 = linear_block.ln1
        self.ln2 = linear_block.ln2
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """
        Forward pass using SVD-decomposed weights.
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # ─── Self-Attention ─────────────────────────────────────────────────
        # Layer norm + per-head QKV projection using SVD
        normed = self.ln1(hidden_states)  # [batch, seq, d_model]
        
        # Apply per-head SVD decomposition for Q, K, V
        def apply_per_head_svd(x, U_heads, V_heads, bias_heads):
            """Apply per-head SVD: x -> [batch, n_heads, seq, head_dim]"""
            # x: [batch, seq, d_model]
            # U_heads: [n_heads, head_dim, r], V_heads: [n_heads, r, d_model], bias_heads: [n_heads, head_dim]
            outputs = []
            for h in range(self.n_heads):
                # Project with per-head SVD: x @ V_h^T @ U_h^T + bias_h
                h_out = (x @ V_heads[h].T) @ U_heads[h].T + bias_heads[h]  # [batch, seq, head_dim]
                outputs.append(h_out)
            
            # Stack and reshape: [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]
            return torch.stack(outputs, dim=2).transpose(1, 2)  # [batch, n_heads, seq, head_dim]
        
        q = apply_per_head_svd(normed, self.q_U, self.q_V, self.q_bias)  # [batch, n_heads, seq, head_dim]
        k = apply_per_head_svd(normed, self.k_U, self.k_V, self.k_bias)  # [batch, n_heads, seq, head_dim]  
        v = apply_per_head_svd(normed, self.v_U, self.v_V, self.v_bias)  # [batch, n_heads, seq, head_dim]
        
        # Attention computation (same as linear format)
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
            attn_scores = attn_scores.masked_fill(~attention_mask.bool(), float('-inf'))
        
        # Softmax and apply to values
        attn_probs = F.softmax(attn_scores, dim=-1)  # [batch, n_heads, seq, seq]
        attn_output = torch.matmul(attn_probs, v)    # [batch, n_heads, seq, head_dim]
        
        # Reshape and project output using SVD
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq, n_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, d_model)  # [batch, seq, d_model]
        attn_output = (attn_output @ self.out_V.T) @ self.out_U.T + self.out_bias  # [batch, seq, d_model]
        
        # Residual connection
        hidden_states = hidden_states + attn_output
        
        # ─── Feed-Forward Network ─────────────────────────────────────────────
        # Layer norm + FFN using SVD
        normed = self.ln2(hidden_states)         # [batch, seq, d_model]
        ff_output = (normed @ self.fc1_V.T) @ self.fc1_U.T + self.fc1_bias  # [batch, seq, intermediate]
        ff_output = F.gelu(ff_output)            # GELU activation
        ff_output = (ff_output @ self.fc2_V.T) @ self.fc2_U.T + self.fc2_bias  # [batch, seq, d_model]
        
        # Residual connection
        hidden_states = hidden_states + ff_output
        
        return (hidden_states,)


if __name__ == "__main__":
    BATCH_SIZE = 8
    SEQ_LEN    = 256
    device     = "cuda"
    RANK_ATTN  = 64     # Full rank per head (head_dim = 64)
    RANK_FF    = 768   # Full rank for FFN (min(768, 3072) = 768) - exact reconstruction  
    RANK_WO    = 768   # Full rank for output projection [768, 768] - exact reconstruction  
    
    # Load dataset and tokenizer
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokz = AutoTokenizer.from_pretrained("gpt2")
    tokz.pad_token = tokz.eos_token
    
    def tokenize_fn(batch):
        return tokz(batch["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=SEQ_LEN)

    ds = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: {
                            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
                            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                        })
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.config.use_cache = False
    model = model.to(device).eval()
    
    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False
    
    # Benchmark helper for perplexity
    @torch.no_grad()
    def perplexity_peak_time(mdl):
        mdl.eval()
        total_loss, total_tokens = 0.0, 0
        start = time.perf_counter()
        torch.cuda.synchronize()
        
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            
            # Process in smaller chunks to reduce peak memory
            chunk_size = 4
            num_chunks = (batch["input_ids"].size(0) + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, batch["input_ids"].size(0))
                
                chunk_batch = {
                    "input_ids": batch["input_ids"][start_idx:end_idx],
                    "attention_mask": batch["attention_mask"][start_idx:end_idx]
                }
                
                outputs = mdl(input_ids=chunk_batch["input_ids"],
                             attention_mask=chunk_batch["attention_mask"],
                             use_cache=False)

                # Calculate loss for language modeling
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = chunk_batch["input_ids"][..., 1:].contiguous()
                mask = chunk_batch["attention_mask"][..., 1:].contiguous()
                
                valid_positions = mask.bool()
                
                if valid_positions.sum() > 0:
                    valid_logits = shift_logits[valid_positions]
                    valid_labels = shift_labels[valid_positions]
                    
                    loss = F.cross_entropy(valid_logits, valid_labels)
                    total_loss += loss.item() * valid_positions.sum().item()
                    total_tokens += valid_positions.sum().item()
                    
                    del valid_logits, valid_labels, valid_positions
                
                del outputs, shift_logits, shift_labels, mask
                torch.cuda.empty_cache()
            
        torch.cuda.synchronize()
        t = (time.perf_counter() - start)*1000.0/len(loader)
        peak = torch.cuda.max_memory_allocated()/(1024**2)
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity, peak, t

    # Create SVD Model
    svd_model = GPT2LMHeadModel.from_pretrained("gpt2")
    svd_model.config.use_cache = False
    svd_model = svd_model.to(device).eval()
    
    # Disable gradients
    for param in svd_model.parameters():
        param.requires_grad = False
    
    # Convert each layer to SVD format
    for i, layer in enumerate(svd_model.transformer.h):
        linear_block = LinearGPT2Block(layer)
        svd_block = LinearSVDBlock(linear_block, RANK_ATTN, RANK_FF, RANK_WO)
        svd_model.transformer.h[i] = LayerShim(svd_block).to(device).eval()
    
    # Reset memory tracking
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    SVD_CACHED_MEM = torch.cuda.max_memory_allocated()/1024**2
    
    # Evaluate SVD model
    perplexity, peak_svd, t_svd = perplexity_peak_time(svd_model)
    
    # Print results in the requested format
    print(f"{'Model':<15} | {'Storage (MiB)':<12} | {'Peak (MiB)':<10} | {'Transient (MiB)':<14} | {'Time (ms/b)':<10} | {'Perplexity':<10}")
    print("-" * 90)
    print(f"{'SVD':<15} | {SVD_CACHED_MEM:<12.1f} | {peak_svd:<10.1f} | {peak_svd-SVD_CACHED_MEM:<14.1f} | {t_svd:<10.1f} | {perplexity:<10.4f}")


