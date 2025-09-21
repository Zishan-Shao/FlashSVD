# profile_svd_kv.py
# we should expect this file did exact work as profile_svd_full.py
# but it should uses the KV-Cache to reduce the memory footprint

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

# ─── helpers ─────────────────────────────────────────────────────────────────
def _apply_attention_mask_additively(attn_scores, attention_mask, batch_size, query_len, kv_len):
    """
    Make attn_scores respect attention_mask following Hugging Face semantics:
    - If attention_mask is 2D [B, K] with 1=keep, 0=pad, convert to additive and add.
    - If attention_mask is 4D additive [B,1,1,K] (or [B,1,Q,K]), just add it.
    Shapes after this:
      attn_scores: [B, H, Q, K]
    """
    if attention_mask is None:
        return attn_scores

    if attention_mask.dim() == 2:
        # [B, K_partial] with 1=keep, 0=pad. Resize to K.
        if attention_mask.size(-1) < kv_len:
            pad = kv_len - attention_mask.size(-1)
            # Pad with 1s: the missing (past) tokens are real tokens, not padding.
            attention_mask = F.pad(attention_mask, (0, pad), value=1)
        elif attention_mask.size(-1) > kv_len:
            attention_mask = attention_mask[:, -kv_len:]

        # Convert to additive: 0 -> keep (0.0), 1 -> keep; (1 - mask) -> 1 for PAD
        additive = (1.0 - attention_mask.float()) * -1e4
        additive = additive.view(batch_size, 1, 1, kv_len).to(attn_scores.dtype)
        return attn_scores + additive

    elif attention_mask.dim() == 4:
        # Already additive. Ensure last dim matches kv_len.
        if attention_mask.size(-1) != kv_len:
            if attention_mask.size(-1) < kv_len:
                pad = kv_len - attention_mask.size(-1)
                # Pad with 0.0 (no extra masking on the added past tokens)
                attention_mask = F.pad(attention_mask, (0, pad), value=0.0)
            else:
                attention_mask = attention_mask[..., -kv_len:]

        # We don't need to expand to Q explicitly; broadcasting will handle [B,1,1,K] vs [B,H,Q,K].
        return attn_scores + attention_mask.to(dtype=attn_scores.dtype)

    else:
        # Unexpected shape; be conservative and do nothing.
        return attn_scores

# ─── LinearGPT2Block: Clean Linear Format for SVD Base ──────────────────────
class LinearGPT2Block(nn.Module):
    """
    Custom GPT-2 block that converts Conv1D weights to standard Linear format
    and implements a clean forward pass for easier SVD implementation.
    Supports KV-cache for memory-efficient inference.
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
        # GPT-2 uses Conv1D (weight [out, in]); Linear also stores [out, in] (applied as x @ W^T + b).
        # Copy weights directly after transposing HF's internal storage.
        qkv_weight = hf_layer.attn.c_attn.weight.data.t()  # -> [3*d_model, d_model]
        qkv_bias = hf_layer.attn.c_attn.bias.data          # [3*d_model]
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.qkv_proj.weight.data = qkv_weight
        self.qkv_proj.bias.data = qkv_bias
        
        out_weight = hf_layer.attn.c_proj.weight.data.t()  # [d_model, d_model]
        out_bias = hf_layer.attn.c_proj.bias.data          # [d_model]
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj.weight.data = out_weight
        self.out_proj.bias.data = out_bias
        
        # ─── Convert FFN Weights to Linear Format ─────────────────────────────
        fc1_weight = hf_layer.mlp.c_fc.weight.data.t()     # [intermediate, d_model]
        fc1_bias = hf_layer.mlp.c_fc.bias.data             # [intermediate]
        self.fc1 = nn.Linear(d_model, intermediate_size, bias=True)
        self.fc1.weight.data = fc1_weight
        self.fc1.bias.data = fc1_bias
        
        fc2_weight = hf_layer.mlp.c_proj.weight.data.t()   # [d_model, intermediate]
        fc2_bias = hf_layer.mlp.c_proj.bias.data           # [d_model]
        self.fc2 = nn.Linear(intermediate_size, d_model, bias=True)
        self.fc2.weight.data = fc2_weight
        self.fc2.bias.data = fc2_bias
        
        # ─── Copy Layer Norms ─────────────────────────────────────────────────
        self.ln1 = hf_layer.ln_1
        self.ln2 = hf_layer.ln_2
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False, **kwargs):
        """
        Clean forward pass using standard linear operations with KV-cache support.
        Args:
            hidden_states: [B, Q, D]
            attention_mask: [B, K] (1/0) or [B,1,1,K] additive
            past_key_value: tuple (K_prev, V_prev) each [B, H, P, Hd]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # ─── Self-Attention ─────────────────────────────────────────────────
        normed = self.ln1(hidden_states)                # [B, Q, D]
        qkv = self.qkv_proj(normed)                     # [B, Q, 3D]
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # [3, B, H, Q, Hd]
        q, k, v = qkv[0], qkv[1], qkv[2]                # each [B, H, Q, Hd]
        
        # KV cache concat
        if past_key_value is not None and len(past_key_value) > 0:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=-2)        # [B, H, K, Hd]
            v = torch.cat([past_value, v], dim=-2)      # [B, H, K, Hd]
        kv_len = k.size(-2)
        past_len = kv_len - seq_len
        present_key_value = (k, v) if use_cache else None
        
        # Scores + causal
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, Q, K]
        i_idx = torch.arange(seq_len, device=hidden_states.device).view(seq_len, 1)
        j_idx = torch.arange(kv_len, device=hidden_states.device).view(1, kv_len)
        causal = j_idx <= (past_len + i_idx)            # [Q, K]
        attn_scores = attn_scores.masked_fill(~causal, float('-inf'))
        
        # Additive attention mask (HF semantics)
        attn_scores = _apply_attention_mask_additively(attn_scores, attention_mask, batch_size, seq_len, kv_len)
        
        # Softmax and apply to values
        attn_probs = F.softmax(attn_scores, dim=-1)     # [B, H, Q, K]
        attn_output = torch.matmul(attn_probs, v)       # [B, H, Q, Hd]
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_output = self.out_proj(attn_output)        # [B, Q, D]
        hidden_states = hidden_states + attn_output
        
        # ─── Feed-Forward Network ─────────────────────────────────────────────
        normed = self.ln2(hidden_states)
        ff_output = self.fc2(F.gelu(self.fc1(normed)))
        hidden_states = hidden_states + ff_output
        
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present_key_value,)
        return outputs

# ─── LayerShim ────────────────────────────────────────────────────────────────
class LayerShim(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, 
                head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, 
                use_cache=False, output_attentions=False, **kwargs):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        outputs = self.block(hidden_states, attention_mask=attention_mask, 
                             past_key_value=past_key_value, use_cache=use_cache, **kwargs)
        if use_cache:
            hidden_states, present_key_value = outputs
            return (hidden_states, present_key_value)
        else:
            return outputs

# ─── LinearSVDBlock: SVD decomposition of LinearGPT2Block ───────────────────
class LinearSVDBlock(nn.Module):
    """
    SVD-decomposed version of LinearGPT2Block for memory-efficient inference.
    """
    def __init__(self, linear_block: LinearGPT2Block, rank_attn: int, rank_ff: int, rank_out: int):
        super().__init__()
        
        self.d_model = linear_block.d_model
        self.n_heads = linear_block.n_heads
        self.head_dim = linear_block.head_dim
        self.scale = linear_block.scale
        
        # ─── Decompose Q, K, V per head (QR) ──────────────────────────────────
        qkv_weight = linear_block.qkv_proj.weight.data  # [3D, D]
        qkv_bias = linear_block.qkv_proj.bias.data      # [3D]
        q_weight = qkv_weight[:self.d_model, :]
        k_weight = qkv_weight[self.d_model:2*self.d_model, :]
        v_weight = qkv_weight[2*self.d_model:, :]
        q_bias = qkv_bias[:self.d_model]
        k_bias = qkv_bias[self.d_model:2*self.d_model]
        v_bias = qkv_bias[2*self.d_model:]

        def decompose_per_head(weight, rank):
            Us, Vs = [], []
            for h in range(self.n_heads):
                head_weight = weight[h*self.head_dim:(h+1)*self.head_dim, :].float()
                Q, R = torch.linalg.qr(head_weight)
                r = min(rank, Q.shape[1], R.shape[0])
                U_r = Q[:, :r].to(weight.dtype)
                V_r = R[:r, :].to(weight.dtype)
                U_r = torch.clamp(U_r, min=-1e6, max=1e6)
                V_r = torch.clamp(V_r, min=-1e6, max=1e6)
                Us.append(U_r); Vs.append(V_r)
            return torch.stack(Us, 0), torch.stack(Vs, 0)  # [H, Hd, r], [H, r, D]
        
        self.q_U, self.q_V = decompose_per_head(q_weight, rank_attn)
        self.k_U, self.k_V = decompose_per_head(k_weight, rank_attn)
        self.v_U, self.v_V = decompose_per_head(v_weight, rank_attn)
        self.q_bias = nn.Parameter(q_bias.view(self.n_heads, self.head_dim))
        self.k_bias = nn.Parameter(k_bias.view(self.n_heads, self.head_dim))
        self.v_bias = nn.Parameter(v_bias.view(self.n_heads, self.head_dim))
        
        # Output projection (QR)
        out_weight = linear_block.out_proj.weight.data
        out_bias = linear_block.out_proj.bias.data
        Q, R = torch.linalg.qr(out_weight.float())
        r_out = min(rank_out, Q.shape[1], R.shape[0])
        self.out_U = nn.Parameter(torch.clamp(Q[:, :r_out].to(out_weight.dtype), -1e6, 1e6))
        self.out_V = nn.Parameter(torch.clamp(R[:r_out, :].to(out_weight.dtype), -1e6, 1e6))
        self.out_bias = nn.Parameter(out_bias)
        
        # FFN (QR)
        fc1_weight = linear_block.fc1.weight.data
        fc1_bias = linear_block.fc1.bias.data
        Q, R = torch.linalg.qr(fc1_weight.float())
        r_fc1 = min(rank_ff, Q.shape[1], R.shape[0])
        self.fc1_U = nn.Parameter(torch.clamp(Q[:, :r_fc1].to(fc1_weight.dtype), -1e6, 1e6))
        self.fc1_V = nn.Parameter(torch.clamp(R[:r_fc1, :].to(fc1_weight.dtype), -1e6, 1e6))
        self.fc1_bias = nn.Parameter(fc1_bias)
        
        fc2_weight = linear_block.fc2.weight.data
        fc2_bias = linear_block.fc2.bias.data
        Q, R = torch.linalg.qr(fc2_weight.float())
        r_fc2 = min(rank_ff, Q.shape[1], R.shape[0])
        self.fc2_U = nn.Parameter(torch.clamp(Q[:, :r_fc2].to(fc2_weight.dtype), -1e6, 1e6))
        self.fc2_V = nn.Parameter(torch.clamp(R[:r_fc2, :].to(fc2_weight.dtype), -1e6, 1e6))
        self.fc2_bias = nn.Parameter(fc2_bias)
        
        self.ln1 = linear_block.ln1
        self.ln2 = linear_block.ln2
        
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False, **kwargs):
        """
        Forward pass using SVD-decomposed weights with KV-cache support.
        hidden_states: [B, Q, D]; past_key_value: (K_prev, V_prev)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # ─── Self-Attention ─────────────────────────────────────────────────
        normed = self.ln1(hidden_states)  # [B, Q, D]

        def apply_per_head_svd(x, U_heads, V_heads, bias_heads):
            outs = []
            for h in range(self.n_heads):
                h_out = (x @ V_heads[h].T) @ U_heads[h].T + bias_heads[h]  # [B, Q, Hd]
                h_out = torch.clamp(h_out, min=-1e6, max=1e6)
                outs.append(h_out)
            return torch.stack(outs, dim=1)  # [B, H, Q, Hd]
        
        q = apply_per_head_svd(normed, self.q_U, self.q_V, self.q_bias)  # [B, H, Q, Hd]
        k = apply_per_head_svd(normed, self.k_U, self.k_V, self.k_bias)  # [B, H, Q, Hd]
        v = apply_per_head_svd(normed, self.v_U, self.v_V, self.v_bias)  # [B, H, Q, Hd]
        
        if past_key_value is not None and len(past_key_value) > 0:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
        kv_len = k.size(-2)
        past_len = kv_len - seq_len
        present_key_value = (k, v) if use_cache else None
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, Q, K]
        i_idx = torch.arange(seq_len, device=hidden_states.device).view(seq_len, 1)
        j_idx = torch.arange(kv_len, device=hidden_states.device).view(1, kv_len)
        causal = j_idx <= (past_len + i_idx)                                  # [Q, K]
        attn_scores = attn_scores.masked_fill(~causal, float('-inf'))

        # Additive attention mask (HF semantics)
        attn_scores = _apply_attention_mask_additively(attn_scores, attention_mask, batch_size, seq_len, kv_len)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)                              # [B, H, Q, Hd]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_output = (attn_output @ self.out_V.T) @ self.out_U.T + self.out_bias
        attn_output = torch.clamp(attn_output, min=-1e6, max=1e6)
        hidden_states = hidden_states + attn_output
        
        normed = self.ln2(hidden_states)
        ff_output = (F.gelu((normed @ self.fc1_V.T) @ self.fc1_U.T + self.fc1_bias) @ self.fc2_V.T) @ self.fc2_U.T + self.fc2_bias
        ff_output = torch.clamp(ff_output, min=-1e6, max=1e6)
        hidden_states = hidden_states + ff_output
        
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present_key_value,)
        return outputs

if __name__ == "__main__":
    BATCH_SIZE = 8
    SEQ_LEN    = 128*8
    device     = "cuda"
    RANK_ATTN  = 64 // 2
    RANK_FF    = 768 // 2
    RANK_WO    = 768 // 2
    
    # Load dataset and tokenizer
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokz = AutoTokenizer.from_pretrained("gpt2")
    tokz.pad_token = tokz.eos_token
    
    def tokenize_fn(batch):
        return tokz(batch["text"], padding="max_length", truncation=True, max_length=SEQ_LEN)

    ds = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: {
                            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
                            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                        })
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.config.use_cache = True
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    
    @torch.no_grad()
    def perplexity_peak_time_kv_cache(mdl):
        mdl.eval()
        total_loss, total_tokens = 0.0, 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        validate = os.getenv("VALIDATE_CHUNK", "0") == "1"
        chosen_chunk = int(os.getenv("CHUNK_SIZE", "1024"))

        for batch_idx, batch in enumerate(loader):
            batch = {k:v.to(device) for k,v in batch.items()}
            batch_size, seq_len = batch["input_ids"].shape

            if validate and batch_idx == 0:
                # Full pass
                out_full = mdl(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"],
                               use_cache=False)
                full_logits = out_full.logits
                shift_logits = full_logits[..., :-1, :].contiguous()
                shift_labels = batch["input_ids"][..., 1:].contiguous()
                mask = batch["attention_mask"][..., 1:].contiguous().bool()
                loss_full = F.cross_entropy(shift_logits[mask], shift_labels[mask]).item()

                # Chunked pass
                all_logits_val, past_kv = [], None
                for s in range(0, seq_len, chosen_chunk):
                    e = min(s + chosen_chunk, seq_len)
                    ids = batch["input_ids"][:, s:e]
                    am  = batch["attention_mask"][:, :e]
                    o = mdl(input_ids=ids, attention_mask=am, past_key_values=past_kv, use_cache=True)
                    all_logits_val.append(o.logits)
                    past_kv = o.past_key_values
                logits_chunked = torch.cat(all_logits_val, dim=1)
                sl = logits_chunked[..., :-1, :].contiguous()
                lb = batch["input_ids"][..., 1:].contiguous()
                mk = batch["attention_mask"][..., 1:].contiguous().bool()
                loss_chunk = F.cross_entropy(sl[mk], lb[mk]).item()
                print(f"Chunk validate (chunk={chosen_chunk}): full CE={loss_full:.6f}, chunked CE={loss_chunk:.6f}, diff={abs(loss_full-loss_chunk):.6f}")

            # Normal chunked pass
            chunk_size = int(os.getenv("CHUNK_SIZE", "256"))
            all_logits, past_key_values = [], None
            for s in range(0, seq_len, chunk_size):
                e = min(s + chunk_size, seq_len)
                current_ids = batch["input_ids"][:, s:e]
                current_mask = batch["attention_mask"][:, :e]
                outputs = mdl(input_ids=current_ids,
                              attention_mask=current_mask,
                              past_key_values=past_key_values,
                              use_cache=True)
                all_logits.append(outputs.logits)
                past_key_values = outputs.past_key_values
            
            full_logits = torch.cat(all_logits, dim=1)  # [B, L, V]
            shift_logits = full_logits[..., :-1, :].contiguous()
            shift_labels = batch["input_ids"][..., 1:].contiguous()
            mask = batch["attention_mask"][..., 1:].contiguous().bool()
            
            if mask.any():
                loss = F.cross_entropy(shift_logits[mask], shift_labels[mask], reduction="mean")
                total_loss += loss.item() * mask.sum().item()
                total_tokens += mask.sum().item()

            # clear for safety
            del past_key_values
            torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        t = (time.perf_counter() - start)*1000.0/len(loader)
        peak = torch.cuda.max_memory_allocated()/(1024**2)
        
        if total_tokens == 0:
            return float('nan'), peak, t
        
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        return (ppl if math.isfinite(ppl) else float('nan')), peak, t

    # Create SVD Model
    svd_model = GPT2LMHeadModel.from_pretrained("gpt2")
    svd_model.config.use_cache = True
    svd_model = svd_model.to(device).eval()
    for p in svd_model.parameters():
        p.requires_grad = False
    
    # Replace each layer by SVD block
    for i, layer in enumerate(svd_model.transformer.h):
        linear_block = LinearGPT2Block(layer)
        svd_block = LinearSVDBlock(linear_block, RANK_ATTN, RANK_FF, RANK_WO)
        svd_model.transformer.h[i] = LayerShim(svd_block).to(device).eval()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    SVD_CACHED_MEM = torch.cuda.max_memory_allocated()/1024**2
    
    perplexity, peak_svd, t_svd = perplexity_peak_time_kv_cache(svd_model)
    
    print(f"{'Model':<15} | {'Storage (MiB)':<12} | {'Peak (MiB)':<10} | {'Transient (MiB)':<14} | {'Time (ms/b)':<10} | {'Perplexity':<10}")
    print("-" * 90)
    print(f"{'SVD KV-Cache':<15} | {SVD_CACHED_MEM:<12.1f} | {peak_svd:<10.1f} | {peak_svd-SVD_CACHED_MEM:<14.1f} | {t_svd:<10.1f} | {perplexity:<10.4f}")
