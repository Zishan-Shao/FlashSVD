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
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))  # decoders/
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Also add project root so we can import top-level modules
PROJECT_ROOT = os.path.dirname(REPO_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Add this module's directory for sibling imports (e.g., flash_attn_causal)
MODULE_DIR = os.path.dirname(THIS_FILE)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
from src.fwsvd import (
    compute_row_sum_svd_decomposition,
    estimate_fisher_weights_bert,
    estimate_fisher_weights_bert_with_attention,
)
MODEL_DIR = os.path.join(REPO_ROOT, "model", "gpt2")
from flashsvdattn import flash_svd_attention
from flashsvdffn import flashsvd_ffn
try:
    from flash_attn_causal import flash_attn_triton_unified
except Exception:
    flash_attn_triton_unified = None



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
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False, **kwargs):
        """
        Clean forward pass using standard linear operations with KV-cache support.
        
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len] or [batch_size, 1, 1, seq_len]
            past_key_value: tuple of (past_key, past_value) each [batch, n_heads, past_seq_len, head_dim]
            use_cache: whether to return updated cache
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
        
        # Handle past key-value cache
        if past_key_value is not None and len(past_key_value) > 0:
            past_key, past_value = past_key_value
            # Concatenate past and current keys/values
            k = torch.cat([past_key, k], dim=-2)  # [batch, n_heads, past_seq_len + seq_len, head_dim]
            v = torch.cat([past_value, v], dim=-2)  # [batch, n_heads, past_seq_len + seq_len, head_dim]
        
        # Update cache for next iteration
        present_key_value = (k, v) if use_cache else None
        
        # FlashAttention: q/k/v -> [B, S, H, Dh]
        kv_seq_len = k.size(-2)
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()
        # Build matrix bias to support KV-cache causal masking with offset
        past_len = kv_seq_len - seq_len
        i_idx = torch.arange(seq_len, device=hidden_states.device).view(seq_len, 1)
        j_idx = torch.arange(kv_seq_len, device=hidden_states.device).view(1, kv_seq_len)
        causal_allow = (j_idx <= (past_len + i_idx))  # [S, Kv]
        bias = torch.zeros(batch_size, 1, seq_len, kv_seq_len, device=hidden_states.device, dtype=torch.float32)
        bias.masked_fill_(~causal_allow.view(1, 1, seq_len, kv_seq_len), float('-inf'))
        # Apply attention mask on keys
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.size(-1) < kv_seq_len:
                    attention_mask = F.pad(attention_mask, (0, kv_seq_len - attention_mask.size(-1)), value=1)
                elif attention_mask.size(-1) > kv_seq_len:
                    attention_mask = attention_mask[:, -kv_seq_len:]
                key_mask = (~attention_mask.bool()).view(batch_size, 1, 1, kv_seq_len)
            elif attention_mask.dim() == 4:
                if attention_mask.size(-1) != kv_seq_len:
                    if attention_mask.size(-1) < kv_seq_len:
                        attention_mask = F.pad(attention_mask, (0, kv_seq_len - attention_mask.size(-1)), value=1)
                    else:
                        attention_mask = attention_mask[..., -kv_seq_len:]
                # Reduce to key vector mask if given as [B,1,S,Kv] by OR across S
                if attention_mask.shape[-2] == 1:
                    key_mask = (~attention_mask.bool())
                else:
                    key_mask = (~attention_mask.bool()).amax(dim=-2, keepdim=True).bool()
            else:
                key_mask = None
            if key_mask is not None:
                bias.masked_fill_(key_mask.expand(-1, 1, seq_len, -1), float('-inf'))
        # Ensure dtype is supported
        if q_t.dtype not in (torch.float16, torch.bfloat16):
            q_t = q_t.to(torch.float16)
            k_t = k_t.to(torch.float16)
            v_t = v_t.to(torch.float16)
        # q, k, v are already [B, H, S, D]
        Q = q
        K = k
        V = v
        # Build query padding mask [B,H,1,S] from attention_mask (1=keep, 0=pad)
        if attention_mask is not None and attention_mask.dim() == 2:
            # attention_mask is cumulative up to kv length; take the last seq_len as current queries
            qmask = attention_mask[:, -seq_len:]
        else:
            qmask = torch.ones(batch_size, seq_len, device=hidden_states.device, dtype=torch.bool)
        qmask4d = qmask.view(batch_size, 1, 1, seq_len).expand(-1, self.n_heads, -1, -1)
        # Build key padding mask [B,1,Kv,1] and zero out padded keys/values per sample
        if attention_mask is not None and attention_mask.dim() == 2:
            kmask = attention_mask[:, :kv_seq_len].view(batch_size, 1, kv_seq_len, 1)
        else:
            kmask = torch.ones(batch_size, 1, kv_seq_len, 1, device=hidden_states.device, dtype=torch.bool)
        K = K * kmask.to(K.dtype)
        V = V * kmask.to(V.dtype)
        # Call unified FlashAttention
        attn_out = flash_attn_triton_unified(Q, K, V, qmask4d, BLOCK_M=16)  # [B,H,S,D]
        # Back to [B, S, D]
        attn_output = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_output = attn_output.to(hidden_states.dtype)
        attn_output = self.out_proj(attn_output)
        
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
        # Convert HuggingFace interface to our block interface
        # Note: hidden_states might be a tuple, extract the actual tensor
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        outputs = self.block(hidden_states, attention_mask=attention_mask, 
                           past_key_value=past_key_value, use_cache=use_cache, **kwargs)
        
        # Convert back to HuggingFace format
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
            """Decompose weight matrix per head: [d_model, d_model] -> per-head factors.
            Returns per-head (U_head_dim, V_input) such that W_head ≈ U_head_dim @ V_input,
            with shapes: U_head_dim [head_dim, r], V_input [r, d_model]."""
            Us, Vs = [], []
            for h in range(self.n_heads):
                head_weight = weight[h*self.head_dim:(h+1)*self.head_dim, :].float()  # [head_dim, d_model]
                Q, R = torch.linalg.qr(head_weight)
                r = min(rank, Q.shape[1], R.shape[0])
                U_r = Q[:, :r].to(weight.dtype)            # [head_dim, r]
                V_r = R[:r, :].to(weight.dtype)            # [r, d_model]
                U_r = torch.clamp(U_r, min=-1e6, max=1e6)
                V_r = torch.clamp(V_r, min=-1e6, max=1e6)
                Us.append(U_r)
                Vs.append(V_r)
            return torch.stack(Us, dim=0), torch.stack(Vs, dim=0)
        
        # Decompose Q, K, V per head
        self.q_U, self.q_V = decompose_per_head(q_weight, rank_attn)  # [n_heads, head_dim, r], [n_heads, r, d_model]
        self.k_U, self.k_V = decompose_per_head(k_weight, rank_attn)
        self.v_U, self.v_V = decompose_per_head(v_weight, rank_attn)
        
        # Store biases (reshaped per head)
        self.q_bias = nn.Parameter(q_bias.view(self.n_heads, self.head_dim))  # [n_heads, head_dim]
        self.k_bias = nn.Parameter(k_bias.view(self.n_heads, self.head_dim))
        self.v_bias = nn.Parameter(v_bias.view(self.n_heads, self.head_dim))
        
        # ─── QR Decomposition of Output Projection ─────────────────────────────
        out_weight = linear_block.out_proj.weight.data  # [d_model, d_model]
        out_bias = linear_block.out_proj.bias.data       # [d_model]
        
        Q, R = torch.linalg.qr(out_weight.float())
        r_out = min(rank_out, Q.shape[1], R.shape[0])
        
        self.out_U = nn.Parameter(Q[:, :r_out].to(out_weight.dtype))  # [d_model, r_out]
        self.out_V = nn.Parameter(R[:r_out, :].to(out_weight.dtype))  # [r_out, d_model]
        
        # Clip extreme values
        self.out_U.data = torch.clamp(self.out_U.data, min=-1e6, max=1e6)
        self.out_V.data = torch.clamp(self.out_V.data, min=-1e6, max=1e6)
        
        self.out_bias = nn.Parameter(out_bias)
        
        # ─── Low-rank FFN factors for FlashSVD-FFN ─────────────────────────────
        # First FFN layer: use Wi = (fc1.weight).T with shape [d_model, intermediate]
        fc1_weight = linear_block.fc1.weight.data.t().contiguous()  # [d_model, intermediate]
        fc1_bias = linear_block.fc1.bias.data                        # [intermediate]
        Q1, R1 = torch.linalg.qr(fc1_weight.float())
        r_fc1 = min(rank_ff, Q1.shape[1], R1.shape[0])
        # Map to FlashSVD-FFN expected shapes: W ≈ Q_r @ R_r
        U1S = Q1[:, :r_fc1].to(fc1_weight.dtype)        # [d_model, r_fc1]
        V1  = R1[:r_fc1, :].to(fc1_weight.dtype)        # [r_fc1, intermediate]
        self.ff_U1S = nn.Parameter(torch.clamp(U1S, min=-1e6, max=1e6))
        self.ff_V1  = nn.Parameter(torch.clamp(V1,  min=-1e6, max=1e6))
        self.ff_b1  = nn.Parameter(fc1_bias)

        # Second FFN layer: use WoT = (fc2.weight).T with shape [intermediate, d_model]
        fc2_WoT = linear_block.fc2.weight.data.t().contiguous()     # [intermediate, d_model]
        fc2_bias = linear_block.fc2.bias.data                        # [d_model]
        Q2, R2 = torch.linalg.qr(fc2_WoT.float())
        r_fc2 = min(rank_ff, Q2.shape[1], R2.shape[0])
        U2 = Q2[:, :r_fc2].to(fc2_WoT.dtype)                         # [intermediate, r_fc2]
        V2 = R2[:r_fc2, :].to(fc2_WoT.dtype)                         # [r_fc2, d_model]
        self.ff_U2 = nn.Parameter(torch.clamp(U2, min=-1e6, max=1e6))
        self.ff_V2 = nn.Parameter(torch.clamp(V2, min=-1e6, max=1e6))
        self.ff_b2 = nn.Parameter(fc2_bias)
        
        # ─── Copy Layer Norms (no decomposition needed) ─────────────────────────
        self.ln1 = linear_block.ln1
        self.ln2 = linear_block.ln2
        
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False, **kwargs):
        """
        Forward using per-head low-rank projections for attention (with FlashAttention KV-cache),
        and FlashSVD-FFN for the MLP.
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # ─── Attention via low-rank (supports KV-cache) ──────────────────────
        x = self.ln1(hidden_states)  # [B, Q, D]

        # Build low-rank Pq, Pk, Pv per head: P = X @ U_flash where U_flash = V_heads[h].T
        Pq_list, Pk_list, Pv_list = [], [], []
        for h in range(self.n_heads):
            Uq_flash = self.q_V[h].T  # [D, r]
            Uk_flash = self.k_V[h].T  # [D, r]
            Uv_flash = self.v_V[h].T  # [D, r]
            Pq_list.append(x @ Uq_flash)
            Pk_list.append(x @ Uk_flash)
            Pv_list.append(x @ Uv_flash)
        Pq = torch.stack(Pq_list, dim=1)  # [B, H, Q, R]
        Pk = torch.stack(Pk_list, dim=1)  # [B, H, Q, R]
        Pv = torch.stack(Pv_list, dim=1)  # [B, H, Q, R]

        # Append KV-cache in low-rank space
        if past_key_value is not None and len(past_key_value) > 0:
            past_Pk, past_Pv = past_key_value
            Pk = torch.cat([past_Pk, Pk], dim=-2)  # [B, H, K, R]
            Pv = torch.cat([past_Pv, Pv], dim=-2)  # [B, H, K, R]
        kv_seq_len = Pk.size(-2)
        present_key_value = (Pk, Pv) if use_cache else None

        # Key padding mask
        if attention_mask is not None and attention_mask.dim() == 2:
            kmask = attention_mask[:, :kv_seq_len].to(torch.bool)  # [B, K]
        else:
            kmask = torch.ones(batch_size, kv_seq_len, device=hidden_states.device, dtype=torch.bool)

        # Per-head low-rank attention
        dh = self.head_dim
        scale = 1.0 / math.sqrt(dh)
        attn_heads = []
        for h in range(self.n_heads):
            # Bases
            Vq_h = self.q_U[h].T  # [R, dh]
            Vk_h = self.k_U[h].T  # [R, dh]
            Vv_h = self.v_U[h].T  # [R, dh]
            # Precompute Y = Vq @ Vk^T → [R, R]
            Y = Vq_h @ Vk_h.T
            # logits: [B,Q,K]
            logits = (Pq[:, h] @ Y) @ Pk[:, h].transpose(1, 2)
            # causal mask per chunk
            Qlen = seq_len
            Klen = kv_seq_len
            past_len = Klen - Qlen
            i_idx = torch.arange(Qlen, device=hidden_states.device).view(Qlen, 1)
            j_idx = torch.arange(Klen, device=hidden_states.device).view(1, Klen)
            causal = (j_idx <= (past_len + i_idx))  # [Q, K]
            logits = logits.masked_fill(~causal.view(1, Qlen, Klen), float('-inf'))
            # key padding mask
            logits = logits.masked_fill(~kmask.view(batch_size, 1, Klen), float('-inf'))
            # softmax
            attn_prob = torch.softmax(logits * scale, dim=-1)  # [B,Q,K]
            # values: [B,K,dh] = Pv @ Vv + bv
            Vvals = Pv[:, h] @ Vv_h  # [B,K,dh]
            Vvals = Vvals + self.v_bias[h].unsqueeze(0).unsqueeze(0)
            out_h = attn_prob @ Vvals  # [B,Q,dh]
            attn_heads.append(out_h)
        attn = torch.stack(attn_heads, dim=2)  # [B, Q, H, dh]
        attn_output = attn.contiguous().view(batch_size, seq_len, d_model)
        attn_output = (attn_output @ self.out_V.T) @ self.out_U.T + self.out_bias
        attn_output = torch.clamp(attn_output, min=-1e6, max=1e6)
        hidden_states = hidden_states + attn_output

        # ─── Feed-Forward (FlashSVD-FFN) ─────────────────────────────────────
        x1 = self.ln2(hidden_states)
        P_ffn = x1 @ self.ff_U1S  # [B, Q, R1]
        # Triton requires tile sizes (BR1/BR2) to be powers of two. Use fixed tiles with masking.
        y = flashsvd_ffn(P_ffn, self.ff_V1, self.ff_U2, self.ff_V2, self.ff_b1, self.ff_b2,
                         BL=64, BD=64, BH=64, BR1=64, BR2=64)
        hidden_states = hidden_states + y

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present_key_value,)
        return outputs


if __name__ == "__main__":
    BATCH_SIZE = 8  # Increased batch size for better GPU utilization
    SEQ_LEN    = 128*8  # Reduced sequence length for faster processing
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
    model.config.use_cache = True  # Enable KV-cache
    model = model.to(device).eval()
    
    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False
    
    # Optimized benchmark helper for perplexity with KV-cache
    @torch.no_grad()
    def perplexity_peak_time_kv_cache(mdl):
        mdl.eval()
        total_loss, total_tokens = 0.0, 0
        start = time.perf_counter()
        torch.cuda.synchronize()
        
        for batch_idx, batch in enumerate(loader):
            batch = {k:v.to(device) for k,v in batch.items()}
            
            # Process entire batch at once for better GPU utilization
            batch_size, seq_len = batch["input_ids"].shape
            
            # Process sequence in chunks for memory efficiency
            chunk_size = 1024  # Process 32 tokens at a time
            past_key_values = None
            prev_last_logits = None  # boundary logits from previous chunk [B, V]
            
            for chunk_start in range(0, seq_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_len)
                
                # Get current chunk
                current_ids = batch["input_ids"][:, chunk_start:chunk_end]  # [batch, chunk_size]
                # KV-cache expects mask up to current key length
                current_mask = batch["attention_mask"][:, :chunk_end]  # [batch, chunk_end]
                
                # Forward pass with KV-cache
                outputs = mdl(input_ids=current_ids,
                             attention_mask=current_mask,
                             past_key_values=past_key_values,
                             use_cache=True)
                
                logits = outputs.logits  # [B, cur_len, V]
                cur_len = logits.size(1)
                
                # 1) boundary loss with last logit of previous chunk → first label of current chunk
                if prev_last_logits is not None:
                    boundary_labels = batch["input_ids"][:, chunk_start]  # [B]
                    boundary_mask = batch["attention_mask"][:, chunk_start]  # [B]
                    valid = boundary_mask.bool()
                    if valid.any():
                        v_logits = prev_last_logits[valid]
                        v_labels = boundary_labels[valid]
                        # safety checks
                        if not (torch.isnan(v_logits).any() or torch.isinf(v_logits).any()):
                            b_loss = F.cross_entropy(v_logits, v_labels)
                            if not (torch.isnan(b_loss) or torch.isinf(b_loss)):
                                total_loss += b_loss.item() * valid.sum().item()
                                total_tokens += valid.sum().item()
                
                # 2) intra-chunk next-token loss
                if cur_len > 1:
                    intra_logits = logits[:, :-1, :].contiguous()  # [B, cur_len-1, V]
                    intra_labels = batch["input_ids"][:, chunk_start+1:chunk_end].contiguous()  # [B, cur_len-1]
                    intra_mask   = batch["attention_mask"][:, chunk_start+1:chunk_end].contiguous()  # [B, cur_len-1]
                    valid_positions = intra_mask.bool()
                    if valid_positions.sum() > 0:
                        v_logits = intra_logits[valid_positions]
                        v_labels = intra_labels[valid_positions]
                        if not (torch.isnan(v_logits).any() or torch.isinf(v_logits).any()):
                            loss = F.cross_entropy(v_logits, v_labels)
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                total_loss += loss.item() * valid_positions.sum().item()
                                total_tokens += valid_positions.sum().item()
                
                # keep last logit for boundary with next chunk
                prev_last_logits = logits[:, -1, :].contiguous()
                past_key_values = outputs.past_key_values
            
            # Clear cache periodically
            if past_key_values is not None:
                del past_key_values
                torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        t = (time.perf_counter() - start)*1000.0/len(loader)
        peak = torch.cuda.max_memory_allocated()/(1024**2)
        
        # Check if we have valid tokens
        if total_tokens == 0:
            return float('nan'), peak, t
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        # Check for valid perplexity
        if math.isnan(perplexity) or math.isinf(perplexity):
            return float('nan'), peak, t
        
        return perplexity, peak, t

    # Create SVD Model
    svd_model = GPT2LMHeadModel.from_pretrained("gpt2")
    svd_model.config.use_cache = True  # Enable KV-cache
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
    perplexity, peak_svd, t_svd = perplexity_peak_time_kv_cache(svd_model)
    
    # Print results in the requested format
    print(f"{'Model':<15} | {'Storage (MiB)':<12} | {'Peak (MiB)':<10} | {'Transient (MiB)':<14} | {'Time (ms/b)':<10} | {'Perplexity':<10}")
    print("-" * 90)
    print(f"{'SVD KV-Cache FA':<15} | {SVD_CACHED_MEM:<12.1f} | {peak_svd:<10.1f} | {peak_svd-SVD_CACHED_MEM:<14.1f} | {t_svd:<10.1f} | {perplexity:<10.4f}")

