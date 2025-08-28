import os, sys, time, math, platform, gc
from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# ─────────────────────────────────────────────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
DECODERS_DIR = os.path.dirname(os.path.dirname(THIS_FILE))
if DECODERS_DIR not in sys.path: sys.path.insert(0, DECODERS_DIR)
PROJECT_ROOT = os.path.dirname(DECODERS_DIR)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
def _repeat_kv(x, n_rep: int):
    if n_rep == 1: return x
    b, h, s, d = x.shape
    return x[:, :, :, None, :].expand(b, h, s, n_rep, d).reshape(b, h * n_rep, s, d)

def _build_full_bias(attention_mask, batch_size, q_len, k_len, device, dtype):
    i = torch.arange(q_len, device=device).view(q_len, 1)
    j = torch.arange(k_len, device=device).view(1, k_len)
    past_len = k_len - q_len
    causal = (j <= (past_len + i))
    causal_bias = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    causal_bias.masked_fill_(~causal, -1e4)
    causal_bias = causal_bias.view(1, 1, q_len, k_len)

    pad_bias = None
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            am = attention_mask
            if am.size(-1) < k_len: am = F.pad(am, (0, k_len - am.size(-1)), value=1)
            elif am.size(-1) > k_len: am = am[:, -k_len:]
            pad_bias = (1.0 - am.float()) * -1e4
            pad_bias = pad_bias.view(batch_size, 1, 1, k_len).to(dtype=dtype, device=device)
        elif attention_mask.dim() == 4:
            pad_bias = attention_mask.to(dtype=dtype, device=device)
            if pad_bias.size(-1) != k_len:
                pad_bias = F.pad(pad_bias, (0, k_len - pad_bias.size(-1)), value=0.0) if pad_bias.size(-1) < k_len else pad_bias[..., -k_len:]
    if pad_bias is None: return causal_bias
    if pad_bias.size(-2) == 1: pad_bias = pad_bias.expand(-1, -1, q_len, -1)
    return causal_bias + pad_bias

# ─────────────────────────────────────────────────────────────────────────────
class SimpleRoPE(torch.nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = int(head_dim); self.base = float(base)
    def forward(self, x: torch.Tensor, seq_len: int):
        device = x.device; dtype = torch.float32
        evens = torch.arange(0, self.head_dim, 2, dtype=dtype, device=device)
        inv_freq = 1.0 / (self.base ** (evens / self.head_dim))
        t = torch.arange(seq_len, dtype=dtype, device=device)
        emb = torch.einsum("i,j->ij", t, inv_freq).repeat_interleave(2, dim=-1)
        cos = emb.cos()[None, None, :, :].to(x.dtype)
        sin = emb.sin()[None, None, :, :].to(x.dtype)
        return cos, sin

# ─────────────────────────────────────────────────────────────────────────────
class LinearLlamaBlock(nn.Module):
    def __init__(self, hf_layer: nn.Module, config):
        super().__init__()
        attn, mlp = hf_layer.self_attn, hf_layer.mlp
        self.d_model = int(getattr(config, "hidden_size"))
        self.n_heads = int(getattr(config, "num_attention_heads"))
        self.n_kv_heads = int(getattr(config, "num_key_value_heads", self.n_heads))
        self.head_dim = self.d_model // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        w_dtype = attn.q_proj.weight.dtype
        def L(in_f, out_f): return nn.Linear(in_f, out_f, bias=False, dtype=w_dtype)

        self.q_proj = L(self.d_model, self.n_heads    * self.head_dim)
        self.k_proj = L(self.d_model, self.n_kv_heads * self.head_dim)
        self.v_proj = L(self.d_model, self.n_kv_heads * self.head_dim)
        self.o_proj = L(self.n_heads * self.head_dim, self.d_model)
        with torch.no_grad():
            self.q_proj.weight.copy_(attn.q_proj.weight.to(w_dtype))
            self.k_proj.weight.copy_(attn.k_proj.weight.to(w_dtype))
            self.v_proj.weight.copy_(attn.v_proj.weight.to(w_dtype))
            self.o_proj.weight.copy_(attn.o_proj.weight.to(w_dtype))

        inter = int(getattr(config, "intermediate_size"))
        self.gate_proj = L(self.d_model, inter)
        self.up_proj   = L(self.d_model, inter)
        self.down_proj = L(inter, self.d_model)
        with torch.no_grad():
            self.gate_proj.weight.copy_(mlp.gate_proj.weight.to(w_dtype))
            self.up_proj.weight.copy_(mlp.up_proj.weight.to(w_dtype))
            self.down_proj.weight.copy_(mlp.down_proj.weight.to(w_dtype))

        self.ln1 = hf_layer.input_layernorm
        self.ln2 = hf_layer.post_attention_layernorm
        self.rotary_emb = getattr(attn, "rotary_emb", None)
        if self.rotary_emb is None:
            try:
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
                max_pos = int(getattr(config, "max_position_embeddings", 4096))
                rope_th = float(getattr(config, "rope_theta", 10000.0))
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=max_pos, base=rope_th)
            except Exception:
                self.rotary_emb = SimpleRoPE(self.head_dim, base=float(getattr(config, "rope_theta", 10000.0)))

    def _rope_qk(self, q, k, *, position_ids=None, past_len: int = 0):
        if position_ids is not None:
            seq_len = int(position_ids.max().item()) + 1
            cos, sin = self.rotary_emb(q, seq_len=seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids); return q, k
        b, _, q_len, _ = q.shape
        pos = torch.arange(past_len, past_len + q_len, device=q.device).view(1, q_len).expand(b, q_len)
        cos, sin = self.rotary_emb(q, seq_len=past_len + q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos); return q, k

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, position_ids=None, use_cache=False, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        x = self.ln1(hidden_states)
        q = self.q_proj(x).view(bsz, q_len, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        past_len = 0
        if past_key_value is not None and len(past_key_value) == 2: past_len = past_key_value[0].size(-2)
        q, k_now = self._rope_qk(q, k, position_ids=position_ids, past_len=past_len)
        if past_len > 0:
            k = torch.cat([past_key_value[0], k_now], dim=-2); v = torch.cat([past_key_value[1], v], dim=-2)
        else: k = k_now
        present = (k, v) if use_cache else None
        k_rep = _repeat_kv(k, self.n_heads // self.n_kv_heads)
        v_rep = _repeat_kv(v, self.n_heads // self.n_kv_heads)
        k_len = k_rep.size(-2)
        bias = _build_full_bias(attention_mask, bsz, q_len, k_len, hidden_states.device, q.dtype)
        attn = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=bias, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        attn = self.o_proj(attn)
        h = hidden_states + attn
        y = self.ln2(h)
        gate = F.silu(self.gate_proj(y)); up = self.up_proj(y); ff = self.down_proj(gate * up)
        h = h + ff
        outputs = (h,); 
        if use_cache: outputs += (present,)
        return outputs

# ─────────────────────────────────────────────────────────────────────────────
class LinearSVDLlamaBlock(nn.Module):
    """
    Whitening + SVD (right-whitened): for a weight W (out x D), compute SVD of (W @ S)
    with S from Cholesky of XX^T (activation Gram). Recompose W = U Σ V^T S^{-1}.
    We store U (out x r) and Vhat := Σ V^T S^{-1} (r x D). Full rank → exact.
    """
    def __init__(self, dense: LinearLlamaBlock, rank_q: int, rank_kv: int, rank_o: int, rank_ff: int, S: torch.Tensor):
        super().__init__()
        self.d_model = dense.d_model; self.n_heads = dense.n_heads
        self.n_kv_heads = dense.n_kv_heads; self.head_dim = self.d_model // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.ln1, self.ln2, self.rotary_emb = dense.ln1, dense.ln2, dense.rotary_emb
        S = S.float()  # lower-triangular from cholesky

        def svd_right_whiten(W, S, r):
            WS = W.float() @ S
            U, Svals, Vh = torch.linalg.svd(WS, full_matrices=False)  # WS = U diag(Svals) Vh^T
            r = min(r, U.shape[1], Vh.shape[1])
            U_r = U[:, :r]
            A = (Vh[:, :r].T * Svals[:r].unsqueeze(1))  # (r x D)
            # Solve (S^T) Y = A^T  =>  Y = (S^T)^{-1} A^T  =>  Vhat = Y^T = A @ S^{-1}
            Y = torch.linalg.solve_triangular(S.T, A.T, upper=True)   # (D x r)
            Vhat = Y.T
            return U_r.to(W.dtype), Vhat.to(W.dtype)

        def decompose_heads_whiten(W, n_heads, head_dim, r, S):
            U_list, V_list = [], []
            for h in range(n_heads):
                W_h = W[h*head_dim:(h+1)*head_dim, :]
                Uh, Vh = svd_right_whiten(W_h, S, r)
                U_list.append(Uh); V_list.append(Vh)
            return nn.Parameter(torch.stack(U_list, 0), requires_grad=False), \
                   nn.Parameter(torch.stack(V_list, 0), requires_grad=False)

        # Q/K/V with whitening on common D-dim input
        self.q_U, self.q_V = decompose_heads_whiten(dense.q_proj.weight.data, self.n_heads,    self.head_dim, rank_q,  S)
        self.k_U, self.k_V = decompose_heads_whiten(dense.k_proj.weight.data, self.n_kv_heads, self.head_dim, rank_kv, S)
        self.v_U, self.v_V = decompose_heads_whiten(dense.v_proj.weight.data, self.n_kv_heads, self.head_dim, rank_kv, S)

        # O, gate, up: same D-dim input → use same S
        Wo = dense.o_proj.weight.data
        Uo, Vo = svd_right_whiten(Wo, S, min(rank_o, self.d_model))
        self.o_U = nn.Parameter(Uo, requires_grad=False)          # [D, r_o]
        self.o_V = nn.Parameter(Vo, requires_grad=False)          # [r_o, D]

        Wg, Wu, Wd = dense.gate_proj.weight.data, dense.up_proj.weight.data, dense.down_proj.weight.data
        Ug, Vg = svd_right_whiten(Wg, S, rank_ff)
        Uu, Vu = svd_right_whiten(Wu, S, rank_ff)
        # down_proj lives in 'intermediate' space; keep exact QR (full-rank exact & short)
        Qd, Rd = torch.linalg.qr(Wd.float(), mode='reduced')
        rd = min(rank_ff, Qd.shape[1], Rd.shape[0])
        Ud, Vd = Qd[:, :rd].to(Wd.dtype), Rd[:rd, :].to(Wd.dtype)

        self.g_U = nn.Parameter(Ug, requires_grad=False); self.g_V = nn.Parameter(Vg, requires_grad=False)
        self.u_U = nn.Parameter(Uu, requires_grad=False); self.u_V = nn.Parameter(Vu, requires_grad=False)
        self.d_U = nn.Parameter(Ud, requires_grad=False); self.d_V = nn.Parameter(Vd, requires_grad=False)

    @torch.no_grad()
    def _proj_per_head(self, x, U, V, n_heads):
        outs = [ (x @ V[h].T) @ U[h].T for h in range(n_heads) ]
        return torch.stack(outs, dim=1)  # [B,H,Q,Dh]

    def _rope_qk(self, q, k, *, position_ids=None, past_len: int = 0):
        if position_ids is not None:
            seq_len = int(position_ids.max().item()) + 1
            cos, sin = self.rotary_emb(q, seq_len=seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids); return q, k
        b, _, q_len, _ = q.shape
        pos = torch.arange(past_len, past_len + q_len, device=q.device).view(1, q_len).expand(b, q_len)
        cos, sin = self.rotary_emb(q, seq_len=past_len + q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos); return q, k

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, position_ids=None, use_cache=False, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        x = self.ln1(hidden_states)
        q = self._proj_per_head(x, self.q_U, self.q_V, self.n_heads)
        k = self._proj_per_head(x, self.k_U, self.k_V, self.n_kv_heads)
        v = self._proj_per_head(x, self.v_U, self.v_V, self.n_kv_heads)
        past_len = 0
        if past_key_value is not None and len(past_key_value) == 2: past_len = past_key_value[0].size(-2)
        q, k_now = self._rope_qk(q, k, position_ids=position_ids, past_len=past_len)
        if past_len > 0:
            k = torch.cat([past_key_value[0], k_now], dim=-2); v = torch.cat([past_key_value[1], v], dim=-2)
        else: k = k_now
        present = (k, v) if use_cache else None
        k_rep = _repeat_kv(k, self.n_heads // self.n_kv_heads)
        v_rep = _repeat_kv(v, self.n_heads // self.n_kv_heads)
        k_len = k_rep.size(-2)
        bias = _build_full_bias(attention_mask, bsz, q_len, k_len, hidden_states.device, q.dtype)
        attn = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=bias, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        attn = (attn @ self.o_V.T) @ self.o_U.T
        h = hidden_states + attn
        y = self.ln2(h)
        gate = F.silu((y @ self.g_V.T) @ self.g_U.T)
        up   =        (y @ self.u_V.T) @ self.u_U.T
        ff   = ((gate * up) @ self.d_V.T) @ self.d_U.T
        h = h + ff
        outputs = (h,); 
        if use_cache: outputs += (present,)
        return outputs

# ─────────────────────────────────────────────────────────────────────────────
class LayerShim(nn.Module):
    def __init__(self, block: nn.Module): super().__init__(); self.block = block
    def forward(self, hidden_states, past_key_value=None, attention_mask=None,
                position_ids=None, use_cache=False, output_attentions=False, **kwargs):
        if isinstance(hidden_states, tuple): hidden_states = hidden_states[0]
        return self.block(hidden_states, attention_mask=attention_mask,
                          past_key_value=past_key_value, position_ids=position_ids, use_cache=use_cache)

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def perplexity_peak_time_kv_cache(mdl, loader, device):
    mdl.eval(); total_loss, total_tokens = 0.0, 0
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.perf_counter()
    chunk_size = int(os.getenv("CHUNK_SIZE", "256"))
    use_safer_sdpa = os.getenv("PPL_SAFE_SDPA", "1") == "1"
    ppl_debug = os.getenv("PPL_DEBUG", "0") == "1"; dropped_rows = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B, L = batch["input_ids"].shape
        past_kv = None; prev_last_logits = None; prev_last_mask = None
        for s in range(0, L, chunk_size):
            e = min(s + chunk_size, L)
            ids = batch["input_ids"][:, s:e]
            am  = batch["attention_mask"][:, :e]
            cm = torch.backends.cuda.sdp_kernel
            ctx = cm(enable_flash=False, enable_mem_efficient=True, enable_math=True) if (use_safer_sdpa and device == "cuda") else nullcontext()
            with ctx: out = mdl(input_ids=ids, attention_mask=am, past_key_values=past_kv, use_cache=True)
            logits = out.logits; past_kv = out.past_key_values; cur_len = logits.size(1)

            if prev_last_logits is not None and prev_last_mask is not None and cur_len > 0:
                cur_first_mask = batch["attention_mask"][:, s].bool()
                both_valid = (prev_last_mask & cur_first_mask)
                if both_valid.any():
                    v_logits = prev_last_logits[both_valid].float()
                    v_labels = batch["input_ids"][both_valid, s]
                    finite = torch.isfinite(v_logits).all(dim=-1)
                    if finite.any():
                        loss = F.cross_entropy(v_logits[finite], v_labels[finite], reduction="sum")
                        total_loss += loss.item(); total_tokens += int(finite.sum().item())
                    if ppl_debug: dropped_rows += int((~finite).sum().item())

            if cur_len > 1:
                intra_logits = logits[:, :-1, :].contiguous()
                intra_labels = batch["input_ids"][:, s+1:e].contiguous()
                intra_mask   = batch["attention_mask"][:, s+1:e].contiguous().bool()
                if intra_mask.any():
                    v_logits = intra_logits[intra_mask].float()
                    v_labels = intra_labels[intra_mask]
                    finite = torch.isfinite(v_logits).all(dim=-1)
                    if finite.any():
                        loss = F.cross_entropy(v_logits[finite], v_labels[finite], reduction="sum")
                        total_loss += loss.item(); total_tokens += int(finite.sum().item())
                    if ppl_debug: dropped_rows += int((~finite).sum().item())

            last_mask = batch["attention_mask"][:, e-1].bool() if cur_len > 0 else torch.zeros(B, dtype=torch.bool, device=device)
            prev_last_logits = logits[:, -1, :].contiguous() if cur_len > 0 else None
            prev_last_mask   = last_mask if cur_len > 0 else None

        del past_kv
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t = (time.perf_counter() - start) * 1000.0 / len(loader)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    if ppl_debug: print(f"[ppl_debug] dropped rows (non‑finite logits): {dropped_rows}")
    if total_tokens == 0: return float('nan'), peak, t
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if math.isfinite(avg_loss) else float('nan')
    return ppl, peak, t

# ─────────────────────────────────────────────────────────────────────────────
def print_llama_summary(cfg, model_name: str, *, device: str, dtype: torch.dtype,
                        batch_size: int, seq_len: int, chunk_size: int,
                        rank_q: int, rank_kv: int, rank_o: int, rank_ff: int,
                        for_layer=None):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    n_kv = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    n_rep = cfg.num_attention_heads // n_kv
    print("\n================== LLaMA / Run Configuration ==================")
    print(f"Model:              {model_name}")
    print(f"Transformers:       {transformers.__version__}")
    print(f"Torch:              {torch.__version__}")
    print(f"Python:             {platform.python_version()}")
    print(f"Device / dtype:     {device} / {dtype}")
    print(f"Batch / Seq / Chunk:{batch_size} / {seq_len} / {chunk_size}")
    print("---------------------------------------------------------------")
    print(f"hidden_size:        {cfg.hidden_size}")
    print(f"n_layers:           {cfg.num_hidden_layers}")
    print(f"n_heads:            {cfg.num_attention_heads}")
    print(f"n_kv_heads:         {n_kv}  (GQA n_rep={n_rep})")
    print(f"head_dim:           {head_dim}")
    print(f"intermediate_size:  {cfg.intermediate_size}")
    print(f"vocab_size:         {cfg.vocab_size}")
    print(f"rope_theta:         {getattr(cfg, 'rope_theta', 'N/A')}")
    print(f"max_pos_emb:        {getattr(cfg, 'max_position_embeddings', 'N/A')}")
    print("---------------------------------------------------------------")
    print(f"SVD ranks:          q={rank_q}, kv={rank_kv}, o={rank_o}, ff={rank_ff}")
    if for_layer is not None:
        attn0 = for_layer.self_attn; mlp0  = for_layer.mlp
        print("Layer[0] shapes:")
        print(f"  q_proj: {tuple(attn0.q_proj.weight.shape)}  "
              f"k_proj: {tuple(attn0.k_proj.weight.shape)}  "
              f"v_proj: {tuple(attn0.v_proj.weight.shape)}  "
              f"o_proj: {tuple(attn0.o_proj.weight.shape)}")
        print(f"  gate/up/down: {tuple(mlp0.gate_proj.weight.shape)} / "
              f"{tuple(mlp0.up_proj.weight.shape)} / {tuple(mlp0.down_proj.weight.shape)}")
    print("===============================================================\n")

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_global_whitening(base_model, loader, layer_idx=0, max_batches=1, eps=1e-6):
    """
    Collect X from layer[layer_idx].input_layernorm output (shape [B,L,D]) and
    build S via Cholesky on X X^T (truncation-aware whitening, SVD-LLM).
    """
    device = "cpu"
    layer = base_model.model.layers[layer_idx]
    D = base_model.config.hidden_size
    G = torch.zeros(D, D, dtype=torch.float32)  # Gram = sum(X^T X)
    handle = layer.input_layernorm.register_forward_hook(lambda m, inp, out: _accum_gram(out, G))
    seen = 0
    for batch in loader:
        batch = {k: v.to(device) for k,v in batch.items()}
        base_model(**batch)  # FP32 CPU forward, tiny number of batches
        seen += 1
        if seen >= max_batches: break
    handle.remove()
    # Cholesky of XX^T (note: here G = X^T X; it's the same symmetric pos.def.), add eps*I
    G = (G + eps * torch.eye(D))
    S = torch.linalg.cholesky(G)  # lower-triangular
    return S

def _accum_gram(x, G_store):
    x = x.detach().to(torch.float32)
    N, D = x.numel() // x.shape[-1], x.shape[-1]
    x2d = x.reshape(N, D)
    G_store += x2d.t().matmul(x2d)  # X^T X

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16

    MODEL_NAME = os.getenv("LLAMA_MODEL", "meta-llama/Llama-2-7b-hf")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
    SEQ_LEN    = int(os.getenv("SEQ_LEN", "1024"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
    CALIB_BATCHES = int(os.getenv("CALIB_BATCHES", "1"))  # tiny, increase if you like

    # 0) Data (do this early so calibration can reuse it)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    def tokenize_fn(batch): return tok(batch["text"], padding="max_length", truncation=True, max_length=SEQ_LEN)
    ds = raw.map(tokenize_fn, batched=True, remove_columns=["text"]); ds.set_format("torch")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]),
                                              "attention_mask": torch.stack([x["attention_mask"] for x in b])})

    # 1) Load a base model on CPU in FP32 (safe to run for calibration) and print summary.
    base_cpu = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map="cpu")
    cfg = base_cpu.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    # Full‑rank defaults (parity)
    RANK_Q  = int(os.getenv("RANK_Q",  str(head_dim)))
    RANK_KV = int(os.getenv("RANK_KV", str(head_dim)))
    RANK_O  = int(os.getenv("RANK_O",  str(cfg.hidden_size)))
    RANK_FF = int(os.getenv("RANK_FF", str(cfg.intermediate_size)))

    print_llama_summary(cfg, MODEL_NAME, device=device, dtype=dtype,
                        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, chunk_size=CHUNK_SIZE,
                        rank_q=RANK_Q, rank_kv=RANK_KV, rank_o=RANK_O, rank_ff=RANK_FF,
                        for_layer=base_cpu.model.layers[0])

    # 1.5) Calibration for whitening (Layer 0 LN1 activations) → S (D x D, lower-triangular).
    S = compute_global_whitening(base_cpu, loader, layer_idx=0, max_batches=CALIB_BATCHES, eps=1e-5)

    # Free base CPU if you wish
    del base_cpu; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 2) Load the SVD model on CPU, replace layers with whitened-SVD blocks, stream to device.
    svd_model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="cpu")
    svd_model.config.use_cache = True
    for p in svd_model.parameters(): p.requires_grad = False

    for i in range(len(svd_model.model.layers)):
        layer_cpu = svd_model.model.layers[i]
        dense_block = LinearLlamaBlock(layer_cpu, cfg)
        svd_block = LinearSVDLlamaBlock(dense_block, rank_q=RANK_Q, rank_kv=RANK_KV, rank_o=RANK_O, rank_ff=RANK_FF, S=S)
        svd_block = svd_block.to(dtype=torch.float16)
        shim = LayerShim(svd_block).to(device=device, dtype=torch.float16)
        svd_model.model.layers[i] = shim
        del layer_cpu, dense_block, svd_block, shim
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    svd_model.model.embed_tokens = svd_model.model.embed_tokens.to(device=device, dtype=torch.float16)
    svd_model.model.norm         = svd_model.model.norm.to(device=device, dtype=torch.float16)
    svd_model.lm_head            = svd_model.lm_head.to(device=device, dtype=torch.float16)

    # 3) Measure perplexity + memory + time
    torch.cuda.reset_peak_memory_stats()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    ppl, peak_mem, time_ms = perplexity_peak_time_kv_cache(svd_model, loader, device)
    svd_cached_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    print(f"{'Model':<15} | {'Storage (MiB)':<12} | {'Peak (MiB)':<10} | {'Transient (MiB)':<14} | {'Time (ms/b)':<10} | {'Perplexity':<10}")
    print("-" * 90)
    print(f"{'LLaMA SVD+W':<15} | {svd_cached_mem:<12.1f} | {peak_mem:<10.1f} | {peak_mem - svd_cached_mem:<14.1f} | {time_ms:<10.1f} | {ppl:<10.4f}")
