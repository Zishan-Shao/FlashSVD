import os
import math
import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM


@torch.no_grad()
def _apply_svd_to_linear(linear: nn.Linear, rank: int) -> nn.Module:
    out_features, in_features = linear.out_features, linear.in_features
    max_rank = min(out_features, in_features)
    if rank is None or rank >= max_rank:
        return linear
    # SVD on float32 for stability
    device = linear.weight.device
    dtype = linear.weight.dtype
    W = linear.weight.detach().to(torch.float32)
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except Exception:
        U, S, Vh = torch.linalg.svd(W.cpu(), full_matrices=False)
        U, S, Vh = U.to(device), S.to(device), Vh.to(device)
    r = max(1, min(int(rank), U.shape[1], Vh.shape[0]))
    U_r = U[:, :r].contiguous() * S[:r].unsqueeze(0)
    V_r = Vh[:r, :].contiguous()

    # Two small matmuls at inference: (x @ V^T) @ U^T + bias
    class _LowRankLinear(nn.Module):
        def __init__(self, U_lr, V_r, bias):
            super().__init__()
            self.U = nn.Parameter(U_lr.to(dtype), requires_grad=False)
            self.V = nn.Parameter(V_r.to(dtype), requires_grad=False)
            if bias is not None:
                self.bias = nn.Parameter(bias.detach().to(dtype), requires_grad=False)
            else:
                self.bias = None

        def forward(self, x):
            tmp = F.linear(x, self.V, None)
            return F.linear(tmp, self.U, self.bias)

    return _LowRankLinear(U_r, V_r, linear.bias)


@torch.no_grad()
def apply_svd_to_model(
    model: LlamaForCausalLM,
    rank_q: Optional[int],
    rank_kv: Optional[int],
    rank_o: Optional[int],
    rank_ff: Optional[int],
) -> None:
    """Replace attention and MLP linears with low-rank factors in-place."""
    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        if rank_q is not None:
            attn.q_proj = _apply_svd_to_linear(attn.q_proj, rank_q)
        if rank_kv is not None:
            attn.k_proj = _apply_svd_to_linear(attn.k_proj, rank_kv)
            attn.v_proj = _apply_svd_to_linear(attn.v_proj, rank_kv)
        if rank_o is not None:
            attn.o_proj = _apply_svd_to_linear(attn.o_proj, rank_o)

        if rank_ff is not None:
            mlp.gate_proj = _apply_svd_to_linear(mlp.gate_proj, rank_ff)
            mlp.up_proj = _apply_svd_to_linear(mlp.up_proj, rank_ff)
            mlp.down_proj = _apply_svd_to_linear(mlp.down_proj, rank_ff)


@torch.no_grad()
def evaluate_perplexity_kv(
    model: LlamaForCausalLM,
    loader: DataLoader,
    device: str,
    chunk_size: int,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    dropped_rows = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    use_safer_sdpa = os.getenv("PPL_SAFE_SDPA", "1") == "1"

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B, L = batch["input_ids"].shape
        past_kv = None
        prev_last_logits = None
        prev_last_mask = None

        for s in range(0, L, chunk_size):
            e = min(s + chunk_size, L)
            ids = batch["input_ids"][:, s:e]
            am = batch["attention_mask"][:, :e]

            cm = torch.backends.cuda.sdp_kernel
            ctx = (
                cm(enable_flash=False, enable_mem_efficient=True, enable_math=True)
                if (use_safer_sdpa and device == "cuda")
                else nullcontext()
            )
            with ctx:
                out = model(input_ids=ids, attention_mask=am, past_key_values=past_kv, use_cache=True)

            logits = out.logits  # [B, cur_len, V]
            past_kv = out.past_key_values
            cur_len = logits.size(1)

            # Boundary loss between chunks
            if prev_last_logits is not None and prev_last_mask is not None and cur_len > 0:
                cur_first_mask = batch["attention_mask"][:, s].bool()
                both_valid = (prev_last_mask & cur_first_mask)
                if both_valid.any():
                    v_logits = prev_last_logits[both_valid].float()
                    v_labels = batch["input_ids"][both_valid, s]
                    finite = torch.isfinite(v_logits).all(dim=-1)
                    if finite.any():
                        loss = F.cross_entropy(v_logits[finite], v_labels[finite], reduction="sum")
                        total_loss += loss.item()
                        total_tokens += int(finite.sum().item())
                    else:
                        dropped_rows += int((~finite).sum().item())

            # Intra-chunk loss
            if cur_len > 1:
                intra_logits = logits[:, :-1, :].contiguous()
                intra_labels = batch["input_ids"][:, s + 1 : e].contiguous()
                intra_mask = batch["attention_mask"][:, s + 1 : e].contiguous().bool()
                if intra_mask.any():
                    v_logits = intra_logits[intra_mask].float()
                    v_labels = intra_labels[intra_mask]
                    finite = torch.isfinite(v_logits).all(dim=-1)
                    if finite.any():
                        loss = F.cross_entropy(v_logits[finite], v_labels[finite], reduction="sum")
                        total_loss += loss.item()
                        total_tokens += int(finite.sum().item())
                    else:
                        dropped_rows += int((~finite).sum().item())

            last_mask = batch["attention_mask"][:, e - 1].bool() if cur_len > 0 else torch.zeros(B, dtype=torch.bool, device=device)
            prev_last_logits = logits[:, -1, :].contiguous() if cur_len > 0 else None
            prev_last_mask = last_mask if cur_len > 0 else None

        del past_kv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(loader))
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    if total_tokens == 0:
        return float("nan"), peak_mem, time_ms
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if math.isfinite(avg_loss) else float("nan")
    return ppl, peak_mem, time_ms


def main():
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = os.getenv("LLAMA_MODEL", "meta-llama/Llama-2-7b-hf")
    batch_size = int(os.getenv("BATCH_SIZE", "1"))
    seq_len = int(os.getenv("SEQ_LEN", "1024"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "256"))
    dtype_str = os.getenv("DTYPE", "float16").lower()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_str]

    print(f"Loading model {model_name} ...")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad = False

    # Ranks (defaults keep full rank)
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    rank_q = int(os.getenv("RANK_Q", str(head_dim)))
    rank_kv = int(os.getenv("RANK_KV", str(head_dim)))
    rank_o = int(os.getenv("RANK_O", str(cfg.hidden_size)))
    rank_ff = int(os.getenv("RANK_FF", str(cfg.intermediate_size)))

    # If the installed transformers LlamaForCausalLM already has .apply_svd (from local edits), use it; otherwise use helper.
    if hasattr(model, "apply_svd") and callable(getattr(model, "apply_svd")):
        print("Applying SVD via model.apply_svd(...) ...")
        model.apply_svd(rank_q=rank_q, rank_kv=rank_kv, rank_o=rank_o, rank_ff=rank_ff)
    else:
        print("Applying SVD via helper (no built-in apply_svd found) ...")
        apply_svd_to_model(model, rank_q=rank_q, rank_kv=rank_kv, rank_o=rank_o, rank_ff=rank_ff)

    print("Preparing dataset (wikitext-2-raw-v1, test split) ...")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    def tokenize_fn(batch):
        return tok(batch["text"], padding="max_length", truncation=True, max_length=seq_len)

    ds = raw.map(tokenize_fn, batched=True, remove_columns=["text"])  # type: ignore[arg-type]
    ds.set_format("torch")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
        },
    )

    ppl, peak_mem, time_ms = evaluate_perplexity_kv(model, loader, device=device, chunk_size=chunk_size)
    cached_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    print(f"{'Model':<15} | {'Storage (MiB)':<12} | {'Peak (MiB)':<10} | {'Transient (MiB)':<14} | {'Time (ms/b)':<10} | {'Perplexity':<10}")
    print("-" * 90)
    print(f"{'LLaMA SVD KV':<15} | {cached_mem:<12.1f} | {peak_mem:<10.1f} | {peak_mem - cached_mem:<14.1f} | {time_ms:<10.1f} | {ppl:<10.4f}")


if __name__ == "__main__":
    main()


