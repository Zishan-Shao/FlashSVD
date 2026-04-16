# profile the dense model
# in this case, the code is the same as the run_llama_dense.py
# the only difference is the name of the script
# this profile will be used to generate the dense profile

import os
import math
import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM

# CUDA_VISIBLE_DEVICES=5 python3 run_llama_dense.py MAX_EVAL_SAMPLES=200 MAX_EVAL_BATCHES=5 CHUNK_SIZE=64

# CUDA_VISIBLE_DEVICES=3,4,5,6,7 LLAMA_MODEL=meta-llama/Llama-2-7b-hf DTYPE=float16 BATCH_SIZE=1 SEQ_LEN=1024 CHUNK_SIZE=128  MAX_EVAL_SAMPLES=200 MAX_EVAL_BATCHES=5 python run_llama_dense.py

@torch.no_grad()
def evaluate_perplexity_kv(
    model: LlamaForCausalLM,
    loader: DataLoader,
    device: str,
    chunk_size: int,
    max_eval_batches: int = 0,
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

    batch_idx = 0
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
        batch_idx += 1
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    processed_batches = max(1, batch_idx)
    time_ms = (time.perf_counter() - start) * 1000.0 / processed_batches
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
    max_eval_samples = int(os.getenv("MAX_EVAL_SAMPLES", "0"))
    max_eval_batches = int(os.getenv("MAX_EVAL_BATCHES", "0"))

    print(f"Loading dense model {model_name} ...")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad = False

    print("Preparing dataset (wikitext-2-raw-v1, test split) ...")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    if max_eval_samples > 0:
        raw = raw.select(range(min(max_eval_samples, len(raw))))  # type: ignore[arg-type]

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

    ppl, peak_mem, time_ms = evaluate_perplexity_kv(
        model, loader, device=device, chunk_size=chunk_size, max_eval_batches=max_eval_batches
    )
    storage_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

    print(f"{'Model':<15} | {'Storage (MiB)':<12} | {'Peak (MiB)':<10} | {'Transient (MiB)':<14} | {'Time (ms/b)':<10} | {'Perplexity':<10}")
    print("-" * 90)
    print(f"{'LLaMA Dense KV':<15} | {storage_mem:<12.1f} | {peak_mem:<10.1f} | {peak_mem - storage_mem:<14.1f} | {time_ms:<10.1f} | {ppl:<10.4f}")


if __name__ == "__main__":
    main()


