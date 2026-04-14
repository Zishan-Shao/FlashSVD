#!/usr/bin/env python3
"""
Profile ModernBERT on IMDB reviews with long contexts.

- Loads IMDB (HuggingFace 'imdb' dataset)
- Packs multiple reviews into long sequences up to --seq-len
- Runs ModernBERT sequence classification model (SDPA backend)
- Reports latency, tokens/s, and GPU peak memory

Usage:
  python ModernBERT/BERT_LONG/profile_imdb.py \
    --model-dir ../model/modernbert-base-imdb \
    --seq-len 2048 --batch-size 16 --max-batches 50

  python ModernBERT/BERT_LONG/profile_imdb.py \
    --model-dir ../model/modernbert-base-imdb \
    --seq-len 4096 --batch-size 8 --max-batches 50
"""

import os
import time
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _get_text_field(ex):
    # IMDB uses 'text'; support fallbacks for robustness
    for k in ("text", "review", "content", "sentence"):
        if k in ex:
            return ex[k]
    # last resort: stringify
    return str(ex)


class PackedIMDBDataset(Dataset):
    """Packs multiple IMDB reviews into sequences near target seq_len.

    Each item returns tensors: input_ids, attention_mask.
    Labels are not used for profiling.
    """
    def __init__(self, hf_ds: HFDataset, tokenizer, seq_len: int, max_packs: int):
        self.items: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._build(hf_ds, tokenizer, seq_len, max_packs)

    def _build(self, hf_ds: HFDataset, tokenizer, seq_len: int, max_packs: int):
        acc_text = ""
        packs = 0

        def flush_pack(text: str):
            nonlocal packs
            if not text:
                return
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            )
            self.items.append((enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)))
            packs += 1

        for ex in hf_ds:
            snippet = _get_text_field(ex)
            cur = snippet if not acc_text else (acc_text + "\n\n" + snippet)
            # Estimate tokenized length (no padding/truncation here)
            try:
                enc_len = len(tokenizer(cur, truncation=False)["input_ids"])  # type: ignore
            except Exception:
                # Some tokenizers may require truncation; approximate with truncation
                enc_len = seq_len + 1

            if enc_len >= seq_len:
                if acc_text:
                    flush_pack(acc_text)
                else:
                    flush_pack(snippet)
                acc_text = ""
                if packs >= max_packs:
                    break
            else:
                acc_text = cur

            if packs >= max_packs:
                break

        if packs < max_packs and acc_text:
            flush_pack(acc_text)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ids, mask = self.items[idx]
        return {"input_ids": ids, "attention_mask": mask}


@torch.no_grad()
def profile_peak_time(model, loader, device: str):
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    steps = 0
    tokens = 0
    start = time.perf_counter()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        tokens += int(batch["attention_mask"].sum().item())
        steps += 1
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_s = max(1e-9, time.perf_counter() - start)

    if torch.cuda.is_available() and device.startswith("cuda"):
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_mib = 0.0

    return {
        "steps": steps,
        "ms_per_batch": (elapsed_s * 1000.0 / max(1, steps)),
        "tok_per_s": (tokens / elapsed_s) if tokens else 0.0,
        "peak_mib": peak_mib,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser("ModernBERT IMDB long-context profiler")
    ap.add_argument("--model-dir", default="../model/modernbert-base-imdb")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-batches", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, config=cfg, trust_remote_code=True
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        persistent_baseline = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Persistent model storage: {persistent_baseline:6.1f} MiB")

    print(f"Loading IMDB split={args.split} ...")
    hf_ds = load_dataset("imdb", split=args.split)
    print(f"Loaded {len(hf_ds)} reviews")

    max_packs = max(1, args.max_batches * args.batch_size)
    packed = PackedIMDBDataset(hf_ds, tok, seq_len=args.seq_len, max_packs=max_packs)
    print(f"Built {len(packed)} packed sequences @ L={args.seq_len}")

    loader = DataLoader(
        packed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    stats = profile_peak_time(model, loader, device)
    print(
        f"IMDB profile | steps={stats['steps']} | bs={args.batch_size} | L={args.seq_len} | "
        f"{stats['ms_per_batch']:6.1f} ms/b | {stats['tok_per_s']/1e6:5.2f}M tok/s | peak={stats['peak_mib']:6.1f} MiB"
    )


if __name__ == "__main__":
    main()

