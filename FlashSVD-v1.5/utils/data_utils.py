from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

try:
    from datasets import load_dataset
except Exception as e:  # pragma: no cover
    load_dataset = None


def _first_text_field(example: Dict) -> str:
    if "text" in example and isinstance(example["text"], str):
        return example["text"]
    if "sentence" in example and isinstance(example["sentence"], str):
        return example["sentence"]
    # fallback: pick the first string field
    for v in example.values():
        if isinstance(v, str):
            return v
    raise KeyError(f"Could not find text field in dataset example keys={list(example.keys())}")


def _iter_texts(dataset_name: str, split: str):
    if load_dataset is None:
        raise ImportError("datasets is required for SVD-LLM utilities; please `pip install datasets`.")

    name = dataset_name.lower()
    if name in {"wikitext2", "wikitext-2", "wiki2"}:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        for ex in ds:
            yield _first_text_field(ex)
        return
    if name in {"ptb", "penn_treebank"}:
        ds = load_dataset("ptb_text_only", split=split)
        for ex in ds:
            yield _first_text_field(ex)
        return
    if name in {"c4", "c4_en"}:
        # Use streaming to avoid downloading huge shards when only a few samples are needed.
        ds = load_dataset("c4", "en", split=split, streaming=True)
        for ex in ds:
            yield _first_text_field(ex)
        return

    raise ValueError(f"Unsupported dataset: {dataset_name} (supported: wikitext2, ptb, c4)")


def _build_token_stream(
    dataset_name: str,
    tokenizer,
    split: str,
    *,
    min_tokens: Optional[int] = None,
    max_texts: Optional[int] = None,
) -> List[int]:
    eos = tokenizer.eos_token_id
    ids: List[int] = []
    n = 0
    for txt in _iter_texts(dataset_name, split):
        if not txt:
            continue
        # Keep tokenization simple and stable.
        ids.extend(tokenizer.encode(txt, add_special_tokens=False))
        if eos is not None:
            ids.append(int(eos))
        n += 1
        if min_tokens is not None and len(ids) >= int(min_tokens):
            break
        if max_texts is not None and n >= max_texts:
            break
    return ids


def get_loaders(
    dataset_name: str,
    *,
    nsamples: int,
    seed: int,
    tokenizer,
    seqlen: int,
) -> Tuple[DataLoader, None]:
    """Return a calibration/update loader compatible with SVDLLM scripts.

    The returned loader yields tuples like `(input_ids,)` where `input_ids` is
    an `int64` tensor of shape `[1, seqlen]` (batch size fixed to 1), which
    matches the expectations in `SVDLLM_flashsvd.py`.
    """
    rng = random.Random(int(seed))
    # Collect only as many tokens as we need (important for streaming datasets like C4).
    need = int(nsamples) * int(seqlen) + int(seqlen) + 1
    stream = _build_token_stream(dataset_name, tokenizer, split="train", min_tokens=need, max_texts=None)
    if len(stream) < need:
        raise ValueError(
            f"Not enough tokens in {dataset_name} train split to build sequences "
            f"(need>={need}, got={len(stream)})."
        )

    # Pick a random offset (<= seqlen) for variety.
    max_start = len(stream) - (int(nsamples) * int(seqlen) + 1)
    samples: List[Tuple[torch.Tensor]] = []
    base = rng.randint(0, max_start) if max_start > 0 else 0
    for i in range(int(nsamples)):
        s = base + i * int(seqlen)
        ids = torch.tensor(stream[s : s + int(seqlen)], dtype=torch.long)
        samples.append((ids,))

    loader = DataLoader(samples, batch_size=1, shuffle=False, collate_fn=lambda b: (torch.stack([x[0] for x in b]),))
    return loader, None


def get_calib_train_data(
    dataset_name: str,
    tokenizer,
    nsamples: int,
    *,
    seqlen: int,
    seed: int = 0,
) -> DataLoader:
    """Return a loader of dict batches for whitening/profiling.

    Each batch is a dict with keys: `input_ids`, `attention_mask`.
    """
    rng = random.Random(int(seed))
    need = int(nsamples) * int(seqlen) + int(seqlen) + 1
    stream = _build_token_stream(dataset_name, tokenizer, split="train", min_tokens=need, max_texts=None)
    if len(stream) < need:
        raise ValueError(
            f"Not enough tokens in {dataset_name} train split to build sequences "
            f"(need>={need}, got={len(stream)})."
        )

    max_start = len(stream) - (int(nsamples) * int(seqlen) + 1)
    base = rng.randint(0, max_start) if max_start > 0 else 0

    samples: List[Dict[str, torch.Tensor]] = []
    for i in range(int(nsamples)):
        s = base + i * int(seqlen)
        ids = torch.tensor(stream[s : s + int(seqlen)], dtype=torch.long)
        samples.append(
            {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids, dtype=torch.long),
            }
        )

    def collate(batch: List[Dict[str, torch.Tensor]]):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        }

    return DataLoader(samples, batch_size=1, shuffle=False, collate_fn=collate)


def get_mixed_calib_train_data(
    dataset_names: Sequence[str],
    tokenizer,
    nsamples: int,
    *,
    seqlen: int,
    seed: int = 0,
) -> DataLoader:
    """Simple mixed calibration loader: round-robin over datasets."""
    names = [str(x).strip() for x in dataset_names if str(x).strip()]
    if not names:
        raise ValueError("dataset_names must be non-empty")
    per = max(1, int(nsamples) // len(names))
    batches: List[Dict[str, torch.Tensor]] = []
    for i, n in enumerate(names):
        l = get_calib_train_data(n, tokenizer, per, seqlen=seqlen, seed=seed + i * 17)
        for b in l:
            # flatten batch dimension (always 1 here)
            batches.append(
                {
                    "input_ids": b["input_ids"][0].cpu(),
                    "attention_mask": b["attention_mask"][0].cpu(),
                }
            )
    batches = batches[: int(nsamples)]

    def collate(batch: List[Dict[str, torch.Tensor]]):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        }

    return DataLoader(batches, batch_size=1, shuffle=False, collate_fn=collate)
