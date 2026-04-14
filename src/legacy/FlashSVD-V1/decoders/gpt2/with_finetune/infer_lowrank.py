#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_lowrank.py
------------------------------------------------------------
Run inference with a GPT-2 whose blocks were replaced by
low-rank factors, loading fine-tuned factors from disk.

Assumes lowrank_gpt2.build_lowrank_gpt2 supports preloading
factors and (recommended) auto-adopts ranks from the files.

Example:
  python infer_lowrank.py \
    --factors-dir ./checkpoints/lowrank_gpt2/best_rankA0.6_M0.6_lowrank+ln+bias \
    --prompt "The future of AI" \
    --max-new-tokens 100 \
    --do-sample --top-p 0.9 --temperature 0.8
"""
import argparse
import time
from typing import Optional

import torch
from transformers import AutoTokenizer

from lowrank_gpt2_new import set_seed, build_lowrank_gpt2


def top_k_top_p_filtering(logits: torch.Tensor,
                          top_k: int = 0,
                          top_p: float = 1.0) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p).
    Returns logits with filtered indices set to -inf.
    """
    logits = logits.clone()

    # Top-K
    if top_k > 0 and top_k < logits.size(-1):
        topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
        kth = topk_vals[..., -1, None]
        mask = logits < kth
        logits[mask] = float("-inf")

    # Top-P (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        # mask tokens with cumulative prob above threshold (keep first above too)
        cutoff = cumprobs > top_p
        # shift right to always keep at least one token
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits[cutoff] = float("-inf")
        # unsort back
        logits.scatter_(-1, sorted_idx, sorted_logits)

    return logits


@torch.no_grad()
def generate_lowrank(
    model,
    tokenizer,
    prompt_text: str,
    device: str,
    max_new_tokens: int = 100,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    print_every: int = 10,
    stop_at_eos: bool = True,
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id

    # Prefill (with cache) timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits
    prefill_time = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        prefill_peak = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.reset_peak_memory_stats()
    else:
        prefill_peak = 0.0

    # Decode loop
    generated = input_ids
    pieces = []
    t1 = time.perf_counter()

    for i in range(max_new_tokens):
        last_logits = logits[:, -1, :]

        if do_sample:
            # temperature
            if temperature != 1.0:
                last_logits = last_logits / max(temperature, 1e-5)
            # filter
            last_logits = top_k_top_p_filtering(last_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(last_logits, dim=-1, keepdim=True)

        if (i + 1) % max(1, print_every) == 0 or i == 0:
            tok = tokenizer.decode(next_id[0], skip_special_tokens=True)
            print(f"[Decode] {i+1}/{max_new_tokens}: '{tok}'")

        generated = torch.cat([generated, next_id], dim=1)

        if stop_at_eos and eos_id is not None and int(next_id[0]) == int(eos_id):
            print("[Decode] Hit <eos>; stopping early.")
            break

        # feed one token step with KV cache
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        logits = out.logits
        past = out.past_key_values

    decode_time = time.perf_counter() - t1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        decode_peak = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        decode_peak = 0.0

    toks_per_s = (generated.size(1) - input_ids.size(1)) / max(decode_time, 1e-6)

    print("\n=== Inference Stats ===")
    print(f"Prefill: {prefill_time*1000:.1f} ms | peak={prefill_peak:.1f} MiB")
    print(f"Decode : {decode_time*1000:.1f} ms | peak={decode_peak:.1f} MiB | {toks_per_s:.2f} tok/s")

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors-dir", type=str, required=True,
                    help="Directory containing gpt2_block_*.pt (fine-tuned low-rank factors)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # prompt / generation
    ap.add_argument("--prompt", type=str, default=None, help="Inline prompt text")
    ap.add_argument("--prompt-file", type=str, default=None, help="File containing the prompt text")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--no-stop-at-eos", action="store_true", help="If set, do not stop early at EOS token")

    # ranks are unused when preloading IF your lowrank_gpt2 adopts preload ranks.
    # They remain for compatibility if you kept the older version.
    ap.add_argument("--rank-ratio-attn", type=float, default=1.0)
    ap.add_argument("--rank-ratio-mlp", type=float, default=1.0)

    args = ap.parse_args()
    set_seed(args.seed)

    # prompt source
    if args.prompt is not None:
        prompt_text = args.prompt
    elif args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    else:
        prompt_text = (
            "The future of artificial intelligence holds tremendous promise for transforming "
            "how we live, work, and interact with technology."
        )

    # tokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    # build model with preloaded factors (trainable=False)
    model = build_lowrank_gpt2(
        rank_ratio_attn=args.rank_ratio_attn,
        rank_ratio_mlp=args.rank_ratio_mlp,
        preload_dir=args.factors_dir,
        device=args.device,
        trainable=False,
    )
    model.eval()

    # generate
    full_text = generate_lowrank(
        model, tok, prompt_text,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        print_every=args.print_every,
        stop_at_eos=not args.no_stop_at_eos,
    )

    print("\n=== Generated Text ===")
    print(full_text)


if __name__ == "__main__":
    main()


'''
greedy decode

python infer_lowrank.py \
  --factors-dir ./checkpoints/lowrank_gpt2/best_rankA0.6_M0.6_lowrank+ln+bias \
  --prompt "The future of AI" \
  --max-new-tokens 120


sampling with top-k and top-p

python infer_lowrank.py \
  --factors-dir ./checkpoints/lowrank_gpt2/best_rankA0.6_M0.6_lowrank+ln+bias \
  --prompt-file ./my_prompt.txt \
  --max-new-tokens 200 \
  --do-sample --top-p 0.9 --top-k 50 --temperature 0.8

'''