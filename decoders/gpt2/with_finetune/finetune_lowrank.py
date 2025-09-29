# finetune_lowrank.py
# ------------------------------------------------------------
# Fine-tune low-rank GPT-2 on WikiText-2/PTB with AMP + warmup.
# ------------------------------------------------------------
import math
import os
import time
import argparse
from typing import Optional, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from lowrank_gpt2 import (
    set_seed,
    build_lowrank_gpt2,
    get_param_groups,
    save_lowrank_factors,
    evaluate_perplexity,
)


def make_dataloaders(
    dataset: str,
    max_length: int,
    batch_size: int,
    train_samples: Optional[int] = None,
    eval_samples: Optional[int] = None,
):
    if dataset.lower() in ("wikitext2", "wikitext-2", "wiki"):
        train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        textcol = "text"
    elif dataset.lower() in ("ptb", "penn", "ptb_text_only"):
        train = load_dataset("ptb_text_only", "penn_treebank", split="train")
        val = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        textcol = "sentence"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    def tok_fn(batch):
        return tok(
            batch[textcol],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train = train.map(tok_fn, batched=True, remove_columns=[textcol])
    val = val.map(tok_fn, batched=True, remove_columns=[textcol])

    if train_samples is not None:
        train = train.select(range(min(train_samples, len(train))))
    if eval_samples is not None:
        val = val.select(range(min(eval_samples, len(val))))

    train.set_format("torch")
    val.set_format("torch")

    def collate(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        }

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


# def train_one_epoch(
#     model,
#     loader,
#     optimizer,
#     device: str,
#     scaler: Optional[torch.cuda.amp.GradScaler] = None,
#     grad_accum_steps: int = 1,
#     scheduler=None,
#     mixed_precision: Optional[str] = None,  # "fp16" | "bf16" | None
#     max_norm: Optional[float] = 1.0,
# ): 
#     model.train()
#     use_amp = (mixed_precision in ("fp16", "bf16"))
#     amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16

#     loss_meter = 0.0
#     tokens_meter = 0
#     step = 0

#     t0 = time.perf_counter()
#     for it, batch in enumerate(loader):
#         input_ids = batch["input_ids"].to(device)
#         attn = batch["attention_mask"].to(device)

#         with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
#             out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
#             logits = out.logits
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = input_ids[..., 1:].contiguous()
#             m = attn[..., 1:].contiguous().bool()

#             if m.any():
#                 valid_logits = shift_logits[m]
#                 valid_labels = shift_labels[m]
#                 loss = F.cross_entropy(valid_logits, valid_labels)
#             else:
#                 loss = torch.tensor(0.0, device=device, requires_grad=True)

#         loss = loss / grad_accum_steps

#         if scaler is not None:
#             scaler.scale(loss).backward()
#         else:
#             loss.backward()

#         if (it + 1) % grad_accum_steps == 0:
#             if scaler is not None:
#                 if max_norm is not None:
#                     scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 if max_norm is not None:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#                 optimizer.step()
#             optimizer.zero_grad(set_to_none=True)
#             if scheduler is not None:
#                 scheduler.step()
#             step += 1

#         # meters
#         loss_meter += loss.item() * m.sum().item() * grad_accum_steps
#         tokens_meter += m.sum().item() * grad_accum_steps

#     elapsed = time.perf_counter() - t0
#     avg_loss = loss_meter / max(tokens_meter, 1)
#     ppl = math.exp(avg_loss) if tokens_meter > 0 else float("inf")
#     return ppl, elapsed



# --- change train_one_epoch signature ---
def train_one_epoch(
    model,
    loader,
    optimizer,
    device: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_accum_steps: int = 1,
    scheduler=None,
    mixed_precision: Optional[str] = None,  # "fp16" | "bf16" | None
    max_norm: Optional[float] = 1.0,
    log_interval: int = 100,
    step_offset: int = 0,
):
    # --- inside train_one_epoch, replace the body with this updated version ---
    model.train()
    use_amp = (mixed_precision in ("fp16", "bf16"))
    amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16

    loss_meter = 0.0
    tokens_meter = 0
    step = 0  # number of optimizer updates this epoch

    # window for periodic logging
    win_loss = 0.0
    win_tokens = 0
    win_t0 = time.perf_counter()

    t0 = time.perf_counter()
    for it, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            m = attn[..., 1:].contiguous().bool()

            if m.any():
                valid_logits = shift_logits[m]
                valid_labels = shift_labels[m]
                loss = F.cross_entropy(valid_logits, valid_labels)
                n_tok = int(m.sum().item())
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                n_tok = 0

        # scale by grad accumulation
        loss = loss / max(grad_accum_steps, 1)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        did_update = False
        last_gn = None

        if (it + 1) % grad_accum_steps == 0:
            # clip + step
            if scaler is not None:
                if max_norm is not None:
                    scaler.unscale_(optimizer)
                    last_gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_norm is not None:
                    last_gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            step += 1
            did_update = True

            # periodic logging (every N optimizer updates)
            global_step = step_offset + step
            if log_interval > 0 and (global_step % log_interval == 0):
                dt = max(time.perf_counter() - win_t0, 1e-6)
                win_avg_loss = (win_loss / max(win_tokens, 1)) if win_tokens > 0 else float("inf")
                win_ppl = math.exp(win_avg_loss) if win_tokens > 0 else float("inf")
                lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
                tps = win_tokens / dt
                msg = (f"[Step {global_step}] "
                       f"lr={lr:.2e} "
                       f"win_loss={win_avg_loss:.4f} "
                       f"win_ppl={win_ppl:.3f} "
                       f"win_tok={win_tokens} "
                       f"tok/s={tps:.1f}")
                if last_gn is not None:
                    msg += f" grad_norm={last_gn:.2f}"
                print(msg, flush=True)
                # reset window
                win_loss = 0.0
                win_tokens = 0
                win_t0 = time.perf_counter()

        # meters (undo the 1/grad_accum scaling)
        loss_meter += float(loss.item()) * n_tok * max(grad_accum_steps, 1)
        tokens_meter += n_tok * max(grad_accum_steps, 1)
        # window meters
        win_loss += float(loss.item()) * n_tok * max(grad_accum_steps, 1)
        win_tokens += n_tok * max(grad_accum_steps, 1)

    elapsed = time.perf_counter() - t0
    avg_loss = loss_meter / max(tokens_meter, 1)
    ppl = math.exp(avg_loss) if tokens_meter > 0 else float("inf")
    return ppl, elapsed, step



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # ranks
    ap.add_argument("--rank-ratio-attn", type=float, default=0.5)
    ap.add_argument("--rank-ratio-mlp", type=float, default=0.5)

    # data
    ap.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb"])
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-samples", type=int, default=20000, help="Num training sequences (None uses full)")
    ap.add_argument("--eval-samples", type=int, default=2048)

    # train
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="If >0, overrides epochs.")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    ap.add_argument("--update", type=str, default="lowrank-only", choices=["lowrank-only", "lowrank+ln+bias"])

    # IO
    ap.add_argument("--preload-factors", type=str, default=None, help="Optional dir with precomputed low-rank factors")
    ap.add_argument("--save-dir", type=str, default="./checkpoints/lowrank_gpt2")
    ap.add_argument("--save-best", action="store_true", help="Save factors when validation perplexity improves")
    ap.add_argument("--log-interval", type=int, default=100, help="Log every N optimizer updates")

    args = ap.parse_args()
    set_seed(args.seed)

    device = args.device
    mp = None if args.mixed_precision == "none" else args.mixed_precision

    # data
    train_loader, val_loader = make_dataloaders(
        dataset=args.dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_samples=None if args.train_samples <= 0 else args.train_samples,
        eval_samples=None if args.eval_samples <= 0 else args.eval_samples,
    )

    # model (low-rank)
    model = build_lowrank_gpt2(
        rank_ratio_attn=args.rank_ratio_attn,
        rank_ratio_mlp=args.rank_ratio_mlp,
        preload_dir=args.preload_factors,
        device=device,
        trainable=True,
    )

    # optimizer param groups
    groups = get_param_groups(
        model,
        update=args.update,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)

    # scheduler
    steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # scaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(mp == "fp16"))

    # baseline perplexity before finetune
    with torch.no_grad():
        base_ppl = evaluate_perplexity(model, val_loader, device=device, mixed_precision=mp)
    print(f"[Eval] Initial low-rank PPL: {base_ppl:.3f}")

    # # training loop
    # global_step = 0
    # best_ppl = float("inf")
    # for epoch in range(max(args.epochs, 1_000_000)):  # will break by max_steps
    #     train_ppl, train_time = train_one_epoch(
    #         model, train_loader, optimizer, device,
    #         scaler=scaler, grad_accum_steps=args.grad_accum,
    #         scheduler=scheduler, mixed_precision=mp,
    #         max_norm=args.max_grad_norm
    #     )
    #     global_step += steps_per_epoch
    #     print(f"[Train] epoch={epoch} ppl={train_ppl:.3f} time={train_time:.1f}s")

    #     val_ppl = evaluate_perplexity(model, val_loader, device=device, mixed_precision=mp)
    #     print(f"[Eval ] epoch={epoch} ppl={val_ppl:.3f} (best={best_ppl:.3f})")

    #     if args.save_best and val_ppl < best_ppl:
    #         best_ppl = val_ppl
    #         save_dir = os.path.join(args.save_dir, f"best_rankA{args.rank_ratio_attn}_M{args.rank_ratio_mlp}_{args.update}")
    #         os.makedirs(save_dir, exist_ok=True)
    #         save_lowrank_factors(model, save_dir)
    #         print(f"[Save] Improved perplexity; saved low-rank factors to: {save_dir}")

    #     if args.max_steps > 0 and global_step >= args.max_steps:
    #         print(f"[Stop] Reached max_steps={args.max_steps}.")
    #         break
    # --- in main(), where you call train_one_epoch ---
    global_step = 0
    best_ppl = float("inf")
    for epoch in range(max(args.epochs, 1_000_000)):  # will break by max_steps
        train_ppl, train_time, steps_done = train_one_epoch(
            model, train_loader, optimizer, device,
            scaler=scaler, grad_accum_steps=args.grad_accum,
            scheduler=scheduler, mixed_precision=mp,
            max_norm=args.max_grad_norm,
            log_interval=args.log_interval,
            step_offset=global_step,
        )
        global_step += steps_done
        print(f"[Train] epoch={epoch} ppl={train_ppl:.3f} time={train_time:.1f}s")

        val_ppl = evaluate_perplexity(model, val_loader, device=device, mixed_precision=mp)
        print(f"[Eval ] epoch={epoch} ppl={val_ppl:.3f} (best={best_ppl:.3f})")

    # final save
    final_dir = os.path.join(args.save_dir, f"final_rankA{args.rank_ratio_attn}_M{args.rank_ratio_mlp}_{args.update}")
    os.makedirs(final_dir, exist_ok=True)
    save_lowrank_factors(model, final_dir)
    print(f"[Save] Final low-rank factors saved to: {final_dir}")

    # final eval
    final_ppl = evaluate_perplexity(model, val_loader, device=device, mixed_precision=mp)
    print(f"[Eval] Final low-rank PPL: {final_ppl:.3f}")


if __name__ == "__main__":
    main()


'''
python finetune_lowrank.py \
  --dataset wikitext2 \
  --rank-ratio-attn 0.6 \
  --rank-ratio-mlp 0.6 \
  --batch-size 8 \
  --max-length 256 \
  --epochs 1 \
  --grad-accum 2 \
  --lr 2e-4 \
  --weight-decay 0.01 \
  --mixed-precision bf16 \
  --update lowrank+ln+bias \
  --preload-factors ./factors/gpt2_svdllm \
  --save-best \
  --save-dir ./checkpoints/lowrank_gpt2
'''