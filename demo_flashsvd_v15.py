#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.demo_support import configure_runtime, generate_text, load_for_inference


LOWRANKARENA_EXAMPLE = (
    "LowRankArena::"
    "llama_7b/Basis_Sharing/share_llama-7b_20"
)
LOWRANKARENA_REPO_ENV = "FLASHSVD_LOWRANKARENA_REPO_ID"


def _maybe_sync(device: str | torch.device) -> None:
    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device=target)


def _run_timed_generation(*, model, tokenizer, prompt: str, max_new_tokens: int, device: str, do_sample: bool, temperature: float, top_p: float):
    _maybe_sync(device)
    t0 = time.perf_counter()
    result = generate_text(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    _maybe_sync(device)
    return result, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a single-prompt FlashSVD v1.5 generation demo",
        epilog=(
            "Example LowRankArena checkpoint: "
            f"{LOWRANKARENA_EXAMPLE}"
        ),
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Checkpoint source. Accepts a local directory, a trusted local .pt checkpoint, "
            "a Hugging Face repo id, or a repo subfolder like namespace/repo::path/to/export. "
            f"Review alias example: {LOWRANKARENA_EXAMPLE}"
        ),
    )
    ap.add_argument(
        "--lowrankarena-repo-id",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face repo id backing the LowRankArena:: review alias. "
            f"Defaults to ${LOWRANKARENA_REPO_ENV}."
        ),
    )
    ap.add_argument(
        "--prompt",
        default="FlashSVD accelerates low-rank language models by",
        help="Prompt text used for generation.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    ap.add_argument("--hf-token", type=str, default=None)
    ap.add_argument("--mode", choices=["flashsvd", "hf"], default="flashsvd")
    ap.add_argument("--ffn-backend", type=str, default="auto")
    ap.add_argument("--mlp-graph", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mlp-graph-scope", choices=["auto", "mlp", "layer_tail"], default="layer_tail")
    ap.add_argument("--flash-dense-attn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dense-decode-backend", choices=["auto", "exact", "packed"], default="auto")
    ap.add_argument("--dense-decode-graph", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dense-decode-autotune-scope", choices=["attention", "reconstruct"], default="attention")
    ap.add_argument("--cutlass-ropeattn", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument(
        "--warmup-tokens",
        type=int,
        default=0,
        help="Optional extra untimed warmup decode length inserted between the cold and steady-state measurements.",
    )
    args = ap.parse_args()

    if args.lowrankarena_repo_id:
        os.environ[LOWRANKARENA_REPO_ENV] = str(args.lowrankarena_repo_id).strip()

    runtime_cfg = configure_runtime(
        mode=args.mode,
        ffn_backend=args.ffn_backend,
        enable_mlp_graph=bool(args.mlp_graph),
        mlp_graph_scope=args.mlp_graph_scope,
        enable_flash_dense_attn=bool(args.flash_dense_attn),
        dense_decode_backend=args.dense_decode_backend,
        enable_dense_decode_graph=bool(args.dense_decode_graph),
        dense_decode_autotune_scope=args.dense_decode_autotune_scope,
        enable_cutlass_rope_attn=bool(args.cutlass_ropeattn),
    )

    load_t0 = time.perf_counter()
    model, tokenizer = load_for_inference(
        args.checkpoint,
        hf_token=args.hf_token,
        dtype=args.dtype,
        device=args.device,
    )
    load_dt = time.perf_counter() - load_t0

    cold_result, cold_dt = _run_timed_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=int(args.max_new_tokens),
        device=args.device,
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )
    warmup_dt = None
    if int(args.warmup_tokens) > 0:
        _, warmup_dt = _run_timed_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=int(args.warmup_tokens),
            device=args.device,
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )
    steady_result, steady_dt = _run_timed_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=int(args.max_new_tokens),
        device=args.device,
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )

    cold_tok_s = float(cold_result["generated_token_count"]) / cold_dt if cold_dt > 0 else float("inf")
    steady_tok_s = float(steady_result["generated_token_count"]) / steady_dt if steady_dt > 0 else float("inf")
    if bool(args.do_sample):
        output_relation = "sampling_enabled"
    else:
        output_relation = "identical" if cold_result["output_ids"] == steady_result["output_ids"] else "different"
    print("== FlashSVD v1.5 Demo ==")
    print(f"checkpoint: {args.checkpoint}")
    print(f"mode: {args.mode}")
    print(f"device: {args.device} | dtype: {args.dtype}")
    print(f"load_time_s: {load_dt:.3f}")
    print(f"cold_generation_time_s: {cold_dt:.3f}")
    print(f"cold_generated_tokens_per_s: {cold_tok_s:.2f}")
    print(f"steady_state_generation_time_s: {steady_dt:.3f}")
    print(f"steady_state_generated_tokens_per_s: {steady_tok_s:.2f}")
    if warmup_dt is not None:
        print(f"extra_warmup_tokens: {int(args.warmup_tokens)}")
        print(f"extra_warmup_time_s: {warmup_dt:.3f}")
    print(f"steady_state_output_vs_cold: {output_relation}")
    print("runtime_env:")
    print(json.dumps(runtime_cfg, indent=2, sort_keys=True))
    print()
    print("prompt:")
    print(args.prompt)
    print()
    print("completion:")
    print(cold_result["completion_text"])
    print()
    print("full_text:")
    print(cold_result["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
