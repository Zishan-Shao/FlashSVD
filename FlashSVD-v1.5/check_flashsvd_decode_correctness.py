#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from utils.model_utils import get_model_from_huggingface, get_model_from_local


def _dtype_from_name(name: str) -> torch.dtype:
    raw = str(name).strip().lower()
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_model(source: str, hf_token: Optional[str]):
    path = Path(source)
    if path.exists():
        return get_model_from_local(str(path))
    return get_model_from_huggingface(source, hf_token=hf_token)


def _load_prompt_text(args: argparse.Namespace) -> Optional[str]:
    if getattr(args, "prompt_text", None):
        return str(args.prompt_text)
    prompt_file = getattr(args, "prompt_file", None)
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    return None


def _build_input_ids(
    *,
    model,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, int, bool]:
    prompt_text = _load_prompt_text(args)
    vocab = int(getattr(model.config, "vocab_size", 32000))
    decode_steps = int(args.decode_steps)
    batch_size = int(args.batch_size)
    if prompt_text is None:
        prompt_len = int(args.prompt_len)
        total_len = prompt_len + decode_steps
        input_ids = torch.randint(
            0,
            vocab,
            (batch_size, total_len),
            device=device,
            dtype=torch.long,
        )
        return input_ids, prompt_len, False

    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=bool(args.add_special_tokens),
    )
    prompt_ids = enc["input_ids"]
    if prompt_ids.numel() == 0:
        raise ValueError("Tokenized prompt is empty.")
    prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
    prompt_len = int(prompt_ids.shape[1])
    total_len = prompt_len + decode_steps
    input_ids = torch.randint(
        0,
        vocab,
        (batch_size, total_len),
        device=device,
        dtype=torch.long,
    )
    input_ids[:, :prompt_len] = prompt_ids.repeat(batch_size, 1)
    return input_ids, prompt_len, True


def _set_env(name: str, value: Optional[str]) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _backend_needs_experimental_ffn(backend: str) -> bool:
    raw = str(backend).strip().lower()
    return raw in {
        "dual_split_cublas",
        "dual_split_kernel",
        "dual_split_kernel_v2",
        "dual_split_kernel_v2_sm80",
        "dual_split_kernel_v3",
    }


def _configure_mode(
    mode: str,
    *,
    backend: str,
    use_graph: bool,
    graph_scope: str,
    enable_flash_dense_attn: bool,
    enable_baseline_dense_kvcache: bool,
    enable_reference_dense_attn: bool,
) -> None:
    mode = str(mode).strip().lower()
    if mode == "baseline":
        _set_env("SVDLLM_FLASH_FALLBACK", "1")
        _set_env("FLASH_SVD_DISABLE_FFN", "1")
        _set_env("FLASH_SVD_BASELINE_LR_KVCACHE", "0")
        _set_env("FLASH_SVD_ENABLE_DENSE_ATTN_DECODE", "0")
        _set_env("FLASH_SVD_BASELINE_DENSE_KVCACHE", "1" if enable_baseline_dense_kvcache else "0")
        _set_env("FLASH_SVD_REFERENCE_DENSE_ATTN", "1" if enable_reference_dense_attn else "0")
        _set_env("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN", "0")
        _set_env("FLASH_SVD_MLP_CUDA_GRAPH", "0")
        _set_env("FLASH_SVD_MLP_CUDA_GRAPH_SCOPE", "mlp")
        _set_env("FLASH_SVD_FFN_BACKEND", "dual_split_cublas_legacy")
        return

    if mode == "flash":
        _set_env("SVDLLM_FLASH_FALLBACK", "0")
        _set_env("FLASH_SVD_DISABLE_FFN", "0")
        _set_env("FLASH_SVD_BASELINE_LR_KVCACHE", "0")
        _set_env("FLASH_SVD_ENABLE_DENSE_ATTN_DECODE", "1" if enable_flash_dense_attn else "0")
        _set_env("FLASH_SVD_BASELINE_DENSE_KVCACHE", "0")
        _set_env("FLASH_SVD_REFERENCE_DENSE_ATTN", "1" if enable_reference_dense_attn else "0")
        _set_env("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN", "1" if _backend_needs_experimental_ffn(backend) else "0")
        _set_env("FLASH_SVD_MLP_CUDA_GRAPH", "1" if use_graph else "0")
        _set_env("FLASH_SVD_MLP_CUDA_GRAPH_SCOPE", str(graph_scope))
        _set_env("FLASH_SVD_FFN_BACKEND", str(backend))
        return

    raise ValueError(f"Unsupported mode: {mode}")


def _clear_runtime_caches(model) -> None:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for name in (
                "_flashsvd_graph_cache",
                "_flashsvd_layer_graph_cache",
            ):
                cache = getattr(mlp, name, None)
                if isinstance(cache, dict):
                    cache.clear()


def _max_abs(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x - y).abs().max().item())


def _mean_abs(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x - y).abs().mean().item())


def _rel_linf(x: torch.Tensor, y: torch.Tensor) -> float:
    denom = float(x.abs().max().item())
    if denom == 0.0:
        return 0.0
    return float((x - y).abs().max().item()) / denom


def _cache_for_mode(model, *, batch_size: int, max_cache_len: int, device: torch.device, dtype: torch.dtype, dense_kv: bool):
    if dense_kv:
        from flashsvd_component.dense_cache import FlashSVDDenseKVCache

        return FlashSVDDenseKVCache(
            model.config,
            max_batch_size=int(batch_size),
            max_cache_len=int(max_cache_len),
            device=device,
            dtype=dtype,
        )

    from transformers.cache_utils import StaticCache

    return StaticCache(
        model.config,
        max_batch_size=int(batch_size),
        max_cache_len=int(max_cache_len),
        device=device,
        dtype=dtype,
    )


@torch.inference_mode()
def _run_fullseq(model, input_ids: torch.Tensor) -> torch.Tensor:
    attn = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    return out.logits.detach().clone()


@torch.inference_mode()
def _run_decode_teacher_forced(
    model,
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    decode_steps: int,
    flash_cache: bool,
) -> torch.Tensor:
    device = input_ids.device
    dtype = next(iter(model.parameters())).dtype
    bsz = int(input_ids.shape[0])
    total_len = int(input_ids.shape[1])
    max_cache_len = total_len + 4
    cache = _cache_for_mode(
        model,
        batch_size=bsz,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
        dense_kv=flash_cache,
    )

    prefix = input_ids[:, :prompt_len]
    prefix_attn = torch.ones_like(prefix, dtype=torch.long, device=device)
    cache_pos = torch.arange(int(prompt_len), device=device, dtype=torch.long)
    out = model(
        input_ids=prefix,
        attention_mask=prefix_attn,
        use_cache=True,
        past_key_values=cache,
        cache_position=cache_pos,
    )
    logits = [out.logits[:, -1, :].detach().clone()]

    for step in range(int(decode_steps) - 1):
        tok = input_ids[:, prompt_len + step : prompt_len + step + 1]
        pos = torch.tensor([int(prompt_len) + step], device=device, dtype=torch.long)
        out = model(
            input_ids=tok,
            use_cache=True,
            past_key_values=cache,
            cache_position=pos,
        )
        logits.append(out.logits[:, -1, :].detach().clone())
    return torch.stack(logits, dim=1)


def _cross_entropy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    return float(loss.item())


@torch.inference_mode()
def _run_decode_greedy(
    model,
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    decode_steps: int,
    flash_cache: bool,
) -> torch.Tensor:
    device = input_ids.device
    dtype = next(iter(model.parameters())).dtype
    bsz = int(input_ids.shape[0])
    max_cache_len = int(prompt_len) + int(decode_steps) + 4
    cache = _cache_for_mode(
        model,
        batch_size=bsz,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
        dense_kv=flash_cache,
    )

    prefix = input_ids[:, :prompt_len]
    prefix_attn = torch.ones_like(prefix, dtype=torch.long, device=device)
    cache_pos = torch.arange(int(prompt_len), device=device, dtype=torch.long)
    out = model(
        input_ids=prefix,
        attention_mask=prefix_attn,
        use_cache=True,
        past_key_values=cache,
        cache_position=cache_pos,
    )
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [next_token.detach().clone()]

    for step in range(int(decode_steps) - 1):
        pos = torch.tensor([int(prompt_len) + step], device=device, dtype=torch.long)
        out = model(
            input_ids=next_token,
            use_cache=True,
            past_key_values=cache,
            cache_position=pos,
        )
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token.detach().clone())
    return torch.cat(tokens, dim=1)


def main() -> int:
    ap = argparse.ArgumentParser("Check FlashSVD decode correctness against the low-rank baseline")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--prompt_len", type=int, default=512)
    ap.add_argument("--decode_steps", type=int, default=32)
    ap.add_argument("--prompt_text", type=str, default=None)
    ap.add_argument("--prompt_file", type=str, default=None)
    ap.add_argument("--add_special_tokens", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--legacy_backend", type=str, default="dual_split_cublas_legacy")
    ap.add_argument("--test_backend", type=str, default="dual_split_kernel_v2_sm80")
    ap.add_argument("--graph", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--graph_scope", type=str, default="mlp", choices=["auto", "mlp", "layer_tail"])
    ap.add_argument("--flash_dense_attn", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--baseline_dense_kvcache", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--reference_dense_attn", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--print_tokens", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--print_text", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this correctness check.")

    torch.manual_seed(int(args.seed))
    dtype = _dtype_from_name(args.dtype)
    model, _tokenizer = _load_model(args.checkpoint, hf_token=args.hf_token)
    model.eval().to(args.device)
    model = model.to(dtype=dtype)

    prev_env = {
        "SVDLLM_FLASH_FALLBACK": os.getenv("SVDLLM_FLASH_FALLBACK"),
        "FLASH_SVD_DISABLE_FFN": os.getenv("FLASH_SVD_DISABLE_FFN"),
        "FLASH_SVD_BASELINE_LR_KVCACHE": os.getenv("FLASH_SVD_BASELINE_LR_KVCACHE"),
        "FLASH_SVD_ENABLE_DENSE_ATTN_DECODE": os.getenv("FLASH_SVD_ENABLE_DENSE_ATTN_DECODE"),
        "FLASH_SVD_BASELINE_DENSE_KVCACHE": os.getenv("FLASH_SVD_BASELINE_DENSE_KVCACHE"),
        "FLASH_SVD_REFERENCE_DENSE_ATTN": os.getenv("FLASH_SVD_REFERENCE_DENSE_ATTN"),
        "FLASH_SVD_ENABLE_EXPERIMENTAL_FFN": os.getenv("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN"),
        "FLASH_SVD_MLP_CUDA_GRAPH": os.getenv("FLASH_SVD_MLP_CUDA_GRAPH"),
        "FLASH_SVD_MLP_CUDA_GRAPH_SCOPE": os.getenv("FLASH_SVD_MLP_CUDA_GRAPH_SCOPE"),
        "FLASH_SVD_FFN_BACKEND": os.getenv("FLASH_SVD_FFN_BACKEND"),
    }
    try:
        input_ids, actual_prompt_len, using_text_prompt = _build_input_ids(
            model=model,
            tokenizer=_tokenizer,
            args=args,
            device=torch.device(args.device),
        )

        baseline_dense_cache = bool(args.baseline_dense_kvcache)
        flash_dense_cache = bool(args.flash_dense_attn)
        modes = [
            ("baseline", "baseline", baseline_dense_cache),
            ("flash_legacy", str(args.legacy_backend), flash_dense_cache),
            ("flash_test", str(args.test_backend), flash_dense_cache),
        ]

        outputs = {}
        for label, backend, dense_cache in modes:
            _configure_mode(
                "baseline" if label == "baseline" else "flash",
                backend=backend,
                use_graph=bool(args.graph),
                graph_scope=str(args.graph_scope),
                enable_flash_dense_attn=flash_dense_cache if label != "baseline" else False,
                enable_baseline_dense_kvcache=baseline_dense_cache if label == "baseline" else False,
                enable_reference_dense_attn=bool(args.reference_dense_attn),
            )
            _clear_runtime_caches(model)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            full_logits = _run_fullseq(model, input_ids[:, :actual_prompt_len])
            decode_logits = _run_decode_teacher_forced(
                model,
                input_ids,
                prompt_len=actual_prompt_len,
                decode_steps=int(args.decode_steps),
                flash_cache=dense_cache,
            )
            greedy_tokens = _run_decode_greedy(
                model,
                input_ids,
                prompt_len=actual_prompt_len,
                decode_steps=int(args.decode_steps),
                flash_cache=dense_cache,
            )
            outputs[label] = {
                "full_logits": full_logits,
                "decode_logits": decode_logits,
                "greedy_tokens": greedy_tokens,
            }

        full_targets = input_ids[:, 1:actual_prompt_len]
        decode_targets = input_ids[:, actual_prompt_len : actual_prompt_len + int(args.decode_steps)]

        base_full = outputs["baseline"]["full_logits"][:, :-1, :]
        base_decode = outputs["baseline"]["decode_logits"]

        print("==== FlashSVD Decode Correctness ====")
        print(
            f"Config: B={args.batch_size} prompt_len={actual_prompt_len} decode_steps={args.decode_steps} "
            f"dtype={args.dtype} graph={int(args.graph)} flash_dense_attn={int(args.flash_dense_attn)} "
            f"baseline_dense_kvcache={int(args.baseline_dense_kvcache)} reference_dense_attn={int(args.reference_dense_attn)} "
            f"test_backend={args.test_backend} prompt_source={'text' if using_text_prompt else 'random'}"
        )

        print(
            f"baseline full_seq_ce={_cross_entropy_from_logits(base_full, full_targets):.6f} "
            f"decode_ce={_cross_entropy_from_logits(base_decode, decode_targets):.6f}"
        )
        for label in ("flash_legacy", "flash_test"):
            cur_full = outputs[label]["full_logits"][:, :-1, :]
            cur_decode = outputs[label]["decode_logits"]
            cur_tokens = outputs[label]["greedy_tokens"]
            base_tokens = outputs["baseline"]["greedy_tokens"]
            token_match = float((cur_tokens == base_tokens).to(torch.float32).mean().item())
            print(
                f"- {label}: "
                f"full_max_abs={_max_abs(base_full, cur_full):.6g} "
                f"full_mean_abs={_mean_abs(base_full, cur_full):.6g} "
                f"full_rel_linf={_rel_linf(base_full, cur_full):.6g} "
                f"full_ce={_cross_entropy_from_logits(cur_full, full_targets):.6f} "
                f"| decode_max_abs={_max_abs(base_decode, cur_decode):.6g} "
                f"decode_mean_abs={_mean_abs(base_decode, cur_decode):.6g} "
                f"decode_rel_linf={_rel_linf(base_decode, cur_decode):.6g} "
                f"decode_ce={_cross_entropy_from_logits(cur_decode, decode_targets):.6f} "
                f"| greedy_token_match={token_match:.6f}"
            )

        if args.print_tokens:
            if using_text_prompt:
                print(f"Prompt token ids: {input_ids[0, :actual_prompt_len].tolist()}")
            print("Greedy token ids:")
            base_tokens = outputs["baseline"]["greedy_tokens"][0].tolist()
            print(f"- baseline    : {base_tokens}")
            for label in ("flash_legacy", "flash_test"):
                cur_tokens = outputs[label]["greedy_tokens"][0].tolist()
                print(f"- {label:<12}: {cur_tokens}")

        if args.print_text:
            print("Greedy decoded text:")
            try:
                if using_text_prompt:
                    prompt_txt = _tokenizer.decode(
                        input_ids[0, :actual_prompt_len].tolist(),
                        skip_special_tokens=False,
                    )
                    print(f"- prompt       : {prompt_txt!r}")
                base_txt = _tokenizer.decode(outputs["baseline"]["greedy_tokens"][0].tolist(), skip_special_tokens=False)
                print(f"- baseline    : {base_txt!r}")
                for label in ("flash_legacy", "flash_test"):
                    cur_txt = _tokenizer.decode(outputs[label]["greedy_tokens"][0].tolist(), skip_special_tokens=False)
                    print(f"- {label:<12}: {cur_txt!r}")
            except Exception as exc:
                print(f"[warn] failed to decode greedy tokens as text: {exc!r}")
    finally:
        for name, value in prev_env.items():
            _set_env(name, value)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
