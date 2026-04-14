#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import os
from pathlib import Path
import re
import hashlib
import statistics
import sys
import time

import torch


THIS = Path(__file__).resolve()
STUDY_DIR = THIS.parents[1]
RESULTS_DIR = STUDY_DIR.parent
ROOT = RESULTS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.decode.bench_flashsvd_vs_svd_decode import (  # noqa: E402
    _cast_model_for_eval,
    _configure_mode,
    _load_model_and_tokenizer,
)
from benchmark.decode.check_flashsvd_decode_correctness import _run_decode_greedy  # noqa: E402
from utils.evaluator import decode_kvcache_eval  # noqa: E402


RAW_DIR = STUDY_DIR / "raw"
PROFILE_DIR = STUDY_DIR / "profiles"
TABLE_DIR = STUDY_DIR / "tables"
EXAMPLE_DIR = STUDY_DIR / "examples"

RATIO_PATHS = {
    "0.5": "/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.5/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.5_hf",
    "0.6": "/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.6/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.6_hf",
    "0.7": "/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.7/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.7_hf",
    "0.8": "/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.8/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.8_hf",
}

MODES = {
    "static": {"mode": "svd", "baseline_dense": False, "flash_cache": False, "label": "StaticCache"},
    "densekv": {"mode": "svd", "baseline_dense": True, "flash_cache": True, "label": "DenseKVCacheBaseline"},
    "flashsvd": {"mode": "flashsvd", "baseline_dense": False, "flash_cache": True, "label": "FlashSVD-v1.5"},
}

GRAPH_VARIANTS = {
    "nograph": {"dense_decode_graph": "0", "mlp_graph": False, "scope": "mlp"},
    "split": {"dense_decode_graph": "1", "mlp_graph": True, "scope": "mlp"},
    "layer": {"dense_decode_graph": "1", "mlp_graph": True, "scope": "layer_tail"},
}

PROMPTS = [
    "The capital of France is",
    "FlashSVD accelerates low-rank language models by",
    "In a future where AI systems assist scientific research,",
]

TRACKED_OPS = [
    "cudaLaunchKernel",
    "cudaGraphLaunch",
    "aten::copy_",
    "aten::clone",
    "aten::to",
    "aten::_to_copy",
    "cudaMemcpyAsync",
]


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROFILE_DIR, TABLE_DIR, EXAMPLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sync_if_needed(device: str) -> None:
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()


def stable_seed(*parts: object, offset: int = 0) -> int:
    payload = "::".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    return int(offset) + (int(digest[:8], 16) % 1000)


def configure_runtime(mode_key: str, *, dense_decode_graph: str = "1", mlp_graph: bool = True, scope: str = "layer_tail") -> tuple[bool, bool, bool]:
    cfg = MODES[mode_key]
    os.environ["FLASH_SVD_DENSE_DECODE_BACKEND"] = "packed"
    os.environ["FLASH_SVD_DENSE_DECODE_GRAPH"] = str(dense_decode_graph)
    return _configure_mode(
        cfg["mode"],
        ffn_backend="flashsvd_mlp_dual_split_prod",
        enable_mlp_graph=bool(mlp_graph),
        mlp_graph_scope=str(scope),
        graph_alias_output=False,
        enable_flash_dense_attn=True,
        enable_cutlass_rope_attn=False,
        enable_baseline_dense_kvcache=bool(cfg["baseline_dense"]),
    )


def load_model(mode_key: str, source: str, *, device: str, dtype_name: str) -> tuple:
    lowrank_cache, flashsvd_dense_cache, baseline_dense_kvcache = configure_runtime(mode_key)
    model, tokenizer = _load_model_and_tokenizer(source, hf_token=None)
    model.eval()
    model = _cast_model_for_eval(model, dtype_name)
    model = model.to(device)
    return model, tokenizer, lowrank_cache, flashsvd_dense_cache, baseline_dense_kvcache


def unload_model(model, tokenizer) -> None:
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def parse_module_profile(text: str) -> dict:
    out: dict[str, object] = {}
    total_match = re.search(r"total_forward\s+([0-9.]+)\s+ms/token", text)
    if total_match:
        out["total_forward_ms"] = float(total_match.group(1))
    bucket_re = re.compile(r"^\s+([A-Za-z0-9_.]+)\s*:\s*([0-9.]+)\s+ms\s+\(([0-9.]+)%\)", re.MULTILINE)
    buckets = {}
    for name, ms, pct in bucket_re.findall(text):
        buckets[name] = {"ms": float(ms), "pct": float(pct)}
    out["buckets"] = buckets
    return out


def run_module_profile(
    *,
    model,
    device: str,
    lowrank_cache: bool,
    flashsvd_dense_cache: bool,
    baseline_dense_kvcache: bool,
    prompt_len: int,
    new_tokens: int,
    warmup: int,
    profile_steps: int,
    seed: int,
) -> tuple[dict, str]:
    set_seed(seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = decode_kvcache_eval(
            model,
            prompt_len=prompt_len,
            new_tokens=new_tokens,
            warmup=warmup,
            max_cache_len=None,
            batch_size=1,
            device=device,
            lowrank_cache=bool(lowrank_cache),
            flashsvd_dense_cache=bool(flashsvd_dense_cache),
            baseline_dense_kvcache=bool(baseline_dense_kvcache),
            profile_decode=True,
            profile_decode_steps=profile_steps,
        )
    text = buf.getvalue()
    parsed = parse_module_profile(text)
    parsed["prefill_time_s"] = float(result["prefill_time_s"])
    parsed["decode_ms_per_token"] = float(result["decode_ms_per_token"])
    parsed["profile_steps"] = int(profile_steps)
    return parsed, text


def _cache_for_mode(model, *, batch_size: int, max_cache_len: int, device: torch.device, dtype: torch.dtype, dense_kv: bool):
    if dense_kv:
        from runtime.cache.attn_dense_kv import FlashSVDV15DenseKVCache

        return FlashSVDV15DenseKVCache(
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


def _tracked_summary(prof, *, steps: int) -> dict:
    by_name = {evt.key: evt for evt in prof.key_averages()}
    tracked = {}
    for name in TRACKED_OPS:
        evt = by_name.get(name)
        if evt is None:
            tracked[name] = {"count": 0, "cpu_ms": 0.0, "self_cpu_ms": 0.0, "cuda_ms": 0.0, "self_cuda_ms": 0.0}
            continue
        tracked[name] = {
            "count": int(evt.count),
            "cpu_ms": float(evt.cpu_time_total) / 1000.0,
            "self_cpu_ms": float(evt.self_cpu_time_total) / 1000.0,
            "cuda_ms": float(getattr(evt, "cuda_time_total", 0.0)) / 1000.0,
            "self_cuda_ms": float(getattr(evt, "self_cuda_time_total", 0.0)) / 1000.0,
        }
    out = {
        "tracked_ops": tracked,
        "cudaLaunchKernel_per_token": tracked["cudaLaunchKernel"]["count"] / float(steps),
        "cudaGraphLaunch_per_token": tracked["cudaGraphLaunch"]["count"] / float(steps),
        "copy_per_token": tracked["aten::copy_"]["count"] / float(steps),
        "clone_per_token": tracked["aten::clone"]["count"] / float(steps),
        "to_copy_per_token": (tracked["aten::to"]["count"] + tracked["aten::_to_copy"]["count"]) / float(steps),
        "launch_cpu_ms_per_token": (tracked["cudaLaunchKernel"]["cpu_ms"] + tracked["cudaGraphLaunch"]["cpu_ms"]) / float(steps),
        "copy_clone_cpu_ms_per_token": (tracked["aten::copy_"]["cpu_ms"] + tracked["aten::clone"]["cpu_ms"]) / float(steps),
    }
    return out


def _random_prompt_ids(model, *, prompt_len: int, batch_size: int, device: torch.device, seed: int) -> torch.Tensor:
    set_seed(seed)
    vocab = int(getattr(model.config, "vocab_size", 32000))
    return torch.randint(0, vocab, (int(batch_size), int(prompt_len)), device=device, dtype=torch.long)


def run_op_profile(
    *,
    model,
    device: str,
    dense_cache: bool,
    prompt_len: int,
    decode_steps: int,
    warmup: int,
    seed: int,
) -> tuple[dict, str]:
    dev = torch.device(device)
    dtype = next(iter(model.parameters())).dtype
    batch_size = 1
    input_ids = _random_prompt_ids(model, prompt_len=prompt_len, batch_size=batch_size, device=dev, seed=seed)
    max_cache_len = int(prompt_len) + int(warmup) + int(decode_steps) + 8
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    # Prefill profile
    cache = _cache_for_mode(
        model,
        batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=dev,
        dtype=dtype,
        dense_kv=dense_cache,
    )
    prefix_attn = torch.ones_like(input_ids, dtype=torch.long, device=dev)
    cache_pos = torch.arange(int(prompt_len), device=dev, dtype=torch.long)
    sync_if_needed(device)
    with torch.no_grad(), torch.profiler.profile(activities=activities) as prof_prefill:
        out = model(
            input_ids=input_ids,
            attention_mask=prefix_attn,
            use_cache=True,
            past_key_values=cache,
            cache_position=cache_pos,
        )
    sync_if_needed(device)

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for t in range(int(warmup)):
        pos = torch.tensor([int(prompt_len) + t], device=dev, dtype=torch.long)
        out = model(input_ids=next_token, use_cache=True, past_key_values=cache, cache_position=pos)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync_if_needed(device)

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    with torch.no_grad(), torch.profiler.profile(activities=activities) as prof_decode:
        for t in range(int(decode_steps)):
            pos = torch.tensor([int(prompt_len) + int(warmup) + t], device=dev, dtype=torch.long)
            out = model(input_ids=next_token, use_cache=True, past_key_values=cache, cache_position=pos)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    ev1.record()
    sync_if_needed(device)
    decode_total_ms = float(ev0.elapsed_time(ev1))

    prefill_table = prof_prefill.key_averages().table(sort_by="cpu_time_total", row_limit=60)
    decode_table = prof_decode.key_averages().table(sort_by="cpu_time_total", row_limit=80)
    text = (
        "=== Prefill Profile ===\n"
        + prefill_table
        + "\n\n=== Decode Profile ===\n"
        + decode_table
        + "\n"
    )
    out_summary = {
        "prompt_len": int(prompt_len),
        "decode_steps": int(decode_steps),
        "decode_total_ms": decode_total_ms,
        "decode_ms_per_token": decode_total_ms / float(max(1, decode_steps)),
        "prefill": _tracked_summary(prof_prefill, steps=1),
        "decode": _tracked_summary(prof_decode, steps=int(decode_steps)),
    }
    return out_summary, text


def run_graph_fragmentation_profile(
    *,
    source: str,
    ratio: str,
    device: str,
    dtype_name: str,
    prompt_len: int,
    new_tokens: int,
    profile_steps: int,
) -> dict:
    results = {}
    for variant, cfg in GRAPH_VARIANTS.items():
        lowrank_cache, flashsvd_dense_cache, baseline_dense_kvcache = configure_runtime(
            "flashsvd",
            dense_decode_graph=cfg["dense_decode_graph"],
            mlp_graph=cfg["mlp_graph"],
            scope=cfg["scope"],
        )
        model, tokenizer = _load_model_and_tokenizer(source, hf_token=None)
        model.eval()
        model = _cast_model_for_eval(model, dtype_name).to(device)
        try:
            summary, text = run_op_profile(
                model=model,
                device=device,
                dense_cache=True,
                prompt_len=prompt_len,
                decode_steps=profile_steps,
                warmup=3,
                seed=stable_seed("graph", ratio, variant, offset=7000),
            )
            text_path = PROFILE_DIR / f"graph_fragmentation_{ratio}_{variant}.txt"
            text_path.write_text(text)
            results[variant] = summary["decode"]
            results[variant]["decode_ms_per_token"] = summary["decode_ms_per_token"]
        finally:
            unload_model(model, tokenizer)
    return results


def run_generation_examples(
    *,
    model,
    tokenizer,
    mode_key: str,
    device: str,
    decode_steps: int,
) -> list[dict]:
    rows = []
    dev = torch.device(device)
    dense_cache = MODES[mode_key]["flash_cache"]
    for prompt in PROMPTS:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device=dev, dtype=torch.long)
        greedy_tokens = _run_decode_greedy(
            model,
            input_ids,
            prompt_len=int(input_ids.shape[1]),
            decode_steps=int(decode_steps),
            flash_cache=bool(dense_cache),
        )
        text = tokenizer.decode(greedy_tokens[0].tolist(), skip_special_tokens=False)
        rows.append({"prompt": prompt, "generated_text": text})
    return rows


def write_examples_markdown(all_examples: dict) -> None:
    lines = ["# Decode Examples", ""]
    for ratio in sorted(all_examples):
        lines.append(f"## Ratio {ratio}")
        lines.append("")
        for mode_key in ("static", "densekv", "flashsvd"):
            lines.append(f"### {MODES[mode_key]['label']}")
            lines.append("")
            for idx, item in enumerate(all_examples[ratio][mode_key], start=1):
                lines.append(f"Prompt {idx}: `{item['prompt']}`")
                lines.append("")
                lines.append("```text")
                lines.append(item["generated_text"])
                lines.append("```")
                lines.append("")
    (EXAMPLE_DIR / "decode_examples.md").write_text("\n".join(lines))


def write_experiment_settings(args: argparse.Namespace) -> None:
    text = f"""# Ratio Profile Study

Date: 2026-03-31

## Goal

Collect ratio-wise profiling beyond end-to-end speed for SVD-LLM v1 exported checkpoints at ratios:

- {", ".join(args.ratios)}

The study emphasizes:

- module-wise decode timing
- op-level CPU/CUDA profiler output
- decode-path launch and staging overhead
- graph-fragmentation counts for FlashSVD runtime variants
- qualitative greedy decode examples for paper figures or appendix

## Machine / Software

- Host cwd: `/home/zs89/FlashSVD`
- GPU target: `CUDA_VISIBLE_DEVICES={args.gpu}`
- Python env: `/home/zs89/miniconda3/envs/flashsvd15/bin/python`
- Python: `3.13.2`
- PyTorch: `2.7.1+cu128`
- Triton: `3.3.1`
- Transformers: `4.53.0`
- FlashAttention: `2.8.3`

## Runtime Recipe

Default active path:

```bash
FLASH_SVD_DENSE_DECODE_BACKEND=packed
FLASH_SVD_DENSE_DECODE_GRAPH=1
--experimental_flash_dense_attn
--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod
--mlp_cuda_graph
--mlp_cuda_graph_scope layer_tail
```

## Profiling Matrix

For each ratio and each runtime mode (`StaticCache`, `DenseKVCacheBaseline`, `FlashSVD-v1.5`):

- module-wise decode profile on `prompt_len={args.profile_prompt_len}`, `new_tokens={args.profile_new_tokens}`, `profile_decode_steps={args.profile_decode_steps}`
- op-level `torch.profiler` for:
  - prefill
  - decode

For each ratio and FlashSVD only:

- graph fragmentation profile on variants:
  - `nograph`
  - `split`
  - `layer`

Greedy decode examples:

- prompts: {len(PROMPTS)}
- generated tokens per prompt: `{args.example_decode_steps}`
"""
    (STUDY_DIR / "EXPERIMENT_SETTINGS.md").write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser("Collect ratio-wise module/op profiles and decode examples")
    ap.add_argument("--ratios", nargs="+", default=["0.5", "0.6", "0.7", "0.8"])
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--profile_prompt_len", type=int, default=512)
    ap.add_argument("--profile_new_tokens", type=int, default=32)
    ap.add_argument("--profile_decode_steps", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--example_decode_steps", type=int, default=64)
    args = ap.parse_args()

    ensure_dirs()
    write_experiment_settings(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["FLASH_SVD_DENSE_DECODE_BACKEND"] = "packed"
    os.environ["FLASH_SVD_DENSE_DECODE_GRAPH"] = "1"

    device = "cuda"
    module_profiles = {}
    op_profiles = {}
    graph_profiles = {}
    all_examples = {}

    start = time.perf_counter()
    for ratio in args.ratios:
        source = RATIO_PATHS[ratio]
        module_profiles[ratio] = {}
        op_profiles[ratio] = {}
        all_examples[ratio] = {}
        print(f"\n==== Ratio {ratio} :: {source} ====")
        for mode_key in ("static", "densekv", "flashsvd"):
            print(f"-- Profiling mode={mode_key}")
            model, tokenizer, lowrank_cache, flashsvd_dense_cache, baseline_dense_kvcache = load_model(
                mode_key, source, device=device, dtype_name=args.dtype
            )
            try:
                op_summary, op_text = run_op_profile(
                    model=model,
                    device=device,
                    dense_cache=bool(MODES[mode_key]["flash_cache"]),
                    prompt_len=args.profile_prompt_len,
                    decode_steps=args.profile_decode_steps,
                    warmup=args.warmup,
                    seed=stable_seed(ratio, mode_key, "op", offset=2000),
                )
                (PROFILE_DIR / f"op_profile_{ratio}_{mode_key}.txt").write_text(op_text)
                op_profiles[ratio][mode_key] = op_summary

                parsed_module, module_text = run_module_profile(
                    model=model,
                    device=device,
                    lowrank_cache=lowrank_cache,
                    flashsvd_dense_cache=flashsvd_dense_cache,
                    baseline_dense_kvcache=baseline_dense_kvcache,
                    prompt_len=args.profile_prompt_len,
                    new_tokens=args.profile_new_tokens,
                    warmup=args.warmup,
                    profile_steps=args.profile_decode_steps,
                    seed=stable_seed(ratio, mode_key, "module", offset=1000),
                )
                (PROFILE_DIR / f"module_profile_{ratio}_{mode_key}.txt").write_text(module_text)
                module_profiles[ratio][mode_key] = parsed_module

                examples = run_generation_examples(
                    model=model,
                    tokenizer=tokenizer,
                    mode_key=mode_key,
                    device=device,
                    decode_steps=args.example_decode_steps,
                )
                (EXAMPLE_DIR / f"decode_examples_{ratio}_{mode_key}.json").write_text(json.dumps(examples, indent=2))
                all_examples[ratio][mode_key] = examples
            finally:
                unload_model(model, tokenizer)

        print(f"-- Graph fragmentation ratio={ratio}")
        graph_profiles[ratio] = run_graph_fragmentation_profile(
            source=source,
            ratio=ratio,
            device=device,
            dtype_name=args.dtype,
            prompt_len=args.profile_prompt_len,
            new_tokens=args.profile_new_tokens,
            profile_steps=args.profile_decode_steps,
        )

    (TABLE_DIR / "module_profiles.json").write_text(json.dumps(module_profiles, indent=2))
    (TABLE_DIR / "op_profiles.json").write_text(json.dumps(op_profiles, indent=2))
    (TABLE_DIR / "graph_fragmentation_profiles.json").write_text(json.dumps(graph_profiles, indent=2))
    write_examples_markdown(all_examples)

    summary_rows = []
    for ratio in args.ratios:
        for mode_key in ("static", "densekv", "flashsvd"):
            m = module_profiles[ratio][mode_key]
            o = op_profiles[ratio][mode_key]["decode"]
            summary_rows.append(
                {
                    "ratio": ratio,
                    "mode": MODES[mode_key]["label"],
                    "module_total_forward_ms": m.get("total_forward_ms", 0.0),
                    "decode_ms_per_token": op_profiles[ratio][mode_key]["decode_ms_per_token"],
                    "cudaLaunchKernel_per_token": o["cudaLaunchKernel_per_token"],
                    "cudaGraphLaunch_per_token": o["cudaGraphLaunch_per_token"],
                    "copy_per_token": o["copy_per_token"],
                    "to_copy_per_token": o["to_copy_per_token"],
                    "launch_cpu_ms_per_token": o["launch_cpu_ms_per_token"],
                    "copy_clone_cpu_ms_per_token": o["copy_clone_cpu_ms_per_token"],
                }
            )
    with (TABLE_DIR / "profile_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ratio",
                "mode",
                "module_total_forward_ms",
                "decode_ms_per_token",
                "cudaLaunchKernel_per_token",
                "cudaGraphLaunch_per_token",
                "copy_per_token",
                "to_copy_per_token",
                "launch_cpu_ms_per_token",
                "copy_clone_cpu_ms_per_token",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    elapsed = time.perf_counter() - start
    (STUDY_DIR / "SUMMARY.md").write_text(
        "# Ratio Profile Study\n\n"
        f"- elapsed: `{elapsed/60.0:.1f}` min\n"
        f"- ratios: `{', '.join(args.ratios)}`\n"
        f"- module profiles: `{len(args.ratios) * 3}`\n"
        f"- op profiles: `{len(args.ratios) * 3}`\n"
        f"- graph fragmentation summaries: `{len(args.ratios) * 3}` flash variants\n"
        f"- example bundles: `{len(args.ratios) * 3}` mode-specific json files\n\n"
        "Key outputs:\n\n"
        f"- [module_profiles.json]({TABLE_DIR / 'module_profiles.json'})\n"
        f"- [op_profiles.json]({TABLE_DIR / 'op_profiles.json'})\n"
        f"- [graph_fragmentation_profiles.json]({TABLE_DIR / 'graph_fragmentation_profiles.json'})\n"
        f"- [profile_summary.csv]({TABLE_DIR / 'profile_summary.csv'})\n"
        f"- [decode_examples.md]({EXAMPLE_DIR / 'decode_examples.md'})\n"
    )
    print(f"\nFinished ratio profile study in {elapsed/60.0:.1f} min")
    print(f"Study directory: {STUDY_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
