#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
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
TAB_DIR = STUDY_DIR / "tables"
EXAMPLE_DIR = STUDY_DIR / "examples"

RATIO_PATHS = {
    "0.5": "/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.5/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.5_hf",
    "0.6": "/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.6/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.6_hf",
    "0.7": "/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.7/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.7_hf",
    "0.8": "/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.8/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.8_hf",
}

MODES = {
    "static": {"mode": "svd", "baseline_dense": False, "label": "StaticCache"},
    "densekv": {"mode": "svd", "baseline_dense": True, "label": "DenseKVCacheBaseline"},
    "flashsvd": {"mode": "flashsvd", "baseline_dense": False, "label": "FlashSVD-v1.5"},
}

LONG_CONTEXT_CONFIGS = {
    "ctx4096": {"prompt_len": 4096, "new_tokens": 128},
    "ctx8192": {"prompt_len": 8192, "new_tokens": 128},
}

DEFAULT_DECODE_SWEEP_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
EXAMPLE_PROMPT = "Reducing kernel launch overhead helps autoregressive decoding because"


def ensure_dirs() -> None:
    for path in (RAW_DIR, TAB_DIR, EXAMPLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def stable_seed(*parts: object, offset: int = 0) -> int:
    payload = "::".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    return int(offset) + (int(digest[:8], 16) % 100000)


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_runtime(mode_key: str) -> tuple[bool, bool, bool]:
    cfg = MODES[mode_key]
    os.environ["FLASH_SVD_DENSE_DECODE_BACKEND"] = "packed"
    os.environ["FLASH_SVD_DENSE_DECODE_GRAPH"] = "1"
    return _configure_mode(
        cfg["mode"],
        ffn_backend="flashsvd_mlp_dual_split_prod",
        enable_mlp_graph=True,
        mlp_graph_scope="layer_tail",
        graph_alias_output=False,
        enable_flash_dense_attn=True,
        enable_cutlass_rope_attn=False,
        enable_baseline_dense_kvcache=bool(cfg["baseline_dense"]),
    )


def load_model(mode_key: str, source: str, *, dtype_name: str, device: str):
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


def run_once(
    *,
    model,
    device: str,
    lowrank_cache: bool,
    flashsvd_dense_cache: bool,
    baseline_dense_kvcache: bool,
    prompt_len: int,
    new_tokens: int,
    warmup: int,
    seed: int,
) -> dict[str, float | int | bool]:
    set_seed(seed)
    result = decode_kvcache_eval(
        model,
        prompt_len=int(prompt_len),
        new_tokens=int(new_tokens),
        warmup=int(warmup),
        max_cache_len=None,
        batch_size=1,
        device=device,
        lowrank_cache=bool(lowrank_cache),
        flashsvd_dense_cache=bool(flashsvd_dense_cache),
        baseline_dense_kvcache=bool(baseline_dense_kvcache),
        profile_decode=False,
    )
    result["total_time_s"] = float(result["prefill_time_s"]) + float(result["decode_time_s"])
    return result


def generate_example(
    *,
    model,
    tokenizer,
    device: str,
    prompt_text: str,
    decode_steps: int,
    flash_cache: bool,
) -> dict[str, object]:
    dev = torch.device(device)
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device=dev, dtype=torch.long)
    greedy_tokens = _run_decode_greedy(
        model,
        input_ids,
        prompt_len=int(input_ids.shape[1]),
        decode_steps=int(decode_steps),
        flash_cache=bool(flash_cache),
    )
    text = tokenizer.decode(greedy_tokens[0].tolist(), skip_special_tokens=False)
    return {
        "prompt": prompt_text,
        "prompt_token_len": int(input_ids.shape[1]),
        "new_tokens": int(decode_steps),
        "generated_text": text,
        "generated_characters": len(text),
    }


def metric_stats(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "runs": list(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "stdev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_records(records: list[dict]) -> dict[str, dict]:
    keys = [
        "prefill_time_s",
        "decode_time_s",
        "decode_ms_per_token",
        "prefill_tok_s",
        "decode_tok_s",
        "total_time_s",
    ]
    return {key: metric_stats([float(r[key]) for r in records]) for key in keys}


def write_experiment_settings(args: argparse.Namespace, *, stage_contexts: list[int], decode_lengths: list[int]) -> None:
    text = f"""# Long-Context Decode Sweep

Date: 2026-03-31

## Goal

This study extends the earlier ratio/runtime sweep in two directions:

1. Long-context stage study with larger prompt lengths while still recording both prefill and decode.
2. Decode-length sweep with fixed prompt length and `new_tokens` ranging from `64` to `16384`.

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

```bash
FLASH_SVD_DENSE_DECODE_BACKEND=packed
FLASH_SVD_DENSE_DECODE_GRAPH=1
--experimental_flash_dense_attn
--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod
--mlp_cuda_graph
--mlp_cuda_graph_scope layer_tail
```

## Stage Study

- Ratios: {", ".join(args.ratios)}
- Baselines / target: `StaticCache`, `DenseKVCacheBaseline`, `FlashSVD-v1.5`
- Configurations:
{os.linesep.join(f"  - `prompt_len={ctx}, new_tokens={args.stage_new_tokens}`" for ctx in stage_contexts)}
- Repeats: `{args.stage_repeats}` timed runs per ratio/mode/config after one burn-in run

## Decode-Length Sweep

- Fixed prompt length: `prompt_len={args.decode_prompt_len}`
- `new_tokens`: {", ".join(str(x) for x in decode_lengths)}
- Repeats: `{args.decode_repeats}` timed runs per ratio/mode/length after one burn-in run

## Decode Examples

- Mode: `FlashSVD-v1.5`
- Prompt: `{EXAMPLE_PROMPT}`
- Example lengths: {", ".join(str(x) for x in decode_lengths)}
"""
    (STUDY_DIR / "EXPERIMENT_SETTINGS.md").write_text(text)


def write_example_markdown(example_rows: list[dict]) -> None:
    lines = ["# Decode Examples", ""]
    for ratio in sorted({row["ratio"] for row in example_rows}, key=float):
        lines.append(f"## Ratio {ratio}")
        lines.append("")
        for row in [r for r in example_rows if r["ratio"] == ratio]:
            lines.append(f"### new_tokens={row['new_tokens']}")
            lines.append("")
            lines.append(f"Prompt: `{row['prompt']}`")
            lines.append("")
            preview = row["generated_text"][:1200]
            if len(row["generated_text"]) > 1200:
                preview += "\n...[truncated in markdown preview]..."
            lines.append("```text")
            lines.append(preview)
            lines.append("```")
            lines.append("")
    (EXAMPLE_DIR / "decode_examples.md").write_text("\n".join(lines))


def write_summary(stage_summary: dict, decode_summary: dict, *, stage_contexts: list[int], decode_lengths: list[int]) -> None:
    lines = [
        "# Long-Context Decode Sweep Summary",
        "",
        "## Long-Context Stage Study",
        "",
        "| Ratio | Config | Baseline | Baseline decode (mean ms/token) | FlashSVD decode (mean ms/token) | Decode speedup | Baseline prefill (mean s) | FlashSVD prefill (mean s) | Baseline total (mean s) | FlashSVD total (mean s) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in sorted(stage_summary, key=float):
        for pair_key in ("static_vs_flashsvd", "densekv_vs_flashsvd"):
            for cfg_key in [f"ctx{ctx}" for ctx in stage_contexts]:
                item = stage_summary[ratio][f"{pair_key}:{cfg_key}"]
                lines.append(
                    f"| `{ratio}` | `{cfg_key}` | `{item['baseline_label']}` | "
                    f"{item['baseline.decode_ms_per_token']['mean']:.3f} | "
                    f"{item['flashsvd.decode_ms_per_token']['mean']:.3f} | "
                    f"{item['decode_speedup_mean']:.2f}x | "
                    f"{item['baseline.prefill_time_s']['mean']:.3f} | "
                    f"{item['flashsvd.prefill_time_s']['mean']:.3f} | "
                    f"{item['baseline.total_time_s']['mean']:.3f} | "
                    f"{item['flashsvd.total_time_s']['mean']:.3f} |"
                )
    lines.extend(
        [
            "",
            "## Decode-Length Sweep",
            "",
            "| Ratio | new_tokens | Baseline | Baseline decode (mean ms/token) | FlashSVD decode (mean ms/token) | Decode speedup |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for ratio in sorted(decode_summary, key=float):
        for pair_key in ("static_vs_flashsvd", "densekv_vs_flashsvd"):
            for token_key in [str(x) for x in decode_lengths]:
                item = decode_summary[ratio][f"{pair_key}:tok{token_key}"]
                lines.append(
                    f"| `{ratio}` | `{token_key}` | `{item['baseline_label']}` | "
                    f"{item['baseline.decode_ms_per_token']['mean']:.3f} | "
                    f"{item['flashsvd.decode_ms_per_token']['mean']:.3f} | "
                    f"{item['decode_speedup_mean']:.2f}x |"
                )
    (STUDY_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser("Run long-context stage study and decode-length sweep across ratios")
    ap.add_argument("--ratios", nargs="+", default=["0.8", "0.7", "0.6", "0.5"])
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--stage_new_tokens", type=int, default=128)
    ap.add_argument("--stage_repeats", type=int, default=5)
    ap.add_argument("--stage_contexts", nargs="+", type=int, default=[4096, 8192])
    ap.add_argument("--decode_prompt_len", type=int, default=512)
    ap.add_argument("--decode_repeats", type=int, default=2)
    ap.add_argument("--decode_lengths", nargs="+", type=int, default=DEFAULT_DECODE_SWEEP_LENGTHS)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    ensure_dirs()
    stage_contexts = [int(x) for x in args.stage_contexts]
    stage_configs = {f"ctx{ctx}": {"prompt_len": int(ctx), "new_tokens": int(args.stage_new_tokens)} for ctx in stage_contexts}
    decode_lengths = [int(x) for x in args.decode_lengths]
    write_experiment_settings(args, stage_contexts=stage_contexts, decode_lengths=decode_lengths)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["FLASH_SVD_DENSE_DECODE_BACKEND"] = "packed"
    os.environ["FLASH_SVD_DENSE_DECODE_GRAPH"] = "1"

    device = "cuda"
    raw_rows = []
    stage_summary: dict[str, dict[str, dict]] = {}
    decode_summary: dict[str, dict[str, dict]] = {}
    example_rows: list[dict] = []

    start = time.perf_counter()
    for ratio in args.ratios:
        source = RATIO_PATHS[ratio]
        stage_summary[ratio] = {}
        decode_summary[ratio] = {}
        print(f"\n==== Ratio {ratio} :: {source} ====")
        for mode_key in ("static", "densekv", "flashsvd"):
            print(f"-- Loading mode={mode_key}")
            model, tokenizer, lowrank_cache, flashsvd_dense_cache, baseline_dense_kvcache = load_model(
                mode_key,
                source,
                dtype_name=args.dtype,
                device=device,
            )
            mode_stage_records: dict[str, list[dict]] = {}
            mode_decode_records: dict[str, list[dict]] = {}
            try:
                for cfg_key, cfg in stage_configs.items():
                    prompt_len = int(cfg["prompt_len"])
                    new_tokens = int(args.stage_new_tokens)
                    print(f"   stage {cfg_key}: prompt_len={prompt_len} new_tokens={new_tokens}")
                    burn = run_once(
                        model=model,
                        device=device,
                        lowrank_cache=lowrank_cache,
                        flashsvd_dense_cache=flashsvd_dense_cache,
                        baseline_dense_kvcache=baseline_dense_kvcache,
                        prompt_len=prompt_len,
                        new_tokens=new_tokens,
                        warmup=args.warmup,
                        seed=stable_seed(ratio, mode_key, cfg_key, "burn", offset=1000),
                    )
                    (RAW_DIR / f"stage_{ratio}_{mode_key}_{cfg_key}_burnin.json").write_text(json.dumps(burn, indent=2))

                    mode_stage_records[cfg_key] = []
                    for run_idx in range(1, int(args.stage_repeats) + 1):
                        rec = run_once(
                            model=model,
                            device=device,
                            lowrank_cache=lowrank_cache,
                            flashsvd_dense_cache=flashsvd_dense_cache,
                            baseline_dense_kvcache=baseline_dense_kvcache,
                            prompt_len=prompt_len,
                            new_tokens=new_tokens,
                            warmup=args.warmup,
                            seed=stable_seed(ratio, mode_key, cfg_key, run_idx, offset=2000),
                        )
                        rec.update(
                            {
                                "study": "stage",
                                "ratio": ratio,
                                "mode": mode_key,
                                "mode_label": MODES[mode_key]["label"],
                                "config": cfg_key,
                                "run_idx": run_idx,
                            }
                        )
                        raw_rows.append(rec.copy())
                        mode_stage_records[cfg_key].append(rec)
                        (RAW_DIR / f"stage_{ratio}_{mode_key}_{cfg_key}_run{run_idx}.json").write_text(json.dumps(rec, indent=2))

                for new_tokens in decode_lengths:
                    print(f"   decode sweep new_tokens={new_tokens}")
                    burn = run_once(
                        model=model,
                        device=device,
                        lowrank_cache=lowrank_cache,
                        flashsvd_dense_cache=flashsvd_dense_cache,
                        baseline_dense_kvcache=baseline_dense_kvcache,
                        prompt_len=int(args.decode_prompt_len),
                        new_tokens=int(new_tokens),
                        warmup=args.warmup,
                        seed=stable_seed(ratio, mode_key, "tok", new_tokens, "burn", offset=3000),
                    )
                    (RAW_DIR / f"decode_{ratio}_{mode_key}_tok{new_tokens}_burnin.json").write_text(json.dumps(burn, indent=2))

                    key = str(new_tokens)
                    mode_decode_records[key] = []
                    for run_idx in range(1, int(args.decode_repeats) + 1):
                        rec = run_once(
                            model=model,
                            device=device,
                            lowrank_cache=lowrank_cache,
                            flashsvd_dense_cache=flashsvd_dense_cache,
                            baseline_dense_kvcache=baseline_dense_kvcache,
                            prompt_len=int(args.decode_prompt_len),
                            new_tokens=int(new_tokens),
                            warmup=args.warmup,
                            seed=stable_seed(ratio, mode_key, "tok", new_tokens, run_idx, offset=4000),
                        )
                        rec.update(
                            {
                                "study": "decode_sweep",
                                "ratio": ratio,
                                "mode": mode_key,
                                "mode_label": MODES[mode_key]["label"],
                                "config": f"tok{new_tokens}",
                                "run_idx": run_idx,
                            }
                        )
                        raw_rows.append(rec.copy())
                        mode_decode_records[key].append(rec)
                        (RAW_DIR / f"decode_{ratio}_{mode_key}_tok{new_tokens}_run{run_idx}.json").write_text(json.dumps(rec, indent=2))

                if mode_key == "flashsvd":
                    for new_tokens in decode_lengths:
                        print(f"   example new_tokens={new_tokens}")
                        example = generate_example(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            prompt_text=EXAMPLE_PROMPT,
                            decode_steps=int(new_tokens),
                            flash_cache=True,
                        )
                        example.update({"ratio": ratio, "mode": mode_key, "mode_label": MODES[mode_key]["label"]})
                        example_rows.append(example)
                        (EXAMPLE_DIR / f"decode_example_{ratio}_tok{new_tokens}.json").write_text(json.dumps(example, indent=2))
            finally:
                unload_model(model, tokenizer)

            # per-mode summaries
            for cfg_key, records in mode_stage_records.items():
                key = f"{mode_key}:{cfg_key}"
                stage_summary[ratio][key] = summarize_records(records)
            for new_tokens, records in mode_decode_records.items():
                key = f"{mode_key}:tok{new_tokens}"
                decode_summary[ratio][key] = summarize_records(records)

        # pairwise summaries
        for cfg_key in stage_configs:
            for baseline_mode, pair_key in (("static", "static_vs_flashsvd"), ("densekv", "densekv_vs_flashsvd")):
                b = stage_summary[ratio][f"{baseline_mode}:{cfg_key}"]
                f = stage_summary[ratio][f"flashsvd:{cfg_key}"]
                stage_summary[ratio][f"{pair_key}:{cfg_key}"] = {
                    "baseline_label": MODES[baseline_mode]["label"],
                    "baseline.prefill_time_s": b["prefill_time_s"],
                    "baseline.decode_ms_per_token": b["decode_ms_per_token"],
                    "baseline.total_time_s": b["total_time_s"],
                    "flashsvd.prefill_time_s": f["prefill_time_s"],
                    "flashsvd.decode_ms_per_token": f["decode_ms_per_token"],
                    "flashsvd.total_time_s": f["total_time_s"],
                    "decode_speedup_mean": b["decode_ms_per_token"]["mean"] / f["decode_ms_per_token"]["mean"],
                    "decode_speedup_median": b["decode_ms_per_token"]["median"] / f["decode_ms_per_token"]["median"],
                    "total_speedup_mean": b["total_time_s"]["mean"] / f["total_time_s"]["mean"],
                    "total_speedup_median": b["total_time_s"]["median"] / f["total_time_s"]["median"],
                }
        for new_tokens in decode_lengths:
            token_key = str(new_tokens)
            for baseline_mode, pair_key in (("static", "static_vs_flashsvd"), ("densekv", "densekv_vs_flashsvd")):
                b = decode_summary[ratio][f"{baseline_mode}:tok{token_key}"]
                f = decode_summary[ratio][f"flashsvd:tok{token_key}"]
                decode_summary[ratio][f"{pair_key}:tok{token_key}"] = {
                    "baseline_label": MODES[baseline_mode]["label"],
                    "baseline.decode_ms_per_token": b["decode_ms_per_token"],
                    "baseline.total_time_s": b["total_time_s"],
                    "flashsvd.decode_ms_per_token": f["decode_ms_per_token"],
                    "flashsvd.total_time_s": f["total_time_s"],
                    "decode_speedup_mean": b["decode_ms_per_token"]["mean"] / f["decode_ms_per_token"]["mean"],
                    "decode_speedup_median": b["decode_ms_per_token"]["median"] / f["decode_ms_per_token"]["median"],
                }

    # Write aggregated outputs.
    with (TAB_DIR / "all_runs.csv").open("w", newline="") as f:
        fieldnames = [
            "study",
            "ratio",
            "mode",
            "mode_label",
            "config",
            "run_idx",
            "prompt_len",
            "new_tokens",
            "batch_size",
            "prefill_time_s",
            "decode_time_s",
            "prefill_tok_s",
            "decode_tok_s",
            "decode_ms_per_token",
            "total_time_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    (TAB_DIR / "stage_summary.json").write_text(json.dumps(stage_summary, indent=2))
    (TAB_DIR / "decode_sweep_summary.json").write_text(json.dumps(decode_summary, indent=2))
    (EXAMPLE_DIR / "decode_examples.json").write_text(json.dumps(example_rows, indent=2))
    write_example_markdown(example_rows)
    write_summary(stage_summary, decode_summary, stage_contexts=stage_contexts, decode_lengths=decode_lengths)

    elapsed = time.perf_counter() - start
    (STUDY_DIR / "SUMMARY.md").write_text((STUDY_DIR / "SUMMARY.md").read_text() + f"\nElapsed wall time: `{elapsed/3600.0:.2f} h`\n")
    print(f"\nFinished long-context/decode sweep in {elapsed/3600.0:.2f} h")
    print(f"Study directory: {STUDY_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
