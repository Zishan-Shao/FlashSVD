#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
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
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.decode.bench_flashsvd_vs_svd_decode import (  # noqa: E402
    _cast_model_for_eval,
    _configure_mode,
    _load_model_and_tokenizer,
)
from utils.evaluator import decode_kvcache_eval  # noqa: E402


RAW_DIR = STUDY_DIR / "raw"
TAB_DIR = STUDY_DIR / "tables"

RATIO_PATHS = {
    "0.5": "/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.5/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.5_hf",
    "0.6": "/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.6/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.6_hf",
    "0.7": "/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.7/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.7_hf",
    "0.8": "/home/zs89/FlashSVD/models/lowrankarena/svdllm_whitening_only_0.8/llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_0.8_hf",
}

CONFIGS = {
    "short": {"prompt_len": 512, "new_tokens": 32},
    "long": {"prompt_len": 2048, "new_tokens": 128},
}

MODES = {
    "static": {"mode": "svd", "baseline_dense": False, "label": "StaticCache"},
    "densekv": {"mode": "svd", "baseline_dense": True, "label": "DenseKVCacheBaseline"},
    "flashsvd": {"mode": "flashsvd", "baseline_dense": False, "label": "FlashSVD-v1.5"},
}

METRIC_KEYS = [
    "prefill_time_s",
    "decode_time_s",
    "decode_ms_per_token",
    "prefill_tok_s",
    "decode_tok_s",
    "total_time_s",
]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def configure_runtime(mode_key: str) -> tuple[bool, bool, bool]:
    cfg = MODES[mode_key]
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


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_model(mode_key: str, source: str, *, dtype_name: str = "bf16", device: str = "cuda"):
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


def metric_stats(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "runs": list(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "stdev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_mode(records: list[dict]) -> dict[str, dict]:
    out = {}
    for key in METRIC_KEYS:
        vals = [float(r[key]) for r in records]
        out[key] = metric_stats(vals)
    return out


def format_seconds(x: float) -> str:
    return f"{x:.3f}"


def format_ms(x: float) -> str:
    return f"{x:.3f}"


def write_experiment_settings() -> None:
    text = f"""# Ratio Sweep Runtime Study

Date: 2026-03-31

## Goal

This study extends the existing `0.5` FlashSVD-v1.5 runtime recipe to SVD-LLM v1 checkpoints at ratios:

- `0.5`
- `0.6`
- `0.7`
- `0.8`

The purpose is to measure how the active `FlashSVD-v1.5` serving stack scales with compression ratio, using the same end-to-end benchmark recipe as the main `0.5` study.

## Machine / Software

- Host cwd: `/home/zs89/FlashSVD`
- GPU target: `CUDA_VISIBLE_DEVICES=5`
- Python env: `/home/zs89/miniconda3/envs/flashsvd15/bin/python`
- Python: `3.13.2`
- PyTorch: `2.7.1+cu128`
- Triton: `3.3.1`
- Transformers: `4.53.0`
- FlashAttention: `2.8.3`

## Runtime Recipe

FlashSVD runtime knobs:

```bash
FLASH_SVD_DENSE_DECODE_BACKEND=packed
FLASH_SVD_DENSE_DECODE_GRAPH=1
```

Benchmark flags:

```bash
--experimental_flash_dense_attn
--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod
--mlp_cuda_graph
--mlp_cuda_graph_scope layer_tail
```

## Benchmarked Checkpoints

{os.linesep.join(f"- `{ratio}`: `{path}`" for ratio, path in RATIO_PATHS.items())}

## Benchmark Matrix

For each ratio:

- Baselines:
  - `StaticCache`
  - `DenseKVCacheBaseline`
- Target:
  - `FlashSVD-v1.5`
- Configurations:
  - `prompt_len=512, new_tokens=32`
  - `prompt_len=2048, new_tokens=128`
- Repeats:
  - `n=5` timed runs per mode/configuration
- Warmup:
  - `warmup=3`
  - plus one untimed compile/burn-in run per mode/configuration because this sweep harness reuses a loaded model instance

## Notes

- All checkpoints are loaded from their exported HuggingFace local directories for consistency across ratios.
- This study focuses on repeated end-to-end latency only. It does not re-run the full profiler stack per ratio.
"""
    (STUDY_DIR / "EXPERIMENT_SETTINGS.md").write_text(text)


def write_summary(summary: dict, best_decode: dict, best_total: dict) -> None:
    lines = [
        "# Ratio Sweep Summary",
        "",
        "This study uses the same FlashSVD-v1.5 runtime recipe as the main `0.5` result, but sweeps SVD-LLM v1 ratios `0.5 / 0.6 / 0.7 / 0.8` under the exported HuggingFace checkpoint layout.",
        "",
        "## Headline Table",
        "",
        "| Ratio | Config | Baseline | Baseline decode (median ms/token) | FlashSVD decode (median ms/token) | Decode speedup | Baseline total (median s) | FlashSVD total (median s) | Total speedup |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in ("0.5", "0.6", "0.7", "0.8"):
        for pair_key in ("static_vs_flashsvd", "densekv_vs_flashsvd"):
            for cfg in ("short", "long"):
                item = summary[ratio][f"{pair_key}:{cfg}"]
                lines.append(
                    f"| `{ratio}` | `{cfg}` | `{item['baseline_label']}` | "
                    f"{format_ms(item['svd.decode_ms_per_token']['median'])} | "
                    f"{format_ms(item['flashsvd.decode_ms_per_token']['median'])} | "
                    f"{item['decode_speedup_median']:.2f}x | "
                    f"{format_seconds(item['svd.total_time_s']['median'])} | "
                    f"{format_seconds(item['flashsvd.total_time_s']['median'])} | "
                    f"{item['total_speedup_median']:.2f}x |"
                )
    lines.extend(
        [
            "",
            "## Best Ratios",
            "",
            f"- Best decode speedup vs `StaticCache`: ratio `{best_decode['static']['ratio']}`, config `{best_decode['static']['config']}`, `{best_decode['static']['speedup']:.2f}x`.",
            f"- Best decode speedup vs `DenseKVCacheBaseline`: ratio `{best_decode['densekv']['ratio']}`, config `{best_decode['densekv']['config']}`, `{best_decode['densekv']['speedup']:.2f}x`.",
            f"- Best total speedup vs `StaticCache`: ratio `{best_total['static']['ratio']}`, config `{best_total['static']['config']}`, `{best_total['static']['speedup']:.2f}x`.",
            f"- Best total speedup vs `DenseKVCacheBaseline`: ratio `{best_total['densekv']['ratio']}`, config `{best_total['densekv']['config']}`, `{best_total['densekv']['speedup']:.2f}x`.",
            "",
            "## Interpretation",
            "",
            "- This sweep is directly comparable across ratios because the runtime recipe, GPU, dtype, and benchmark shapes are fixed.",
            "- The `DenseKVCacheBaseline` remains the cleaner aligned performance baseline.",
            "- `StaticCache` remains the practical baseline.",
            "",
            "Structured outputs:",
            "",
            f"- [{(TAB_DIR / 'ratio_sweep_repeated_runs.csv').name}]({TAB_DIR / 'ratio_sweep_repeated_runs.csv'})",
            f"- [{(TAB_DIR / 'ratio_sweep_summary.json').name}]({TAB_DIR / 'ratio_sweep_summary.json'})",
        ]
    )
    (STUDY_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ensure_dirs()
    write_experiment_settings()
    os.environ["FLASH_SVD_DENSE_DECODE_BACKEND"] = "packed"
    os.environ["FLASH_SVD_DENSE_DECODE_GRAPH"] = "1"

    device = "cuda"
    dtype_name = "bf16"
    warmup = 3
    repeats = 5
    burnin = 1

    run_rows: list[dict] = []
    summary: dict[str, dict] = {}

    start_all = time.perf_counter()
    for ratio, checkpoint in RATIO_PATHS.items():
        summary[ratio] = {}
        print(f"\n==== Ratio {ratio} :: {checkpoint} ====")
        for mode_key in ("static", "densekv", "flashsvd"):
            print(f"\n-- Loading mode={mode_key} --")
            model, tokenizer, lowrank_cache, flashsvd_dense_cache, baseline_dense_kvcache = load_model(
                mode_key, checkpoint, dtype_name=dtype_name, device=device
            )
            try:
                for cfg_key, cfg in CONFIGS.items():
                    print(f"  > Burn-in mode={mode_key} cfg={cfg_key}")
                    _ = run_once(
                        model=model,
                        device=device,
                        lowrank_cache=lowrank_cache,
                        flashsvd_dense_cache=flashsvd_dense_cache,
                        baseline_dense_kvcache=baseline_dense_kvcache,
                        prompt_len=cfg["prompt_len"],
                        new_tokens=cfg["new_tokens"],
                        warmup=warmup,
                        seed=1000 + burnin,
                    )
                    for run_idx in range(1, repeats + 1):
                        print(f"  > Timed run ratio={ratio} mode={mode_key} cfg={cfg_key} run={run_idx}/{repeats}")
                        result = run_once(
                            model=model,
                            device=device,
                            lowrank_cache=lowrank_cache,
                            flashsvd_dense_cache=flashsvd_dense_cache,
                            baseline_dense_kvcache=baseline_dense_kvcache,
                            prompt_len=cfg["prompt_len"],
                            new_tokens=cfg["new_tokens"],
                            warmup=warmup,
                            seed=1000 + run_idx,
                        )
                        row = {
                            "ratio": ratio,
                            "mode": mode_key,
                            "baseline_label": MODES[mode_key]["label"],
                            "config": cfg_key,
                            "prompt_len": cfg["prompt_len"],
                            "new_tokens": cfg["new_tokens"],
                            "run": run_idx,
                            **result,
                        }
                        run_rows.append(row)
                        log_path = RAW_DIR / f"ratio_{ratio}_{mode_key}_{cfg_key}_run{run_idx}.json"
                        log_path.write_text(json.dumps(row, indent=2))
            finally:
                unload_model(model, tokenizer)

        for pair_key, baseline_mode in (("static_vs_flashsvd", "static"), ("densekv_vs_flashsvd", "densekv")):
            for cfg_key in CONFIGS:
                base_rows = [r for r in run_rows if r["ratio"] == ratio and r["mode"] == baseline_mode and r["config"] == cfg_key]
                flash_rows = [r for r in run_rows if r["ratio"] == ratio and r["mode"] == "flashsvd" and r["config"] == cfg_key]
                base_summary = summarize_mode(base_rows)
                flash_summary = summarize_mode(flash_rows)
                entry = {
                    "ratio": ratio,
                    "config": cfg_key,
                    "baseline_label": MODES[baseline_mode]["label"],
                    "svd.prefill_time_s": base_summary["prefill_time_s"],
                    "svd.decode_time_s": base_summary["decode_time_s"],
                    "svd.decode_ms_per_token": base_summary["decode_ms_per_token"],
                    "svd.prefill_tok_s": base_summary["prefill_tok_s"],
                    "svd.decode_tok_s": base_summary["decode_tok_s"],
                    "svd.total_time_s": base_summary["total_time_s"],
                    "flashsvd.prefill_time_s": flash_summary["prefill_time_s"],
                    "flashsvd.decode_time_s": flash_summary["decode_time_s"],
                    "flashsvd.decode_ms_per_token": flash_summary["decode_ms_per_token"],
                    "flashsvd.prefill_tok_s": flash_summary["prefill_tok_s"],
                    "flashsvd.decode_tok_s": flash_summary["decode_tok_s"],
                    "flashsvd.total_time_s": flash_summary["total_time_s"],
                }
                entry["decode_speedup_median"] = (
                    entry["svd.decode_ms_per_token"]["median"] / entry["flashsvd.decode_ms_per_token"]["median"]
                )
                entry["total_speedup_median"] = (
                    entry["svd.total_time_s"]["median"] / entry["flashsvd.total_time_s"]["median"]
                )
                summary[ratio][f"{pair_key}:{cfg_key}"] = entry

    elapsed = time.perf_counter() - start_all

    csv_path = TAB_DIR / "ratio_sweep_repeated_runs.csv"
    fieldnames = [
        "ratio",
        "mode",
        "baseline_label",
        "config",
        "prompt_len",
        "new_tokens",
        "run",
        "prefill_time_s",
        "decode_time_s",
        "prefill_tok_s",
        "decode_tok_s",
        "decode_ms_per_token",
        "total_time_s",
        "lowrank_cache",
        "flashsvd_dense_cache",
        "baseline_dense_kvcache",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    summary_payload = {
        "elapsed_s": elapsed,
        "ratios": summary,
    }
    (TAB_DIR / "ratio_sweep_summary.json").write_text(json.dumps(summary_payload, indent=2))

    best_decode = {
        "static": {"speedup": -1.0},
        "densekv": {"speedup": -1.0},
    }
    best_total = {
        "static": {"speedup": -1.0},
        "densekv": {"speedup": -1.0},
    }
    for ratio, entries in summary.items():
        for key, item in entries.items():
            base = "static" if item["baseline_label"] == "StaticCache" else "densekv"
            if item["decode_speedup_median"] > best_decode[base]["speedup"]:
                best_decode[base] = {
                    "ratio": ratio,
                    "config": item["config"],
                    "speedup": item["decode_speedup_median"],
                }
            if item["total_speedup_median"] > best_total[base]["speedup"]:
                best_total[base] = {
                    "ratio": ratio,
                    "config": item["config"],
                    "speedup": item["total_speedup_median"],
                }

    write_summary(summary, best_decode, best_total)
    print(f"\nFinished ratio sweep in {elapsed/60.0:.1f} min")
    print(f"Study directory: {STUDY_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
