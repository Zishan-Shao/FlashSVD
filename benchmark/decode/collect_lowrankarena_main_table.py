#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time
import traceback

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.decode.bench_flashsvd_vs_svd_decode import _bench_one_mode


LOWRANKARENA_REPO_ALIAS = "LowRankArena"
LOWRANKARENA_REPO_ENV = "FLASHSVD_LOWRANKARENA_REPO_ID"
COMMON_RATIOS = ("0.4", "0.5", "0.6")


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    family_name: str

    def hf_subdir(self, ratio_text: str) -> str:
        raise NotImplementedError

    def local_root(self, ratio_text: str) -> Path:
        raise NotImplementedError

    def checkpoint_path(self, ratio_text: str) -> Path:
        return self.local_root(ratio_text) / self.hf_subdir(ratio_text)


class SVDLLMV1Spec(FamilySpec):
    def hf_subdir(self, ratio_text: str) -> str:
        return f"llama_7b/SVDLLM/jeffwan_llama_7b_hf_whitening_only_{ratio_text}_hf"

    def local_root(self, ratio_text: str) -> Path:
        return WORKSPACE / "models" / "lowrankarena" / f"svdllm_whitening_only_{ratio_text}"


class SVDLLMV2Spec(FamilySpec):
    def hf_subdir(self, ratio_text: str) -> str:
        # Public LowRankArena SVD-LLM v2 checkpoints are named by removed ratio
        # (e.g. keep=0.8 -> r20, keep=0.7 -> r30).
        pct = int(round((1.0 - float(ratio_text)) * 100))
        return f"llama_7b/SVDLLMv2/jeffwan_llama_7b_hf_svdllmv2_r{pct}_hf"

    def local_root(self, ratio_text: str) -> Path:
        return WORKSPACE / "models" / "lowrankarena" / f"svdllmv2_keep{ratio_text}"


class BasisSharingSpec(FamilySpec):
    def hf_subdir(self, ratio_text: str) -> str:
        # Public LowRankArena Basis Sharing checkpoints are also named by removed ratio
        # (e.g. keep=0.8 -> share_llama-7b_20, keep=0.7 -> share_llama-7b_30).
        pct = int(round((1.0 - float(ratio_text)) * 100))
        return f"llama_7b/Basis_Sharing/share_llama-7b_{pct}"

    def local_root(self, ratio_text: str) -> Path:
        pct = int(round((1.0 - float(ratio_text)) * 100))
        return WORKSPACE / "models" / "lowrankarena" / f"basis_sharing_{pct}"


FAMILY_SPECS: dict[str, FamilySpec] = {
    "svdllm_v1": SVDLLMV1Spec("svdllm_v1", "SVD-LLM v1"),
    "svdllm_v2": SVDLLMV2Spec("svdllm_v2", "SVD-LLM v2"),
    "basis_sharing": BasisSharingSpec("basis_sharing", "Basis Sharing"),
}


def _lowrankarena_repo_id() -> str:
    repo_id = os.environ.get(LOWRANKARENA_REPO_ENV, "").strip().strip("/")
    if repo_id:
        return repo_id
    raise RuntimeError(
        f"Set {LOWRANKARENA_REPO_ENV}=<namespace>/<repo> before downloading "
        f"{LOWRANKARENA_REPO_ALIAS} checkpoints."
    )


def _download_checkpoint(spec: FamilySpec, ratio_text: str) -> Path:
    ckpt_path = spec.checkpoint_path(ratio_text)
    cfg_path = ckpt_path / "config.json"
    if cfg_path.is_file():
        return ckpt_path

    local_root = spec.local_root(ratio_text)
    local_root.mkdir(parents=True, exist_ok=True)
    subdir = spec.hf_subdir(ratio_text)
    snapshot_download(
        repo_id=_lowrankarena_repo_id(),
        repo_type="model",
        allow_patterns=[f"{subdir}/*"],
        local_dir=str(local_root),
    )
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Expected downloaded checkpoint at: {ckpt_path}")
    return ckpt_path


def _run_pair(
    *,
    checkpoint: str,
    dtype: str,
    device: str,
    prompt_len: int,
    new_tokens: int,
    warmup: int,
    batch_size: int,
    ffn_backend: str,
    mlp_graph_scope: str,
):
    baseline = _bench_one_mode(
        mode="svd",
        source=checkpoint,
        hf_token=None,
        dtype_name=dtype,
        device=device,
        prompt_len=prompt_len,
        new_tokens=new_tokens,
        warmup=warmup,
        batch_size=batch_size,
        max_cache_len=0,
        ffn_backend=ffn_backend,
        enable_mlp_graph=True,
        mlp_graph_scope=mlp_graph_scope,
        graph_alias_output=False,
        enable_flash_dense_attn=False,
        enable_cutlass_rope_attn=False,
        enable_baseline_dense_kvcache=True,
    )
    flashsvd = _bench_one_mode(
        mode="flashsvd",
        source=checkpoint,
        hf_token=None,
        dtype_name=dtype,
        device=device,
        prompt_len=prompt_len,
        new_tokens=new_tokens,
        warmup=warmup,
        batch_size=batch_size,
        max_cache_len=0,
        ffn_backend=ffn_backend,
        enable_mlp_graph=True,
        mlp_graph_scope=mlp_graph_scope,
        graph_alias_output=False,
        enable_flash_dense_attn=True,
        enable_cutlass_rope_attn=False,
        enable_baseline_dense_kvcache=False,
    )
    return baseline, flashsvd


def _format_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _aggregate_repeat_rows(repeats: list[dict[str, object]]) -> dict[str, object]:
    ok_rows = [row for row in repeats if row.get("status") == "ok"]
    if not ok_rows:
        first = repeats[0] if repeats else {}
        return {
            "status": "error",
            "ok_repeats": 0,
            "requested_repeats": len(repeats),
            "error": first.get("error", "all repeats failed"),
            "traceback": first.get("traceback"),
        }

    metric_keys = [
        "baseline_prefill_time_s",
        "baseline_decode_time_s",
        "flash_prefill_time_s",
        "flash_decode_time_s",
        "baseline_prefill_tok_s",
        "flash_prefill_tok_s",
        "baseline_decode_ms",
        "flash_decode_ms",
        "baseline_decode_tok_s",
        "flash_decode_tok_s",
        "prefill_speedup_x",
        "decode_speedup_x",
        "baseline_e2e_time_s",
        "flash_e2e_time_s",
        "e2e_speedup_x",
    ]
    summary: dict[str, object] = {
        "status": "ok" if len(ok_rows) == len(repeats) else "partial",
        "ok_repeats": len(ok_rows),
        "requested_repeats": len(repeats),
    }
    for key in metric_keys:
        values = [float(row[key]) for row in ok_rows]
        summary[key] = float(statistics.fmean(values))
        summary[f"{key}_std"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        summary[f"{key}_min"] = float(min(values))
        summary[f"{key}_max"] = float(max(values))
    return summary


def _build_markdown(
    *,
    results: list[dict[str, object]],
    config: dict[str, object],
    md_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("# LowRankArena Main Table Results")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Benchmark Config")
    lines.append("")
    lines.append(f"- GPU: `{config['gpu']}`")
    lines.append(f"- Device: `{config['device']}`")
    lines.append(f"- Dtype: `{config['dtype']}`")
    lines.append(f"- Prompt length: `{config['prompt_len']}`")
    lines.append(f"- New tokens: `{config['new_tokens']}`")
    lines.append(f"- Warmup: `{config['warmup']}`")
    lines.append(f"- Batch size: `{config['batch_size']}`")
    lines.append(f"- Repeats: `{config['repeats']}`")
    lines.append(f"- FlashSVD MLP backend: `{config['ffn_backend']}`")
    lines.append(f"- MLP CUDA graph scope: `{config['mlp_graph_scope']}`")
    lines.append("- Compare mode: baseline dense-KV vs FlashSVD production packed path")
    lines.append("")
    lines.append("## Main Table")
    lines.append("")
    lines.append("| Family | Ratio | n | Baseline Prefill (s) | FlashSVD Prefill (s) | Prefill Speedup | Baseline Decode (ms/token) | FlashSVD Decode (ms/token) | Decode Speedup | Baseline E2E (s) | FlashSVD E2E (s) | E2E Speedup |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results:
        if row["status"] == "error":
            lines.append(
                f"| {row['family_name']} | {row['ratio']} | 0/{row.get('requested_repeats', 0)} | fail | fail | fail | fail | fail | fail | fail | fail | fail |"
            )
            continue
        lines.append(
            f"| {row['family_name']} | {row['ratio']} | {row['ok_repeats']}/{row['requested_repeats']} | "
            f"{_format_float(row['baseline_prefill_time_s'])} | "
            f"{_format_float(row['flash_prefill_time_s'])} | "
            f"{_format_float(row['prefill_speedup_x'], 2)}x ± {_format_float(row['prefill_speedup_x_std'], 2)} | "
            f"{_format_float(row['baseline_decode_ms'])} | "
            f"{_format_float(row['flash_decode_ms'])} | "
            f"{_format_float(row['decode_speedup_x'], 2)}x ± {_format_float(row['decode_speedup_x_std'], 2)} | "
            f"{_format_float(row['baseline_e2e_time_s'])} | "
            f"{_format_float(row['flash_e2e_time_s'])} | "
            f"{_format_float(row['e2e_speedup_x'], 2)}x ± {_format_float(row['e2e_speedup_x_std'], 2)} |"
        )
    lines.append("")
    lines.append("## Detailed Throughput")
    lines.append("")
    lines.append("| Family | Ratio | n | Baseline Prefill Tok/s | FlashSVD Prefill Tok/s | Baseline Decode Tok/s | FlashSVD Decode Tok/s | Checkpoint |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in results:
        if row["status"] == "error":
            lines.append(
                f"| {row['family_name']} | {row['ratio']} | 0/{row.get('requested_repeats', 0)} | fail | fail | fail | fail | `{row.get('checkpoint', '')}` |"
            )
            continue
        lines.append(
            f"| {row['family_name']} | {row['ratio']} | {row['ok_repeats']}/{row['requested_repeats']} | "
            f"{_format_float(row['baseline_prefill_tok_s'], 0)} | "
            f"{_format_float(row['flash_prefill_tok_s'], 0)} | "
            f"{_format_float(row['baseline_decode_tok_s'], 0)} | "
            f"{_format_float(row['flash_decode_tok_s'], 0)} | "
            f"`{row['checkpoint']}` |"
        )
    lines.append("")
    lines.append("## Per-repeat Results")
    lines.append("")
    lines.append("| Family | Ratio | Repeat | Prefill Speedup | Decode Speedup | E2E Speedup | Baseline Decode (ms/token) | FlashSVD Decode (ms/token) | Status |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in results:
        for repeat in row.get("repeats", []):
            if repeat.get("status") != "ok":
                lines.append(
                    f"| {row['family_name']} | {row['ratio']} | {repeat.get('repeat_index', '?')} | fail | fail | fail | fail | fail | {repeat.get('error', 'error')} |"
                )
                continue
            lines.append(
                f"| {row['family_name']} | {row['ratio']} | {repeat['repeat_index']} | "
                f"{_format_float(repeat['prefill_speedup_x'], 2)}x | "
                f"{_format_float(repeat['decode_speedup_x'], 2)}x | "
                f"{_format_float(repeat['e2e_speedup_x'], 2)}x | "
                f"{_format_float(repeat['baseline_decode_ms'])} | "
                f"{_format_float(repeat['flash_decode_ms'])} | ok |"
            )
    failed = [row for row in results if row["status"] != "ok"]
    if failed:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for row in failed:
            lines.append(f"- `{row['family_name']} {row['ratio']}`: `{row['error']}`")
    lines.append("")
    lines.append(f"_Saved to `{md_path}`._")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser("Collect FlashSVD main-table results on LowRankArena checkpoints")
    ap.add_argument(
        "--families",
        type=str,
        default="svdllm_v1,svdllm_v2,basis_sharing",
        help="Comma-separated family ids.",
    )
    ap.add_argument(
        "--ratios",
        type=str,
        default=",".join(COMMON_RATIOS),
        help="Comma-separated ratio strings shared across the selected families.",
    )
    ap.add_argument("--gpu", type=str, default="4", help="CUDA_VISIBLE_DEVICES value.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--prompt_len", type=int, default=512)
    ap.add_argument("--new_tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--ffn_backend", type=str, default="flashsvd_mlp_dual_split_prod")
    ap.add_argument("--mlp_graph_scope", type=str, default="mlp")
    ap.add_argument(
        "--md_out",
        type=str,
        default=str(WORKSPACE / "results" / "docs_notes_archive" / "lowrankarena_main_table_2026-03-17.md"),
    )
    ap.add_argument(
        "--json_out",
        type=str,
        default=str(WORKSPACE / "results" / "docs_notes_archive" / "lowrankarena_main_table_2026-03-17.json"),
    )
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    ratios = [r.strip() for r in args.ratios.split(",") if r.strip()]

    bad_families = [f for f in families if f not in FAMILY_SPECS]
    if bad_families:
        raise ValueError(f"Unknown families: {bad_families}")

    results: list[dict[str, object]] = []
    for family_id in families:
        spec = FAMILY_SPECS[family_id]
        for ratio_text in ratios:
            row: dict[str, object] = {
                "family_id": spec.family_id,
                "family_name": spec.family_name,
                "ratio": ratio_text,
            }
            try:
                ckpt = _download_checkpoint(spec, ratio_text)
                row["checkpoint"] = str(ckpt)
                repeats: list[dict[str, object]] = []
                for repeat_index in range(1, int(args.repeats) + 1):
                    repeat_row: dict[str, object] = {
                        "repeat_index": repeat_index,
                        "checkpoint": str(ckpt),
                    }
                    try:
                        baseline, flashsvd = _run_pair(
                            checkpoint=str(ckpt),
                            dtype=args.dtype,
                            device=args.device,
                            prompt_len=args.prompt_len,
                            new_tokens=args.new_tokens,
                            warmup=args.warmup,
                            batch_size=args.batch_size,
                            ffn_backend=args.ffn_backend,
                            mlp_graph_scope=args.mlp_graph_scope,
                        )
                        repeat_row.update(
                            {
                                "status": "ok",
                                "baseline_prefill_time_s": float(baseline["prefill_time_s"]),
                                "baseline_decode_time_s": float(baseline["decode_time_s"]),
                                "flash_prefill_time_s": float(flashsvd["prefill_time_s"]),
                                "flash_decode_time_s": float(flashsvd["decode_time_s"]),
                                "baseline_prefill_tok_s": float(baseline["prefill_tok_s"]),
                                "flash_prefill_tok_s": float(flashsvd["prefill_tok_s"]),
                                "baseline_decode_ms": float(baseline["decode_ms_per_token"]),
                                "flash_decode_ms": float(flashsvd["decode_ms_per_token"]),
                                "baseline_decode_tok_s": float(baseline["decode_tok_s"]),
                                "flash_decode_tok_s": float(flashsvd["decode_tok_s"]),
                            }
                        )
                        repeat_row["prefill_speedup_x"] = float(repeat_row["flash_prefill_tok_s"]) / max(
                            1e-12, float(repeat_row["baseline_prefill_tok_s"])
                        )
                        repeat_row["decode_speedup_x"] = float(repeat_row["baseline_decode_ms"]) / max(
                            1e-12, float(repeat_row["flash_decode_ms"])
                        )
                        repeat_row["baseline_e2e_time_s"] = float(repeat_row["baseline_prefill_time_s"]) + float(
                            repeat_row["baseline_decode_time_s"]
                        )
                        repeat_row["flash_e2e_time_s"] = float(repeat_row["flash_prefill_time_s"]) + float(
                            repeat_row["flash_decode_time_s"]
                        )
                        repeat_row["e2e_speedup_x"] = float(repeat_row["baseline_e2e_time_s"]) / max(
                            1e-12, float(repeat_row["flash_e2e_time_s"])
                        )
                    except Exception as exc:
                        repeat_row["status"] = "error"
                        repeat_row["error"] = f"{type(exc).__name__}: {exc}"
                        repeat_row["traceback"] = traceback.format_exc()
                    repeats.append(repeat_row)
                row["repeats"] = repeats
                row.update(_aggregate_repeat_rows(repeats))
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()
            results.append(row)

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "gpu": args.gpu,
        "device": args.device,
        "dtype": args.dtype,
        "prompt_len": args.prompt_len,
        "new_tokens": args.new_tokens,
        "warmup": args.warmup,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "ffn_backend": args.ffn_backend,
        "mlp_graph_scope": args.mlp_graph_scope,
    }
    md_text = _build_markdown(results=results, config=config, md_path=md_path)
    md_path.write_text(md_text, encoding="utf-8")
    print(md_text)
    print("")
    print(f"JSON saved to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
