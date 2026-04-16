#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_QUANT_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_QUANT_ROOT)
for _path in (_REPO_ROOT, _QUANT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from quant.awq_quantize_ckpt import build_parser as build_awq_parser
from quant.awq_quantize_ckpt import quantize_checkpoint as run_awq_quantize
from quant.fuse_lora_ckpt import build_parser as build_fuse_parser
from quant.fuse_lora_ckpt import fuse_checkpoint
from quant.gptq_quantize_ckpt import build_parser as build_gptq_parser
from quant.gptq_quantize_ckpt import quantize_checkpoint as run_gptq_quantize


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _parse_wbits(raw: str) -> List[int]:
    vals = []
    for item in _parse_csv(raw):
        vals.append(int(item))
    if not vals:
        raise ValueError("--wbits is empty")
    bad = [x for x in vals if x not in (4, 8)]
    if bad:
        raise ValueError(f"Only W4/W8 are supported here, got: {bad}")
    return vals


def _parse_methods(raw: str) -> List[str]:
    vals = [x.lower() for x in _parse_csv(raw)]
    if not vals:
        raise ValueError("--method is empty")
    bad = [x for x in vals if x not in {"awq", "gptq"}]
    if bad:
        raise ValueError(f"Unsupported methods: {bad}")
    return vals


def _base_stem(path: str) -> str:
    stem = Path(path).stem
    if stem.endswith("_linear_fused"):
        stem = stem[: -len("_linear_fused")]
    return stem


def _default_out_dir() -> str:
    return str(Path(_REPO_ROOT) / "checkpoints" / "quantized")


def _default_namespace(parser: argparse.ArgumentParser, required: Sequence[str]) -> argparse.Namespace:
    return parser.parse_args(list(required))


def _prepare_fused_ckpt(args: argparse.Namespace) -> str:
    if not bool(args.prepare_linear_fused):
        return str(args.in_ckpt)

    base = _base_stem(str(args.in_ckpt))
    out_dir = Path(str(args.out_dir or _default_out_dir()))
    out_dir.mkdir(parents=True, exist_ok=True)
    fused_ckpt = str(Path(args.prepared_ckpt) if args.prepared_ckpt else (out_dir / f"{base}_linear_fused.pt"))
    if os.path.exists(fused_ckpt) and not bool(args.overwrite_prepare):
        print(f"[Prepare] reusing fused checkpoint: {fused_ckpt}")
        return fused_ckpt

    fuse_args = _default_namespace(
        build_fuse_parser(),
        ["--in_ckpt", str(args.in_ckpt), "--out_ckpt", fused_ckpt],
    )
    fuse_args.device = str(args.prepare_device)
    fuse_args.dtype = str(args.prepare_dtype)
    fuse_args.overwrite = True
    fuse_checkpoint(fuse_args)
    return fused_ckpt


def _build_awq_args(
    cli_args: argparse.Namespace,
    *,
    work_ckpt: str,
    out_ckpt: str,
    wbits: int,
) -> argparse.Namespace:
    args = _default_namespace(build_awq_parser(), ["--in_ckpt", work_ckpt])
    args.in_ckpt = str(work_ckpt)
    args.out_ckpt = str(out_ckpt)
    args.overwrite = bool(cli_args.overwrite)
    args.wbits = int(wbits)
    args.groupsize = int(cli_args.groupsize)
    args.zero_point = bool(cli_args.zero_point)
    args.no_zero_point = bool(not cli_args.zero_point)
    args.version = str(cli_args.awq_version)
    args.calib_dataset = str(cli_args.calib_dataset)
    args.calib_nsamples = int(cli_args.calib_nsamples)
    args.calib_seqlen = int(cli_args.calib_seqlen)
    args.seed = int(cli_args.seed)
    args.device = str(cli_args.device)
    args.dtype = str(cli_args.dtype)
    args.max_chunk_memory = int(cli_args.max_chunk_memory)
    args.n_parallel_calib_samples = int(cli_args.n_parallel_calib_samples)
    args.max_calib_samples = int(cli_args.max_calib_samples)
    args.max_calib_seq_len = int(cli_args.max_calib_seq_len)
    args.smoke_seq_len = int(cli_args.smoke_seq_len)
    args.skip_save = bool(cli_args.skip_save)
    args.skip_smoke = bool(cli_args.skip_smoke)
    args.apply_clip = bool(cli_args.apply_clip)
    args.merge_lora = False
    return args


def _build_gptq_args(
    cli_args: argparse.Namespace,
    *,
    work_ckpt: str,
    out_ckpt: str,
    wbits: int,
) -> argparse.Namespace:
    args = _default_namespace(build_gptq_parser(), ["--in_ckpt", work_ckpt])
    args.in_ckpt = str(work_ckpt)
    args.out_ckpt = str(out_ckpt)
    args.wbits = int(wbits)
    args.groupsize = int(cli_args.groupsize)
    args.percdamp = float(cli_args.percdamp)
    args.sym = bool(cli_args.sym)
    args.act_order = bool(cli_args.act_order)
    args.static_groups = bool(cli_args.static_groups)
    args.true_sequential = bool(cli_args.true_sequential)
    args.skip_lora = bool(cli_args.skip_lora)
    args.nearest = bool(cli_args.nearest)
    args.calib_dataset = str(cli_args.calib_dataset)
    args.calib_nsamples = int(cli_args.calib_nsamples)
    args.calib_seqlen = int(cli_args.calib_seqlen)
    args.seed = int(cli_args.seed)
    args.eval_datasets = str(cli_args.eval_datasets)
    args.eval_seqlen = int(cli_args.eval_seqlen)
    args.eval_batch_size = int(cli_args.eval_batch_size)
    args.max_eval_batches = int(cli_args.max_eval_batches)
    args.device = str(cli_args.device)
    args.dtype = str(cli_args.dtype)
    args.eval_before = bool(cli_args.eval_before)
    args.skip_save = bool(cli_args.skip_save)
    args.real_pack = bool(cli_args.real_pack)
    args.merge_lora = False
    args.pack_compute_dtype = str(cli_args.pack_compute_dtype)
    args.save_legacy = bool(cli_args.save_legacy)
    args.cpu_acts = bool(cli_args.cpu_acts)
    args.cpu_hessian = bool(cli_args.cpu_hessian)
    return args


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Unified repo-native quantization path for FlashSVD checkpoints.")
    ap.add_argument("--in_ckpt", type=str, required=True, help="Input repo-native .pt checkpoint.")
    ap.add_argument("--out_dir", type=str, default=_default_out_dir(), help="Directory for quantized outputs.")
    ap.add_argument("--method", type=str, default="awq,gptq", help="Comma-separated: awq,gptq")
    ap.add_argument("--wbits", type=str, default="4,8", help="Comma-separated: 4,8")
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite final quantized checkpoints.")
    ap.add_argument("--skip_save", action="store_true", help="Run quantization but do not save outputs.")
    ap.add_argument("--skip_smoke", action="store_true", help="Disable backend smoke forward where supported.")

    ap.add_argument("--prepare_linear_fused", action=argparse.BooleanOptionalAction, default=True, help="Fuse LoRA/ActLoRA wrappers into plain Linear before quantization.")
    ap.add_argument("--prepared_ckpt", type=str, default="", help="Optional path for the prepared fused checkpoint.")
    ap.add_argument("--overwrite_prepare", action="store_true", help="Rebuild the fused checkpoint even if it already exists.")
    ap.add_argument("--prepare_device", type=str, default="cuda")
    ap.add_argument("--prepare_dtype", type=str, default="float16")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib_dataset", type=str, default="wikitext2")
    ap.add_argument("--calib_nsamples", type=int, default=1)
    ap.add_argument("--calib_seqlen", type=int, default=64)

    ap.add_argument("--max_chunk_memory", type=int, default=1024 * 1024 * 1024)
    ap.add_argument("--n_parallel_calib_samples", type=int, default=1)
    ap.add_argument("--max_calib_samples", type=int, default=1)
    ap.add_argument("--max_calib_seq_len", type=int, default=64)
    ap.add_argument("--smoke_seq_len", type=int, default=8)

    ap.add_argument("--zero_point", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--apply_clip", action="store_true")
    ap.add_argument("--awq_version", type=str, default="GEMM")

    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--act_order", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--static_groups", action="store_true")
    ap.add_argument("--true_sequential", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip_lora", action="store_true")
    ap.add_argument("--nearest", action="store_true", help="Use RTN instead of GPTQ.")
    ap.add_argument("--eval_before", action="store_true")
    ap.add_argument("--eval_datasets", type=str, default="wikitext2")
    ap.add_argument("--eval_seqlen", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=1)
    ap.add_argument("--max_eval_batches", type=int, default=1)
    ap.add_argument("--real_pack", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pack_compute_dtype", type=str, default="auto")
    ap.add_argument("--save_legacy", action="store_true")
    ap.add_argument("--cpu_acts", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cpu_hessian", action=argparse.BooleanOptionalAction, default=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    methods = _parse_methods(args.method)
    wbits_list = _parse_wbits(args.wbits)

    out_dir = Path(str(args.out_dir or _default_out_dir()))
    out_dir.mkdir(parents=True, exist_ok=True)

    work_ckpt = _prepare_fused_ckpt(args)
    base = _base_stem(str(args.in_ckpt))
    results: List[Dict[str, object]] = []

    print("==== Unified Quantization Path ====")
    print(
        f"in_ckpt={args.in_ckpt} work_ckpt={work_ckpt} methods={methods} "
        f"wbits={wbits_list} groupsize={int(args.groupsize)} device={args.device} dtype={args.dtype}"
    )

    for method in methods:
        for wbits in wbits_list:
            out_ckpt = str(out_dir / f"{base}_{method}_w{int(wbits)}_real.pt")
            print(f"\n[Job] method={method} wbits={int(wbits)} out={out_ckpt}")
            if method == "awq":
                job_args = _build_awq_args(args, work_ckpt=work_ckpt, out_ckpt=out_ckpt, wbits=wbits)
                result = run_awq_quantize(job_args)
            else:
                job_args = _build_gptq_args(args, work_ckpt=work_ckpt, out_ckpt=out_ckpt, wbits=wbits)
                result = run_gptq_quantize(job_args)
            results.append(result)

    print("\n==== Summary ====")
    for item in results:
        print(
            f"{item['method']} W{int(item['wbits'])}: "
            f"saved={int(bool(item['saved']))} out={item['out_ckpt']}"
        )


if __name__ == "__main__":
    main()
