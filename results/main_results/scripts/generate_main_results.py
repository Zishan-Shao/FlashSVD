#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = OUT_DIR.parent
REPO_DIR = RESULTS_DIR.parent
MAIN_DIR = OUT_DIR / "main"
ABLATION_DIR = OUT_DIR / "ablation"
PAPER_SCRIPT = RESULTS_DIR / "paper_results" / "scripts" / "generate_paper_results.py"


def load_paper_helpers():
    spec = importlib.util.spec_from_file_location("paper_results_helpers", PAPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load helper module from {PAPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper = load_paper_helpers()
SVG = paper.SVG
rasterize_svg = paper.rasterize_svg


COLORS = {
    "bg": "#FFFFFF",
    "panel_border": "#D6DEE8",
    "panel_title": "#111827",
    "text": "#1F2937",
    "muted": "#6B7280",
    "grid": "#E5EAF1",
    "axis": "#334155",
    "flashsvd": "#007F73",
    "flashsvd_light": "#40B8A7",
    "static": "#5B6577",
    "densekv": "#8C6E5C",
    "nograph": "#7A8799",
    "split": "#F39C12",
    "layer": "#0B8F5A",
    "attn": "#2F6BFF",
    "mlp": "#E67E22",
    "ln": "#8E44AD",
    "other": "#95A5A6",
    "dense": "#B56576",
    "sparse": "#C0392B",
    "sparse_fa2": "#8E44AD",
    "gain": "#0F766E",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def ensure_dirs() -> None:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)


def avg(values: list[float]) -> float:
    return mean(values)


def fmt_number(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def fmt_speedup(value: float) -> str:
    return f"{value:.2f}x"


def nice_upper(value: float) -> float:
    if value <= 1:
        step = 0.2
    elif value <= 3:
        step = 0.25
    elif value <= 8:
        step = 0.5
    elif value <= 20:
        step = 2.0
    elif value <= 80:
        step = 5.0
    elif value <= 300:
        step = 20.0
    elif value <= 800:
        step = 50.0
    else:
        step = 100.0
    return math.ceil(value * 1.08 / step) * step


def figure_title(svg: SVG, width: int, title: str, subtitle: str) -> None:
    svg.text(90, 72, title, size=44, weight="bold", fill=COLORS["panel_title"])
    svg.text(90, 112, subtitle, size=24, fill=COLORS["muted"])
    svg.line(90, 138, width - 90, 138, stroke=COLORS["panel_border"], stroke_width=2)


def panel_header(svg: SVG, rect: tuple[int, int, int, int], label: str, title: str, note: str | None = None) -> None:
    x, y, w, h = rect
    svg.rect(x, y, w, h, fill=COLORS["bg"], stroke=COLORS["panel_border"], stroke_width=2, rx=24)
    svg.rect(x + 26, y + 22, 40, 40, fill="#ECFDF5", stroke="none", rx=10)
    svg.text(x + 46, y + 51, label, size=24, weight="bold", fill=COLORS["gain"], anchor="middle")
    svg.text(x + 82, y + 50, title, size=28, weight="bold", fill=COLORS["panel_title"])
    if note:
        svg.text(x + w - 28, y + 50, note, size=20, anchor="end", fill=COLORS["muted"])


def draw_legend(svg: SVG, x: float, y: float, items: list[dict], *, text_size: int = 20, gap: int = 12, swatch: int = 18) -> None:
    cursor_x = x
    for item in items:
        svg.rect(cursor_x, y - swatch + 2, swatch, swatch, fill=item["color"], rx=4)
        svg.text(cursor_x + swatch + 8, y, item["label"], size=text_size, fill=COLORS["text"])
        cursor_x += swatch + 8 + max(80, len(item["label"]) * text_size * 0.62) + gap


def draw_grouped_bars_panel(
    svg: SVG,
    rect: tuple[int, int, int, int],
    *,
    title: str,
    categories: list[str],
    series: list[dict],
    y_label: str,
    panel_label: str,
    note: str | None = None,
    y_max: float | None = None,
    group_notes: list[str] | None = None,
    reference_y: float | None = None,
) -> None:
    x, y, w, h = rect
    panel_header(svg, rect, panel_label, title, note)
    left = x + 92
    right = x + w - 32
    top = y + 96
    bottom = y + h - 92
    plot_w = right - left
    plot_h = bottom - top

    if y_max is None:
        y_max = max(max(s["values"]) for s in series)
    y_max = nice_upper(y_max)

    ticks = 5
    for i in range(ticks + 1):
        frac = i / ticks
        yy = bottom - frac * plot_h
        val = frac * y_max
        svg.line(left, yy, right, yy, stroke=COLORS["grid"], stroke_width=2)
        svg.text(left - 12, yy + 8, fmt_number(val), size=20, anchor="end", fill=COLORS["muted"])

    if reference_y is not None and 0 <= reference_y <= y_max:
        ref_y = bottom - reference_y / y_max * plot_h
        svg.line(left, ref_y, right, ref_y, stroke=COLORS["muted"], stroke_width=2, dash="10 8")

    svg.line(left, top, left, bottom, stroke=COLORS["axis"], stroke_width=3)
    svg.line(left, bottom, right, bottom, stroke=COLORS["axis"], stroke_width=3)
    svg.text(x + 28, top + plot_h / 2, y_label, size=22, anchor="middle", rotate=-90, fill=COLORS["text"])

    group_w = plot_w / len(categories)
    inner_w = group_w * 0.74
    bar_w = inner_w / len(series)
    for idx, cat in enumerate(categories):
        gx = left + idx * group_w + (group_w - inner_w) / 2
        for sidx, s in enumerate(series):
            value = s["values"][idx]
            bar_h = 0 if y_max == 0 else value / y_max * plot_h
            bx = gx + sidx * bar_w
            by = bottom - bar_h
            svg.rect(bx, by, bar_w - 10, bar_h, fill=s["color"], rx=8)
            svg.text(bx + (bar_w - 10) / 2, by - 12, fmt_number(value), size=18, anchor="middle", fill=COLORS["text"])
        svg.multiline_text(gx + inner_w / 2, bottom + 34, cat, size=20, anchor="middle", line_gap=22)
        if group_notes:
            svg.text(gx + inner_w / 2, bottom + 66, group_notes[idx], size=18, anchor="middle", fill=COLORS["gain"])

    draw_legend(svg, x + 92, y + 74, [{"label": s["label"], "color": s["color"]} for s in series], text_size=20)


def draw_stacked_bars_panel(
    svg: SVG,
    rect: tuple[int, int, int, int],
    *,
    title: str,
    rows: list[dict],
    stacks: list[dict],
    y_label: str,
    panel_label: str,
    note: str | None = None,
) -> None:
    x, y, w, h = rect
    panel_header(svg, rect, panel_label, title, note)
    left = x + 92
    right = x + w - 32
    top = y + 96
    bottom = y + h - 92
    plot_w = right - left
    plot_h = bottom - top

    y_max = nice_upper(max(sum(row[s["key"]] for s in stacks) for row in rows))
    ticks = 5
    for i in range(ticks + 1):
        frac = i / ticks
        yy = bottom - frac * plot_h
        val = frac * y_max
        svg.line(left, yy, right, yy, stroke=COLORS["grid"], stroke_width=2)
        svg.text(left - 12, yy + 8, fmt_number(val), size=20, anchor="end", fill=COLORS["muted"])

    svg.line(left, top, left, bottom, stroke=COLORS["axis"], stroke_width=3)
    svg.line(left, bottom, right, bottom, stroke=COLORS["axis"], stroke_width=3)
    svg.text(x + 28, top + plot_h / 2, y_label, size=22, anchor="middle", rotate=-90, fill=COLORS["text"])

    bar_w = plot_w / (len(rows) * 1.7)
    gap = bar_w * 0.7
    cursor = left + gap
    for row in rows:
        running = 0.0
        for stack in stacks:
            value = row[stack["key"]]
            bar_h = value / y_max * plot_h
            by = bottom - (running + value) / y_max * plot_h
            svg.rect(cursor, by, bar_w, bar_h, fill=stack["color"], rx=6)
            running += value
        svg.multiline_text(cursor + bar_w / 2, bottom + 34, row["label"], size=20, anchor="middle", line_gap=22)
        svg.text(cursor + bar_w / 2, bottom - running / y_max * plot_h - 12, fmt_number(running), size=18, anchor="middle")
        cursor += bar_w + gap

    draw_legend(svg, x + 92, y + 74, [{"label": s["label"], "color": s["color"]} for s in stacks], text_size=20)


def draw_line_panel(
    svg: SVG,
    rect: tuple[int, int, int, int],
    *,
    title: str,
    x_values: list[float],
    series: list[dict],
    y_label: str,
    x_label: str,
    panel_label: str,
    note: str | None = None,
) -> None:
    x, y, w, h = rect
    panel_header(svg, rect, panel_label, title, note)
    left = x + 92
    right = x + w - 32
    top = y + 96
    bottom = y + h - 92
    plot_w = right - left
    plot_h = bottom - top

    x_min, x_max = min(x_values), max(x_values)
    y_max = nice_upper(max(max(s["values"]) for s in series))
    ticks = 5

    for i in range(ticks + 1):
        frac = i / ticks
        yy = bottom - frac * plot_h
        val = frac * y_max
        svg.line(left, yy, right, yy, stroke=COLORS["grid"], stroke_width=2)
        svg.text(left - 12, yy + 8, fmt_number(val), size=20, anchor="end", fill=COLORS["muted"])

    svg.line(left, top, left, bottom, stroke=COLORS["axis"], stroke_width=3)
    svg.line(left, bottom, right, bottom, stroke=COLORS["axis"], stroke_width=3)
    svg.text(x + 28, top + plot_h / 2, y_label, size=22, anchor="middle", rotate=-90, fill=COLORS["text"])
    svg.text(left + plot_w / 2, y + h - 28, x_label, size=22, anchor="middle", fill=COLORS["text"])

    def xmap(value: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def ymap(value: float) -> float:
        return bottom - value / y_max * plot_h

    for xv in x_values:
        xx = xmap(xv)
        svg.line(xx, bottom, xx, bottom + 10, stroke=COLORS["axis"], stroke_width=2)
        svg.text(xx, bottom + 34, f"{int(xv)}", size=20, anchor="middle", fill=COLORS["text"])

    for s in series:
        pts = [(xmap(xv), ymap(yv)) for xv, yv in zip(x_values, s["values"])]
        svg.polyline(pts, stroke=s["color"], stroke_width=5)
        for px, py in pts:
            svg.circle(px, py, r=7, fill=s["color"])

    draw_legend(svg, x + 92, y + 74, [{"label": s["label"], "color": s["color"]} for s in series], text_size=20)


def main_runtime_rows() -> list[dict]:
    data = load_json(RESULTS_DIR / "runtime_study_2026-03-31" / "tables" / "e2e_summary.json")
    rows = []
    order = [
        ("StaticCache", "512/32", "static_vs_flashsvd:short"),
        ("StaticCache", "2048/128", "static_vs_flashsvd:long"),
        ("DenseKV", "512/32", "densekv_vs_flashsvd:short"),
        ("DenseKV", "2048/128", "densekv_vs_flashsvd:long"),
    ]
    for baseline, config, key in order:
        item = data[key]
        rows.append(
            {
                "baseline": baseline,
                "config": config,
                "category": f"{baseline}\n{config}",
                "baseline_decode_ms": item["svd.decode_ms_per_token"]["median"],
                "flash_decode_ms": item["flashsvd.decode_ms_per_token"]["median"],
                "baseline_total_s": item["svd.total_time_s"]["median"],
                "flash_total_s": item["flashsvd.total_time_s"]["median"],
                "decode_speedup": item["svd.decode_ms_per_token"]["median"] / item["flashsvd.decode_ms_per_token"]["median"],
                "e2e_speedup": item["svd.total_time_s"]["median"] / item["flashsvd.total_time_s"]["median"],
            }
        )
    return rows


def family_summary_rows() -> list[dict]:
    rows = load_json(REPO_DIR / "docs" / "notes" / "lowrankarena_main_table_extended_2026-03-17.json")
    grouped: dict[str, list[dict]] = defaultdict(list)
    all_rows: list[dict] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        e2e_speedup = (row["baseline_prefill_time_s"] + 32.0 * row["baseline_decode_ms"] / 1000.0) / (
            row["flash_prefill_time_s"] + 32.0 * row["flash_decode_ms"] / 1000.0
        )
        packed = {
            "family": row["family_name"],
            "ratio": row["ratio"],
            "prefill_speedup": row["prefill_speedup_x"],
            "decode_speedup": row["decode_speedup_x"],
            "e2e_speedup": e2e_speedup,
        }
        grouped[row["family_name"]].append(packed)
        all_rows.append(packed)

    ordered = []
    for family in ("SVD-LLM v1", "SVD-LLM v2", "Basis Sharing"):
        items = grouped[family]
        ordered.append(
            {
                "family": family,
                "n": len(items),
                "prefill_speedup": avg([item["prefill_speedup"] for item in items]),
                "decode_speedup": avg([item["decode_speedup"] for item in items]),
                "e2e_speedup": avg([item["e2e_speedup"] for item in items]),
            }
        )
    ordered.append(
        {
            "family": "Overall",
            "n": len(all_rows),
            "prefill_speedup": avg([item["prefill_speedup"] for item in all_rows]),
            "decode_speedup": avg([item["decode_speedup"] for item in all_rows]),
            "e2e_speedup": avg([item["e2e_speedup"] for item in all_rows]),
        }
    )
    return ordered


def family_detailed_rows() -> list[dict]:
    rows = load_json(REPO_DIR / "docs" / "notes" / "lowrankarena_main_table_extended_2026-03-17.json")
    packed = []
    family_order = {"SVD-LLM v1": 0, "SVD-LLM v2": 1, "Basis Sharing": 2}
    for row in rows:
        if row.get("status") != "ok":
            continue
        e2e_speedup = (row["baseline_prefill_time_s"] + 32.0 * row["baseline_decode_ms"] / 1000.0) / (
            row["flash_prefill_time_s"] + 32.0 * row["flash_decode_ms"] / 1000.0
        )
        packed.append(
            {
                "family": row["family_name"],
                "ratio": str(row["ratio"]),
                "prefill_speedup": row["prefill_speedup_x"],
                "decode_speedup": row["decode_speedup_x"],
                "e2e_speedup": e2e_speedup,
                "sort_key": (family_order[row["family_name"]], float(row["ratio"])),
            }
        )
    packed.sort(key=lambda item: item["sort_key"])
    return packed


def dense_baseline_stats() -> dict[str, float]:
    text = (REPO_DIR / "docs" / "notes" / "speedup_vs_dense_and_lowrank_2026-03-17.md").read_text()

    def extract(pattern: str) -> float:
        match = re.search(pattern, text)
        if match is None:
            raise RuntimeError(f"failed to parse dense baseline stat with pattern: {pattern}")
        return float(match.group(1))

    return {
        "prefill_s": extract(r"Dense prefill: `([0-9.]+) s`"),
        "decode_ms": extract(r"Dense decode: `([0-9.]+) ms/token`"),
        "e2e_s": extract(r"Dense end-to-end: `([0-9.]+) s`"),
    }


def unified_family_rows() -> list[dict]:
    rows = load_json(REPO_DIR / "docs" / "notes" / "lowrankarena_main_table_extended_2026-03-17.json")
    dense = dense_baseline_stats()
    family_order = {"SVD-LLM v1": 0, "SVD-LLM v2": 1, "Basis Sharing": 2}
    packed = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        flash_total = row["flash_prefill_time_s"] + 32.0 * row["flash_decode_ms"] / 1000.0
        lowrank_total = row["baseline_prefill_time_s"] + 32.0 * row["baseline_decode_ms"] / 1000.0
        packed.append(
            {
                "family": row["family_name"],
                "ratio": str(row["ratio"]),
                "flash_total_s": flash_total,
                "flash_decode_ms": row["flash_decode_ms"],
                "densekv_decode_speedup": row["decode_speedup_x"],
                "densekv_e2e_speedup": lowrank_total / flash_total,
                "dense_decode_speedup": dense["decode_ms"] / row["flash_decode_ms"],
                "dense_e2e_speedup": dense["e2e_s"] / flash_total,
                "sort_key": (family_order[row["family_name"]], float(row["ratio"])),
            }
        )
    packed.sort(key=lambda item: item["sort_key"])
    return packed


def long_prompt_rows() -> list[dict]:
    data = load_json(
        RESULTS_DIR / "runtime_study_full_2026-03-31_long_context_decode_sweep" / "tables" / "stage_summary.json"
    )
    rows = []
    for baseline_key, baseline_label in (("static", "StaticCache"), ("densekv", "DenseKV")):
        for ctx in (4096, 8192):
            baseline_decode = []
            flash_decode = []
            baseline_total = []
            flash_total = []
            for ratio in sorted(data):
                baseline_decode.append(data[ratio][f"{baseline_key}:ctx{ctx}"]["decode_ms_per_token"]["median"])
                flash_decode.append(data[ratio][f"flashsvd:ctx{ctx}"]["decode_ms_per_token"]["median"])
                baseline_total.append(data[ratio][f"{baseline_key}:ctx{ctx}"]["total_time_s"]["median"])
                flash_total.append(data[ratio][f"flashsvd:ctx{ctx}"]["total_time_s"]["median"])
            b_decode = avg(baseline_decode)
            f_decode = avg(flash_decode)
            b_total = avg(baseline_total)
            f_total = avg(flash_total)
            rows.append(
                {
                    "baseline": baseline_label,
                    "prompt": f"{ctx // 1024}K",
                    "category": f"{baseline_label}\n{ctx // 1024}K prompt",
                    "baseline_decode_ms": b_decode,
                    "flash_decode_ms": f_decode,
                    "baseline_total_s": b_total,
                    "flash_total_s": f_total,
                    "decode_speedup": b_decode / f_decode,
                    "e2e_speedup": b_total / f_total,
                }
            )
    return rows


def decode_sweep_rows() -> list[dict]:
    data = load_json(
        RESULTS_DIR / "runtime_study_full_2026-03-31_long_context_decode_sweep" / "tables" / "decode_sweep_summary.json"
    )
    rows = []
    for baseline_key, baseline_label in (("static", "StaticCache"), ("densekv", "DenseKV")):
        for tok in (64, 1024, 4096, 8192, 16384):
            baseline_decode = []
            flash_decode = []
            for ratio in sorted(data):
                baseline_decode.append(data[ratio][f"{baseline_key}:tok{tok}"]["decode_ms_per_token"]["median"])
                flash_decode.append(data[ratio][f"flashsvd:tok{tok}"]["decode_ms_per_token"]["median"])
            b_decode = avg(baseline_decode)
            f_decode = avg(flash_decode)
            rows.append(
                {
                    "baseline": baseline_label,
                    "tokens": tok,
                    "baseline_decode_ms": b_decode,
                    "flash_decode_ms": f_decode,
                    "decode_speedup": b_decode / f_decode,
                }
            )
    return rows


def ratio_sweep_rows() -> list[dict]:
    data = load_json(RESULTS_DIR / "runtime_study_full_2026-03-31_ratio_sweep" / "tables" / "ratio_sweep_summary.json")[
        "ratios"
    ]
    rows = []
    for ratio in sorted(data, key=float):
        for config, config_label in (("short", "512/32"), ("long", "2048/128")):
            static = data[ratio][f"static_vs_flashsvd:{config}"]
            dense = data[ratio][f"densekv_vs_flashsvd:{config}"]
            rows.append(
                {
                    "ratio": ratio,
                    "config": config_label,
                    "static_decode_speedup": static["decode_speedup_median"],
                    "static_e2e_speedup": static["total_speedup_median"],
                    "dense_decode_speedup": dense["decode_speedup_median"],
                    "dense_e2e_speedup": dense["total_speedup_median"],
                }
            )
    return rows


def graph_rows() -> list[dict]:
    data = load_json(RESULTS_DIR / "runtime_study_2026-03-31" / "tables" / "graph_ablation_summary.json")
    rows = []
    for config, label in (("short", "512/32"), ("long", "2048/128")):
        entry = {"config": config, "label": label}
        for mode in ("nograph", "split", "layer"):
            entry[mode] = data[f"{config}:{mode}"]["decode_ms_mean"]
        rows.append(entry)
    return rows


def motivation_rows() -> list[dict]:
    rows = load_json(RESULTS_DIR / "motivation" / "graph_fusion_2026-03-31" / "graph_fusion_summary.json")["modes"]["short"]
    packed = []
    for mode, label in (("nograph", "No graph"), ("split", "Split graph"), ("layer", "Per-layer")):
        item = rows[mode]
        packed.append(
            {
                "label": label,
                "decode_ms": item["decode_ms_per_token"],
                "launches": item["cudaLaunchKernel_count_per_token"] + item["cudaGraphLaunch_count_per_token"],
                "copies": item["copy_count_per_token"] + item["clone_count_per_token"] + item["to_copy_count_per_token"],
                "launch_cpu_ms": item["launch_cpu_ms_per_token"],
                "copy_cpu_ms": item["copy_clone_cpu_ms_per_token"],
            }
        )
    return packed


def module_rows() -> list[dict]:
    data = load_json(RESULTS_DIR / "runtime_study_2026-03-31" / "tables" / "module_profiles.json")
    order = [
        ("static", "StaticCache"),
        ("densekv", "DenseKV"),
        ("flashsvd_no_graph", "FlashSVD\nno-graph"),
    ]
    rows = []
    for key, label in order:
        item = data[key]
        rows.append(
            {
                "label": label,
                "attn_ms": item["attn_total"]["ms"],
                "mlp_ms": item["mlp_total"]["ms"],
                "ln_ms": item["ln1_total"]["ms"] + item["ln2_total"]["ms"],
                "other_ms": item["other"]["ms"],
            }
        )
    return rows


def attn_route_rows() -> list[dict]:
    return load_json(RESULTS_DIR / "runtime_study_2026-03-31" / "tables" / "attn_route_microbench.json")


def attn_reconstruct_rows() -> list[dict]:
    data = load_json(RESULTS_DIR / "runtime_study_2026-03-31" / "tables" / "attn_reconstruct_summary.json")
    return [
        {"kind": "Attention reconstruct", "variant": "Exact", "ms": data["exact_ms_mean"]},
        {"kind": "Attention reconstruct", "variant": "Packed linear", "ms": data["packed_linear_ms_mean"]},
        {"kind": "Attention reconstruct", "variant": "Packed Triton", "ms": data["packed_flat_ms_mean"]},
    ]


def mlp_rows() -> list[dict]:
    data = load_json(RESULTS_DIR / "runtime_study_2026-03-31" / "tables" / "mlp_backend_summary.json")
    return [
        {"kind": "MLP backend", "variant": "Baseline eager", "ms": data["baseline_eager_ms_mean"]},
        {"kind": "MLP backend", "variant": "Prod graph", "ms": data["prod_graph_ms_mean"]},
    ]


def save_svg_pdf(svg: SVG, stem: Path) -> None:
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    svg.save(svg_path)
    rasterize_svg(svg_path, png_path)
    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, "PDF", resolution=300.0)


def build_main_figure() -> None:
    width, height = 3200, 2400
    svg = SVG(width, height)
    figure_title(
        svg,
        width,
        "FlashSVD v1.5 Main Results",
        "Compact main-text figure: repeated-run medians, family coverage, and extended long-prompt results (all batch size 1).",
    )
    margin_x = 90
    panel_w = 1480
    panel_h = 980
    gap_x = 60
    gap_y = 60
    top = 180
    left = margin_x
    rects = [
        (left, top, panel_w, panel_h),
        (left + panel_w + gap_x, top, panel_w, panel_h),
        (left, top + panel_h + gap_y, panel_w, panel_h),
        (left + panel_w + gap_x, top + panel_h + gap_y, panel_w, panel_h),
    ]

    runtime = main_runtime_rows()
    draw_grouped_bars_panel(
        svg,
        rects[0],
        title="Decode Latency on Main Repeated-Run Settings",
        panel_label="A",
        y_label="ms / token",
        categories=[row["category"] for row in runtime],
        series=[
            {"label": "Baseline", "color": COLORS["static"], "values": [row["baseline_decode_ms"] for row in runtime]},
            {"label": "FlashSVD v1.5", "color": COLORS["flashsvd"], "values": [row["flash_decode_ms"] for row in runtime]},
        ],
        group_notes=[fmt_speedup(row["decode_speedup"]) for row in runtime],
        note="gain over matched baseline",
    )

    draw_grouped_bars_panel(
        svg,
        rects[1],
        title="End-to-End Latency on Main Repeated-Run Settings",
        panel_label="B",
        y_label="seconds",
        categories=[row["category"] for row in runtime],
        series=[
            {"label": "Baseline", "color": COLORS["static"], "values": [row["baseline_total_s"] for row in runtime]},
            {"label": "FlashSVD v1.5", "color": COLORS["flashsvd"], "values": [row["flash_total_s"] for row in runtime]},
        ],
        group_notes=[fmt_speedup(row["e2e_speedup"]) for row in runtime],
        note="prefill + decode",
    )

    families = family_summary_rows()
    draw_grouped_bars_panel(
        svg,
        rects[2],
        title="Coverage Across Low-Rank Checkpoint Families",
        panel_label="C",
        y_label="speedup (x)",
        categories=[
            "SVD-LLM v1",
            "SVD-LLM v2",
            "Basis\nSharing",
            "Overall",
        ],
        series=[
            {"label": "Prefill", "color": COLORS["flashsvd_light"], "values": [row["prefill_speedup"] for row in families]},
            {"label": "Decode", "color": COLORS["flashsvd"], "values": [row["decode_speedup"] for row in families]},
        ],
        reference_y=1.0,
        note="avg over matched checkpoints",
    )

    long_prompt = long_prompt_rows()
    draw_grouped_bars_panel(
        svg,
        rects[3],
        title="Extended Long-Prompt Decode Latency",
        panel_label="D",
        y_label="ms / token",
        categories=[row["category"] for row in long_prompt],
        series=[
            {"label": "Baseline avg", "color": COLORS["densekv"], "values": [row["baseline_decode_ms"] for row in long_prompt]},
            {"label": "FlashSVD avg", "color": COLORS["flashsvd"], "values": [row["flash_decode_ms"] for row in long_prompt]},
        ],
        group_notes=[fmt_speedup(row["decode_speedup"]) for row in long_prompt],
        note="avg over ratios 0.5-0.8",
    )

    save_svg_pdf(svg, MAIN_DIR / "figure_main")


def build_ablation_figure() -> None:
    width, height = 3200, 2400
    svg = SVG(width, height)
    figure_title(
        svg,
        width,
        "FlashSVD v1.5 Ablations",
        "Compact ablation figure: graph granularity is the main unlock; attention-route changes matter more than MLP backend changes.",
    )
    margin_x = 90
    panel_w = 1480
    panel_h = 980
    gap_x = 60
    gap_y = 60
    top = 180
    left = margin_x
    rects = [
        (left, top, panel_w, panel_h),
        (left + panel_w + gap_x, top, panel_w, panel_h),
        (left, top + panel_h + gap_y, panel_w, panel_h),
        (left + panel_w + gap_x, top + panel_h + gap_y, panel_w, panel_h),
    ]

    graphs = graph_rows()
    draw_grouped_bars_panel(
        svg,
        rects[0],
        title="Decode Graph Granularity",
        panel_label="A",
        y_label="ms / token",
        categories=[row["label"] for row in graphs],
        series=[
            {"label": "No graph", "color": COLORS["nograph"], "values": [row["nograph"] for row in graphs]},
            {"label": "Split graph", "color": COLORS["split"], "values": [row["split"] for row in graphs]},
            {"label": "Per-layer graph", "color": COLORS["layer"], "values": [row["layer"] for row in graphs]},
        ],
        group_notes=[fmt_speedup(row["nograph"] / row["layer"]) for row in graphs],
        note="mean over repeated runs",
    )

    motivation = motivation_rows()
    draw_grouped_bars_panel(
        svg,
        rects[1],
        title="Runtime Bookkeeping per Generated Token",
        panel_label="B",
        y_label="ops / token",
        categories=[row["label"] for row in motivation],
        series=[
            {"label": "Launches", "color": COLORS["dense"], "values": [row["launches"] for row in motivation]},
            {"label": "Copies", "color": COLORS["densekv"], "values": [row["copies"] for row in motivation]},
        ],
        note="512/32 profile",
    )

    draw_stacked_bars_panel(
        svg,
        rects[2],
        title="Module Breakdown Before Full Graph Fusion",
        panel_label="C",
        y_label="ms / token",
        rows=module_rows(),
        stacks=[
            {"key": "attn_ms", "label": "Attention", "color": COLORS["attn"]},
            {"key": "mlp_ms", "label": "MLP", "color": COLORS["mlp"]},
            {"key": "ln_ms", "label": "LayerNorm", "color": COLORS["ln"]},
            {"key": "other_ms", "label": "Other", "color": COLORS["other"]},
        ],
        note="prompt 512, decode 32",
    )

    route = attn_route_rows()
    draw_line_panel(
        svg,
        rects[3],
        title="Attention-Route Microbenchmark",
        panel_label="D",
        x_values=[row["L"] for row in route],
        series=[
            {"label": "FlashSVD v1.5", "color": COLORS["flashsvd"], "values": [row["flashsvd_v15_ms"] for row in route]},
            {"label": "Dense + FA2", "color": COLORS["dense"], "values": [row["dense_fa2_only_ms"] for row in route]},
            {"label": "Sparse + FA2", "color": COLORS["sparse_fa2"], "values": [row["sparse_fa2_only_ms"] for row in route]},
            {"label": "Sparse legacy", "color": COLORS["sparse"], "values": [row["sparse_ms"] for row in route]},
        ],
        y_label="step latency (ms)",
        x_label="cached sequence length",
        note="current-token decode step",
    )

    save_svg_pdf(svg, ABLATION_DIR / "figure_ablation")


def build_main_tables() -> None:
    dense = dense_baseline_stats()
    family_rows = unified_family_rows()
    family_summary = family_summary_rows()
    long_prompt = long_prompt_rows()
    sweep_rows = decode_sweep_rows()
    long_prompt_decode_min = min(row["decode_speedup"] for row in long_prompt)
    long_prompt_decode_max = max(row["decode_speedup"] for row in long_prompt)
    long_prompt_e2e_min = min(row["e2e_speedup"] for row in long_prompt)
    long_prompt_e2e_max = max(row["e2e_speedup"] for row in long_prompt)
    sweep_min = min(row["decode_speedup"] for row in sweep_rows)
    sweep_max = max(row["decode_speedup"] for row in sweep_rows)
    overall = next(row for row in family_summary if row["family"] == "Overall")
    dense_decode_avg = avg([row["dense_decode_speedup"] for row in family_rows])
    dense_e2e_avg = avg([row["dense_e2e_speedup"] for row in family_rows])

    lines = [
        "% Auto-generated by results/main_results/scripts/generate_main_results.py",
        "% Suggested packages: booktabs, threeparttable",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.6pt}",
        "\\renewcommand{\\arraystretch}{0.96}",
        "\\begin{threeparttable}",
        (
            "\\caption{Unified main results on public LowRankArena checkpoints under one serving configuration "
            "(A100, \\texttt{bf16}, batch size 1, prompt length 512, decode length 32). "
            "Each row corresponds to one checkpoint family and ratio. FlashSVD is compared against the matched "
            "compressed checkpoint running with the DenseKV baseline and against a standard dense LLaMA-7B serving path. "
            f"Averaged over 13 checkpoints, FlashSVD achieves {overall['decode_speedup']:.2f}$\\times$ decode and "
            f"{overall['e2e_speedup']:.2f}$\\times$ end-to-end speedup over DenseKV, and {dense_decode_avg:.2f}$\\times$ "
            f"decode and {dense_e2e_avg:.2f}$\\times$ end-to-end speedup over the dense baseline.}}"
        ),
        "\\label{tab:flashsvd-main}",
        "% Legacy labels retained for paper-draft compatibility.",
        "\\label{tab:flashsvd-main-runtime}",
        "\\label{tab:flashsvd-family-coverage}",
        "\\label{tab:flashsvd-long-prompt}",
        "\\begin{tabular}{@{}llrrcccc@{}}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{FlashSVD} & \\multicolumn{2}{c}{vs DenseKV Baseline} & \\multicolumn{2}{c}{vs Dense LLaMA-7B} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
        "Family & Ratio & Total (s) & Decode (ms/tok) & Decode & E2E & Decode & E2E \\\\",
        "\\midrule",
    ]
    previous_family = None
    for row in family_rows:
        if row["family"] != previous_family and previous_family is not None:
            lines.append("\\midrule")
        family_label = row["family"] if row["family"] != previous_family else ""
        lines.append(
            f"{family_label} & {row['ratio']} & "
            f"{row['flash_total_s']:.3f} & {row['flash_decode_ms']:.3f} & "
            f"{row['densekv_decode_speedup']:.2f}$\\times$ & {row['densekv_e2e_speedup']:.2f}$\\times$ & "
            f"{row['dense_decode_speedup']:.2f}$\\times$ & {row['dense_e2e_speedup']:.2f}$\\times$ \\\\"
        )
        previous_family = row["family"]
    lines.extend(
        [
            "\\midrule",
            (
                f"Overall avg & {overall['n']} ckpts & -- & -- & "
                f"{overall['decode_speedup']:.2f}$\\times$ & {overall['e2e_speedup']:.2f}$\\times$ & "
                f"{dense_decode_avg:.2f}$\\times$ & {dense_e2e_avg:.2f}$\\times$ \\\\"
            ),
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]\\footnotesize",
            "\\item DenseKV baseline uses the same compressed checkpoint with the aligned dense-KV runtime from the LowRankArena benchmark. Dense LLaMA-7B uses \\texttt{jeffwan/llama-7b-hf} under the standard dense serving path.",
            (
                f"\\item Dense LLaMA-7B baseline numbers are prefill $={dense['prefill_s']:.3f}$ s, decode "
                f"$={dense['decode_ms']:.3f}$ ms/token, end-to-end $={dense['e2e_s']:.3f}$ s."
            ),
            "\\item End-to-end latency is prefill plus 32-token decode. Basis Sharing checkpoints at ratios 0.7 and 0.8 are unavailable in the public release and are therefore omitted.",
            f"\\item Appendix results retain long-prompt averages of {long_prompt_decode_min:.2f}$\\times$--{long_prompt_decode_max:.2f}$\\times$ decode and {long_prompt_e2e_min:.2f}$\\times$--{long_prompt_e2e_max:.2f}$\\times$ end-to-end speedup, plus a decode-length sweep with {sweep_min:.2f}$\\times$--{sweep_max:.2f}$\\times$ average decode speedup over 64--16384 generated tokens.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
        ]
    )
    (MAIN_DIR / "table_main.tex").write_text("\n".join(lines))


def build_ablation_tables() -> None:
    graphs = graph_rows()
    motivation = motivation_rows()
    route = attn_route_rows()
    kernel_rows = attn_reconstruct_rows() + mlp_rows()

    lines = [
        "% Auto-generated by results/main_results/scripts/generate_main_results.py",
        "% Suggested packages: booktabs, threeparttable",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{threeparttable}",
        "\\caption{Graph-granularity ablation on the two headline decoder settings.}",
        "\\label{tab:flashsvd-graph-ablation}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Config & No graph & Split graph & Per-layer graph & No graph $\\rightarrow$ per-layer \\\\",
        "\\midrule",
    ]
    for row in graphs:
        lines.append(
            f"{row['label']} & {row['nograph']:.3f} & {row['split']:.3f} & {row['layer']:.3f} & {(row['nograph'] / row['layer']):.2f}$\\times$ \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]\\footnotesize",
            "\\item Values are decode latency in ms/token. Per-layer graph is the dominant systems improvement in the current runtime.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{threeparttable}",
            "\\caption{Runtime bookkeeping counts on the short-profile motivation study (prompt length 512, decode length 32).}",
            "\\label{tab:flashsvd-fragmentation}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Mode & Decode ms/token & Launches/token & Copies/token & Launch CPU ms/token & Copy CPU ms/token \\\\",
            "\\midrule",
        ]
    )
    for row in motivation:
        lines.append(
            f"{row['label']} & {row['decode_ms']:.3f} & {row['launches']:.0f} & {row['copies']:.0f} & {row['launch_cpu_ms']:.3f} & {row['copy_cpu_ms']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]\\footnotesize",
            "\\item Launches aggregates direct CUDA kernel launches and CUDA graph launches. Copies aggregates copy, clone, and to-copy operators.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{threeparttable}",
            "\\caption{Attention-route microbenchmark for the current-token decode step.}",
            "\\label{tab:flashsvd-attn-route}",
            "\\begin{tabular}{rrrrr}",
            "\\toprule",
            "Cached length & FlashSVD v1.5 & Dense + FA2 & Sparse + FA2 & Sparse legacy \\\\",
            "\\midrule",
        ]
    )
    for row in route:
        lines.append(
            f"{row['L']} & {row['flashsvd_v15_ms']:.4f} & {row['dense_fa2_only_ms']:.4f} & {row['sparse_fa2_only_ms']:.4f} & {row['sparse_ms']:.4f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]\\footnotesize",
            "\\item All values are step latency in ms. Lower is better.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{threeparttable}",
            "\\caption{Kernel-level micro-ablations retained in the paper-facing summary.}",
            "\\label{tab:flashsvd-kernel-micro}",
            "\\begin{tabular}{llr}",
            "\\toprule",
            "Component & Variant & Mean latency (ms) \\\\",
            "\\midrule",
        ]
    )
    current_kind = None
    for row in kernel_rows:
        kind = row["kind"]
        if current_kind is not None and kind != current_kind:
            lines.append("\\midrule")
        current_kind = kind
        lines.append(f"{kind} & {row['variant']} & {row['ms']:.4f} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]\\footnotesize",
            "\\item The attention reconstruct path benefits materially from packed kernels; the MLP backend change is small in isolation.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
            "",
        ]
    )
    (ABLATION_DIR / "table_ablation.tex").write_text("\n".join(lines))


def build_readme() -> None:
    text = """# Main Results Bundle

This folder contains a compact, main-text-oriented results package derived from the current FlashSVD v1.5 runtime studies.

## Recommended main-paper items

- `main/figure_main.pdf`: one compact figure for headline runtime results, family coverage, and long-prompt behavior.
- `ablation/figure_ablation.pdf`: one compact figure for graph granularity and supporting runtime ablations.
- `main/table_main.tex`: one unified main-text table over the public LowRankArena checkpoint sweep, with one row per family/ratio entry and grouped columns for FlashSVD absolute latency plus speedups against DenseKV and dense LLaMA-7B baselines.
- `ablation/table_ablation.tex`: graph ablation, fragmentation counts, attention-route microbench, and kernel micro-ablation tables.

## Notes

- The figures are exported as SVG, PNG, and PDF. The PDF export is rasterized at 300 DPI.
- The tables are ready to include in LaTeX and assume `booktabs` and `threeparttable`.
"""
    (OUT_DIR / "README.md").write_text(text)


def main() -> None:
    ensure_dirs()
    build_main_figure()
    build_ablation_figure()
    build_main_tables()
    build_ablation_tables()
    build_readme()
    print(f"Wrote compact results bundle to {OUT_DIR}")


if __name__ == "__main__":
    main()
