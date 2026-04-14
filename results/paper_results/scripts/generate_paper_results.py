#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

from PIL import Image, ImageColor, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PAPER_DIR.parent
RUNTIME_DIR = RESULTS_DIR / "runtime_study_2026-03-31"
MOTIVATION_DIR = RESULTS_DIR / "motivation" / "graph_fusion_2026-03-31"
FIG_DIR = PAPER_DIR / "figures"
TAB_DIR = PAPER_DIR / "tables"


COLORS = {
    "baseline": "#5B6577",
    "densekv": "#8C6E5C",
    "flashsvd": "#007F73",
    "flashsvd_light": "#40B8A7",
    "nograph": "#7A8799",
    "split": "#F39C12",
    "layer": "#0B8F5A",
    "attn": "#2F6BFF",
    "mlp": "#E67E22",
    "ln": "#8E44AD",
    "other": "#95A5A6",
    "prefill": "#F4A261",
    "decode": "#2A9D8F",
    "grid": "#D9DEE7",
    "axis": "#2D3748",
    "text": "#1F2937",
    "dense": "#B56576",
    "sparse": "#C0392B",
    "sparse_fa2": "#8E44AD",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(v: float, ndigits: int = 2) -> str:
    return f"{v:.{ndigits}f}"


def escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _font(size: int, weight: str = "normal") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    bold = weight.lower() in {"bold", "600", "700"}
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def _parse_float(value, default=0.0) -> float:
    if value is None:
        return default
    raw = str(value).replace("px", "").strip()
    if not raw:
        return default
    return float(raw)


def _parse_color(value: str | None, opacity: float = 1.0):
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw == "none":
        return None
    rgb = ImageColor.getrgb(raw)
    if len(rgb) == 4:
        r, g, b, a = rgb
        return (r, g, b, int(a * opacity))
    return (*rgb, int(255 * opacity))


def _anchor_x(anchor: str, width: float) -> float:
    if anchor == "middle":
        return width / 2
    if anchor == "end":
        return width
    return 0.0


def rasterize_svg(svg_path: Path, png_path: Path) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    width = int(round(_parse_float(root.attrib.get("width"), 1200)))
    height = int(round(_parse_float(root.attrib.get("height"), 800)))
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    for elem in root:
        tag = elem.tag.split("}")[-1]
        opacity = _parse_float(elem.attrib.get("opacity"), 1.0)
        if tag == "rect":
            x = _parse_float(elem.attrib.get("x"))
            y = _parse_float(elem.attrib.get("y"))
            w = _parse_float(elem.attrib.get("width"))
            h = _parse_float(elem.attrib.get("height"))
            fill = _parse_color(elem.attrib.get("fill"), opacity)
            stroke = _parse_color(elem.attrib.get("stroke"), opacity)
            stroke_width = int(round(_parse_float(elem.attrib.get("stroke-width"), 1)))
            rx = int(round(_parse_float(elem.attrib.get("rx"), 0)))
            if rx > 0 and hasattr(draw, "rounded_rectangle"):
                draw.rounded_rectangle([x, y, x + w, y + h], radius=rx, fill=fill, outline=stroke, width=stroke_width)
            else:
                draw.rectangle([x, y, x + w, y + h], fill=fill, outline=stroke, width=stroke_width)
        elif tag == "line":
            x1 = _parse_float(elem.attrib.get("x1"))
            y1 = _parse_float(elem.attrib.get("y1"))
            x2 = _parse_float(elem.attrib.get("x2"))
            y2 = _parse_float(elem.attrib.get("y2"))
            stroke = _parse_color(elem.attrib.get("stroke"), opacity) or (0, 0, 0, 255)
            stroke_width = int(round(_parse_float(elem.attrib.get("stroke-width"), 1)))
            draw.line([x1, y1, x2, y2], fill=stroke, width=stroke_width)
        elif tag == "polyline":
            points = []
            for pair in elem.attrib.get("points", "").split():
                if "," not in pair:
                    continue
                px, py = pair.split(",", 1)
                points.append((_parse_float(px), _parse_float(py)))
            stroke = _parse_color(elem.attrib.get("stroke"), opacity) or (0, 0, 0, 255)
            stroke_width = int(round(_parse_float(elem.attrib.get("stroke-width"), 1)))
            if len(points) >= 2:
                draw.line(points, fill=stroke, width=stroke_width, joint="curve")
        elif tag == "circle":
            cx = _parse_float(elem.attrib.get("cx"))
            cy = _parse_float(elem.attrib.get("cy"))
            r = _parse_float(elem.attrib.get("r"))
            fill = _parse_color(elem.attrib.get("fill"), opacity)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        elif tag == "text":
            text = elem.text or ""
            size = int(round(_parse_float(elem.attrib.get("font-size"), 16)))
            fill = _parse_color(elem.attrib.get("fill"), opacity) or (0, 0, 0, 255)
            weight = elem.attrib.get("font-weight", "normal")
            anchor = elem.attrib.get("text-anchor", "start")
            font = _font(size, weight)
            x = _parse_float(elem.attrib.get("x"))
            y = _parse_float(elem.attrib.get("y"))
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            ax = _anchor_x(anchor, tw)
            ty = y - th * 0.82
            tx = x - ax
            transform = elem.attrib.get("transform")
            if transform and transform.startswith("rotate("):
                payload = transform[len("rotate(") : -1]
                parts = [p for p in payload.split() if p]
                angle = float(parts[0])
                cx = float(parts[1]) if len(parts) > 1 else x
                cy = float(parts[2]) if len(parts) > 2 else y
                pad = max(20, size)
                canvas_w = int(tw + pad * 4)
                canvas_h = int(th + pad * 4)
                layer = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
                layer_draw = ImageDraw.Draw(layer, "RGBA")
                base_x = canvas_w / 2 - ax
                base_y = canvas_h / 2 - th * 0.82
                layer_draw.text((base_x, base_y), text, font=font, fill=fill)
                rotated = layer.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
                image.alpha_composite(rotated, (int(round(cx - rotated.width / 2)), int(round(cy - rotated.height / 2))))
            else:
                draw.text((tx, ty), text, font=font, fill=fill)

    image.save(png_path)


def rasterize_figure_bundle() -> None:
    for svg_path in sorted(FIG_DIR.glob("*.svg")):
        rasterize_svg(svg_path, svg_path.with_suffix(".png"))


class SVG:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts: list[str] = []

    def rect(self, x, y, w, h, fill="none", stroke="none", stroke_width=1, rx=0, opacity=1.0):
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" rx="{rx}" opacity="{opacity}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke=COLORS["axis"], stroke_width=1, dash: str | None = None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr}/>'
        )

    def text(
        self,
        x,
        y,
        value,
        *,
        size=16,
        fill=COLORS["text"],
        weight="normal",
        anchor="start",
        family="Helvetica, Arial, sans-serif",
        rotate: float | None = None,
    ):
        transform = f' transform="rotate({rotate:.2f} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}" font-family="{family}"{transform}>{escape(value)}</text>'
        )

    def multiline_text(
        self,
        x,
        y,
        value,
        *,
        size=16,
        fill=COLORS["text"],
        weight="normal",
        anchor="start",
        line_gap=18,
    ):
        lines = str(value).replace("\\n", "\n").splitlines() or [str(value)]
        start_y = y - line_gap * (len(lines) - 1) / 2
        for i, line in enumerate(lines):
            self.text(
                x,
                start_y + i * line_gap,
                line,
                size=size,
                fill=fill,
                weight=weight,
                anchor=anchor,
            )

    def polyline(self, points: Iterable[tuple[float, float]], stroke, stroke_width=3, fill="none"):
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(
            f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def circle(self, cx, cy, r=4, fill=COLORS["text"]):
        self.parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}"/>')

    def save(self, path: Path):
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'<rect width="{self.width}" height="{self.height}" fill="white"/>\n'
            + "\n".join(self.parts)
            + "\n</svg>\n"
        )


def draw_grouped_bars(
    path: Path,
    *,
    title: str,
    subtitle: str,
    ylabel: str,
    categories: list[str],
    series: list[dict],
    value_key: str,
    y_max: float | None = None,
    y_ticks: int = 5,
    annotations: list[str] | None = None,
):
    width, height = 1240, 760
    left, right, top, bottom = 120, 60, 120, 140
    plot_w = width - left - right
    plot_h = height - top - bottom
    svg = SVG(width, height)
    svg.text(left, 52, title, size=30, weight="bold")
    svg.text(left, 84, subtitle, size=16, fill="#4B5563")

    if y_max is None:
        y_max = max(max(item[value_key] for item in s["values"]) for s in series) * 1.15
    y_max = max(y_max, 1e-6)

    for i in range(y_ticks + 1):
        frac = i / y_ticks
        y = top + plot_h - frac * plot_h
        value = frac * y_max
        svg.line(left, y, width - right, y, stroke=COLORS["grid"], stroke_width=1)
        svg.text(left - 12, y + 5, fmt(value), size=14, anchor="end", fill="#6B7280")

    svg.line(left, top, left, top + plot_h, stroke=COLORS["axis"], stroke_width=2)
    svg.line(left, top + plot_h, width - right, top + plot_h, stroke=COLORS["axis"], stroke_width=2)
    svg.text(28, top + plot_h / 2, ylabel, size=16, anchor="middle", rotate=-90)

    group_w = plot_w / len(categories)
    inner_w = group_w * 0.72
    bar_w = inner_w / len(series)

    for idx, cat in enumerate(categories):
        gx = left + idx * group_w + (group_w - inner_w) / 2
        for sidx, s in enumerate(series):
            value = s["values"][idx][value_key]
            h = 0 if y_max == 0 else value / y_max * plot_h
            x = gx + sidx * bar_w
            y = top + plot_h - h
            svg.rect(x, y, bar_w - 8, h, fill=s["color"], rx=4)
            svg.text(x + (bar_w - 8) / 2, y - 10, fmt(value), size=13, anchor="middle")
        svg.multiline_text(gx + inner_w / 2, top + plot_h + 34, cat, size=15, anchor="middle", line_gap=18)
        if annotations:
            svg.text(gx + inner_w / 2, top + plot_h + 68, annotations[idx], size=13, anchor="middle", fill="#0F766E")

    legend_x = width - right - 240
    legend_y = 38
    for i, s in enumerate(series):
        yy = legend_y + i * 26
        svg.rect(legend_x, yy - 12, 18, 18, fill=s["color"], rx=3)
        svg.text(legend_x + 28, yy + 2, s["label"], size=14)

    svg.save(path)


def draw_grouped_stacked_bars(
    path: Path,
    *,
    title: str,
    subtitle: str,
    ylabel: str,
    groups: list[dict],
    stacks: list[dict],
):
    width, height = 1280, 780
    left, right, top, bottom = 120, 80, 120, 150
    plot_w = width - left - right
    plot_h = height - top - bottom
    svg = SVG(width, height)
    svg.text(left, 52, title, size=30, weight="bold")
    svg.text(left, 84, subtitle, size=16, fill="#4B5563")

    ymax = 0.0
    for g in groups:
        for bar in g["bars"]:
            ymax = max(ymax, sum(bar[s["key"]] for s in stacks))
    ymax *= 1.18

    ticks = 5
    for i in range(ticks + 1):
        frac = i / ticks
        y = top + plot_h - frac * plot_h
        val = frac * ymax
        svg.line(left, y, width - right, y, stroke=COLORS["grid"], stroke_width=1)
        svg.text(left - 12, y + 5, fmt(val), size=14, anchor="end", fill="#6B7280")

    svg.line(left, top, left, top + plot_h, stroke=COLORS["axis"], stroke_width=2)
    svg.line(left, top + plot_h, width - right, top + plot_h, stroke=COLORS["axis"], stroke_width=2)
    svg.text(30, top + plot_h / 2, ylabel, size=16, anchor="middle", rotate=-90)

    group_w = plot_w / len(groups)
    inner_w = group_w * 0.7
    bar_w = inner_w / 2

    for idx, g in enumerate(groups):
        gx = left + idx * group_w + (group_w - inner_w) / 2
        for bidx, bar in enumerate(g["bars"]):
            x = gx + bidx * bar_w
            running = 0.0
            for stack in stacks:
                value = bar[stack["key"]]
                h = value / ymax * plot_h
                y = top + plot_h - running / ymax * plot_h - h
                svg.rect(x, y, bar_w - 10, h, fill=stack["color"], rx=3)
                running += value
            svg.text(x + (bar_w - 10) / 2, top + plot_h + 22, bar["label"], size=13, anchor="middle")
            svg.text(x + (bar_w - 10) / 2, top + plot_h - running / ymax * plot_h - 10, fmt(running), size=12, anchor="middle")
        svg.multiline_text(gx + inner_w / 2, top + plot_h + 58, g["label"], size=15, anchor="middle", weight="bold", line_gap=18)
        if "note" in g:
            svg.text(gx + inner_w / 2, top + plot_h + 92, g["note"], size=13, anchor="middle", fill="#0F766E")

    legend_x = width - right - 240
    legend_y = 38
    for i, stack in enumerate(stacks):
        yy = legend_y + i * 26
        svg.rect(legend_x, yy - 12, 18, 18, fill=stack["color"], rx=3)
        svg.text(legend_x + 28, yy + 2, stack["label"], size=14)

    svg.save(path)


def draw_stacked_module_breakdown(
    path: Path,
    *,
    title: str,
    subtitle: str,
    rows: list[dict],
):
    width, height = 1180, 760
    left, right, top, bottom = 120, 80, 120, 140
    plot_w = width - left - right
    plot_h = height - top - bottom
    svg = SVG(width, height)
    svg.text(left, 52, title, size=30, weight="bold")
    svg.text(left, 84, subtitle, size=16, fill="#4B5563")

    ymax = max(row["total_ms"] for row in rows) * 1.15
    ticks = 5
    for i in range(ticks + 1):
        frac = i / ticks
        y = top + plot_h - frac * plot_h
        val = frac * ymax
        svg.line(left, y, width - right, y, stroke=COLORS["grid"], stroke_width=1)
        svg.text(left - 12, y + 5, fmt(val), size=14, anchor="end", fill="#6B7280")

    svg.line(left, top, left, top + plot_h, stroke=COLORS["axis"], stroke_width=2)
    svg.line(left, top + plot_h, width - right, top + plot_h, stroke=COLORS["axis"], stroke_width=2)
    svg.text(30, top + plot_h / 2, "ms / token", size=16, anchor="middle", rotate=-90)

    bar_w = plot_w / (len(rows) * 1.8)
    gap = bar_w * 0.8
    x = left + gap
    parts = [
        ("attn_ms", COLORS["attn"], "Attention"),
        ("mlp_ms", COLORS["mlp"], "MLP"),
        ("ln_ms", COLORS["ln"], "LayerNorm"),
        ("other_ms", COLORS["other"], "Other"),
    ]
    for row in rows:
        running = 0.0
        for key, color, _label in parts:
            value = row[key]
            h = value / ymax * plot_h
            y = top + plot_h - running / ymax * plot_h - h
            svg.rect(x, y, bar_w, h, fill=color, rx=3)
            running += value
        svg.multiline_text(x + bar_w / 2, top + plot_h + 32, row["label"], size=15, anchor="middle", line_gap=18)
        svg.text(x + bar_w / 2, top + plot_h - running / ymax * plot_h - 10, fmt(row["total_ms"]), size=13, anchor="middle")
        x += bar_w + gap

    legend_x = width - right - 220
    legend_y = 38
    for i, (_key, color, label) in enumerate(parts):
        yy = legend_y + i * 26
        svg.rect(legend_x, yy - 12, 18, 18, fill=color, rx=3)
        svg.text(legend_x + 28, yy + 2, label, size=14)

    svg.save(path)


def draw_line_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    ylabel: str,
    xlabel: str,
    x_values: list[float],
    series: list[dict],
):
    width, height = 1240, 760
    left, right, top, bottom = 110, 70, 120, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    svg = SVG(width, height)
    svg.text(left, 52, title, size=30, weight="bold")
    svg.text(left, 84, subtitle, size=16, fill="#4B5563")

    x_min, x_max = min(x_values), max(x_values)
    y_max = max(max(v for v in s["values"]) for s in series) * 1.15
    y_ticks = 5

    for i in range(y_ticks + 1):
        frac = i / y_ticks
        y = top + plot_h - frac * plot_h
        val = frac * y_max
        svg.line(left, y, width - right, y, stroke=COLORS["grid"], stroke_width=1)
        svg.text(left - 12, y + 5, fmt(val), size=14, anchor="end", fill="#6B7280")

    svg.line(left, top, left, top + plot_h, stroke=COLORS["axis"], stroke_width=2)
    svg.line(left, top + plot_h, width - right, top + plot_h, stroke=COLORS["axis"], stroke_width=2)
    svg.text(32, top + plot_h / 2, ylabel, size=16, anchor="middle", rotate=-90)
    svg.text(left + plot_w / 2, height - 28, xlabel, size=16, anchor="middle")

    def xmap(x):
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def ymap(y):
        return top + plot_h - y / y_max * plot_h

    for x in x_values:
        xx = xmap(x)
        svg.line(xx, top + plot_h, xx, top + plot_h + 8, stroke=COLORS["axis"], stroke_width=1.5)
        svg.text(xx, top + plot_h + 30, str(int(x)), size=14, anchor="middle")

    for s in series:
        pts = [(xmap(x), ymap(y)) for x, y in zip(x_values, s["values"])]
        svg.polyline(pts, stroke=s["color"], stroke_width=3)
        for (px, py), raw in zip(pts, s["values"]):
            svg.circle(px, py, r=5, fill=s["color"])
            svg.text(px, py - 12, fmt(raw, 3), size=12, anchor="middle")

    legend_x = width - right - 250
    legend_y = 38
    for i, s in enumerate(series):
        yy = legend_y + i * 26
        svg.line(legend_x, yy - 3, legend_x + 22, yy - 3, stroke=s["color"], stroke_width=4)
        svg.circle(legend_x + 11, yy - 3, r=4, fill=s["color"])
        svg.text(legend_x + 32, yy + 2, s["label"], size=14)

    svg.save(path)


def draw_horizontal_bars(
    path: Path,
    *,
    title: str,
    subtitle: str,
    xlabel: str,
    rows: list[dict],
    value_key: str,
    color: str,
):
    width, height = 1280, 760
    left, right, top, bottom = 300, 80, 120, 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    svg = SVG(width, height)
    svg.text(left, 52, title, size=30, weight="bold")
    svg.text(left, 84, subtitle, size=16, fill="#4B5563")

    ymax = max(row[value_key] for row in rows) * 1.15
    ticks = 5
    for i in range(ticks + 1):
        frac = i / ticks
        x = left + frac * plot_w
        val = frac * ymax
        svg.line(x, top, x, top + plot_h, stroke=COLORS["grid"], stroke_width=1)
        svg.text(x, top + plot_h + 24, fmt(val, 1), size=13, anchor="middle", fill="#6B7280")

    svg.line(left, top, left, top + plot_h, stroke=COLORS["axis"], stroke_width=2)
    svg.line(left, top + plot_h, width - right, top + plot_h, stroke=COLORS["axis"], stroke_width=2)
    svg.text(left + plot_w / 2, height - 24, xlabel, size=16, anchor="middle")

    row_h = plot_h / len(rows)
    bar_h = row_h * 0.6
    for idx, row in enumerate(rows):
        y = top + idx * row_h + (row_h - bar_h) / 2
        w = row[value_key] / ymax * plot_w
        svg.rect(left, y, w, bar_h, fill=color, rx=4)
        svg.text(left - 12, y + bar_h / 2 + 5, row["label"], size=14, anchor="end")
        svg.text(left + w + 8, y + bar_h / 2 + 5, fmt(row[value_key], 1), size=13)

    svg.save(path)


def extract_e2e_records() -> list[dict]:
    data = load_json(RUNTIME_DIR / "tables" / "e2e_summary.json")
    records: list[dict] = []
    cfg_map = {"short": "512 / 32", "long": "2048 / 128"}
    baseline_map = {
        "static_vs_flashsvd": "StaticCache",
        "densekv_vs_flashsvd": "DenseKVCacheBaseline",
    }
    order = [
        "static_vs_flashsvd:short",
        "static_vs_flashsvd:long",
        "densekv_vs_flashsvd:short",
        "densekv_vs_flashsvd:long",
    ]
    for key in order:
        value = data[key]
        pair, cfg = key.split(":")
        baseline = baseline_map[pair]
        rec = {
            "pair_key": key,
            "baseline": baseline,
            "config": cfg_map[cfg],
            "label": f"{baseline}\n{cfg_map[cfg]}",
            "baseline_prefill_s": value["svd.prefill_time_s"]["median"],
            "baseline_decode_s": value["svd.decode_time_s"]["median"],
            "baseline_decode_ms": value["svd.decode_ms_per_token"]["median"],
            "baseline_total_s": value["svd.total_time_s"]["median"],
            "flashsvd_prefill_s": value["flashsvd.prefill_time_s"]["median"],
            "flashsvd_decode_s": value["flashsvd.decode_time_s"]["median"],
            "flashsvd_decode_ms": value["flashsvd.decode_ms_per_token"]["median"],
            "flashsvd_total_s": value["flashsvd.total_time_s"]["median"],
        }
        rec["decode_speedup_median"] = rec["baseline_decode_ms"] / rec["flashsvd_decode_ms"]
        rec["total_speedup_median"] = rec["baseline_total_s"] / rec["flashsvd_total_s"]
        records.append(rec)
    return records


def extract_graph_records() -> list[dict]:
    data = load_json(RUNTIME_DIR / "tables" / "graph_ablation_summary.json")
    rows = []
    for cfg in ("short", "long"):
        for mode in ("nograph", "split", "layer"):
            item = data[f"{cfg}:{mode}"]
            rows.append(
                {
                    "config": cfg,
                    "mode": mode,
                    "decode_ms_mean": item["decode_ms_mean"],
                    "decode_ms_median": item["decode_ms_median"],
                    "prefill_s_mean": item["prefill_s_mean"],
                    "total_s_mean": item["total_s_mean"],
                }
            )
    return rows


def extract_module_rows() -> list[dict]:
    data = load_json(RUNTIME_DIR / "tables" / "module_profiles.json")
    rows = []
    label_map = {
        "static": "StaticCache",
        "densekv": "DenseKV",
        "flashsvd_no_graph": "FlashSVD\nno-graph",
    }
    for key in ("static", "densekv", "flashsvd_no_graph"):
        item = data[key]
        rows.append(
            {
                "label": label_map[key],
                "total_ms": item["profile_total_forward_ms"],
                "attn_ms": item["attn_total"]["ms"],
                "mlp_ms": item["mlp_total"]["ms"],
                "ln_ms": item["ln1_total"]["ms"] + item["ln2_total"]["ms"],
                "other_ms": item["other"]["ms"],
            }
        )
    return rows


def extract_correctness_rows() -> list[dict]:
    data = load_json(RUNTIME_DIR / "tables" / "correctness_gold_reference_summary.json")
    systems = data["systems"]
    label_map = {
        "static_cache": "StaticCache bf16",
        "densekv_baseline": "DenseKV bf16",
        "flashsvd_prod": "FlashSVD-v1.5 bf16",
    }
    rows = []
    for key in ("static_cache", "densekv_baseline", "flashsvd_prod"):
        item = systems[key]
        rows.append(
            {
                "label": label_map[key],
                "exact_match_rate": item["exact_match_rate"],
                "mean_token_match": item["mean_token_match"],
                "first_token_match_rate": item["first_token_match_rate"],
                "mean_first_divergence_step_over_divergent": item["mean_first_divergence_step_over_divergent"] or 0.0,
            }
        )
    return rows


def extract_prompt_rows() -> list[dict]:
    data = load_json(RUNTIME_DIR / "raw" / "correctness_gold_reference_details.json")["results"]
    return [{"prompt_id": item["prompt_id"], "prompt": item["prompt"]} for item in data]


def extract_motivation_rows() -> list[dict]:
    data = load_json(MOTIVATION_DIR / "graph_fusion_summary.json")
    rows = []
    for cfg in ("short", "long"):
        for mode in ("nograph", "split", "layer"):
            item = data["modes"][cfg][mode]
            rows.append(
                {
                    "config": cfg,
                    "mode": mode,
                    "decode_ms_per_token": item["decode_ms_per_token"],
                    "cudaLaunchKernel_per_token": item["cudaLaunchKernel_count_per_token"],
                    "cudaGraphLaunch_per_token": item["cudaGraphLaunch_count_per_token"],
                    "copy_per_token": item["copy_count_per_token"],
                    "clone_per_token": item["clone_count_per_token"],
                    "to_copy_per_token": item["to_copy_count_per_token"],
                    "launch_cpu_ms_per_token": item["launch_cpu_ms_per_token"],
                    "copy_clone_cpu_ms_per_token": item["copy_clone_cpu_ms_per_token"],
                }
            )
    return rows


def extract_short_top_ops() -> list[dict]:
    data = load_json(MOTIVATION_DIR / "graph_fusion_summary.json")
    ops = data["modes"]["short"]["nograph"]["top_cpu_ops"][:10]
    rows = []
    profile_steps = data["profile_steps"]
    for item in ops:
        rows.append(
            {
                "label": item["name"],
                "count_per_token": item["count"] / profile_steps,
                "cpu_ms_per_token": item["cpu_ms"] / profile_steps,
                "self_cuda_ms_per_token": item["self_cuda_ms"] / profile_steps,
            }
        )
    return rows


def extract_attn_route_rows() -> list[dict]:
    return load_json(RUNTIME_DIR / "tables" / "attn_route_microbench.json")


def extract_attn_reconstruct_rows() -> list[dict]:
    summary = load_json(RUNTIME_DIR / "tables" / "attn_reconstruct_summary.json")
    return [
        {"label": "Exact", "ms": summary["exact_ms_mean"]},
        {"label": "Packed linear", "ms": summary["packed_linear_ms_mean"]},
        {"label": "Packed Triton", "ms": summary["packed_flat_ms_mean"]},
    ]


def extract_mlp_rows() -> list[dict]:
    summary = load_json(RUNTIME_DIR / "tables" / "mlp_backend_summary.json")
    return [
        {"label": "Baseline eager", "ms": summary["baseline_eager_ms_mean"]},
        {"label": "Prod graph", "ms": summary["prod_graph_ms_mean"]},
    ]


def build_tables() -> None:
    e2e_rows = extract_e2e_records()
    graph_rows = extract_graph_records()
    module_rows = extract_module_rows()
    correctness_rows = extract_correctness_rows()
    motivation_rows = extract_motivation_rows()
    prompt_rows = extract_prompt_rows()
    top_ops = extract_short_top_ops()
    attn_route_rows = extract_attn_route_rows()
    attn_reconstruct_rows = extract_attn_reconstruct_rows()
    mlp_rows = extract_mlp_rows()

    write_csv(
        TAB_DIR / "main_results.csv",
        e2e_rows,
        [
            "pair_key",
            "baseline",
            "config",
            "label",
            "baseline_prefill_s",
            "baseline_decode_s",
            "baseline_decode_ms",
            "baseline_total_s",
            "flashsvd_prefill_s",
            "flashsvd_decode_s",
            "flashsvd_decode_ms",
            "flashsvd_total_s",
            "decode_speedup_median",
            "total_speedup_median",
        ],
    )
    write_csv(
        TAB_DIR / "graph_ablation.csv",
        graph_rows,
        ["config", "mode", "decode_ms_mean", "decode_ms_median", "prefill_s_mean", "total_s_mean"],
    )
    write_csv(
        TAB_DIR / "module_breakdown.csv",
        module_rows,
        ["label", "total_ms", "attn_ms", "mlp_ms", "ln_ms", "other_ms"],
    )
    write_csv(
        TAB_DIR / "correctness_summary.csv",
        correctness_rows,
        [
            "label",
            "exact_match_rate",
            "mean_token_match",
            "first_token_match_rate",
            "mean_first_divergence_step_over_divergent",
        ],
    )
    write_csv(
        TAB_DIR / "motivation_counts.csv",
        motivation_rows,
        [
            "config",
            "mode",
            "decode_ms_per_token",
            "cudaLaunchKernel_per_token",
            "cudaGraphLaunch_per_token",
            "copy_per_token",
            "clone_per_token",
            "to_copy_per_token",
            "launch_cpu_ms_per_token",
            "copy_clone_cpu_ms_per_token",
        ],
    )
    write_csv(
        TAB_DIR / "correctness_prompts.csv",
        prompt_rows,
        ["prompt_id", "prompt"],
    )
    write_csv(
        TAB_DIR / "nograph_top_op_classes_short.csv",
        top_ops,
        ["label", "count_per_token", "cpu_ms_per_token", "self_cuda_ms_per_token"],
    )
    write_csv(
        TAB_DIR / "attn_route_microbench.csv",
        attn_route_rows,
        ["L", "direct_winner", "winner_ms", "flashsvd_v15_ms", "sparse_ms", "sparse_fa2_only_ms", "dense_fa2_only_ms"],
    )
    write_csv(TAB_DIR / "attn_reconstruct_summary.csv", attn_reconstruct_rows, ["label", "ms"])
    write_csv(TAB_DIR / "mlp_backend_summary.csv", mlp_rows, ["label", "ms"])


def build_figures() -> None:
    e2e_rows = extract_e2e_records()
    graph_rows = extract_graph_records()
    module_rows = extract_module_rows()
    correctness_rows = extract_correctness_rows()
    motivation_rows = extract_motivation_rows()
    top_ops = extract_short_top_ops()
    attn_route_rows = extract_attn_route_rows()
    attn_reconstruct_rows = extract_attn_reconstruct_rows()
    mlp_rows = extract_mlp_rows()

    categories = [row["label"] for row in e2e_rows]
    draw_grouped_bars(
        FIG_DIR / "fig01_main_decode_latency.svg",
        title="Main Result: End-to-End Decode Latency",
        subtitle="Median steady-state decode latency over repeated full-model runs. Lower is better.",
        ylabel="decode ms / token",
        categories=categories,
        series=[
            {"label": "Baseline", "color": COLORS["baseline"], "values": [{"value": row["baseline_decode_ms"]} for row in e2e_rows]},
            {"label": "FlashSVD-v1.5", "color": COLORS["flashsvd"], "values": [{"value": row["flashsvd_decode_ms"]} for row in e2e_rows]},
        ],
        value_key="value",
        annotations=[f"{row['decode_speedup_median']:.2f}x" for row in e2e_rows],
    )

    stage_groups = []
    for row in e2e_rows:
        stage_groups.append(
            {
                "label": row["label"],
                "note": f"{row['total_speedup_median']:.2f}x total",
                "bars": [
                    {
                        "label": "Baseline",
                        "prefill": row["baseline_prefill_s"],
                        "decode": row["baseline_decode_s"],
                    },
                    {
                        "label": "FlashSVD",
                        "prefill": row["flashsvd_prefill_s"],
                        "decode": row["flashsvd_decode_s"],
                    },
                ],
            }
        )
    draw_grouped_stacked_bars(
        FIG_DIR / "fig02_stage_breakdown.svg",
        title="Stage Breakdown: Prefill and Decode",
        subtitle="Median stage times. Each bar is a full-model run, split into prefill and decode stages.",
        ylabel="time (s)",
        groups=stage_groups,
        stacks=[
            {"key": "prefill", "label": "Prefill", "color": COLORS["prefill"]},
            {"key": "decode", "label": "Decode", "color": COLORS["decode"]},
        ],
    )

    graph_categories = ["512 / 32", "2048 / 128"]
    graph_series = []
    for mode, color, label in [
        ("nograph", COLORS["nograph"], "No graph"),
        ("split", COLORS["split"], "Split graph"),
        ("layer", COLORS["layer"], "Per-layer graph"),
    ]:
        values = []
        for cfg in ("short", "long"):
            row = next(r for r in graph_rows if r["config"] == cfg and r["mode"] == mode)
            values.append({"value": row["decode_ms_mean"]})
        graph_series.append({"label": label, "color": color, "values": values})
    layer_rows = [next(r for r in graph_rows if r["config"] == cfg and r["mode"] == "layer") for cfg in ("short", "long")]
    nograph_rows = [next(r for r in graph_rows if r["config"] == cfg and r["mode"] == "nograph") for cfg in ("short", "long")]
    annotations = [f"{n['decode_ms_mean']/l['decode_ms_mean']:.2f}x" for n, l in zip(nograph_rows, layer_rows)]
    draw_grouped_bars(
        FIG_DIR / "fig03_graph_ablation.svg",
        title="Ablation: Graph Granularity",
        subtitle="Per-layer graph dominates the runtime improvement. Lower is better.",
        ylabel="decode ms / token",
        categories=graph_categories,
        series=graph_series,
        value_key="value",
        annotations=annotations,
    )

    motivation_short = [r for r in motivation_rows if r["config"] == "short"]
    count_categories = ["cudaLaunch", "cudaGraph", "copy_", "to/_to_copy"]
    count_series = []
    for mode, color, label in [
        ("nograph", COLORS["nograph"], "No graph"),
        ("split", COLORS["split"], "Split graph"),
        ("layer", COLORS["layer"], "Per-layer graph"),
    ]:
        row = next(r for r in motivation_short if r["mode"] == mode)
        count_series.append(
            {
                "label": label,
                "color": color,
                "values": [
                    {"value": row["cudaLaunchKernel_per_token"]},
                    {"value": row["cudaGraphLaunch_per_token"]},
                    {"value": row["copy_per_token"]},
                    {"value": row["to_copy_per_token"]},
                ],
            }
        )
    draw_grouped_bars(
        FIG_DIR / "fig04_kernel_fragmentation_counts.svg",
        title="Why the Old Path Was Fragmented",
        subtitle="Per-token runtime bookkeeping on prompt_len=512. Counts are nearly unchanged at prompt_len=2048.",
        ylabel="calls / token",
        categories=count_categories,
        series=count_series,
        value_key="value",
    )

    cpu_categories = ["launch CPU", "copy/clone CPU"]
    cpu_series = []
    for mode, color, label in [
        ("nograph", COLORS["nograph"], "No graph"),
        ("split", COLORS["split"], "Split graph"),
        ("layer", COLORS["layer"], "Per-layer graph"),
    ]:
        row = next(r for r in motivation_short if r["mode"] == mode)
        cpu_series.append(
            {
                "label": label,
                "color": color,
                "values": [
                    {"value": row["launch_cpu_ms_per_token"]},
                    {"value": row["copy_clone_cpu_ms_per_token"]},
                ],
            }
        )
    draw_grouped_bars(
        FIG_DIR / "fig05_runtime_overhead_cpu.svg",
        title="CPU Overhead Around Graph Replay",
        subtitle="Most residual overhead after fusion is host launch and staging traffic.",
        ylabel="CPU ms / token",
        categories=cpu_categories,
        series=cpu_series,
        value_key="value",
    )

    draw_stacked_module_breakdown(
        FIG_DIR / "fig06_module_breakdown_nograph.svg",
        title="Module Breakdown Before Full Graph Fusion",
        subtitle="Representative decode profile at prompt_len=512, new_tokens=32. The graph path collapses this body into a replayed layer unit.",
        rows=module_rows,
    )

    corr_categories = ["Exact match", "Mean token match", "First-token match"]
    corr_series = []
    for row, color in zip(correctness_rows, [COLORS["baseline"], COLORS["densekv"], COLORS["flashsvd"]]):
        corr_series.append(
            {
                "label": row["label"],
                "color": color,
                "values": [
                    {"value": row["exact_match_rate"]},
                    {"value": row["mean_token_match"]},
                    {"value": row["first_token_match_rate"]},
                ],
            }
        )
    draw_grouped_bars(
        FIG_DIR / "fig07_correctness_audit.svg",
        title="Correctness Audit Against fp32 No-Cache Gold",
        subtitle="All cached bf16 systems drift on some prompts; FlashSVD-v1.5 is not worse than the baselines.",
        ylabel="rate",
        categories=corr_categories,
        series=corr_series,
        value_key="value",
        y_max=1.05,
    )

    xs = [row["L"] for row in attn_route_rows]
    draw_line_chart(
        FIG_DIR / "fig08_attention_route_microbench.svg",
        title="Attention Route Microbenchmark",
        subtitle="Current-token decode attention step on the real uniform-rank checkpoint. Lower is better.",
        ylabel="step latency (ms)",
        xlabel="cached sequence length",
        x_values=xs,
        series=[
            {"label": "FlashSVD-v1.5+graph", "color": COLORS["flashsvd"], "values": [row["flashsvd_v15_ms"] for row in attn_route_rows]},
            {"label": "Dense + FA2 only", "color": COLORS["dense"], "values": [row["dense_fa2_only_ms"] for row in attn_route_rows]},
            {"label": "Sparse + FA2 only", "color": COLORS["sparse_fa2"], "values": [row["sparse_fa2_only_ms"] for row in attn_route_rows]},
            {"label": "Sparse legacy", "color": COLORS["sparse"], "values": [row["sparse_ms"] for row in attn_route_rows]},
        ],
    )

    draw_grouped_bars(
        FIG_DIR / "fig09_attention_reconstruct_ablation.svg",
        title="Attention Token-Reconstruct Kernel Ablation",
        subtitle="Real checkpoint, all 32 uniform-rank layers. Lower is better.",
        ylabel="mean ms",
        categories=[row["label"] for row in attn_reconstruct_rows],
        series=[
            {"label": "Latency", "color": COLORS["flashsvd"], "values": [{"value": row["ms"]} for row in attn_reconstruct_rows]},
        ],
        value_key="value",
    )

    draw_grouped_bars(
        FIG_DIR / "fig10_mlp_backend_ablation.svg",
        title="MLP Backend Contribution",
        subtitle="Layer-wise mean token latency. The main end-to-end win is not an MLP-only story.",
        ylabel="mean ms",
        categories=[row["label"] for row in mlp_rows],
        series=[
            {"label": "Latency", "color": COLORS["mlp"], "values": [{"value": row["ms"]} for row in mlp_rows]},
        ],
        value_key="value",
    )

    draw_horizontal_bars(
        FIG_DIR / "fig11_nograph_top_op_classes.svg",
        title="Top Op Classes Before Graph Fusion",
        subtitle="Prompt_len=512, no-graph FlashSVD path. This is why the old serving path felt noisy.",
        xlabel="count / token",
        rows=top_ops,
        value_key="count_per_token",
        color=COLORS["baseline"],
    )


def build_index_files() -> None:
    readme = PAPER_DIR / "README.md"
    setup = PAPER_DIR / "EXPERIMENT_SETUP_AND_DATASETS.md"
    taxonomy = PAPER_DIR / "KERNEL_CALL_TAXONOMY.md"

    readme_text = """# Paper Results Bundle

This folder packages the current FlashSVD-v1.5 runtime study into paper-facing figures, compact tables, and setup notes.

## Figures

- [fig01_main_decode_latency.svg](./figures/fig01_main_decode_latency.svg)
- [fig02_stage_breakdown.svg](./figures/fig02_stage_breakdown.svg)
- [fig03_graph_ablation.svg](./figures/fig03_graph_ablation.svg)
- [fig04_kernel_fragmentation_counts.svg](./figures/fig04_kernel_fragmentation_counts.svg)
- [fig05_runtime_overhead_cpu.svg](./figures/fig05_runtime_overhead_cpu.svg)
- [fig06_module_breakdown_nograph.svg](./figures/fig06_module_breakdown_nograph.svg)
- [fig07_correctness_audit.svg](./figures/fig07_correctness_audit.svg)
- [fig08_attention_route_microbench.svg](./figures/fig08_attention_route_microbench.svg)
- [fig09_attention_reconstruct_ablation.svg](./figures/fig09_attention_reconstruct_ablation.svg)
- [fig10_mlp_backend_ablation.svg](./figures/fig10_mlp_backend_ablation.svg)
- [fig11_nograph_top_op_classes.svg](./figures/fig11_nograph_top_op_classes.svg)

## Tables

- [main_results.csv](./tables/main_results.csv)
- [graph_ablation.csv](./tables/graph_ablation.csv)
- [module_breakdown.csv](./tables/module_breakdown.csv)
- [correctness_summary.csv](./tables/correctness_summary.csv)
- [motivation_counts.csv](./tables/motivation_counts.csv)
- [attn_route_microbench.csv](./tables/attn_route_microbench.csv)
- [attn_reconstruct_summary.csv](./tables/attn_reconstruct_summary.csv)
- [mlp_backend_summary.csv](./tables/mlp_backend_summary.csv)
- [nograph_top_op_classes_short.csv](./tables/nograph_top_op_classes_short.csv)
- [correctness_prompts.csv](./tables/correctness_prompts.csv)

## Notes

- Experiment setup and dataset notes: [EXPERIMENT_SETUP_AND_DATASETS.md](./EXPERIMENT_SETUP_AND_DATASETS.md)
- Kernel-call explanation: [KERNEL_CALL_TAXONOMY.md](./KERNEL_CALL_TAXONOMY.md)

## Main takeaways

- The active per-layer graph runtime is the dominant systems win. It reduces launch and staging overhead enough to turn FlashSVD-v1.5 into a large end-to-end decode speedup over both `StaticCache` and `DenseKVCacheBaseline`.
- The fairness story is solid when phrased as a same-condition `bf16` serving comparison, with `fp32 no-cache` plus `fp32 StaticCache cached` used as the correctness anchor.
- The remaining bottleneck is not one giant math kernel. It is the thin-serving problem: launch count, graph boundaries, copies, and dtype/layout traffic.
"""
    setup_text = """# Experiment Setup And Datasets

## Scope

This paper bundle is derived from the existing runtime study at:

- [`results/runtime_study_2026-03-31`](/home/zs89/FlashSVD/FlashSVD-v1.5/results/runtime_study_2026-03-31)
- [`results/motivation/graph_fusion_2026-03-31`](/home/zs89/FlashSVD/FlashSVD-v1.5/results/motivation/graph_fusion_2026-03-31)

No new measurements were introduced while building `paper_results/`; this folder is a visualization and packaging layer on top of those frozen artifacts.

## Machine And Software

- Host cwd: `/home/zs89/FlashSVD`
- GPU: `A100 80GB`
- Main runtime study GPU: `CUDA_VISIBLE_DEVICES=5`
- Graph-fusion motivation study GPU: `CUDA_VISIBLE_DEVICES=7`
- Python env: `/home/zs89/miniconda3/envs/flashsvd15/bin/python`
- Python: `3.13.2`
- PyTorch: `2.7.1+cu128`
- Triton: `3.3.1`
- Transformers: `4.53.0`
- FlashAttention: `2.8.3`
- Git commit recorded in the runtime study: `c6a067b304b8a541b3f1c4f24d8bbe0ecbe21869`

## Model

- Checkpoint: `/home/zs89/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt`
- Compression family: uniform-rank `SVDLLM v1` style checkpoint
- Attention ranks: `Rq=Rk=Rv=1024` for all 32 layers
- MLP ranks: `Rgate=Rup=Rdown=1492` for all 32 layers
- Main inference dtype: `bf16`
- Batch size: `1`

## Runtime Definitions

- `StaticCache`:
  standard SVD runtime with HuggingFace-style static KV cache.
- `DenseKVCacheBaseline`:
  aligned dense-KV reference path with reference QKV reconstruct, external RoPE, and `flash_attn_with_kvcache`.
- `FlashSVD-v1.5`:
  packed rank projection + token reconstruct + internal RoPE + FA2 KV-cache decode + active per-layer CUDA graph.

Default FlashSVD-v1.5 runtime knobs used in the main study:

```bash
FLASH_SVD_DENSE_DECODE_BACKEND=packed
FLASH_SVD_DENSE_DECODE_GRAPH=1
```

Default benchmark flags:

```bash
--experimental_flash_dense_attn
--flashsvd_ffn_backend flashsvd_mlp_dual_split_prod
--mlp_cuda_graph
--mlp_cuda_graph_scope layer_tail
```

## Datasets And Inputs

### 1. Latency And Ablation Benchmarks

The end-to-end latency measurements are not tied to a natural-text corpus. The decode benchmark uses synthetic token IDs generated with `torch.randint(...)` inside [`decode_kvcache_eval`](/home/zs89/FlashSVD/FlashSVD-v1.5/utils/evaluator.py), with fixed prompt lengths and decode lengths.

Primary settings:

- `prompt_len=512, new_tokens=32`
- `prompt_len=2048, new_tokens=128`

This is appropriate for systems benchmarking because the goal is to isolate serving-path runtime cost from dataset-specific tokenization or sampling behavior.

### 2. Correctness Audit Prompt Set

The correctness audit uses 20 manually curated short prompts. The prompt list is exported in:

- [correctness_prompts.csv](./tables/correctness_prompts.csv)

Gold reference:

- `fp32 no-cache` full recomputation

Correctness anchor:

- `fp32 StaticCache cached`, which matches the no-cache gold `20/20`

Serving-path comparison:

- `bf16 StaticCache`
- `bf16 DenseKVCacheBaseline`
- `bf16 FlashSVD-v1.5`

### 3. Kernel Microbenchmarks

The attention-route and kernel microbenchmarks use the real checkpoint weights and real layer ranks, but synthetic decode-shape inputs. They are best interpreted as operator-level latency studies, not text-generation quality experiments.

## Paper-Facing Usage

- Use `DenseKVCacheBaseline` as the main aligned performance baseline.
- Use `StaticCache` as the practical baseline and `fp32 no-cache` plus `fp32 StaticCache cached` as the correctness anchor.
- Use the graph-fusion figures to motivate why runtime thinness matters even when the underlying algorithm is unchanged.
"""
    taxonomy_text = """# Kernel Call Taxonomy

This note explains why the pre-fusion decode path had so many calls.

## Important distinction

Profiler rows like `aten::linear` or `aten::to` are not all standalone custom kernels. They are operator classes that trigger kernels, launch kernels, or create staging traffic. The single strongest fragmentation signal is still `cudaLaunchKernel`.

## Main call families before per-layer graph

### 1. Projection / GEMM family

- `aten::linear`
- `aten::matmul`
- `aten::mm`

These come from current-token low-rank projections, QKV reconstruct, output projection, and MLP gate/up/down work.

### 2. Norm / activation / residual family

- `aten::pow`
- `aten::mean`
- `aten::rsqrt`
- `aten::mul`
- `aten::add`
- `aten::silu`

These come from RMSNorm, residual connections, and SwiGLU.

### 3. Data movement / dtype conversion family

- `aten::to`
- `aten::_to_copy`
- `aten::copy_`
- `cudaMemcpyAsync`

These are the ugly but real serving-path costs: bf16/fp32 casts, static-buffer writes, graph staging, and intermediate materialization.

### 4. Layout / view family

- `aten::transpose`
- `aten::reshape`
- `aten::slice`
- `aten::t`
- `aten::as_strided`
- `aten::empty_strided`

These are not necessarily dominant compute, but they are good evidence that the path is tensor-fragmented and layout-noisy.

### 5. Launch overhead family

- `cudaLaunchKernel`
- `cudaGraphLaunch`

This is the systems story. Before per-layer fusion, the decode path paid too many launches and too many graph boundaries per token.

## What the numbers say

On the motivation study (`prompt_len=512`):

- no graph:
  - `cudaLaunchKernel`: `1174 / token`
  - `copy_`: `199 / token`
  - `to/_to_copy`: `402 / token`
- split graph:
  - `cudaLaunchKernel`: `630 / token`
  - `cudaGraphLaunch`: `64 / token`
  - `copy_`: `327 / token`
  - `to/_to_copy`: `402 / token`
- per-layer graph:
  - `cudaLaunchKernel`: `54 / token`
  - `cudaGraphLaunch`: `32 / token`
  - `copy_`: `135 / token`
  - `to/_to_copy`: `82 / token`

So the old path was not failing because one kernel was slow. It was failing because the token-serving path was too fragmented.
"""
    readme.write_text(readme_text)
    setup.write_text(setup_text)
    taxonomy.write_text(taxonomy_text)


def main() -> None:
    ensure_dirs()
    build_tables()
    build_figures()
    rasterize_figure_bundle()
    build_index_files()
    print(f"Wrote paper results bundle to {PAPER_DIR}")


if __name__ == "__main__":
    main()
