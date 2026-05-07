#!/usr/bin/env python3
"""
Render comparison figures for dyck_languages evaluation results.

Figures produced:
  1. dyck_languages_accuracy_by_answer.svg   — baseline accuracy, pure vs hybrid
  2. dyck_languages_ablation_comparison.svg  — accuracy under each ablation group
  3. dyck_languages_accuracy_by_depth.svg    — accuracy vs max nesting depth (no ablation)
  4. dyck_languages_layerwise_ablation.svg   — per-layer ablation, 2×2 subplots
  5. dyck_languages_patching_effect.svg      — activation patching effect, 2×2 subplots

Usage:
    python make_comparison_plot.py --prompt_mode chat
    python make_comparison_plot.py --prompt_mode chat --skip_layerwise --skip_patching
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from xml.sax.saxutils import escape

_ROOT       = Path(__file__).resolve().parent
_OUTPUT_DIR = _ROOT / "output"
_FIGS_DIR   = _ROOT.parents[2] / "Figs" / "State-Tracking"

MODEL_ORDER = [
    ("allenai/Olmo-3-7B-Instruct-SFT",       "OLMo-3 (pure)",   "#2ca02c", False),
    ("Qwen/Qwen3-8B",                         "Qwen3-8B (pure)", "#1f7a1f", False),
    ("allenai/Olmo-Hybrid-Instruct-SFT-7B",  "OLMo-Hybrid",     "#d96bc0", True),
    ("Qwen/Qwen3.5-9B",                       "Qwen3.5-9B",      "#a83d8c", True),
]

ABLATE_GROUPS = [
    ("none",        1.00, "solid"),
    ("self_attn",   0.65, "solid"),
    ("linear_attn", 0.65, "dashed"),
]

LAYER_TYPE_COLORS = {"self_attn": "#4878d0", "linear_attn": "#ee854a"}

_STYLE = (
    '<style>'
    '.title{font:700 22px Arial,sans-serif;fill:#111;}'
    '.subtitle{font:700 14px Arial,sans-serif;fill:#222;}'
    '.axis{font:13px Arial,sans-serif;fill:#333;}'
    '.value{font:11px Arial,sans-serif;fill:#111;}'
    '.ref{font:11px Arial,sans-serif;fill:#666;}'
    '</style>'
)

_HATCH_DEFS = (
    '<defs>'
    '<pattern id="diagonalHatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    '<line x1="0" y1="0" x2="0" y2="6" stroke="#ffffff" stroke-width="2"/>'
    '</pattern>'
    '</defs>'
)


def _model_slug(model_name: str) -> str:
    slug = model_name.replace("/", "--")
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return slug.strip("._-") or "model"


def fmt(v: float) -> str:
    return f"{v:.3f}"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_group_results(prompt_mode: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for path in sorted(_OUTPUT_DIR.glob("*.json")):
        if "_layer" in path.stem or "_patching" in path.stem or "patching_pairs" in path.stem:
            continue
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("task") != "dyck_languages":
            continue
        if data.get("prompt_mode") != prompt_mode:
            continue
        key = (data.get("model_name", ""), data.get("ablate_group", "none"))
        out[key] = data
    return out


def load_layer_results(model_name: str, prompt_mode: str) -> list[dict]:
    slug    = _model_slug(model_name)
    pattern = f"{slug}_{prompt_mode}_layer*.json"
    rows: list[dict] = []
    for path in sorted(_OUTPUT_DIR.glob(pattern)):
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if "layer_idx" in data:
            rows.append(data)
    return sorted(rows, key=lambda d: d["layer_idx"])


def load_patching_results(model_name: str, prompt_mode: str) -> list[dict]:
    slug = _model_slug(model_name)
    path = _OUTPUT_DIR / f"{slug}_{prompt_mode}_patching.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("patching_results", [])


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def _gridlines(parts, ml, mr, plot_bottom, plot_h, max_y=1.0, n=6):
    for tick in range(n):
        y_val = tick * (max_y / (n - 1))
        y     = plot_bottom - (y_val / max_y) * plot_h
        label = fmt(y_val) if y_val < max_y else (fmt(max_y) if max_y != 1.0 else "1.0")
        parts.append(
            f'<line x1="{ml}" y1="{y}" x2="{mr}" y2="{y}" stroke="#e1e1e1" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{ml-10}" y="{y+5}" text-anchor="end" class="axis">{label}</text>'
        )


def _axes(parts, ml, mr, mt, plot_bottom):
    parts.extend([
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{plot_bottom}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{ml}" y1="{plot_bottom}" x2="{mr}" y2="{plot_bottom}" stroke="#333" stroke-width="1.5"/>',
    ])


def _svg_header(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _STYLE,
        _HATCH_DEFS,
        f'<text x="{width/2}" y="38" text-anchor="middle" class="title">{escape(title)}</text>',
    ]


# ---------------------------------------------------------------------------
# Figure 1 — accuracy by answer (baseline, no ablation)
# ---------------------------------------------------------------------------

def build_accuracy_by_answer_svg(results: dict) -> str:
    groups       = ["overall", "yes", "no"]
    group_labels = {"overall": "Overall", "yes": "Answer = yes", "no": "Answer = no"}
    width, height = 860, 560
    ml, mr_pad, mt = 80, 30, 100
    plot_h        = 320
    plot_bottom   = mt + plot_h
    plot_width    = width - ml - mr_pad
    n_models      = len(MODEL_ORDER)
    group_w       = plot_width / len(groups)
    bar_w         = min(28, group_w / (n_models + 1))
    bar_gap       = 4

    parts = _svg_header(width, height, "Dyck Languages — Accuracy by Answer (no ablation)")

    lx, ly = ml, 58
    for i, (_, label, color, is_hybrid) in enumerate(MODEL_ORDER):
        rx = lx + i * 200
        parts.append(f'<rect x="{rx}" y="{ly}" width="16" height="12" fill="{color}"/>')
        if is_hybrid:
            parts.append(f'<rect x="{rx}" y="{ly}" width="16" height="12" fill="url(#diagonalHatch)" opacity="0.55"/>')
        parts.append(f'<text x="{rx+22}" y="{ly+11}" class="axis">{escape(label)}</text>')

    _gridlines(parts, ml, width - mr_pad, plot_bottom, plot_h)
    _axes(parts, ml, width - mr_pad, mt, plot_bottom)

    for gi, grp in enumerate(groups):
        center_x   = ml + (gi + 0.5) * group_w
        cluster_w  = n_models * bar_w + (n_models - 1) * bar_gap
        x0         = center_x - cluster_w / 2
        parts.append(
            f'<text x="{center_x}" y="{plot_bottom+26}" text-anchor="middle" class="axis">{escape(group_labels[grp])}</text>'
        )
        for mi, (model_name, _, color, is_hybrid) in enumerate(MODEL_ORDER):
            key  = (model_name, "none")
            if key not in results:
                continue
            data = results[key]
            acc  = data["overall"]["accuracy"] if grp == "overall" else data.get("by_answer", {}).get(grp, {}).get("accuracy")
            if acc is None:
                continue
            bar_h = acc * plot_h
            x     = x0 + mi * (bar_w + bar_gap)
            y     = plot_bottom - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}" opacity="0.97"/>')
            if is_hybrid:
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="url(#diagonalHatch)" opacity="0.45"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" class="value">{fmt(acc)}</text>')

    parts.extend([
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" class="axis">Answer label</text>',
        f'<text x="22" y="{mt+plot_h/2}" text-anchor="middle" transform="rotate(-90 22 {mt+plot_h/2})" class="axis">Accuracy</text>',
        '</svg>',
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Figure 2 — group ablation
# ---------------------------------------------------------------------------

def build_ablation_svg(results: dict) -> str:
    width, height = 920, 560
    ml, mr_pad, mt = 80, 30, 100
    plot_h        = 320
    plot_bottom   = mt + plot_h
    plot_width    = width - ml - mr_pad
    n_models      = len(MODEL_ORDER)
    n_abl         = len(ABLATE_GROUPS)
    group_w       = plot_width / n_models
    bar_w         = min(28, group_w / (n_abl + 1))
    bar_gap       = 4

    abl_labels  = {"none": "no ablation", "self_attn": "ablate MHA", "linear_attn": "ablate SSM"}
    abl_dash    = {"none": "solid", "self_attn": "solid", "linear_attn": "4,3"}

    parts = _svg_header(width, height, "Dyck Languages — Accuracy by Ablation Group")

    lx, ly = ml, 58
    for i, (ag, opacity, _) in enumerate(ABLATE_GROUPS):
        rx = lx + i * 200
        dash = abl_dash[ag]
        sd   = f' stroke-dasharray="{dash}"' if dash != "solid" else ""
        parts.extend([
            f'<rect x="{rx}" y="{ly}" width="16" height="12" fill="#888" opacity="{opacity}"/>',
            f'<rect x="{rx}" y="{ly}" width="16" height="12" fill="none" stroke="#333" stroke-width="1.5"{sd}/>',
            f'<text x="{rx+22}" y="{ly+11}" class="axis">{escape(abl_labels[ag])}</text>',
        ])

    _gridlines(parts, ml, width - mr_pad, plot_bottom, plot_h)
    _axes(parts, ml, width - mr_pad, mt, plot_bottom)

    for mi, (model_name, model_label, color, is_hybrid) in enumerate(MODEL_ORDER):
        center_x  = ml + (mi + 0.5) * group_w
        cluster_w = n_abl * bar_w + (n_abl - 1) * bar_gap
        x0        = center_x - cluster_w / 2
        parts.append(f'<text x="{center_x}" y="{plot_bottom+26}" text-anchor="middle" class="axis">{escape(model_label)}</text>')
        for ai, (ag, opacity, _) in enumerate(ABLATE_GROUPS):
            key = (model_name, ag)
            if key not in results:
                continue
            acc   = results[key]["overall"]["accuracy"]
            bar_h = acc * plot_h
            x     = x0 + ai * (bar_w + bar_gap)
            y     = plot_bottom - bar_h
            dash  = abl_dash[ag]
            sd    = f' stroke-dasharray="{dash}"' if dash != "solid" else ""
            parts.extend([
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}" opacity="{opacity}"/>',
            ])
            if is_hybrid:
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="url(#diagonalHatch)" opacity="0.45"/>')
            parts.extend([
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="none" stroke="#333" stroke-width="1.5"{sd}/>',
                f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" class="value">{fmt(acc)}</text>',
            ])

    parts.extend([
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" class="axis">Model</text>',
        f'<text x="22" y="{mt+plot_h/2}" text-anchor="middle" transform="rotate(-90 22 {mt+plot_h/2})" class="axis">Accuracy</text>',
        '</svg>',
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Figure 3 — accuracy by nesting depth (no ablation)
# Dyck-specific: shows how accuracy degrades as bracket nesting deepens.
# ---------------------------------------------------------------------------

def build_depth_svg(results: dict) -> str:
    # Collect all depth keys across models
    all_depths: set[int] = set()
    for model_name, _, _, _ in MODEL_ORDER:
        key  = (model_name, "none")
        data = results.get(key, {})
        for k in data.get("by_depth", {}).keys():
            all_depths.add(int(k))
    depths = sorted(all_depths)
    if not depths:
        depths = [1, 2, 3, 4, 5]

    width, height = 860, 560
    ml, mr_pad, mt = 80, 30, 100
    plot_h        = 320
    plot_bottom   = mt + plot_h
    plot_width    = width - ml - mr_pad

    parts = _svg_header(width, height, "Dyck Languages — Accuracy by Max Nesting Depth (no ablation)")

    # Legend
    lx, ly = ml, 58
    for i, (_, label, color, is_hybrid) in enumerate(MODEL_ORDER):
        rx = lx + i * 200
        parts.append(f'<rect x="{rx}" y="{ly}" width="16" height="12" fill="{color}"/>')
        if is_hybrid:
            parts.append(f'<rect x="{rx}" y="{ly}" width="16" height="12" fill="url(#diagonalHatch)" opacity="0.55"/>')
        parts.append(f'<text x="{rx+22}" y="{ly+11}" class="axis">{escape(label)}</text>')

    _gridlines(parts, ml, width - mr_pad, plot_bottom, plot_h)
    _axes(parts, ml, width - mr_pad, mt, plot_bottom)

    n_depths  = len(depths)
    group_w   = plot_width / n_depths
    n_models  = len(MODEL_ORDER)
    bar_w     = min(24, group_w / (n_models + 1))
    bar_gap   = 3

    for di, depth in enumerate(depths):
        center_x  = ml + (di + 0.5) * group_w
        cluster_w = n_models * bar_w + (n_models - 1) * bar_gap
        x0        = center_x - cluster_w / 2
        parts.append(f'<text x="{center_x}" y="{plot_bottom+26}" text-anchor="middle" class="axis">{depth}</text>')
        for mi, (model_name, _, color, is_hybrid) in enumerate(MODEL_ORDER):
            key  = (model_name, "none")
            data = results.get(key, {})
            acc  = data.get("by_depth", {}).get(str(depth), {}).get("accuracy")
            if acc is None:
                continue
            bar_h = acc * plot_h
            x     = x0 + mi * (bar_w + bar_gap)
            y     = plot_bottom - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}" opacity="0.97"/>')
            if is_hybrid:
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="url(#diagonalHatch)" opacity="0.45"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" class="value">{fmt(acc)}</text>')

    parts.extend([
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" class="axis">Max nesting depth</text>',
        f'<text x="22" y="{mt+plot_h/2}" text-anchor="middle" transform="rotate(-90 22 {mt+plot_h/2})" class="axis">Accuracy</text>',
        '</svg>',
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Figure 4 — layerwise ablation (2×2 subplots)
# ---------------------------------------------------------------------------

def _subplot_layerwise(parts, layer_rows, group_results, model_name, model_label,
                       color, is_hybrid, sx, sy, sw, sh):
    if not layer_rows:
        parts.append(f'<text x="{sx+sw/2}" y="{sy+sh/2}" text-anchor="middle" class="ref">no data</text>')
        return

    max_layer   = max(r["layer_idx"] for r in layer_rows)
    plot_bottom = sy + sh
    ml_i = sx + 10
    mr_i = sx + sw - 5

    def lx(idx): return ml_i + (idx / max(max_layer, 1)) * (mr_i - ml_i)
    def ly(acc): return plot_bottom - acc * sh

    if is_hybrid:
        for row in layer_rows:
            bw    = (mr_i - ml_i) / max(max_layer, 1)
            x0    = lx(row["layer_idx"]) - bw / 2
            bg    = "#fff7e6" if row["layer_type"] == "self_attn" else "#e8f4ff"
            parts.append(f'<rect x="{x0:.1f}" y="{sy}" width="{bw:.1f}" height="{sh}" fill="{bg}" opacity="0.7"/>')

    for tick in range(6):
        y = plot_bottom - tick * 0.2 * sh
        parts.append(f'<line x1="{ml_i}" y1="{y:.1f}" x2="{mr_i}" y2="{y:.1f}" stroke="#e1e1e1" stroke-width="0.8"/>')

    parts.extend([
        f'<line x1="{ml_i}" y1="{sy}" x2="{ml_i}" y2="{plot_bottom}" stroke="#555" stroke-width="1"/>',
        f'<line x1="{ml_i}" y1="{plot_bottom}" x2="{mr_i}" y2="{plot_bottom}" stroke="#555" stroke-width="1"/>',
    ])

    for ag, ref_color, ls in [("self_attn", "#2ca02c", "4,3"), ("linear_attn", "#d96bc0", "2,3")]:
        ref_data = group_results.get((model_name, ag))
        if ref_data:
            y_ref = ly(ref_data["overall"]["accuracy"])
            parts.append(f'<line x1="{ml_i}" y1="{y_ref:.1f}" x2="{mr_i}" y2="{y_ref:.1f}" stroke="{ref_color}" stroke-width="1" stroke-dasharray="{ls}" opacity="0.7"/>')

    if is_hybrid:
        for lt, dash in [("self_attn", ""), ("linear_attn", "4,3")]:
            for row in [r for r in layer_rows if r.get("layer_type") == lt]:
                x = lx(row["layer_idx"])
                y = ly(row["overall"]["accuracy"])
                if lt == "self_attn":
                    parts.append(f'<rect x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" fill="{color}" stroke="#333" stroke-width="0.8"/>')
                else:
                    pts = f"{x},{y+4} {x-4},{y-4} {x+4},{y-4}"
                    parts.append(f'<polygon points="{pts}" fill="{color}" opacity="0.75" stroke="#333" stroke-width="0.8"/>')
    else:
        for row in layer_rows:
            x = lx(row["layer_idx"])
            y = ly(row["overall"]["accuracy"])
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#333" stroke-width="0.8"/>')

    parts.append(f'<text x="{sx+sw/2}" y="{sy-6}" text-anchor="middle" class="subtitle">{escape(model_label)}</text>')


def build_layerwise_svg(group_results, layer_by_model) -> str:
    width, height = 960, 640
    ml, mt       = 55, 80
    pad_x, pad_y = 40, 60
    sub_w = (width - ml - 20 - pad_x) / 2
    sub_h = (height - mt - 60 - pad_y) / 2

    parts = _svg_header(width, height, "Dyck Languages — Layerwise Ablation")
    parts.append(f'<text x="16" y="{mt+sub_h}" text-anchor="middle" transform="rotate(-90 16 {mt+sub_h})" class="axis">Accuracy</text>')

    for i, (model_name, model_label, color, is_hybrid) in enumerate(MODEL_ORDER):
        col, row = i % 2, i // 2
        sx = ml + col * (sub_w + pad_x)
        sy = mt + row * (sub_h + pad_y)
        _subplot_layerwise(parts, layer_by_model.get(model_name, []), group_results,
                           model_name, model_label, color, is_hybrid, sx, sy, sub_w, sub_h)

    parts.extend([
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle" class="axis">Layer index</text>',
        '</svg>',
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Figure 5 — activation patching (2×2 subplots)
# ---------------------------------------------------------------------------

def _subplot_patching(parts, patching_rows, model_label, sx, sy, sw, sh):
    if not patching_rows:
        parts.append(f'<text x="{sx+sw/2}" y="{sy+sh/2}" text-anchor="middle" class="ref">no data</text>')
        return

    diffs   = [r["mean_logit_diff"] for r in patching_rows]
    max_abs = max(abs(d) for d in diffs) or 1.0
    y_range = max_abs * 1.15
    pb      = sy + sh
    ml_i    = sx + 12
    mr_i    = sx + sw - 5
    bar_w   = max(3.0, (mr_i - ml_i) / max(len(patching_rows), 1) * 0.7)

    def px(idx): return ml_i + (idx / max(len(patching_rows) - 1, 1)) * (mr_i - ml_i)
    def py(val): return sy + sh / 2 - (val / y_range) * (sh / 2)

    y_zero = sy + sh / 2
    parts.append(f'<line x1="{ml_i}" y1="{y_zero:.1f}" x2="{mr_i}" y2="{y_zero:.1f}" stroke="#555" stroke-width="1" stroke-dasharray="4,3"/>')

    for i, row in enumerate(patching_rows):
        x       = px(i)
        val     = row["mean_logit_diff"]
        bar_h   = abs(val / y_range) * (sh / 2)
        lt      = row.get("layer_type", "self_attn")
        bc      = LAYER_TYPE_COLORS.get(lt, "#888")
        opacity = "0.85" if val >= 0 else "0.5"
        y_top   = y_zero - bar_h if val >= 0 else y_zero
        parts.append(f'<rect x="{x-bar_w/2:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{bc}" opacity="{opacity}"/>')

    parts.extend([
        f'<line x1="{ml_i}" y1="{sy}" x2="{ml_i}" y2="{pb}" stroke="#555" stroke-width="1"/>',
        f'<line x1="{ml_i}" y1="{pb}" x2="{mr_i}" y2="{pb}" stroke="#555" stroke-width="1"/>',
        f'<text x="{sx+sw/2}" y="{sy-6}" text-anchor="middle" class="subtitle">{escape(model_label)}</text>',
    ])


def build_patching_svg(patching_by_model) -> str:
    width, height = 960, 640
    ml, mt       = 55, 80
    pad_x, pad_y = 40, 60
    sub_w = (width - ml - 20 - pad_x) / 2
    sub_h = (height - mt - 60 - pad_y) / 2

    parts = _svg_header(width, height, "Dyck Languages — Activation Patching Effect by Layer")
    parts.append(f'<text x="16" y="{mt+sub_h}" text-anchor="middle" transform="rotate(-90 16 {mt+sub_h})" class="axis">Mean logit diff</text>')

    for i, (model_name, model_label, _, _) in enumerate(MODEL_ORDER):
        col, row = i % 2, i // 2
        sx = ml + col * (sub_w + pad_x)
        sy = mt + row * (sub_h + pad_y)
        _subplot_patching(parts, patching_by_model.get(model_name, []), model_label, sx, sy, sub_w, sub_h)

    lx = ml
    for lt_label, lt_color in [("MHA (self_attn)", "#4878d0"), ("SSM (linear_attn)", "#ee854a")]:
        parts.extend([
            f'<rect x="{lx}" y="{height-38}" width="14" height="10" fill="{lt_color}"/>',
            f'<text x="{lx+18}" y="{height-29}" class="ref">{escape(lt_label)}</text>',
        ])
        lx += 180

    parts.extend([
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle" class="axis">Layer index</text>',
        '</svg>',
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary(results: dict) -> str:
    lines = [
        "# Dyck Languages Results Summary", "",
        "## Overall accuracy by ablation group", "",
        "| Model | none | self_attn | linear_attn |",
        "|-------|------|-----------|-------------|",
    ]
    for model_name, label, _, _ in MODEL_ORDER:
        vals = []
        for ag, _, _ in ABLATE_GROUPS:
            key = (model_name, ag)
            v   = results[key]["overall"]["accuracy"] if key in results else None
            vals.append(fmt(v) if v is not None else "—")
        lines.append(f"| {label} | {' | '.join(vals)} |")

    lines += ["", "## Accuracy by answer (no ablation)", "",
              "| Model | yes | no | overall |", "|-------|-----|----|---------|"]
    for model_name, label, _, _ in MODEL_ORDER:
        key = (model_name, "none")
        if key not in results:
            lines.append(f"| {label} | — | — | — |")
            continue
        d   = results[key]
        yes = fmt(d.get("by_answer", {}).get("yes", {}).get("accuracy", 0.0))
        no  = fmt(d.get("by_answer", {}).get("no",  {}).get("accuracy", 0.0))
        ov  = fmt(d["overall"]["accuracy"])
        lines.append(f"| {label} | {yes} | {no} | {ov} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt_mode", default="chat",
                   choices=("continuation", "chat", "chat_continual"))
    p.add_argument("--skip_depth",     action="store_true", help="Skip Figure 3 (by_depth).")
    p.add_argument("--skip_layerwise", action="store_true", help="Skip Figure 4 (layerwise).")
    p.add_argument("--skip_patching",  action="store_true", help="Skip Figure 5 (patching).")
    p.add_argument("--no_copy_to_figs", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    group_results = load_group_results(args.prompt_mode)
    print(f"Loaded {len(group_results)} group-ablation result(s) for prompt_mode={args.prompt_mode!r}.")

    outputs: list[tuple[Path, Path]] = []

    svg1 = build_accuracy_by_answer_svg(group_results)
    out1 = _OUTPUT_DIR / "dyck_languages_accuracy_by_answer.svg"
    out1.write_text(svg1, encoding="utf-8")
    print(f"Wrote {out1}")
    outputs.append((out1, _FIGS_DIR / "fig_dyck_languages_accuracy.svg"))

    svg2 = build_ablation_svg(group_results)
    out2 = _OUTPUT_DIR / "dyck_languages_ablation_comparison.svg"
    out2.write_text(svg2, encoding="utf-8")
    print(f"Wrote {out2}")
    outputs.append((out2, _FIGS_DIR / "fig_dyck_languages_ablation.svg"))

    if not args.skip_depth:
        svg3 = build_depth_svg(group_results)
        out3 = _OUTPUT_DIR / "dyck_languages_accuracy_by_depth.svg"
        out3.write_text(svg3, encoding="utf-8")
        print(f"Wrote {out3}")
        outputs.append((out3, _FIGS_DIR / "fig_dyck_languages_depth.svg"))

    if not args.skip_layerwise:
        layer_by_model = {m: load_layer_results(m, args.prompt_mode) for m, _, _, _ in MODEL_ORDER}
        svg4 = build_layerwise_svg(group_results, layer_by_model)
        out4 = _OUTPUT_DIR / "dyck_languages_layerwise_ablation.svg"
        out4.write_text(svg4, encoding="utf-8")
        print(f"Wrote {out4}")
        outputs.append((out4, _FIGS_DIR / "fig_dyck_languages_layerwise.svg"))

    if not args.skip_patching:
        patching_by_model = {m: load_patching_results(m, args.prompt_mode) for m, _, _, _ in MODEL_ORDER}
        svg5 = build_patching_svg(patching_by_model)
        out5 = _OUTPUT_DIR / "dyck_languages_patching_effect.svg"
        out5.write_text(svg5, encoding="utf-8")
        print(f"Wrote {out5}")
        outputs.append((out5, _FIGS_DIR / "fig_dyck_languages_patching.svg"))

    summary = build_summary(group_results)
    out_md  = _OUTPUT_DIR / "dyck_languages_summary.md"
    out_md.write_text(summary, encoding="utf-8")
    print(f"Wrote {out_md}")

    if not args.no_copy_to_figs:
        _FIGS_DIR.mkdir(parents=True, exist_ok=True)
        for src, dst in outputs:
            shutil.copy2(src, dst)
            print(f"Copied → {dst}")


if __name__ == "__main__":
    main()
