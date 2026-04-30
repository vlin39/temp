#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUT_SVG = OUTPUT_DIR / "colored_objects_accuracy_by_num_objects.svg"
OUT_ARITHMETIC_SVG = OUTPUT_DIR / "colored_objects_arithmetic_accuracy_by_num_objects.svg"
OUT_SPATIAL_SVG = OUTPUT_DIR / "colored_objects_spatial_accuracy_by_num_objects.svg"
OUT_MD = OUTPUT_DIR / "colored_objects_accuracy_summary.md"

MODEL_ORDER = [
    ("allenai/Olmo-3-7B-Instruct-SFT", "Pure Transformer", "#2ca02c"),
    ("allenai/Olmo-Hybrid-Instruct-SFT-7B", "Hybrid", "#d96bc0"),
]
SUBTYPE_ORDER = ["what_color", "yes_no_color", "neither_color"]
SUBTYPE_TITLES = {
    "what_color": "What Color",
    "yes_no_color": "Yes/No Color",
    "neither_color": "Neither Color",
    "spatial": "Spatial",
    "arithmetic": "Arithmetic",
}


def load_results() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(OUTPUT_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        out.setdefault(data["subtype"], {})[data["model_name"]] = data
    return out


def fmt(v: float) -> str:
    return f"{v:.3f}"


def build_single_subtype_svg(
    results: dict[str, dict[str, dict]],
    subtype: str,
    title: str,
    *,
    width: int = 920,
    legend_width: int = 220,
    legend_offset: int = 250,
) -> str:
    height = 720
    margin_left = 90
    margin_right = 30
    margin_top = 110
    plot_height = 420
    plot_bottom = margin_top + plot_height
    plot_width = width - margin_left - margin_right
    max_y = 1.0
    bins = sorted(
        {
            int(k)
            for model_name, _, _ in MODEL_ORDER
            for k in results[subtype][model_name]["by_num_objects"].keys()
        }
    )
    group_width = plot_width / len(bins)
    bar_width = min(26, max(18, group_width * 0.32))
    bar_gap = min(10, max(4, group_width * 0.08))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        '.title{font: 700 28px Arial, sans-serif; fill:#111;}',
        '.axis{font: 14px Arial, sans-serif; fill:#333;}',
        '.value{font: 12px Arial, sans-serif; fill:#111;}',
        '</style>',
        f'<text x="{width / 2}" y="46" text-anchor="middle" class="title">{escape(title)}</text>',
    ]

    legend_x = width - legend_offset
    legend_y = 95
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="58" rx="8" fill="#fafafa" stroke="#d9d9d9"/>'
    )
    lx = legend_x + 18
    for i, (_, label, color) in enumerate(MODEL_ORDER):
        y = legend_y + 20 + i * 22
        parts.extend([
            f'<rect x="{lx}" y="{y - 11}" width="18" height="12" fill="{color}"/>',
            f'<text x="{lx + 28}" y="{y}" class="axis">{escape(label)}</text>',
        ])

    for tick in range(6):
        y_val = tick * 0.2
        y = plot_bottom - (y_val / max_y) * plot_height
        label = fmt(y_val) if y_val < 1 else "1.0"
        parts.extend([
            f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#e1e1e1" stroke-width="1"/>',
            f'<text x="{margin_left - 10}" y="{y + 5}" text-anchor="end" class="axis">{label}</text>',
        ])

    parts.extend([
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{plot_bottom}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{plot_bottom}" x2="{width - margin_right}" y2="{plot_bottom}" stroke="#333" stroke-width="1.5"/>',
    ])

    for idx, num_objects in enumerate(bins):
        center_x = margin_left + (idx + 0.5) * group_width
        cluster_width = 2 * bar_width + bar_gap
        group_x = center_x - cluster_width / 2
        parts.append(
            f'<text x="{center_x}" y="{plot_bottom + 28}" text-anchor="middle" class="axis">{num_objects}</text>'
        )
        for model_idx, (model_name, _, color) in enumerate(MODEL_ORDER):
            acc = results[subtype][model_name]["by_num_objects"][str(num_objects)]["accuracy"]
            bar_h = (acc / max_y) * plot_height
            x = group_x + model_idx * (bar_width + bar_gap)
            y = plot_bottom - bar_h
            parts.extend([
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" fill="{color}" opacity="0.97"/>',
                f'<text x="{x + bar_width / 2}" y="{y - 6}" text-anchor="middle" class="value">{fmt(acc)}</text>',
            ])

    parts.extend([
        f'<text x="{width / 2}" y="{height - 32}" text-anchor="middle" class="axis">Number of objects</text>',
        f'<text x="28" y="{height / 2}" text-anchor="middle" transform="rotate(-90 28 {height / 2})" class="axis">Accuracy</text>',
        '</svg>',
    ])
    return "\n".join(parts)


def build_svg(results: dict[str, dict[str, dict]]) -> str:
    width = 1600
    height = 720
    margin_left = 90
    margin_right = 30
    margin_top = 110
    margin_bottom = 90
    panel_gap = 35
    panel_width = (width - margin_left - margin_right - panel_gap * 2) / 3
    plot_height = 420
    plot_bottom = margin_top + plot_height
    bar_width = 26
    group_gap = 28
    group_width = 2 * bar_width + group_gap
    max_y = 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        '.title{font: 700 28px Arial, sans-serif; fill:#111;}',
        '.subtitle{font: 700 16px Arial, sans-serif; fill:#222;}',
        '.axis{font: 14px Arial, sans-serif; fill:#333;}',
        '.small{font: 12px Arial, sans-serif; fill:#444;}',
        '.value{font: 12px Arial, sans-serif; fill:#111;}',
        '</style>',
        '<text x="800" y="46" text-anchor="middle" class="title">Colored Objects: Accuracy by Number of Objects</text>',
    ]

    legend_x = width - 330
    legend_y = 95
    parts.extend([
        f'<rect x="{legend_x}" y="{legend_y}" width="280" height="58" rx="8" fill="#fafafa" stroke="#d9d9d9"/>'
    ])
    lx = legend_x + 18
    for i, (_, label, color) in enumerate(MODEL_ORDER):
        y = legend_y + 20 + i * 22
        parts.extend([
            f'<rect x="{lx}" y="{y - 11}" width="18" height="12" fill="{color}"/>',
            f'<text x="{lx + 28}" y="{y}" class="axis">{escape(label)}</text>',
        ])

    for p, subtype in enumerate(SUBTYPE_ORDER):
        panel_left = margin_left + p * (panel_width + panel_gap)
        panel_right = panel_left + panel_width
        bins = sorted(
            {
                int(k)
                for model_name, _, _ in MODEL_ORDER
                for k in results[subtype][model_name]["by_num_objects"].keys()
            }
        )
        plot_width = panel_width - 10
        total_groups_width = len(bins) * group_width
        start_x = panel_left + (plot_width - total_groups_width) / 2

        parts.extend([
            f'<text x="{panel_left + panel_width / 2}" y="{margin_top - 28}" text-anchor="middle" class="subtitle">{escape(SUBTYPE_TITLES[subtype])}</text>',
        ])

        for tick in range(6):
            y_val = tick * 0.2
            y = plot_bottom - (y_val / max_y) * plot_height
            parts.append(f'<line x1="{panel_left}" y1="{y}" x2="{panel_right}" y2="{y}" stroke="#e1e1e1" stroke-width="1"/>')
            label = fmt(y_val) if y_val < 1 else "1.0"
            parts.append(f'<text x="{panel_left - 10}" y="{y + 5}" text-anchor="end" class="axis">{label}</text>')

        parts.extend([
            f'<line x1="{panel_left}" y1="{margin_top}" x2="{panel_left}" y2="{plot_bottom}" stroke="#333" stroke-width="1.5"/>',
            f'<line x1="{panel_left}" y1="{plot_bottom}" x2="{panel_right}" y2="{plot_bottom}" stroke="#333" stroke-width="1.5"/>',
        ])

        for idx, num_objects in enumerate(bins):
            group_x = start_x + idx * group_width
            center_x = group_x + bar_width
            parts.append(
                f'<text x="{center_x}" y="{plot_bottom + 24}" text-anchor="middle" class="axis">{num_objects}</text>'
            )
            for m_idx, (model_name, _, color) in enumerate(MODEL_ORDER):
                acc = results[subtype][model_name]["by_num_objects"][str(num_objects)]["accuracy"]
                bar_h = (acc / max_y) * plot_height
                x = group_x + m_idx * bar_width
                y = plot_bottom - bar_h
                parts.extend([
                    f'<rect x="{x}" y="{y}" width="{bar_width - 2}" height="{bar_h}" fill="{color}" opacity="0.97"/>',
                    f'<text x="{x + (bar_width - 2) / 2}" y="{y - 6}" text-anchor="middle" class="value">{fmt(acc)}</text>',
                ])

    parts.extend([
        '<text x="800" y="665" text-anchor="middle" class="axis">Number of objects</text>',
        '<text x="28" y="340" text-anchor="middle" transform="rotate(-90 28 340)" class="axis">Accuracy</text>',
        '</svg>',
    ])
    return "\n".join(parts)


def build_arithmetic_svg(results: dict[str, dict[str, dict]]) -> str:
    return build_single_subtype_svg(
        results,
        "arithmetic",
        "Colored Objects: Arithmetic Accuracy by Number of Objects",
        width=1200,
        legend_width=280,
        legend_offset=330,
    )


def build_spatial_svg(results: dict[str, dict[str, dict]]) -> str:
    return build_single_subtype_svg(
        results,
        "spatial",
        "Colored Objects: Spatial Accuracy by Number of Objects",
    )


def build_summary(results: dict[str, dict[str, dict]]) -> str:
    summary_subtypes = [subtype for subtype in [*SUBTYPE_ORDER, "spatial", "arithmetic"] if subtype in results]
    lines = [
        "# Colored Objects Results Summary",
        "",
        "Accuracy by number of objects for the pure transformer and hybrid models.",
        "",
    ]
    for subtype in summary_subtypes:
        pure = results[subtype][MODEL_ORDER[0][0]]
        hybrid = results[subtype][MODEL_ORDER[1][0]]
        lines.append(f"## {SUBTYPE_TITLES[subtype]}")
        lines.append(f"- Pure: {fmt(pure['overall']['accuracy'])}")
        lines.append(f"- Hybrid: {fmt(hybrid['overall']['accuracy'])}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    results = load_results()
    svg = build_svg(results)
    arithmetic_svg = build_arithmetic_svg(results)
    spatial_svg = build_spatial_svg(results)
    summary = build_summary(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SVG.write_text(svg, encoding="utf-8")
    OUT_ARITHMETIC_SVG.write_text(arithmetic_svg, encoding="utf-8")
    OUT_SPATIAL_SVG.write_text(spatial_svg, encoding="utf-8")
    OUT_MD.write_text(summary, encoding="utf-8")
    print(f"Wrote {OUT_SVG}")
    print(f"Wrote {OUT_ARITHMETIC_SVG}")
    print(f"Wrote {OUT_SPATIAL_SVG}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
    
