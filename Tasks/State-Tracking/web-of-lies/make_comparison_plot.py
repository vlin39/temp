#!/usr/bin/env python3
"""
Render comparison figures for web_of_lies evaluation results (matplotlib).

Figures produced (PNG + SVG):
  1. web_of_lies_accuracy_by_answer  — baseline accuracy (no ablation), by answer
  2. web_of_lies_ablation_comparison — accuracy under each ablation group, per model
  3. web_of_lies_layerwise_ablation  — accuracy when one layer ablated, by layer index
  4. web_of_lies_patching_effect     — normalized patching effect by layer

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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

_ROOT = Path(__file__).resolve().parent
_OUTPUT_DIR = _ROOT / "output"
_FIGS_DIR = _ROOT.parents[2] / "Figs" / "State-Tracking"

# (HuggingFace model ID, display label, fill hex, is_hybrid)
MODEL_ORDER = [
    ("allenai/Olmo-3-7B-Instruct-SFT",       "OLMo-3 (pure)",    "#2ca02c", False),
    ("Qwen/Qwen3-8B",                         "Qwen3-8B (pure)",  "#1f7a1f", False),
    ("allenai/Olmo-Hybrid-Instruct-SFT-7B",  "OLMo-Hybrid",      "#d96bc0", True),
    ("Qwen/Qwen3.5-9B",                       "Qwen3.5-9B",       "#a83d8c", True),
]

ABLATE_GROUPS = [
    ("none",        1.00, "-"),
    ("self_attn",   0.65, "-"),
    ("linear_attn", 0.65, "--"),
]

LAYER_TYPE_COLORS = {
    "self_attn":   "#4878d0",   # blue   — MHA
    "linear_attn": "#ee854a",   # orange — SSM/linear
}


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
        if data.get("prompt_mode") != prompt_mode:
            continue
        key = (data.get("model_name", ""), data.get("ablate_group", "none"))
        out[key] = data
    return out


def load_layer_results(model_name: str, prompt_mode: str) -> list[dict]:
    slug = _model_slug(model_name)
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
# Figures
# ---------------------------------------------------------------------------

def build_accuracy_by_answer_fig(results: dict[tuple[str, str], dict]):
    """Figure 1: 3 x-groups (overall, yes, no) × 4 models."""
    groups = ["overall", "yes", "no"]
    group_labels = ["Overall", "yes-answer", "no-answer"]
    n_models = len(MODEL_ORDER)
    width = 0.8 / n_models
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (mname, label, color, is_hybrid) in enumerate(MODEL_ORDER):
        key = (mname, "none")
        if key not in results:
            continue
        data = results[key]
        vals = [
            data["overall"]["accuracy"],
            data.get("by_answer", {}).get("yes", {}).get("accuracy", 0.0),
            data.get("by_answer", {}).get("no",  {}).get("accuracy", 0.0),
        ]
        positions = x + (i - n_models / 2 + 0.5) * width
        hatch = "//" if is_hybrid else None
        bars = ax.bar(
            positions, vals, width, color=color, hatch=hatch,
            edgecolor="black", linewidth=0.6, label=label,
        )
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, fmt(v),
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("Web of Lies — Accuracy by answer label (no ablation)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def build_ablation_fig(results: dict[tuple[str, str], dict]):
    """Figure 2: 4 models × 3 ablation groups."""
    n_models = len(MODEL_ORDER)
    n_groups = len(ABLATE_GROUPS)
    width = 0.8 / n_groups
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (ag, opacity, ls) in enumerate(ABLATE_GROUPS):
        for j, (mname, label, color, is_hybrid) in enumerate(MODEL_ORDER):
            key = (mname, ag)
            v = results.get(key, {}).get("overall", {}).get("accuracy", 0.0)
            pos = x[j] + (i - n_groups / 2 + 0.5) * width
            hatch = "//" if is_hybrid else None
            ax.bar(
                pos, v, width, color=color, alpha=opacity, hatch=hatch,
                edgecolor="black", linewidth=1.2, linestyle=ls,
            )
            if key in results:
                ax.text(pos, v + 0.01, fmt(v), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in MODEL_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("Web of Lies — Accuracy under each ablation group")
    ax.grid(axis="y", alpha=0.3)

    # Legend rows: ablation groups, then model families
    ablation_handles = [
        Patch(facecolor="#888888", alpha=op, edgecolor="black",
              linestyle=ls, linewidth=1.2, label=ag)
        for ag, op, ls in ABLATE_GROUPS
    ]
    family_handles = [
        Patch(facecolor=color, hatch=("//" if is_hybrid else None),
              edgecolor="black", label=label)
        for _, label, color, is_hybrid in MODEL_ORDER
    ]
    leg1 = ax.legend(handles=ablation_handles, loc="upper right", fontsize=8,
                     title="Ablation group")
    ax.add_artist(leg1)
    ax.legend(handles=family_handles, loc="upper left", fontsize=8, title="Model")
    fig.tight_layout()
    return fig


def build_layerwise_fig(group_results: dict[tuple[str, str], dict],
                        layer_by_model: dict[str, list[dict]]):
    """Figure 3: 2×2 subplots, accuracy vs layer_idx."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for ax_idx, (mname, label, color, is_hybrid) in enumerate(MODEL_ORDER):
        ax = axes.flat[ax_idx]
        rows = layer_by_model.get(mname, [])
        if not rows:
            ax.text(0.5, 0.5, f"{label}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#777")
            ax.set_title(label)
            ax.set_xlabel("Layer index")
            if ax_idx % 2 == 0:
                ax.set_ylabel("Accuracy")
            continue

        idxs  = [r["layer_idx"] for r in rows]
        accs  = [r["overall"]["accuracy"] for r in rows]
        types = [r.get("layer_type", "unknown") for r in rows]

        if is_hybrid:
            mha_x = [i for i, t in zip(idxs, types) if t == "self_attn"]
            mha_y = [a for a, t in zip(accs, types) if t == "self_attn"]
            ssm_x = [i for i, t in zip(idxs, types) if t == "linear_attn"]
            ssm_y = [a for a, t in zip(accs, types) if t == "linear_attn"]
            ax.plot(mha_x, mha_y, "s-",  color=LAYER_TYPE_COLORS["self_attn"],
                    label="MHA layers", markersize=5)
            ax.plot(ssm_x, ssm_y, "v--", color=LAYER_TYPE_COLORS["linear_attn"],
                    label="SSM layers", markersize=5)
        else:
            ax.plot(idxs, accs, "o-", color=color, label="self_attn layers", markersize=5)

        # Reference lines from group ablation results
        if is_hybrid:
            mha_ref = group_results.get((mname, "self_attn"),   {}).get("overall", {}).get("accuracy")
            ssm_ref = group_results.get((mname, "linear_attn"), {}).get("overall", {}).get("accuracy")
            if mha_ref is not None:
                ax.axhline(mha_ref, color=LAYER_TYPE_COLORS["self_attn"],   ls=":", alpha=0.6,
                           label=f"all-MHA ablated = {mha_ref:.3f}")
            if ssm_ref is not None:
                ax.axhline(ssm_ref, color=LAYER_TYPE_COLORS["linear_attn"], ls=":", alpha=0.6,
                           label=f"all-SSM ablated = {ssm_ref:.3f}")
        else:
            ref = group_results.get((mname, "self_attn"), {}).get("overall", {}).get("accuracy")
            if ref is not None:
                ax.axhline(ref, color="#888", ls=":", alpha=0.6,
                           label=f"all-MHA ablated = {ref:.3f}")

        ax.set_title(label)
        ax.set_xlabel("Layer index")
        if ax_idx % 2 == 0:
            ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Web of Lies — Per-layer ablation accuracy", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def build_patching_fig(patching_by_model: dict[str, list[dict]]):
    """Figure 4: 2×2 subplots, normalized logit diff vs layer_idx."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for ax_idx, (mname, label, _, _) in enumerate(MODEL_ORDER):
        ax = axes.flat[ax_idx]
        rows = patching_by_model.get(mname, [])
        if not rows:
            ax.text(0.5, 0.5, f"{label}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#777")
            ax.set_title(label)
            ax.set_xlabel("Layer index")
            if ax_idx % 2 == 0:
                ax.set_ylabel("Normalized logit diff")
            continue

        idxs  = [r["layer_idx"] for r in rows]
        diffs = [r["mean_normalized_logit_diff"] for r in rows]
        types = [r.get("layer_type", "unknown") for r in rows]
        colors = [LAYER_TYPE_COLORS.get(t, "#999999") for t in types]

        ax.bar(idxs, diffs, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", ls="--", alpha=0.5)
        ax.set_title(label)
        ax.set_xlabel("Layer index")
        if ax_idx % 2 == 0:
            ax.set_ylabel("Normalized logit diff")
        ax.grid(axis="y", alpha=0.3)

    legend_elems = [
        Patch(facecolor=LAYER_TYPE_COLORS["self_attn"],   edgecolor="black", label="MHA layer"),
        Patch(facecolor=LAYER_TYPE_COLORS["linear_attn"], edgecolor="black", label="SSM/linear layer"),
    ]
    fig.legend(handles=legend_elems, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=9)
    fig.suptitle("Web of Lies — Activation patching effect by layer", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(results: dict[tuple[str, str], dict],
                  layer_by_model: dict[str, list[dict]] | None = None,
                  patching_by_model: dict[str, list[dict]] | None = None) -> str:
    lines = [
        "# Web of Lies Results Summary",
        "",
        "## Overall accuracy by ablation group",
        "",
        "| Model | none | self_attn | linear_attn |",
        "|-------|------|-----------|-------------|",
    ]
    for model_name, label, _, _ in MODEL_ORDER:
        row = []
        for ag, _, _ in ABLATE_GROUPS:
            key = (model_name, ag)
            v = results[key]["overall"]["accuracy"] if key in results else None
            row.append(fmt(v) if v is not None else "—")
        lines.append(f"| {label} | {' | '.join(row)} |")

    lines += [
        "",
        "## Accuracy by answer (no ablation)",
        "",
        "| Model | yes | no | overall |",
        "|-------|-----|----|---------|",
    ]
    for model_name, label, _, _ in MODEL_ORDER:
        key = (model_name, "none")
        if key not in results:
            lines.append(f"| {label} | — | — | — |")
            continue
        d = results[key]
        yes_acc = fmt(d.get("by_answer", {}).get("yes", {}).get("accuracy", 0.0))
        no_acc  = fmt(d.get("by_answer", {}).get("no",  {}).get("accuracy", 0.0))
        ov_acc  = fmt(d["overall"]["accuracy"])
        lines.append(f"| {label} | {yes_acc} | {no_acc} | {ov_acc} |")

    if layer_by_model:
        lines += [
            "",
            "## Layerwise ablation — top-3 most-impactful layers per model",
            "",
            "Layers ranked by accuracy drop relative to the unablated baseline.",
            "",
        ]
        for model_name, label, _, _ in MODEL_ORDER:
            rows = layer_by_model.get(model_name, [])
            if not rows:
                continue
            base = results.get((model_name, "none"), {}).get("overall", {}).get("accuracy")
            ranked = sorted(rows, key=lambda r: r["overall"]["accuracy"])[:3]
            lines.append(f"### {label}")
            if base is not None:
                lines.append(f"baseline accuracy: {fmt(base)}")
            lines.append("")
            lines.append("| layer_idx | layer_type | accuracy | drop |")
            lines.append("|-----------|------------|----------|------|")
            for r in ranked:
                acc = r["overall"]["accuracy"]
                drop = (base - acc) if base is not None else None
                lines.append(
                    f"| {r['layer_idx']} | {r.get('layer_type','?')} | {fmt(acc)} | "
                    f"{fmt(drop) if drop is not None else '—'} |"
                )
            lines.append("")

    if patching_by_model:
        lines += [
            "## Activation patching — top-3 layers by recovery",
            "",
            "Higher mean normalized logit diff → that layer's clean activations carry more answer-relevant information.",
            "",
        ]
        for model_name, label, _, _ in MODEL_ORDER:
            rows = patching_by_model.get(model_name, [])
            if not rows:
                continue
            ranked = sorted(rows, key=lambda r: -r.get("mean_normalized_logit_diff", 0.0))[:3]
            lines.append(f"### {label}")
            lines.append("")
            lines.append("| layer_idx | layer_type | norm. logit diff | patched accuracy |")
            lines.append("|-----------|------------|------------------|------------------|")
            for r in ranked:
                lines.append(
                    f"| {r['layer_idx']} | {r.get('layer_type','?')} | "
                    f"{r.get('mean_normalized_logit_diff', 0):.3f} | "
                    f"{r.get('patched_accuracy', 0):.3f} |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def save_fig(fig, base_path: Path, save_svg: bool = True) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Wrote {base_path.with_suffix('.png')}")
    if save_svg:
        fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
        print(f"Wrote {base_path.with_suffix('.svg')}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt_mode", default="chat",
                   choices=("continuation", "chat", "chat_continual"))
    p.add_argument("--skip_layerwise", action="store_true")
    p.add_argument("--skip_patching", action="store_true")
    p.add_argument("--no_svg", action="store_true",
                   help="Save only PNG, not SVG.")
    p.add_argument("--no_copy_to_figs", action="store_true",
                   help="Do not copy PNGs/SVGs to Figs/State-Tracking/.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg = not args.no_svg

    group_results = load_group_results(args.prompt_mode)
    print(f"Loaded {len(group_results)} group-ablation result(s) for prompt_mode={args.prompt_mode!r}.")

    save_fig(build_accuracy_by_answer_fig(group_results),
             _OUTPUT_DIR / "web_of_lies_accuracy_by_answer", save_svg)
    save_fig(build_ablation_fig(group_results),
             _OUTPUT_DIR / "web_of_lies_ablation_comparison", save_svg)

    layer_by_model: dict[str, list[dict]] = {}
    if not args.skip_layerwise:
        layer_by_model = {
            mname: load_layer_results(mname, args.prompt_mode)
            for mname, _, _, _ in MODEL_ORDER
        }
        save_fig(build_layerwise_fig(group_results, layer_by_model),
                 _OUTPUT_DIR / "web_of_lies_layerwise_ablation", save_svg)

    patching_by_model: dict[str, list[dict]] = {}
    if not args.skip_patching:
        patching_by_model = {
            mname: load_patching_results(mname, args.prompt_mode)
            for mname, _, _, _ in MODEL_ORDER
        }
        save_fig(build_patching_fig(patching_by_model),
                 _OUTPUT_DIR / "web_of_lies_patching_effect", save_svg)

    summary_md = build_summary(group_results, layer_by_model, patching_by_model)
    out_md = _OUTPUT_DIR / "web_of_lies_summary.md"
    out_md.write_text(summary_md, encoding="utf-8")
    print(f"Wrote {out_md}")

    if not args.no_copy_to_figs:
        _FIGS_DIR.mkdir(parents=True, exist_ok=True)
        figure_bases = [
            (_OUTPUT_DIR / "web_of_lies_accuracy_by_answer",   _FIGS_DIR / "fig_web_of_lies_accuracy"),
            (_OUTPUT_DIR / "web_of_lies_ablation_comparison",  _FIGS_DIR / "fig_web_of_lies_ablation"),
        ]
        if not args.skip_layerwise:
            figure_bases.append(
                (_OUTPUT_DIR / "web_of_lies_layerwise_ablation",
                 _FIGS_DIR / "fig_web_of_lies_layerwise"),
            )
        if not args.skip_patching:
            figure_bases.append(
                (_OUTPUT_DIR / "web_of_lies_patching_effect",
                 _FIGS_DIR / "fig_web_of_lies_patching"),
            )
        for src_base, dst_base in figure_bases:
            for ext in (".png", ".svg"):
                src = src_base.with_suffix(ext)
                if src.exists():
                    shutil.copy2(src, dst_base.with_suffix(ext))


if __name__ == "__main__":
    main()
