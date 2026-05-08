#!/usr/bin/env python3
"""
Render comparison figures for dyck_languages evaluation results (matplotlib).

Figures produced (PNG + SVG):
  1. dyck_languages_overall              — exact accuracy + token recall, per model (no ablation)
  2. dyck_languages_ablation_comparison  — accuracy under each ablation group, per model
  3. dyck_languages_accuracy_by_depth    — accuracy vs max nesting depth (or target_len fallback)
  4. dyck_languages_layerwise_ablation   — accuracy when one layer ablated, by layer index
  5. dyck_languages_patching_effect      — normalized patching effect by layer

Usage:
    python make_comparison_plot.py --prompt_mode chat
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

LAYER_TYPE_COLORS = {"self_attn": "#4878d0", "linear_attn": "#ee854a"}


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

def build_overall_fig(results: dict[tuple[str, str], dict]):
    metrics = [("exact_accuracy", "Exact accuracy"),
               ("token_recall",   "Token recall")]
    n_models = len(MODEL_ORDER)
    width = 0.8 / n_models
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (mname, label, color, is_hybrid) in enumerate(MODEL_ORDER):
        key = (mname, "none")
        if key not in results:
            continue
        d = results[key]["overall"]
        vals = [d.get(m, 0.0) for m, _ in metrics]
        positions = x + (i - n_models / 2 + 0.5) * width
        hatch = "//" if is_hybrid else None
        bars = ax.bar(positions, vals, width, color=color, hatch=hatch,
                      edgecolor="black", linewidth=0.6, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, fmt(v),
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Dyck Languages — Overall metrics (no ablation)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def build_ablation_fig(results: dict[tuple[str, str], dict]):
    n_models = len(MODEL_ORDER)
    n_groups = len(ABLATE_GROUPS)
    width = 0.8 / n_groups
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (ag, opacity, ls) in enumerate(ABLATE_GROUPS):
        for j, (mname, label, color, is_hybrid) in enumerate(MODEL_ORDER):
            key = (mname, ag)
            v = results.get(key, {}).get("overall", {}).get("exact_accuracy", 0.0)
            pos = x[j] + (i - n_groups / 2 + 0.5) * width
            hatch = "//" if is_hybrid else None
            ax.bar(pos, v, width, color=color, alpha=opacity, hatch=hatch,
                   edgecolor="black", linewidth=1.2, linestyle=ls)
            if key in results:
                ax.text(pos, v + 0.01, fmt(v), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in MODEL_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Exact accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("Dyck Languages — Exact-match accuracy under each ablation group")
    ax.grid(axis="y", alpha=0.3)

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


def build_depth_fig(results: dict[tuple[str, str], dict]):
    fig, ax = plt.subplots(figsize=(11, 6))

    use_depth = False
    for (_, ag), d in results.items():
        if ag == "none" and d.get("by_depth"):
            use_depth = True
            break
    bin_key = "by_depth" if use_depth else "by_target_len"
    x_label = "Max nesting depth" if use_depth else "Target length"

    all_bins: set[int] = set()
    for (_, ag), d in results.items():
        if ag != "none":
            continue
        for k in d.get(bin_key, {}).keys():
            try:
                all_bins.add(int(k))
            except Exception:
                pass
    if not all_bins:
        ax.text(0.5, 0.5, f"no {bin_key} data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return fig
    bins = sorted(all_bins)

    n_models = len(MODEL_ORDER)
    width = 0.8 / n_models
    x = np.arange(len(bins))
    for i, (mname, label, color, is_hybrid) in enumerate(MODEL_ORDER):
        key = (mname, "none")
        if key not in results:
            continue
        per_bin = results[key].get(bin_key, {})
        vals = [per_bin.get(str(b), {}).get("exact_accuracy", 0.0) for b in bins]
        positions = x + (i - n_models / 2 + 0.5) * width
        hatch = "//" if is_hybrid else None
        ax.bar(positions, vals, width, color=color, hatch=hatch,
               edgecolor="black", linewidth=0.6, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bins])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Exact accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Dyck Languages — Accuracy by {x_label.lower()}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def build_layerwise_fig(group_results, layer_by_model):
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
                ax.set_ylabel("Exact accuracy")
            continue

        idxs  = [r["layer_idx"] for r in rows]
        accs  = [r["overall"]["exact_accuracy"] for r in rows]
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

        if is_hybrid:
            mha_ref = group_results.get((mname, "self_attn"),   {}).get("overall", {}).get("exact_accuracy")
            ssm_ref = group_results.get((mname, "linear_attn"), {}).get("overall", {}).get("exact_accuracy")
            if mha_ref is not None:
                ax.axhline(mha_ref, color=LAYER_TYPE_COLORS["self_attn"],   ls=":", alpha=0.6,
                           label=f"all-MHA ablated = {mha_ref:.3f}")
            if ssm_ref is not None:
                ax.axhline(ssm_ref, color=LAYER_TYPE_COLORS["linear_attn"], ls=":", alpha=0.6,
                           label=f"all-SSM ablated = {ssm_ref:.3f}")
        else:
            ref = group_results.get((mname, "self_attn"), {}).get("overall", {}).get("exact_accuracy")
            if ref is not None:
                ax.axhline(ref, color="#888", ls=":", alpha=0.6,
                           label=f"all-MHA ablated = {ref:.3f}")

        ax.set_title(label)
        ax.set_xlabel("Layer index")
        if ax_idx % 2 == 0:
            ax.set_ylabel("Exact accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Dyck Languages — Per-layer ablation accuracy", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def build_patching_fig(patching_by_model):
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
    fig.suptitle("Dyck Languages — Activation patching effect by layer", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def build_summary(results, layer_by_model=None, patching_by_model=None) -> str:
    lines = [
        "# Dyck Languages Results Summary",
        "",
        "## Exact-match accuracy by ablation group",
        "",
        "| Model | none | self_attn | linear_attn |",
        "|-------|------|-----------|-------------|",
    ]
    for model_name, label, _, _ in MODEL_ORDER:
        row = []
        for ag, _, _ in ABLATE_GROUPS:
            key = (model_name, ag)
            v = results.get(key, {}).get("overall", {}).get("exact_accuracy")
            row.append(fmt(v) if v is not None else "—")
        lines.append(f"| {label} | {' | '.join(row)} |")

    lines += [
        "",
        "## Overall: exact / token recall (no ablation)",
        "",
        "| Model | exact | recall |",
        "|-------|-------|--------|",
    ]
    for model_name, label, _, _ in MODEL_ORDER:
        key = (model_name, "none")
        if key not in results:
            lines.append(f"| {label} | — | — |")
            continue
        d = results[key]["overall"]
        lines.append(f"| {label} | {fmt(d.get('exact_accuracy', 0))} | {fmt(d.get('token_recall', 0))} |")

    if layer_by_model:
        lines += [
            "",
            "## Layerwise ablation — top-3 most-impactful layers per model",
            "",
        ]
        for model_name, label, _, _ in MODEL_ORDER:
            rows = layer_by_model.get(model_name, [])
            if not rows:
                continue
            base = results.get((model_name, "none"), {}).get("overall", {}).get("exact_accuracy")
            ranked = sorted(rows, key=lambda r: r["overall"]["exact_accuracy"])[:3]
            lines.append(f"### {label}")
            if base is not None:
                lines.append(f"baseline exact accuracy: {fmt(base)}")
            lines.append("")
            lines.append("| layer_idx | layer_type | exact | drop |")
            lines.append("|-----------|------------|-------|------|")
            for r in ranked:
                acc = r["overall"]["exact_accuracy"]
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
    p.add_argument("--no_svg", action="store_true")
    p.add_argument("--no_copy_to_figs", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg = not args.no_svg

    group_results = load_group_results(args.prompt_mode)
    print(f"Loaded {len(group_results)} group-ablation result(s) for prompt_mode={args.prompt_mode!r}.")

    save_fig(build_overall_fig(group_results),
             _OUTPUT_DIR / "dyck_languages_overall", save_svg)
    save_fig(build_ablation_fig(group_results),
             _OUTPUT_DIR / "dyck_languages_ablation_comparison", save_svg)
    save_fig(build_depth_fig(group_results),
             _OUTPUT_DIR / "dyck_languages_accuracy_by_depth", save_svg)

    layer_by_model: dict[str, list[dict]] = {}
    if not args.skip_layerwise:
        layer_by_model = {
            mname: load_layer_results(mname, args.prompt_mode)
            for mname, _, _, _ in MODEL_ORDER
        }
        save_fig(build_layerwise_fig(group_results, layer_by_model),
                 _OUTPUT_DIR / "dyck_languages_layerwise_ablation", save_svg)

    patching_by_model: dict[str, list[dict]] = {}
    if not args.skip_patching:
        patching_by_model = {
            mname: load_patching_results(mname, args.prompt_mode)
            for mname, _, _, _ in MODEL_ORDER
        }
        save_fig(build_patching_fig(patching_by_model),
                 _OUTPUT_DIR / "dyck_languages_patching_effect", save_svg)

    summary_md = build_summary(group_results, layer_by_model, patching_by_model)
    out_md = _OUTPUT_DIR / "dyck_languages_summary.md"
    out_md.write_text(summary_md, encoding="utf-8")
    print(f"Wrote {out_md}")

    if not args.no_copy_to_figs:
        _FIGS_DIR.mkdir(parents=True, exist_ok=True)
        figure_bases = [
            (_OUTPUT_DIR / "dyck_languages_overall",              _FIGS_DIR / "fig_dyck_languages_overall"),
            (_OUTPUT_DIR / "dyck_languages_ablation_comparison",  _FIGS_DIR / "fig_dyck_languages_ablation"),
            (_OUTPUT_DIR / "dyck_languages_accuracy_by_depth",    _FIGS_DIR / "fig_dyck_languages_depth"),
        ]
        if not args.skip_layerwise:
            figure_bases.append(
                (_OUTPUT_DIR / "dyck_languages_layerwise_ablation",
                 _FIGS_DIR / "fig_dyck_languages_layerwise"),
            )
        if not args.skip_patching:
            figure_bases.append(
                (_OUTPUT_DIR / "dyck_languages_patching_effect",
                 _FIGS_DIR / "fig_dyck_languages_patching"),
            )
        for src_base, dst_base in figure_bases:
            for ext in (".png", ".svg"):
                src = src_base.with_suffix(ext)
                if src.exists():
                    shutil.copy2(src, dst_base.with_suffix(ext))


if __name__ == "__main__":
    main()
