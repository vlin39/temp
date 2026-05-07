"""
Plots for entity-tracking box eval JSONs (see ``run_evaluation.py`` outputs).

Default layout: ``output/boxes{n}_exp2_max3_nops12_zero_shot/<model>_<mode>.json``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from boxes_lm_eval import exact_match, sets_match

OpTypes = Literal["any", "move_only", "no_move"]

_TASK_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_SUBDIR_TEMPLATE = "boxes{n}_exp2_max3_nops12_zero_shot"

NUMOPS_BINS = tuple(range(6))  # 0 .. 5

DEFAULT_N_BOXES_RANGE: tuple[int, ...] = tuple(range(2, 7))  # 2 .. 6

DEFAULT_COMPARISON_TRIPLETS: tuple[tuple[str, str, str], ...] = (
    (
        "allenai--Olmo-3-7B-Instruct-SFT_chat.json",
        "allenai--Olmo-Hybrid-Instruct-SFT-7B_chat.json",
        "Olmo-3 7B vs Hybrid 7B",
    ),
    (
        "Qwen--Qwen3-8B_chat.json",
        "Qwen--Qwen3.5-9B_chat.json",
        "Qwen3 8B vs Qwen3.5 9B",
    ),
    (
        "Qwen--Qwen3-32B_chat.json",
        "Qwen--Qwen3.5-27B_chat.json",
        "Qwen3 32B vs Qwen3.5 27B",
    ),
)

COLOR_TRANSFORMER = "#2ca02c"
COLOR_HYBRID = "#e377c2"
# Ablations: which submodule family was ablated (group or single-layer)
COLOR_ABL_BASELINE = "#5a5a5a"
COLOR_ABL_SELF_ATTN = "#1f77b4"  # e.g. MHA removed / identity
COLOR_ABL_LINEAR_ATTN = "#ff7f0e"  # linear / GatedDeltaNet block
COLOR_ABL_UNKNOWN = "#bcbd22"
# Trivial "always predict nothing" (same for all model runs on the same eval split)
COLOR_NOTHING_PRED = "#8e8d9a"

# Matplotlib font sizes (pt) — slightly larger defaults for readability
FONT_TICK = 11
FONT_AXIS_LABEL = 12
FONT_AX_TITLE = 12
FONT_LEGEND = 10
FONT_SUPTITLE = 13
FONT_NOTE = 11

__all__ = [
    "DEFAULT_COMPARISON_TRIPLETS",
    "DEFAULT_N_BOXES_RANGE",
    "NUMOPS_BINS",
    "OpTypes",
    "OUTPUT_SUBDIR_TEMPLATE",
    "accuracy_by_numops_from_rows",
    "by_numops_for_op_types",
    "color_for_model_path",
    "is_hybrid_model",
    "load_eval_json",
    "metric_series_for_json_path",
    "nothing_predictor_baseline_from_rows",
    "overall_metric_from_eval_data",
    "overall_metric_from_eval_json",
    "attn_kind_from_eval_json",
    "model_name_to_file_slug",
    "output_dir_for_n_boxes",
    "plot_model_ablation_comparison_6box",
    "plot_model_family_comparison_figure",
    "plot_likelihood_pair_by_numops",
    "plot_model_ablation_likelihood_6box",
    "plot_n_boxes_numops_trend_axes",
    "plot_n_boxes_numops_trend_figure",
    "plot_numops_pair_axes",
    "row_matches_op_types",
    "series_across_bins",
    "task_root",
]


def task_root() -> Path:
    """``entity-tracking-lms/`` (parent of ``src``)."""
    return _TASK_ROOT


def model_name_to_file_slug(model_name: str) -> str:
    """Match ``run_evaluation.py`` / ``default_output_json_path`` filename stem for ``model_name``."""
    slug = model_name.replace("/", "--")
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return slug.strip("._-") or "model"


def attn_kind_from_eval_json(data: dict) -> Literal["self_attn", "linear_attn", "none", "unknown"]:
    """
    For ablation runs, infer whether the **ablated** submodule family is
    ``self_attn`` or ``linear_attn``: first from ``ablate_groups`` (group
    ablation), else from the first entry of ``ablated_module_names`` (layerwise).
    """
    ag = str(data.get("ablate_groups") or "none")
    if ag == "self_attn":
        return "self_attn"
    if ag == "linear_attn":
        return "linear_attn"
    names = data.get("ablated_module_names") or []
    if not names:
        return "none"
    n0 = str(names[0])
    tail = n0.rsplit(".", 1)[-1]
    if tail == "self_attn":
        return "self_attn"
    if tail == "linear_attn":
        return "linear_attn"
    return "unknown"


def _results_subdir_path(
    *,
    n_boxes: int,
    results_subdir: str | None,
    root: Path | None,
) -> Path:
    base = root if root is not None else _TASK_ROOT
    if results_subdir is not None:
        return base / "output" / results_subdir
    return output_dir_for_n_boxes(n_boxes, root=root)


def _color_for_attn_kind(
    kind: Literal["self_attn", "linear_attn", "none", "unknown"],
) -> str:
    if kind == "self_attn":
        return COLOR_ABL_SELF_ATTN
    if kind == "linear_attn":
        return COLOR_ABL_LINEAR_ATTN
    if kind == "none":
        return COLOR_ABL_BASELINE
    return COLOR_ABL_UNKNOWN


def output_dir_for_n_boxes(n_boxes: int, root: Path | None = None) -> Path:
    base = root if root is not None else _TASK_ROOT
    return base / "output" / OUTPUT_SUBDIR_TEMPLATE.format(n=n_boxes)


def load_eval_json(path: Path | str) -> dict:
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _primary_op_legend_label(op_types: OpTypes) -> str:
    """Short label for solid lines (primary ``op_types`` filter)."""
    if op_types == "any":
        return "Any ops"
    if op_types == "move_only":
        return "Move only"
    return "Put and remove (no move)"


def _legend_label_from_filename(name: str) -> str:
    stem = Path(name).stem
    stem = stem.replace("_chat", "")
    if "--" in stem:
        return stem.split("--", 1)[-1]
    return stem


def is_hybrid_model(path_or_name: str | Path) -> bool:
    """True for Olmo-Hybrid and all Qwen3.5 checkpoints; False for pure transformer baselines."""
    name = Path(path_or_name).name
    return "Hybrid" in name or "Qwen3.5" in name


def color_for_model_path(path: Path) -> str:
    return COLOR_HYBRID if is_hybrid_model(path) else COLOR_TRANSFORMER


def row_matches_op_types(row: dict, op_types: OpTypes) -> bool:
    """Filter using per-row ``numops_by_op`` (queried box)."""
    if op_types == "any":
        return True
    nb = row.get("numops_by_op")
    if not isinstance(nb, dict):
        return False
    move = int(nb.get("move", 0))
    put = int(nb.get("put", 0))
    rem = int(nb.get("remove", 0))
    if op_types == "no_move":
        return move == 0
    if op_types == "move_only":
        return move >= 1 and put == 0 and rem == 0
    raise ValueError(f"Unknown op_types: {op_types!r}")


def _rows_for_op_types_and_numops(
    rows: list[dict],
    op_types: OpTypes,
    numops: int | None,
) -> list[dict]:
    out = [r for r in rows if row_matches_op_types(r, op_types)]
    if numops is not None:
        out = [r for r in out if int(r.get("numops", -1)) == numops]
    return out


def nothing_predictor_baseline_from_rows(
    rows: list[dict],
    op_types: OpTypes,
    metric: str,
    *,
    numops: int | None = None,
) -> float | None:
    """
    Accuracy (or rate) if the model **always** predicts the text ``"nothing"`` for the
    queried box — i.e. the **empty-set** answer in the task’s string format.

    Uses :func:`boxes_lm_eval.sets_match` for ``set_accuracy`` and
    :func:`boxes_lm_eval.exact_match` for ``exact_accuracy``, same as eval. This value
    depends only on **gold** labels (and ``op_types`` / ``numops`` filters), so it is
    identical across all model checkpoints for the same saved eval JSON.
    """
    if not rows:
        return None
    pred = "nothing"
    filtered = _rows_for_op_types_and_numops(rows, op_types, numops)
    if not filtered:
        return float("nan")
    n = len(filtered)
    if metric == "set_accuracy":
        return sum(1.0 for r in filtered if sets_match(pred, r["gold"])) / n
    if metric == "exact_accuracy":
        return sum(1.0 for r in filtered if exact_match(pred, r["gold"])) / n
    raise ValueError(metric)


def accuracy_by_numops_from_rows(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    bins = sorted({int(r["numops"]) for r in rows})
    out: dict[str, dict[str, float | int]] = {}
    for k in bins:
        sub = [r for r in rows if int(r["numops"]) == k]
        n = len(sub)
        if n == 0:
            continue
        out[str(k)] = {
            "n": n,
            "exact_accuracy": sum(1 for r in sub if r["exact"]) / n,
            "set_accuracy": sum(1 for r in sub if r["set_match"]) / n,
        }
    return out


def by_numops_for_op_types(data: dict, op_types: OpTypes) -> dict[str, dict[str, float | int]]:
    if op_types == "any":
        return dict(data.get("by_numops") or {})
    rows = data.get("rows") or []
    filtered = [r for r in rows if row_matches_op_types(r, op_types)]
    return accuracy_by_numops_from_rows(filtered)


def series_across_bins(
    by_numops: dict[str, dict[str, float | int]],
    metric: str,
    bins: tuple[int, ...] = NUMOPS_BINS,
) -> tuple[list[float], list[int | float]]:
    ys: list[float] = []
    ns: list[int | float] = []
    for b in bins:
        sk = str(b)
        cell = by_numops.get(sk)
        if cell is None:
            ys.append(float("nan"))
            ns.append(float("nan"))
        else:
            ys.append(float(cell[metric]))
            ns.append(int(cell["n"]))
    return ys, ns


def metric_series_for_json_path(
    json_path: Path, op_types: OpTypes, metric: str
) -> list[float] | None:
    """One value per numops bin; ``None`` if file missing."""
    if not json_path.is_file():
        return None
    data = load_eval_json(json_path)
    if op_types != "any" and not data.get("rows"):
        return None
    bnv = by_numops_for_op_types(data, op_types)
    ys, _ = series_across_bins(bnv, metric)
    return ys


def overall_metric_from_eval_data(
    data: dict,
    op_types: OpTypes,
    metric: str,
    *,
    numops: int | None = None,
) -> float | None:
    """
    Pooled ``metric`` over the eval split. When ``op_types == "any"`` and ``numops is
    None``, returns the stored ``overall[metric]`` if present. Otherwise (``numops``
    set, or a non-``any`` op filter) recomputes from ``rows``, optionally keeping only
    rows with ``int(numops) == numops``.
    """
    if numops is None and op_types == "any":
        ov = data.get("overall") or {}
        if metric in ov:
            return float(ov[metric])
    rows = data.get("rows") or []
    if not rows:
        return None
    filtered = _rows_for_op_types_and_numops(rows, op_types, numops)
    if not filtered:
        return float("nan")
    if metric == "set_accuracy":
        return sum(1 for r in filtered if r.get("set_match")) / len(filtered)
    if metric == "exact_accuracy":
        return sum(1 for r in filtered if r.get("exact")) / len(filtered)
    raise ValueError(metric)


def overall_metric_from_eval_json(
    json_path: Path,
    op_types: OpTypes,
    metric: str,
    *,
    numops: int | None = None,
) -> float | None:
    """
    Same as :func:`overall_metric_from_eval_data` for a path (see ``numops`` there).
    ``None`` if the JSON is missing, or for non-``any`` op filters with no ``rows``.
    """
    if not json_path.is_file():
        return None
    data = load_eval_json(json_path)
    if op_types != "any" and not data.get("rows"):
        return None
    return overall_metric_from_eval_data(data, op_types, metric, numops=numops)


def plot_numops_pair_axes(
    ax,
    path_a: Path,
    path_b: Path,
    *,
    op_types: OpTypes = "any",
    metric: str = "set_accuracy",
    title: str | None = None,
) -> None:
    da = load_eval_json(path_a)
    db = load_eval_json(path_b)
    if op_types != "any" and (not da.get("rows") or not db.get("rows")):
        ax.text(
            0.5,
            0.5,
            "Need `rows` + `numops_by_op` in JSON\nfor this op_types filter",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONT_NOTE,
        )
        ax.set_axis_off()
        return

    bna = by_numops_for_op_types(da, op_types)
    bnb = by_numops_for_op_types(db, op_types)
    ya, _ = series_across_bins(bna, metric)
    yb, _ = series_across_bins(bnb, metric)

    x = np.arange(len(NUMOPS_BINS), dtype=float)
    width = 0.36
    label_a = _legend_label_from_filename(path_a.name)
    label_b = _legend_label_from_filename(path_b.name)
    c_a = color_for_model_path(path_a)
    c_b = color_for_model_path(path_b)

    ax.bar(
        x - width / 2,
        ya,
        width,
        label=label_a,
        color=c_a,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        yb,
        width,
        label=label_b,
        color=c_b,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in NUMOPS_BINS])
    ax.set_xlabel("numops (queried box)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel(metric.replace("_", " "), fontsize=FONT_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=FONT_LEGEND, loc="best")
    if title:
        ax.set_title(title, fontsize=FONT_AX_TITLE)


def plot_model_family_comparison_figure(
    n_boxes: int = 5,
    op_types: OpTypes = "any",
    metric: str = "set_accuracy",
    *,
    root: Path | None = None,
    triplets: tuple[tuple[str, str, str], ...] = DEFAULT_COMPARISON_TRIPLETS,
    figsize: tuple[float, float] = (14, 3.8),
    dpi: int = 240,
    suptitle: str | None = None,
):
    """One row, three columns: Olmo pair, Qwen 8/9 pair, Qwen 32/27 pair."""
    out_dir = output_dir_for_n_boxes(n_boxes, root=root)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {out_dir}")

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, constrained_layout=True)
    if suptitle is None:
        suptitle = (
            f"boxes={n_boxes}  |  op_types={op_types!r}  |  metric={metric}  |  {out_dir.name}"
        )
    fig.suptitle(suptitle, fontsize=FONT_SUPTITLE)

    for ax, (fa, fb, sub) in zip(axes, triplets):
        pa = out_dir / fa
        pb = out_dir / fb
        if not pa.is_file() or not pb.is_file():
            miss = pa.name if not pa.is_file() else pb.name
            ax.text(
                0.5,
                0.5,
                f"Missing file:\n{miss}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=FONT_NOTE,
            )
            ax.set_axis_off()
            continue
        plot_numops_pair_axes(ax, pa, pb, op_types=op_types, metric=metric, title=sub)

    return fig, axes


def _likelihood_metric_keys(metric: str) -> tuple[str, str, str]:
    m = str(metric).strip().lower()
    if m in ("log_likelihood", "mean_log_likelihood"):
        return "mean_log_likelihood", "log_likelihood", "mean log likelihood"
    if m in ("probability", "mean_probability"):
        return "mean_probability", "probability", "mean probability"
    raise ValueError(
        "likelihood metric must be one of: "
        "log_likelihood | mean_log_likelihood | probability | mean_probability"
    )


def plot_likelihood_pair_by_numops(
    path_a: Path | str,
    path_b: Path | str,
    *,
    metric: str = "log_likelihood",
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8.2, 4.2),
    dpi: int = 200,
):
    """
    Compare two likelihood-eval JSONs across numops bins.

    Files should come from ``run_evaluation.py --eval_type likelihood`` and include:
    ``overall.{log_likelihood|probability}`` and
    ``by_numops[*].{mean_log_likelihood|mean_probability}``.
    """
    pa = Path(path_a)
    pb = Path(path_b)
    if not pa.is_file():
        raise FileNotFoundError(f"Missing likelihood JSON: {pa}")
    if not pb.is_file():
        raise FileNotFoundError(f"Missing likelihood JSON: {pb}")

    by_numops_key, overall_key, ylabel = _likelihood_metric_keys(metric)
    da = load_eval_json(pa)
    db = load_eval_json(pb)

    if str(da.get("eval_type", "")).lower() != "likelihood":
        raise ValueError(f"{pa.name} is not a likelihood eval JSON (eval_type != 'likelihood').")
    if str(db.get("eval_type", "")).lower() != "likelihood":
        raise ValueError(f"{pb.name} is not a likelihood eval JSON (eval_type != 'likelihood').")

    bna = da.get("by_numops") or {}
    bnb = db.get("by_numops") or {}
    x = np.arange(len(NUMOPS_BINS), dtype=float)
    ya = [float((bna.get(str(k)) or {}).get(by_numops_key, np.nan)) for k in NUMOPS_BINS]
    yb = [float((bnb.get(str(k)) or {}).get(by_numops_key, np.nan)) for k in NUMOPS_BINS]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    width = 0.36
    label_a = _legend_label_from_filename(pa.name.replace("_likelihood", ""))
    label_b = _legend_label_from_filename(pb.name.replace("_likelihood", ""))
    c_a = color_for_model_path(pa)
    c_b = color_for_model_path(pb)
    ax.bar(
        x - width / 2,
        ya,
        width,
        label=label_a,
        color=c_a,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        yb,
        width,
        label=label_b,
        color=c_b,
        edgecolor="white",
        linewidth=0.6,
    )

    ov_a = (da.get("overall") or {}).get(overall_key)
    ov_b = (db.get("overall") or {}).get(overall_key)
    if ov_a is not None:
        ax.axhline(float(ov_a), color=c_a, ls="--", lw=1.5, alpha=0.8)
    if ov_b is not None:
        ax.axhline(float(ov_b), color=c_b, ls="--", lw=1.5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in NUMOPS_BINS])
    ax.set_xlabel("numops (queried box)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=FONT_LEGEND, loc="best")
    if title is None:
        title = f"Likelihood comparison by numops ({ylabel})"
    ax.set_title(title, fontsize=FONT_AX_TITLE)
    return fig, ax


def plot_model_ablation_likelihood_6box(
    model_name: str,
    *,
    prompt_mode: str = "chat",
    n_boxes: int = 6,
    metric: str = "log_likelihood",
    results_subdir: str | None = None,
    root: Path | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 180,
    ax: plt.Axes | None = None,
):
    """
    Single-model bar chart for likelihood ablations on 6-box data.

    Expects filenames:
        {slug}_{prompt_mode}_likelihood.json
        {slug}_{prompt_mode}_ablate_self_attn_likelihood.json
        {slug}_{prompt_mode}_ablate_linear_attn_likelihood.json
        {slug}_{prompt_mode}_layer_{i}_likelihood.json
    """
    out_dir = _results_subdir_path(
        n_boxes=n_boxes, results_subdir=results_subdir, root=root
    )
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {out_dir}")

    by_numops_key, overall_key, ylabel = _likelihood_metric_keys(metric)

    slug = model_name_to_file_slug(model_name)
    base_name = f"{slug}_{prompt_mode}"
    slot: list[tuple[str, Path | None, str]] = []
    p_base = out_dir / f"{base_name}_likelihood.json"
    if p_base.is_file():
        slot.append(("Baseline", p_base, "none"))
    p_self = out_dir / f"{base_name}_ablate_self_attn_likelihood.json"
    if p_self.is_file():
        slot.append(("Abl-SelfA", p_self, "self_attn"))
    p_lin = out_dir / f"{base_name}_ablate_linear_attn_likelihood.json"
    if p_lin.is_file():
        slot.append(("Abl-LinA", p_lin, "linear_attn"))
    layer_prefix = f"{base_name}_layer_"
    layer_suf = "_likelihood.json"
    layer_items: list[tuple[int, Path]] = []
    for p in out_dir.iterdir():
        if not p.is_file() or not p.name.startswith(layer_prefix) or not p.name.endswith(
            layer_suf
        ):
            continue
        m = re.match(
            re.escape(layer_prefix) + r"(\d+)" + re.escape(layer_suf) + r"$", p.name
        )
        if m:
            layer_items.append((int(m.group(1)), p))
    layer_items.sort(key=lambda t: t[0])
    for li, p in layer_items:
        slot.append((f"L{li}", p, "layer"))

    if not slot:
        raise FileNotFoundError(
            f"No likelihood eval JSONs found under {out_dir} for model slug {base_name!r}."
        )

    nbar = len(slot)
    if ax is None:
        w = min(26.0, max(7.0, 0.4 * nbar + 2.0))
        fs = (w, 4.8) if figsize is None else figsize
        _fig, ax = plt.subplots(1, 1, figsize=fs, dpi=dpi)
    else:
        _fig = ax.figure

    xs = np.arange(nbar, dtype=float)
    heights: list[float] = []
    colors: list[str] = []
    for _lab, pth, ckind in slot:
        if pth is None or not pth.is_file():
            heights.append(float("nan"))
            colors.append(COLOR_ABL_UNKNOWN)
            continue
        data = load_eval_json(pth)
        if str(data.get("eval_type", "")).lower() != "likelihood":
            heights.append(float("nan"))
        else:
            ov = data.get("overall") or {}
            heights.append(float(ov.get(overall_key, np.nan)))
        if ckind == "none":
            colors.append(_color_for_attn_kind("none"))
        elif ckind == "layer":
            k = attn_kind_from_eval_json(data)
            if k in ("self_attn", "linear_attn"):
                colors.append(_color_for_attn_kind(k))
            else:
                colors.append(_color_for_attn_kind("unknown"))
        else:
            colors.append(_color_for_attn_kind(str(ckind)))

    ax.bar(
        xs,
        heights,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=2,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [s[0] for s in slot], rotation=60, ha="right", fontsize=FONT_TICK
    )
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    ax.set_xlabel("Condition", fontsize=FONT_AXIS_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.grid(True, axis="y", alpha=0.35)
    out_name = out_dir.name
    ax.set_title(
        f"{model_name}\n{out_name}  |  {prompt_mode}  |  likelihood ({ylabel}, 6-box)",
        fontsize=FONT_AX_TITLE,
    )

    n_by_numops: list[int] = []
    for _lab, pth, _k in slot:
        if pth is None or not pth.is_file():
            continue
        data = load_eval_json(pth)
        bn = data.get("by_numops") or {}
        n_sum = sum(int((bn.get(str(k)) or {}).get("n", 0)) for k in NUMOPS_BINS)
        if n_sum > 0:
            n_by_numops.append(n_sum)
    if n_by_numops:
        ax.text(
            0.99,
            0.98,
            f"n~{max(n_by_numops)} per condition",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=FONT_NOTE,
            color="0.35",
        )

    leg = [
        Patch(
            facecolor=COLOR_ABL_BASELINE,
            edgecolor="white",
            label="Trained model (baseline bar)",
        ),
        Patch(
            facecolor=COLOR_ABL_SELF_ATTN,
            edgecolor="white",
            label="Abl. MHA/self (group or L with self_attn)",
        ),
        Patch(
            facecolor=COLOR_ABL_LINEAR_ATTN,
            edgecolor="white",
            label="Abl. linear (group or L with linear_attn)",
        ),
    ]
    ax.legend(
        handles=leg,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.36),
        ncol=3,
        borderaxespad=0.0,
        fontsize=FONT_LEGEND,
        framealpha=0.95,
    )
    _fig.subplots_adjust(
        left=0.08,
        right=0.95,
        top=0.86,
        bottom=0.34,
    )
    return _fig, ax


def plot_n_boxes_numops_trend_axes(
    ax,
    path_a: Path,
    path_b: Path,
    n_boxes_values: tuple[int, ...],
    *,
    op_types: OpTypes = "any",
    metric: str = "set_accuracy",
    title: str | None = None,
    root: Path | None = None,
    show_no_move_overlay: bool = True,
) -> None:
    """Line plot: x = ``n_boxes``; **2 solid lines** = overall ``metric`` (``op_types``).

    Optional **2 dashed lines** (same green/pink): overall metric on **no_move** rows only.
    """
    xv = np.array(n_boxes_values, dtype=float)

    ya: list[float] = []
    yb: list[float] = []
    missing_for_filter = False
    for nb in n_boxes_values:
        da = output_dir_for_n_boxes(nb, root=root) / path_a.name
        db = output_dir_for_n_boxes(nb, root=root) / path_b.name
        va = overall_metric_from_eval_json(da, op_types, metric)
        vb = overall_metric_from_eval_json(db, op_types, metric)
        if va is None or vb is None:
            missing_for_filter = True
            ya.append(float("nan"))
            yb.append(float("nan"))
        else:
            ya.append(va)
            yb.append(vb)

    if op_types != "any" and missing_for_filter:
        ax.text(
            0.5,
            0.5,
            "Need `rows` + `numops_by_op` in JSON\nfor this op_types filter",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONT_NOTE,
        )
        ax.set_axis_off()
        return

    label_a = _legend_label_from_filename(path_a.name)
    label_b = _legend_label_from_filename(path_b.name)
    c_a = color_for_model_path(path_a)
    c_b = color_for_model_path(path_b)

    ax.plot(
        xv,
        ya,
        marker="o",
        ls="-",
        lw=2.0,
        label="_nolegend_",
        color=c_a,
    )
    ax.plot(
        xv,
        yb,
        marker="s",
        ls="-",
        lw=2.0,
        label="_nolegend_",
        color=c_b,
    )

    has_nm_overlay = bool(show_no_move_overlay and op_types != "no_move")
    if has_nm_overlay:
        y_nm_a: list[float] = []
        y_nm_b: list[float] = []
        for nb in n_boxes_values:
            da = output_dir_for_n_boxes(nb, root=root) / path_a.name
            db = output_dir_for_n_boxes(nb, root=root) / path_b.name
            va_nm = overall_metric_from_eval_json(da, "no_move", metric)
            vb_nm = overall_metric_from_eval_json(db, "no_move", metric)
            y_nm_a.append(float("nan") if va_nm is None else va_nm)
            y_nm_b.append(float("nan") if vb_nm is None else vb_nm)
        ax.plot(
            xv,
            y_nm_a,
            marker="o",
            ls="--",
            lw=1.85,
            label="_nolegend_",
            color=c_a,
        )
        ax.plot(
            xv,
            y_nm_b,
            marker="s",
            ls="--",
            lw=1.85,
            label="_nolegend_",
            color=c_b,
        )

    ax.set_xticks(list(n_boxes_values))
    ax.set_xlabel("n_boxes (# boxes in task)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel(metric.replace("_", " ") + " (overall)", fontsize=FONT_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.35)

    leg_models = ax.legend(
        handles=[
            Line2D([0], [0], color=c_a, lw=2.6, marker="o", ms=6, linestyle="-"),
            Line2D([0], [0], color=c_b, lw=2.6, marker="s", ms=6, linestyle="-"),
        ],
        labels=[label_a, label_b],
        loc="lower left",
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        title="Model",
        title_fontsize=FONT_LEGEND,
    )
    ax.add_artist(leg_models)

    neutral = "0.35"
    solid_lbl = _primary_op_legend_label(op_types)
    style_handles: list[Line2D] = [
        Line2D([0], [0], color=neutral, ls="-", lw=2.2, label=solid_lbl),
    ]
    if has_nm_overlay:
        style_handles.append(
            Line2D(
                [0],
                [0],
                color=neutral,
                ls="--",
                lw=2.0,
                label="Put/Remove",
            )
        )
    leg_styles = ax.legend(
        handles=style_handles,
        loc="lower right",
        fontsize=FONT_LEGEND,
        framealpha=0.95,
        title="Subset",
        title_fontsize=FONT_LEGEND,
    )

    if title:
        ax.set_title(title, fontsize=FONT_AX_TITLE)


def plot_n_boxes_numops_trend_figure(
    *,
    n_boxes_values: tuple[int, ...] = DEFAULT_N_BOXES_RANGE,
    op_types: OpTypes = "any",
    metric: str = "set_accuracy",
    root: Path | None = None,
    triplets: tuple[tuple[str, str, str], ...] = DEFAULT_COMPARISON_TRIPLETS,
    figsize: tuple[float, float] = (15, 4.2),
    dpi: int = 240,
    suptitle: str | None = None,
    show_no_move_overlay: bool = True,
):
    """
    Accuracy vs ``n_boxes`` (default 2..6), one **row × 3** subplots.

    Each subplot has **two solid lines** (one per model): **overall** ``metric`` for
    ``op_types`` (default ``any``). When ``show_no_move_overlay`` is True and
    ``op_types`` is not ``no_move``, adds **two dashed lines** in the same
    green/pink scheme: overall metric on **no_move**-filtered ``rows`` only.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, constrained_layout=True)
    if suptitle is None:
        rng = f"{min(n_boxes_values)}..{max(n_boxes_values)}"
        extra = ""
        if show_no_move_overlay and op_types != "no_move":
            extra = "  |  dashed = no_move"
        suptitle = (
            f"Trend vs n_boxes [{rng}]  |  op_types={op_types!r}  |  metric={metric}"
            f"{extra}"
        )
    fig.suptitle(suptitle, fontsize=FONT_SUPTITLE)

    for ax, (fa, fb, sub) in zip(axes, triplets):
        pa = Path(fa)
        pb = Path(fb)
        plot_n_boxes_numops_trend_axes(
            ax,
            pa,
            pb,
            n_boxes_values,
            op_types=op_types,
            metric=metric,
            title=sub,
            root=root,
            show_no_move_overlay=show_no_move_overlay,
        )

    return fig, axes


def plot_model_ablation_comparison_6box(
    model_name: str,
    *,
    prompt_mode: str = "chat",
    n_boxes: int = 6,
    op_types: OpTypes = "any",
    metric: str = "set_accuracy",
    numops: int | None = None,
    show_nothing_baseline: bool = True,
    results_subdir: str | None = None,
    root: Path | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 180,
    ax: plt.Axes | None = None,
):
    """
    One bar chart for a **single** model on the **6-box** dataset: optional **always-∅**
    constant baseline (:func:`nothing_predictor_baseline_from_rows`), trained baseline,
    group ablations (2 bars), and layerwise ablations (one bar per layer).

    * **Metric** comes from :func:`overall_metric_from_eval_data`. If ``numops`` is
      ``None`` (default), use pooled accuracy over all numops (saved ``overall`` when
      ``op_types="any"``). If ``numops`` is an integer, restrict to
      ``int(row["numops"]) == numops`` (requires ``rows`` in each JSON).

    * **Trivial "nothing" baseline**: shown as a **horizontal dashed line** at the rate
      from :func:`nothing_predictor_baseline_from_rows` (always predict empty / ``"nothing"``).

    Expected JSON filenames next to :func:`output_dir_for_n_boxes` (or
    ``output/<results_subdir>``)::

        {model_slug}_{mode}.json
        {model_slug}_{mode}_ablate_self_attn.json
        {model_slug}_{mode}_ablate_linear_attn.json
        {model_slug}_{mode}_layer_{i}.json  for i = 0 .. N-1
    """
    out_dir = _results_subdir_path(
        n_boxes=n_boxes, results_subdir=results_subdir, root=root
    )
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {out_dir}")

    slug = model_name_to_file_slug(model_name)
    base_name = f"{slug}_{prompt_mode}"
    # Order: baseline → group (self, linear) → layers by index
    slot: list[
        tuple[str, Path | None, str]
    ] = []  # (x tick label, path or None, color_kind: none|self_attn|linear_attn|unknown)
    p_base = out_dir / f"{base_name}.json"
    if p_base.is_file():
        slot.append(("Baseline", p_base, "none"))
    p_self = out_dir / f"{base_name}_ablate_self_attn.json"
    if p_self.is_file():
        slot.append(("Abl-SelfA", p_self, "self_attn"))
    p_lin = out_dir / f"{base_name}_ablate_linear_attn.json"
    if p_lin.is_file():
        slot.append(("Abl-LinA", p_lin, "linear_attn"))
    layer_prefix = f"{base_name}_layer_"
    layer_suf = ".json"
    layer_items: list[tuple[int, Path]] = []
    for p in out_dir.iterdir():
        if not p.is_file() or not p.name.startswith(layer_prefix) or not p.name.endswith(
            layer_suf
        ):
            continue
        m = re.match(
            re.escape(layer_prefix) + r"(\d+)" + re.escape(layer_suf) + r"$", p.name
        )
        if m:
            layer_items.append((int(m.group(1)), p))
    layer_items.sort(key=lambda t: t[0])
    for li, p in layer_items:
        slot.append((f"L{li}", p, "layer"))  # color from JSON

    if not slot:
        raise FileNotFoundError(
            f"No eval JSONs found under {out_dir} for model slug {base_name!r}."
        )

    # Rows for trivial "always nothing" baseline (any run with the same eval split is OK)
    rows_nothing: list[dict] = []
    if p_base.is_file():
        rows_nothing = load_eval_json(p_base).get("rows") or []
    if not rows_nothing:
        for _lab, pth, _k in slot:
            if pth and pth.is_file():
                rows_nothing = load_eval_json(pth).get("rows") or []
                if rows_nothing:
                    break

    nothing_y: float | None = None
    if show_nothing_baseline and rows_nothing:
        nv = nothing_predictor_baseline_from_rows(
            rows_nothing, op_types, metric, numops=numops
        )
        if nv is not None and not (isinstance(nv, float) and np.isnan(nv)):
            nothing_y = float(nv)

    nbar = len(slot)
    if ax is None:
        w = min(26.0, max(7.0, 0.4 * nbar + 2.0))
        fs = (w, 4.8) if figsize is None else figsize
        _fig, ax = plt.subplots(1, 1, figsize=fs, dpi=dpi)
    else:
        _fig = ax.figure

    xs = np.arange(nbar, dtype=float)
    heights: list[float] = []
    colors: list[str] = []
    for lab, pth, ckind in slot:
        if pth is None or not pth.is_file():
            heights.append(float("nan"))
            colors.append(COLOR_ABL_UNKNOWN)
            continue
        data = load_eval_json(pth)
        m = overall_metric_from_eval_data(
            data, op_types, metric, numops=numops
        )
        heights.append(float("nan") if m is None else float(m))
        if ckind == "none":
            colors.append(_color_for_attn_kind("none"))
        elif ckind == "layer":
            k = attn_kind_from_eval_json(data)
            if k in ("self_attn", "linear_attn"):
                colors.append(_color_for_attn_kind(k))
            else:
                colors.append(_color_for_attn_kind("unknown"))
        else:
            colors.append(_color_for_attn_kind(str(ckind)))

    if nothing_y is not None:
        ax.axhline(
            nothing_y,
            color=COLOR_NOTHING_PRED,
            ls="--",
            lw=1.9,
            zorder=0,
        )

    ax.bar(
        xs,
        heights,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=2,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [s[0] for s in slot], rotation=60, ha="right", fontsize=FONT_TICK
    )
    y_label = (metric or "").replace("_", " ") + f" ({op_types})"
    if numops is not None:
        y_label += f"  [numops={numops}]"
    ax.set_ylabel(y_label, fontsize=FONT_AXIS_LABEL)
    ax.set_xlabel("Condition", fontsize=FONT_AXIS_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.35)
    out_name = out_dir.name
    nops_t = f"  |  numops={numops} only" if numops is not None else "  |  all numops (pooled)"
    ax.set_title(
        f"{model_name}\n{out_name}  |  {prompt_mode}  |  {metric} (6-box){nops_t}",
        fontsize=FONT_AX_TITLE,
    )
    if (op_types != "any" or numops is not None) and not any(
        pth is not None
        and pth.is_file()
        and load_eval_json(pth).get("rows")
        for _lab, pth, _k in slot
    ):
        ax.text(
            0.5,
            0.96,
            "Need per-example `rows` in JSONs for this op_types / numops filter",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=FONT_NOTE,
            color="0.35",
        )

    leg: list[Line2D | Patch] = []
    if nothing_y is not None:
        leg.append(
            Line2D(
                [0],
                [0],
                color=COLOR_NOTHING_PRED,
                ls="--",
                lw=2.0,
                label="Trivial: always pred 'nothing' (empty-set)",
            )
        )
    leg.extend(
        [
            Patch(
                facecolor=COLOR_ABL_BASELINE,
                edgecolor="white",
                label="Trained model (baseline bar)",
            ),
            Patch(
                facecolor=COLOR_ABL_SELF_ATTN,
                edgecolor="white",
                label="Abl. MHA/self (group or L with self_attn)",
            ),
            Patch(
                facecolor=COLOR_ABL_LINEAR_ATTN,
                edgecolor="white",
                label="Abl. linear (group or L with linear_attn)",
            ),
        ]
    )
    n_leg = len(leg)
    ncol = max(1, n_leg)
    ax.legend(
        handles=leg,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.52),
        ncol=ncol,
        borderaxespad=0.0,
        fontsize=FONT_LEGEND,
        framealpha=0.95,
    )
    _fig.subplots_adjust(
        left=0.08,
        right=0.95,
        top=0.86,
        bottom=0.40,
    )

    return _fig, ax
