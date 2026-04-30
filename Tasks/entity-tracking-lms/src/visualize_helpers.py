"""
Plots for entity-tracking box eval JSONs (see ``run_evaluation.py`` outputs).

Default layout: ``output/boxes{n}_exp2_max3_nops12_zero_shot/<model>_<mode>.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
    "overall_metric_from_eval_json",
    "output_dir_for_n_boxes",
    "plot_model_family_comparison_figure",
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


def overall_metric_from_eval_json(
    json_path: Path, op_types: OpTypes, metric: str
) -> float | None:
    """
    Overall accuracy pooled across all numops bins: the saved ``overall[metric]`` when
    ``op_types == "any"``, otherwise recomputed from filtered ``rows``.
    ``None`` if the JSON file is missing, or if ``op_types != "any"`` and there are no ``rows``.
    """
    if not json_path.is_file():
        return None
    data = load_eval_json(json_path)
    if op_types == "any":
        ov = data.get("overall") or {}
        if metric not in ov:
            return float("nan")
        return float(ov[metric])
    rows = data.get("rows") or []
    if not rows:
        return None
    filtered = [r for r in rows if row_matches_op_types(r, op_types)]
    if not filtered:
        return float("nan")
    if metric == "set_accuracy":
        return sum(1 for r in filtered if r.get("set_match")) / len(filtered)
    if metric == "exact_accuracy":
        return sum(1 for r in filtered if r.get("exact")) / len(filtered)
    raise ValueError(metric)


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
