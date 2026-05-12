#!/usr/bin/env python3
"""
Run dataset generation (if missing) then model evaluation from a YAML config.

Config format::

    data:
      num_boxes: 5
      expected_num_items_per_box: 2
      max_items_per_box: 3
      num_operations: 12
      num_samples: 2200   # optional; default 2200

    model:
      model_name: allenai/Olmo-3-7B-Instruct-SFT
      prompt_mode: chat   # or continuation
      split: test         # optional
      seed: 0             # optional; forwarded to run_evaluation.py
      num_test_set_per_bin: 300
      max_new_tokens: 64
      eval_type: generation  # optional: generation | likelihood
      output_json: null   # optional; override default result path
      ablate_groups: none # optional: none | self_attn | linear_attn (group-wide identity ablation)
      ablate_layer_idx: null   # optional: int, list of ints (sweep), or the string ``all_layers`` (expand to 0..N-1 per model via HF config)

**Hyperparameter sweeps:** Any value may be a YAML list instead of a scalar. Lists
are expanded with a Cartesian product within the ``data`` block and within the
``model`` block; the pipeline runs every ``data`` combo × every ``model`` combo.

Example sweep::

    data:
      num_boxes: [5, 7]
      expected_num_items_per_box: 2
      ...
    model:
      model_name: [allenai/A, allenai/B]
      prompt_mode: [chat, continuation]

Zero-shot generation is always enabled. The dataset directory under ``data/`` is
derived from the data block so runs are reproducible and namespaced.

Example::

    python scripts/run_boxes_eval_pipeline.py --config configs/boxes_pipeline.example.yaml
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_REQUIRED_DATA_KEYS = (
    "num_boxes",
    "expected_num_items_per_box",
    "max_items_per_box",
    "num_operations",
)
_REQUIRED_MODEL_KEYS = ("model_name", "prompt_mode")

_ROOT = Path(__file__).resolve().parents[1]
_GEN_SCRIPT = _ROOT / "src" / "dataset_generation" / "generate_boxes_data.py"
_EVAL_SCRIPT = _ROOT / "run_evaluation.py"


def expand_sweep(section: dict[str, Any], section_label: str) -> list[dict[str, Any]]:
    """Scalar values are fixed; each list value introduces a sweep dimension (product)."""
    if not isinstance(section, dict):
        sys.exit(f"{section_label} must be a mapping, got {type(section).__name__}")
    sweep_keys: list[str] = []
    sweep_vals: list[list[Any]] = []
    fixed: dict[str, Any] = {}
    for k, v in section.items():
        if isinstance(v, list):
            if not v:
                sys.exit(
                    f"{section_label}.{k}: empty list is invalid; use a scalar or omit the key"
                )
            sweep_keys.append(k)
            sweep_vals.append(v)
        else:
            fixed[k] = v
    if not sweep_keys:
        return [dict(fixed)]
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*sweep_vals):
        row = dict(fixed)
        for key, val in zip(sweep_keys, combo):
            row[key] = val
        out.append(row)
    return out


def dataset_dir_slug(cfg_data: dict) -> str:
    nb = int(cfg_data["num_boxes"])
    exp = int(cfg_data["expected_num_items_per_box"])
    mx = int(cfg_data["max_items_per_box"])
    nops = int(cfg_data["num_operations"])
    return f"boxes{nb}_exp{exp}_max{mx}_nops{nops}_zero_shot"


def splits_jsonl_ready(dataset_dir: Path) -> bool:
    return all((dataset_dir / f"{s}-t5.jsonl").is_file() for s in ("train", "dev", "test"))


def _is_ablate_layer_idx_all_layers(value: Any) -> bool:
    """YAML ``ablate_layer_idx: all_layers`` (case-insensitive) expands via HF ``AutoConfig``."""
    return isinstance(value, str) and value.strip().casefold() == "all_layers"


def _num_decoder_layers_from_pretrained(model_id: str) -> int:
    """Infer stack depth without loading weights (``AutoConfig``)."""
    try:
        from transformers import AutoConfig
    except ImportError as e:
        sys.exit(
            "transformers is required to resolve ablate_layer_idx=all_layers "
            f"(install transformers): {e}"
        )
    c = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    attrs = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
    for attr in attrs:
        v = getattr(c, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    for sub in ("text_config", "llm_config", "model_config", "language_config"):
        subc = getattr(c, sub, None)
        if subc is None:
            continue
        for attr in attrs:
            v = getattr(subc, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    sys.exit(
        f"Could not infer decoder layer count from config for {model_id!r} "
        "(tried num_hidden_layers / n_layer / … on root and text_config / llm_config / …). "
        "Use an explicit list of integers for ablate_layer_idx instead of all_layers."
    )


def expand_ablate_layer_idx_all_layers(
    model_variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Replace ``ablate_layer_idx: all_layers`` with one variant per layer index ``0 .. N-1``,
    using :func:`_num_decoder_layers_from_pretrained` for each variant's ``model_name``.
    """
    out: list[dict[str, Any]] = []
    for m in model_variants:
        raw = m.get("ablate_layer_idx")
        if not _is_ablate_layer_idx_all_layers(raw):
            out.append(m)
            continue
        if _normalize_ablate_groups(m) != "none":
            sys.exit(
                "model: ablate_layer_idx all_layers cannot be combined with "
                "non-none ablate_groups."
            )
        mn = str(m["model_name"])
        n = _num_decoder_layers_from_pretrained(mn)
        print(
            f"Expanded ablate_layer_idx=all_layers -> {n} layer(s) 0..{n - 1} for {mn!r}",
            flush=True,
        )
        for i in range(n):
            row = dict(m)
            row["ablate_layer_idx"] = i
            out.append(row)
    return out


def _normalize_ablate_groups(cfg_model: dict) -> str:
    """YAML ``null`` / missing key both mean ``none`` (matches ``run_evaluation.py`` default)."""
    ag = cfg_model.get("ablate_groups")
    return "none" if ag is None else str(ag)


def _validate_model_ablation(cfg_model: dict) -> None:
    """``run_evaluation.py`` forbids ``ablate_layer_idx`` with non-none ``ablate_groups``."""
    ag = _normalize_ablate_groups(cfg_model)
    layer_idx = cfg_model.get("ablate_layer_idx")
    if layer_idx is None or _is_ablate_layer_idx_all_layers(layer_idx):
        return
    if ag != "none":
        sys.exit(
            "model config: use only one of ablate_layer_idx or ablate_groups (non-none). "
            "Set ablate_groups to none or omit it when using ablate_layer_idx."
        )


def run_generate(dataset_dir: Path, cfg_data: dict) -> None:
    num_samples = int(cfg_data.get("num_samples", 1000))
    cmd = [
        sys.executable,
        str(_GEN_SCRIPT),
        "--output_dir",
        str(dataset_dir),
        "--num_boxes",
        str(cfg_data["num_boxes"]),
        "--expected_num_items_per_box",
        str(cfg_data["expected_num_items_per_box"]),
        "--max_items_per_box",
        str(cfg_data["max_items_per_box"]),
        "--num_operations",
        str(cfg_data["num_operations"]),
        "--num_samples",
        str(num_samples),
        "--zero_shot",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_ROOT), check=True)


def run_eval(dataset_dir: Path, cfg_model: dict) -> None:
    _validate_model_ablation(cfg_model)
    cmd = [
        sys.executable,
        str(_EVAL_SCRIPT),
        "--model_name",
        str(cfg_model["model_name"]),
        "--prompt_mode",
        str(cfg_model["prompt_mode"]),
        "--dataset_dir",
        str(dataset_dir),
    ]
    optional = [
        ("split", "--split"),
        ("seed", "--seed"),
        ("num_test_set_per_bin", "--num_test_set_per_bin"),
        ("max_new_tokens", "--max_new_tokens"),
        ("eval_type", "--eval_type"),
        ("output_json", "--output_json"),
        ("ablate_groups", "--ablate_groups"),
        ("ablate_layer_idx", "--ablate_layer_idx"),
    ]
    for key, flag in optional:
        if key in cfg_model and cfg_model[key] is not None:
            cmd.extend([flag, str(cfg_model[key])])
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run dataset generation (if missing), then run_evaluation.py. "
            "Pass a YAML file with 'data' and 'model' sections (see docstring at top of this file)."
        ),
    )
    ap.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "YAML with 'data' and 'model' sections. Required keys as scalars or lists "
            "(lists are swept via Cartesian product within each section; runs are all "
            "data combinations × all model combinations)."
        ),
    )
    args = ap.parse_args()
    cfg_path = args.config.resolve()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "data" not in cfg or "model" not in cfg:
        sys.exit("Config must be a mapping with 'data' and 'model' keys.")
    raw_data = cfg["data"]
    raw_model = cfg["model"]

    for k in _REQUIRED_DATA_KEYS:
        if k not in raw_data:
            sys.exit(f"data.{k} is required in {cfg_path}")
    for k in _REQUIRED_MODEL_KEYS:
        if k not in raw_model:
            sys.exit(f"model.{k} is required in {cfg_path}")

    data_variants = expand_sweep(raw_data, "data")
    model_variants = expand_sweep(raw_model, "model")
    model_variants = expand_ablate_layer_idx_all_layers(model_variants)
    total_runs = len(data_variants) * len(model_variants)
    run_i = 0
    for cfg_data in data_variants:
        slug = dataset_dir_slug(cfg_data)
        dataset_dir = _ROOT / "data" / slug
        if splits_jsonl_ready(dataset_dir):
            print(f"Dataset present, skipping generation: {dataset_dir}", flush=True)
        else:
            dataset_dir.parent.mkdir(parents=True, exist_ok=True)
            run_generate(dataset_dir, cfg_data)

        for cfg_model in model_variants:
            run_i += 1
            ablate_bits = []
            if cfg_model.get("ablate_layer_idx") is not None:
                ablate_bits.append(f"ablate_layer_idx={cfg_model['ablate_layer_idx']}")
            elif _normalize_ablate_groups(cfg_model) != "none":
                ablate_bits.append(
                    f"ablate_groups={_normalize_ablate_groups(cfg_model)!r}"
                )
            ablate_suffix = (" " + ", ".join(ablate_bits)) if ablate_bits else ""
            print(
                f"=== sweep run {run_i}/{total_runs} "
                f"(data={slug}, model={cfg_model['model_name']!r}, "
                f"prompt_mode={cfg_model['prompt_mode']!r}{ablate_suffix}) ===",
                flush=True,
            )
            run_eval(dataset_dir, cfg_model)


if __name__ == "__main__":
    main()
