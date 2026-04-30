#!/usr/bin/env python3
"""
Evaluate a Causal LM on the box entity-tracking JSONL task (see ``test.ipynb``).

Example:
  python run_evaluation.py --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \\
    --split dev --num_test_set_per_bin 100 --prompt_mode continuation --seed 0

  python run_evaluation.py --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \\
    --dataset_dir data/boxes5_nso_exp2_max3_zero_shot --prompt_mode chat
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from boxes_lm_eval import (
    accuracy_by_numops,
    evaluate_subset,
    load_model_and_tokenizer,
    stratified_indices_by_numops,
)

_DEFAULT_DATA_SUBDIR = "boxes5_nso_exp2_max3_zero_shot"
_OUTPUT_DIR = _ROOT / "output"


def default_output_json_path(
    model_name: str, prompt_mode: str, dataset_slug: str
) -> Path:
    """``output/<dataset_slug>/<safe_model>_<prompt_mode>.json`` next to this script."""
    slug = model_name.replace("/", "--")
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    slug = slug.strip("._-") or "model"
    ds = dataset_slug.strip()
    if not ds or ds in (".", ".."):
        ds = "dataset"
    ds = re.sub(r"[^a-zA-Z0-9._-]", "_", ds).strip("._-") or "dataset"
    return _OUTPUT_DIR / ds / f"{slug}_{prompt_mode}.json"


def default_jsonl_path(split: str, data_subdir: str) -> Path:
    return _ROOT / "data" / data_subdir / f"{split}-t5.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model id (e.g. allenai/Olmo-Hybrid-Instruct-SFT-7B).",
    )
    p.add_argument(
        "--num_test_set_per_bin",
        type=int,
        default=300,
    )
    p.add_argument(
        "--split",
        type=str,
        choices=("dev", "train", "test"),
        default="test",
        help="Which JSONL split to load ({split}-t5.jsonl).",
    )
    p.add_argument(
        "--prompt_mode",
        type=str,
        choices=("continuation", "chat"),
        default="continuation",
        help="continuation: raw prefix ending in 'Box k contains'; chat: apply_chat_template + explicit question.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument(
        "--data_subdir",
        type=str,
        default=_DEFAULT_DATA_SUBDIR,
        help=f"Subfolder under data/ (default: {_DEFAULT_DATA_SUBDIR}). Ignored if --dataset_dir or --data_file is set.",
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help=(
            "Directory containing {split}-t5.jsonl files (e.g. data/my_run/). "
            "Overrides --data_subdir when set (unless --data_file is set)."
        ),
    )
    p.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Override path to a single JSONL file (ignores --split, --dataset_dir, and --data_subdir).",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "JSON path for full results (including rows). "
            "Default: output/<dataset_slug>/<model_name_with_slashes_as_-->_<prompt_mode>.json "
            f"(under {_OUTPUT_DIR.relative_to(_ROOT)})."
        ),
    )
    return p.parse_args()


def resolve_jsonl_path(args: argparse.Namespace) -> Path:
    if args.data_file:
        return Path(args.data_file)
    if args.dataset_dir:
        return Path(args.dataset_dir) / f"{args.split}-t5.jsonl"
    return default_jsonl_path(args.split, args.data_subdir)


def summarize_numops_by_op(rows: list[dict]) -> dict | None:
    """Aggregate ``numops_by_op`` across evaluated rows (when present in data)."""
    keys: set[str] = set()
    totals: dict[str, int] = {}
    for r in rows:
        nb = r.get("numops_by_op")
        if not nb:
            continue
        for k, v in nb.items():
            ks = str(k)
            keys.add(ks)
            totals[ks] = totals.get(ks, 0) + int(v)
    if not keys:
        return None
    return {
        "operation_type_keys": sorted(keys),
        "total_operation_counts_by_type": totals,
    }


def dataset_slug_for_output(args: argparse.Namespace, jsonl: Path) -> str:
    if args.data_file:
        return jsonl.resolve().parent.name
    if args.dataset_dir:
        return Path(args.dataset_dir).resolve().name
    return args.data_subdir


def main() -> None:
    args = parse_args()

    n_per_bin = args.num_test_set_per_bin

    jsonl = resolve_jsonl_path(args)
    if not jsonl.is_file():
        sys.exit(f"Data file not found: {jsonl}")

    dataset = load_dataset("json", data_files=str(jsonl), split="train")
    rng = np.random.default_rng(args.seed)
    try:
        indices, pool_sizes = stratified_indices_by_numops(
            dataset, rng, n_per_bin=n_per_bin
        )
    except ValueError as e:
        sys.exit(str(e))

    print(f"Data: {jsonl} ({len(dataset)} rows)")
    print(f"Pool sizes (numops 0..3): {pool_sizes}")
    print(
        f"Evaluating {len(indices)} examples ({n_per_bin} per numops bin), "
        f"prompt_mode={args.prompt_mode!r}, seed={args.seed}"
    )

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    results = evaluate_subset(
        dataset,
        model,
        tokenizer,
        indices,
        max_new_tokens=args.max_new_tokens,
        prompt_mode=args.prompt_mode,
    )

    by_ops = accuracy_by_numops(results["rows"])
    print(
        f"Overall: exact_acc={results['exact_accuracy']:.4f} "
        f"set_acc={results['set_accuracy']:.4f}"
    )
    for k in sorted(by_ops):
        b = by_ops[k]
        print(
            f"  numops={k}: n={b['n']} exact_acc={b['exact_accuracy']:.4f} "
            f"set_acc={b['set_accuracy']:.4f}"
        )

    ds_slug = dataset_slug_for_output(args, jsonl)
    out_path = (
        Path(args.output_json)
        if args.output_json
        else default_output_json_path(args.model_name, args.prompt_mode, ds_slug)
    )
    op_summary = summarize_numops_by_op(results["rows"])
    res_to_save = {
        "model_name": args.model_name,
        "data_file": str(jsonl),
        "dataset_slug": ds_slug,
        "split": args.split,
        "num_test_set_per_bin": args.num_test_set_per_bin,
        "n_per_numops_bin": n_per_bin,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "prompt_mode": args.prompt_mode,
        "pool_sizes_numops_0_to_3": pool_sizes,
        "overall": {
            "n": results["n"],
            "exact_accuracy": results["exact_accuracy"],
            "set_accuracy": results["set_accuracy"],
        },
        "by_numops": {str(k): v for k, v in by_ops.items()},
        "rows": results["rows"],
    }
    if op_summary is not None:
        res_to_save["operation_types"] = op_summary
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res_to_save, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
