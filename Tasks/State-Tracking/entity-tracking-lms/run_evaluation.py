#!/usr/bin/env python3
"""
Evaluate a Causal LM on the box entity-tracking JSONL task (see ``test.ipynb``).

Example:
  python run_evaluation.py --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \\
    --split dev --num_test_set_per_bin 100 --prompt_mode chat --seed 0

  python run_evaluation.py --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \\
    --dataset_dir data/boxes5_nso_exp2_max3_zero_shot --prompt_mode chat

  python run_evaluation.py --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \\
    --ablate_groups self_attn --split dev --prompt_mode chat

  python run_evaluation.py --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \\
    --ablate_layer_idx 12 --split dev --prompt_mode chat
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
    likelihood_by_numops,
    load_model_and_tokenizer,
    stratified_indices_by_numops,
)
from Models.model_util import ablate_groups, ablate_single_attn_module

_DEFAULT_DATA_SUBDIR = "boxes5_nso_exp2_max3_zero_shot"
_OUTPUT_DIR = _ROOT / "output"


def default_output_json_path(
    model_name: str,
    prompt_mode: str,
    dataset_slug: str,
    eval_type: str = "generation",
    ablate_groups_name: str = "none",
    ablate_layer_idx: int | None = None,
) -> Path:
    """``output/.../…_<prompt_mode>[_layer_N|_ablate_…].json`` next to this script."""
    slug = model_name.replace("/", "--")
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    slug = slug.strip("._-") or "model"
    ds = dataset_slug.strip()
    if not ds or ds in (".", ".."):
        ds = "dataset"
    ds = re.sub(r"[^a-zA-Z0-9._-]", "_", ds).strip("._-") or "dataset"
    suffix = ""
    if ablate_layer_idx is not None:
        suffix = f"_layer_{int(ablate_layer_idx)}"
    elif ablate_groups_name and ablate_groups_name != "none":
        ag = re.sub(r"[^a-zA-Z0-9._-]", "_", ablate_groups_name).strip("._-") or "ablate"
        suffix = f"_ablate_{ag}"
    if eval_type != "generation":
        suffix = f"{suffix}_{eval_type}"
    return _OUTPUT_DIR / ds / f"{slug}_{prompt_mode}{suffix}.json"


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
        "--eval_type",
        type=str,
        choices=("generation", "likelihood"),
        default="generation",
        help=(
            "generation: decode continuation and compute exact/set metrics; "
            "likelihood: score the gold continuation and compute log-likelihood/probability."
        ),
    )
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
            "Default: output/.../<model>_.._<prompt_mode>[_layer_<i>|_ablate_<group>].json "
            f"(under {_OUTPUT_DIR.relative_to(_ROOT)})."
        ),
    )
    p.add_argument(
        "--ablate_groups",
        type=str,
        choices=("none", "self_attn", "linear_attn"),
        default="none",
        help=(
            "Hybrid attention ablation via Models.model_util.ablate_groups: "
            "'none' (default), 'self_attn' (identity full attention), "
            "'linear_attn' (identity linear/GatedDeltaNet blocks). "
            "Mutually exclusive with --ablate_layer_idx."
        ),
    )
    p.add_argument(
        "--ablate_layer_idx",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, ablate only decoder layer N (self_attn or linear_attn in that layer) "
            "via Models.model_util.ablate_single_attn_module. Mutually exclusive with "
            "--ablate_groups other than 'none'."
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

    if args.ablate_layer_idx is not None and args.ablate_groups != "none":
        sys.exit(
            "Use only one of --ablate_layer_idx or --ablate_groups (non-none), not both."
        )

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
    if args.ablate_layer_idx is not None:
        try:
            patched_modules = ablate_single_attn_module(model, args.ablate_layer_idx)
        except ValueError as e:
            sys.exit(str(e))
        print(
            f"Attention ablation: layer_idx={args.ablate_layer_idx}, "
            f"patched {len(patched_modules)} module(s): {patched_modules}"
        )
    else:
        patched_modules = ablate_groups(model, group=args.ablate_groups)
        if args.ablate_groups != "none":
            print(
                f"Attention ablation: group={args.ablate_groups!r}, "
                f"patched {len(patched_modules)} module(s)"
            )

    results = evaluate_subset(
        dataset,
        model,
        tokenizer,
        indices,
        max_new_tokens=args.max_new_tokens,
        prompt_mode=args.prompt_mode,
        eval_type=args.eval_type,
    )

    if args.eval_type == "generation":
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
    else:
        by_ops = likelihood_by_numops(results["rows"])
        print(
            f"Overall: mean_log_likelihood={results['mean_log_likelihood']:.4f} "
            f"mean_probability={results['mean_probability']:.6g}"
        )
        for k in sorted(by_ops):
            b = by_ops[k]
            print(
                f"  numops={k}: n={b['n']} mean_log_likelihood={b['mean_log_likelihood']:.4f} "
                f"mean_probability={b['mean_probability']:.6g}"
            )

    ds_slug = dataset_slug_for_output(args, jsonl)
    out_path = (
        Path(args.output_json)
        if args.output_json
        else default_output_json_path(
            args.model_name,
            args.prompt_mode,
            ds_slug,
            args.eval_type,
            args.ablate_groups,
            args.ablate_layer_idx,
        )
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
        "eval_type": args.eval_type,
        "ablate_groups": args.ablate_groups,
        "ablate_layer_idx": args.ablate_layer_idx,
        "ablated_module_names": patched_modules
        if (args.ablate_groups != "none" or args.ablate_layer_idx is not None)
        else [],
        "pool_sizes_numops_0_to_3": pool_sizes,
        "overall": (
            {
                "n": results["n"],
                "exact_accuracy": results["exact_accuracy"],
                "set_accuracy": results["set_accuracy"],
            }
            if args.eval_type == "generation"
            else {
                "n": results["n"],
                "log_likelihood": results["mean_log_likelihood"],
                "probability": results["mean_probability"],
            }
        ),
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
