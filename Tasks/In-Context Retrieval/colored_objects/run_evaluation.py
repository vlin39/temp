#!/usr/bin/env python3
"""
Evaluate a Causal LM on the colored-objects task.
python run_evaluation.py --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --subtype what_color --num_test_set 250 --prompt_mode chat --seed 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from colored_objects_eval import (  # noqa: E402
    accuracy_by_num_objects,
    accuracy_by_subtype,
    count_objects_in_input,
    evaluate_subset,
    load_task_examples,
    stratified_indices_by_num_objects,
)

_DEFAULT_TASK_JSON = _ROOT / "task.json"
_OUTPUT_DIR = _ROOT / "output"
PROMPT_MODES = ("continuation", "chat", "chat_continual")
ABLATE_GROUPS = ("none", "self_attn", "linear_attn")
DEFAULT_SUBTYPES = ("what_color", "yes_no_color", "neither_color", "spatial", "arithmetic")


def model_slug(model_name: str) -> str:
    slug = model_name.replace("/", "--")
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return slug.strip("._-") or "model"


def default_output_json_path(
    model_name: str,
    prompt_mode: str,
    subtype: str | None,
    ablate_group: str,
    allowed_num_objects: tuple[int, ...] | None,
) -> Path:
    pieces = [model_slug(model_name), subtype or "all", prompt_mode]
    if ablate_group != "none":
        pieces.append(ablate_group)
    if allowed_num_objects:
        pieces.append("numobj-" + "-".join(str(x) for x in allowed_num_objects))
    return _OUTPUT_DIR / ("_".join(pieces) + ".json")


def parse_csv_arg(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_allowed_num_objects(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    values = sorted({int(part) for part in parse_csv_arg(raw)})
    if not values:
        raise ValueError("--allowed_num_objects must contain at least one integer.")
    return tuple(values)


def resolve_prompt_modes(args: argparse.Namespace) -> list[str]:
    modes = list(PROMPT_MODES) if args.sweep_prompt_modes else [args.prompt_mode]
    invalid = [mode for mode in modes if mode not in PROMPT_MODES]
    if invalid:
        raise ValueError(f"Unsupported prompt mode(s): {invalid}")
    return modes


def resolve_ablate_groups(args: argparse.Namespace) -> list[str]:
    groups = list(ABLATE_GROUPS) if args.sweep_ablate_groups else [args.ablate_group]
    invalid = [group for group in groups if group not in ABLATE_GROUPS]
    if invalid:
        raise ValueError(f"Unsupported ablate group(s): {invalid}")
    return groups


def resolve_subtypes(args: argparse.Namespace, examples: list[dict]) -> list[str]:
    if args.sweep_subtypes:
        present = {ex.get("comment", "") for ex in examples}
        return [subtype for subtype in DEFAULT_SUBTYPES if subtype in present]
    return [args.subtype]


def filter_examples_by_num_objects(
    examples: list[dict],
    allowed_num_objects: tuple[int, ...] | None,
) -> list[dict]:
    if allowed_num_objects is None:
        return examples
    allowed = set(allowed_num_objects)
    return [ex for ex in examples if count_objects_in_input(ex["input"]) in allowed]


def resolve_model_name(model_name: str) -> str:
    _DOL_INTERP = _ROOT.parents[2]
    if str(_DOL_INTERP) not in sys.path:
        sys.path.insert(0, str(_DOL_INTERP))
    from Models.model_util import MODEL_IDS

    return MODEL_IDS.get(model_name, model_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model id (e.g. allenai/Olmo-Hybrid-Instruct-SFT-7B).",
    )
    p.add_argument(
        "--num_test_set",
        type=int,
        default=100,
        help=(
            "Total number of evaluated examples. Must be divisible by the number "
            "of available object-count bins for the selected subtype."
        ),
    )
    p.add_argument(
        "--prompt_mode",
        type=str,
        choices=PROMPT_MODES,
        default="continuation",
    )
    p.add_argument(
        "--sweep_prompt_modes",
        action="store_true",
        help="Run continuation, chat, and chat_continual prompt variants.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument(
        "--subtype",
        type=str,
        default="what_color",
        help="Optional task subtype filter (e.g. what_color, yes_no_color, neither_color, arithmetic, spatial).",
    )
    p.add_argument(
        "--sweep_subtypes",
        action="store_true",
        help="Run each available task subtype separately instead of a single subtype or the pooled 'all' setting.",
    )
    p.add_argument(
        "--ablate_group",
        type=str,
        choices=ABLATE_GROUPS,
        default="none",
        help="Attention ablation to apply via Models/model_util.py.",
    )
    p.add_argument(
        "--sweep_ablate_groups",
        action="store_true",
        help="Run none, self_attn, and linear_attn ablations.",
    )
    p.add_argument(
        "--allowed_num_objects",
        type=str,
        default=None,
        help="Comma-separated object-count bins to evaluate, e.g. 3,4 or 6.",
    )
    p.add_argument(
        "--task_json",
        type=str,
        default=str(_DEFAULT_TASK_JSON),
        help="Path to task.json.",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="JSON path for full results. Only valid for a single configuration.",
    )
    return p.parse_args()


def run_single_config(
    *,
    args: argparse.Namespace,
    examples: list[dict],
    model_name: str,
    model,
    tokenizer,
    apply_ablation,
    subtype: str,
    prompt_mode: str,
    ablate_group: str,
    allowed_num_objects: tuple[int, ...] | None,
) -> Path:
    filtered_examples = filter_examples_by_num_objects(examples, allowed_num_objects)
    filtered = [
        ex for ex in filtered_examples if subtype == "all" or ex.get("comment") == subtype
    ]
    available_bins = sorted({count_objects_in_input(ex["input"]) for ex in filtered})
    if not available_bins:
        raise ValueError(
            f"No examples found for subtype={subtype!r}"
            + (
                f" with allowed_num_objects={list(allowed_num_objects)!r}"
                if allowed_num_objects is not None else ""
            )
        )
    if args.num_test_set <= 0 or args.num_test_set % len(available_bins) != 0:
        raise ValueError(
            "--num_test_set must be a positive multiple of the available object-count bins: "
            f"{available_bins}"
        )
    n_per_bin = args.num_test_set // len(available_bins)

    rng = np.random.default_rng(args.seed)
    try:
        indices, pool_sizes = stratified_indices_by_num_objects(
            filtered_examples,
            rng,
            n_per_bin=n_per_bin,
            allowed_bins=tuple(available_bins),
            subtype=subtype if subtype != "all" else None,
        )
    except ValueError as e:
        raise ValueError(str(e)) from e

    print(f"Task: {args.task_json} ({len(examples)} examples)")
    print(f"Pool sizes (num_objects {available_bins}): {pool_sizes}")
    print(
        f"Evaluating {len(indices)} examples ({n_per_bin} per object-count bin), "
        f"subtype={subtype!r}, prompt_mode={prompt_mode!r}, "
        f"ablate_group={ablate_group!r}, seed={args.seed}"
    )

    patched = apply_ablation(ablate_group)
    if ablate_group != "none":
        print(f"Ablated {len(patched)} module(s) for group={ablate_group!r}")
        if not patched:
            print("Warning: no modules matched the requested ablation group.")
    results = evaluate_subset(
        filtered_examples,
        model,
        tokenizer,
        indices,
        max_new_tokens=args.max_new_tokens,
        prompt_mode=prompt_mode,
    )

    by_objects = accuracy_by_num_objects(results["rows"])
    by_subtype = accuracy_by_subtype(results["rows"])
    print(f"Overall: accuracy={results['accuracy']:.4f}")
    for k in sorted(by_objects):
        b = by_objects[k]
        print(f"  num_objects={k}: n={b['n']} accuracy={b['accuracy']:.4f}")
    for k in sorted(by_subtype):
        b = by_subtype[k]
        print(f"  subtype={k}: n={b['n']} accuracy={b['accuracy']:.4f}")

    out_path = (
        Path(args.output_json)
        if args.output_json
        else default_output_json_path(model_name, prompt_mode, subtype, ablate_group, allowed_num_objects)
    )
    res_to_save = {
        "model_name": model_name,
        "task_json": str(args.task_json),
        "num_test_set": args.num_test_set,
        "n_per_num_objects_bin": n_per_bin,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "prompt_mode": prompt_mode,
        "subtype": subtype,
        "ablate_group": ablate_group,
        "ablation_modules": patched,
        "allowed_num_objects": list(allowed_num_objects) if allowed_num_objects is not None else None,
        "available_num_objects_bins": available_bins,
        "pool_sizes_num_objects": pool_sizes,
        "overall": {
            "n": results["n"],
            "accuracy": results["accuracy"],
        },
        "by_num_objects": {str(k): v for k, v in by_objects.items()},
        "by_subtype": by_subtype,
        "rows": results["rows"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res_to_save, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    examples = load_task_examples(args.task_json)
    try:
        allowed_num_objects = parse_allowed_num_objects(args.allowed_num_objects)
        prompt_modes = resolve_prompt_modes(args)
        ablate_group_values = resolve_ablate_groups(args)
        subtypes = resolve_subtypes(args, examples)
        model_name = resolve_model_name(args.model_name)
    except ValueError as e:
        sys.exit(str(e))

    num_runs = len(prompt_modes) * len(ablate_group_values) * len(subtypes)
    if args.output_json and num_runs != 1:
        sys.exit("--output_json can only be used when running a single configuration.")

    print(
        f"Resolved model={model_name!r}; running {num_runs} configuration(s) "
        f"over subtypes={subtypes}, prompt_modes={prompt_modes}, ablate_groups={ablate_group_values}"
    )
    if allowed_num_objects is not None:
        print(f"Restricting evaluation to num_objects bins: {list(allowed_num_objects)}")

    _DOL_INTERP = _ROOT.parents[2]
    if str(_DOL_INTERP) not in sys.path:
        sys.path.insert(0, str(_DOL_INTERP))
    from Models.model_util import ablate_groups, load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(model_name)

    def apply_ablation(group: str) -> list[str]:
        return ablate_groups(model, group)

    failures: list[tuple[str, str, str, str]] = []
    for subtype, prompt_mode, ablate_group in product(subtypes, prompt_modes, ablate_group_values):
        print("\n" + "=" * 80)
        print(
            f"Run config: subtype={subtype}, prompt_mode={prompt_mode}, "
            f"ablate_group={ablate_group}"
        )
        try:
            run_single_config(
                args=args,
                examples=examples,
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                apply_ablation=apply_ablation,
                subtype=subtype,
                prompt_mode=prompt_mode,
                ablate_group=ablate_group,
                allowed_num_objects=allowed_num_objects,
            )
        except ValueError as e:
            failures.append((subtype, prompt_mode, ablate_group, str(e)))
            print(f"Failed: {e}")

    if failures:
        print("\nSome configurations failed:")
        for subtype, prompt_mode, ablate_group, error in failures:
            print(
                f"  subtype={subtype}, prompt_mode={prompt_mode}, "
                f"ablate_group={ablate_group}: {error}"
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
