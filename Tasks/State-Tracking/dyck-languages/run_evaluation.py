#!/usr/bin/env python3
"""
Evaluate a Causal LM on the BBH dyck_languages task.

python run_evaluation.py --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
    --prompt_mode chat --num_test_set 200 --seed 0
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

from dyck_languages_eval import (  # noqa: E402
    accuracy_by_answer,
    accuracy_by_depth,
    accuracy_by_seq_len,
    evaluate_subset,
    load_task_examples,
    stratified_indices_by_answer,
)

_OUTPUT_DIR = _ROOT / "output"
PROMPT_MODES  = ("continuation", "chat", "chat_continual")
ABLATE_GROUPS = ("none", "self_attn", "linear_attn")


def model_slug(model_name: str) -> str:
    slug = model_name.replace("/", "--")
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return slug.strip("._-") or "model"


def default_output_json_path(model_name: str, prompt_mode: str, ablate_group: str) -> Path:
    pieces = [model_slug(model_name), prompt_mode]
    if ablate_group != "none":
        pieces.append(ablate_group)
    return _OUTPUT_DIR / ("_".join(pieces) + ".json")


def default_layer_output_json_path(model_name: str, prompt_mode: str, layer_idx: int) -> Path:
    return _OUTPUT_DIR / f"{model_slug(model_name)}_{prompt_mode}_layer{layer_idx}.json"


def resolve_prompt_modes(args: argparse.Namespace) -> list[str]:
    modes = list(PROMPT_MODES) if args.sweep_prompt_modes else [args.prompt_mode]
    invalid = [m for m in modes if m not in PROMPT_MODES]
    if invalid:
        raise ValueError(f"Unsupported prompt mode(s): {invalid}")
    return modes


def resolve_ablate_groups(args: argparse.Namespace) -> list[str]:
    groups = list(ABLATE_GROUPS) if args.sweep_ablate_groups else [args.ablate_group]
    invalid = [g for g in groups if g not in ABLATE_GROUPS]
    if invalid:
        raise ValueError(f"Unsupported ablate group(s): {invalid}")
    return groups


def resolve_model_name(model_name: str) -> str:
    _DOL_INTERP = _ROOT.parents[2]
    if str(_DOL_INTERP) not in sys.path:
        sys.path.insert(0, str(_DOL_INTERP))
    from Models.model_util import MODEL_IDS
    return MODEL_IDS.get(model_name, model_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", type=str, required=True,
                   help="HuggingFace model ID or key from Models.model_util.MODEL_IDS.")
    p.add_argument("--num_test_set", type=int, default=200,
                   help="Total examples evaluated. Must be even (split equally into yes/no bins).")
    p.add_argument("--prompt_mode", type=str, choices=PROMPT_MODES, default="continuation")
    p.add_argument("--sweep_prompt_modes", action="store_true",
                   help="Run all three prompt modes.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--ablate_group", type=str, choices=ABLATE_GROUPS, default="none")
    p.add_argument("--sweep_ablate_groups", action="store_true",
                   help="Run none, self_attn, and linear_attn ablations.")
    p.add_argument("--sweep_layer_indices", action="store_true",
                   help="Ablate one decoder layer at a time; writes one JSON per layer.")
    p.add_argument("--cache_dir", type=str, default=str(_ROOT / "data"))
    p.add_argument("--output_json", type=str, default=None,
                   help="Override output path (single configuration only).")
    return p.parse_args()


def run_single_config(
    *,
    args: argparse.Namespace,
    examples: list[dict],
    model_name: str,
    model,
    tokenizer,
    apply_ablation,
    prompt_mode: str,
    ablate_group: str,
) -> Path:
    if args.num_test_set <= 0 or args.num_test_set % 2 != 0:
        raise ValueError("--num_test_set must be a positive even number.")
    n_per_bin = args.num_test_set // 2

    rng = np.random.default_rng(args.seed)
    indices, pool_sizes = stratified_indices_by_answer(examples, rng, n_per_bin=n_per_bin)

    print(f"Dataset: {len(examples)} examples total")
    print(f"Pool sizes (by answer): {pool_sizes}")
    print(
        f"Evaluating {len(indices)} examples ({n_per_bin} per answer bin), "
        f"prompt_mode={prompt_mode!r}, ablate_group={ablate_group!r}, seed={args.seed}"
    )

    patched = apply_ablation(ablate_group)
    if ablate_group != "none":
        print(f"Ablated {len(patched)} module(s) for group={ablate_group!r}")

    results = evaluate_subset(
        examples, model, tokenizer, indices,
        max_new_tokens=args.max_new_tokens,
        prompt_mode=prompt_mode,
    )

    by_ans   = accuracy_by_answer(results["rows"])
    by_depth = accuracy_by_depth(results["rows"])
    by_len   = accuracy_by_seq_len(results["rows"])

    print(f"Overall: accuracy={results['accuracy']:.4f}")
    for label in ("yes", "no"):
        b = by_ans[label]
        print(f"  answer={label}: n={b['n']} accuracy={b['accuracy']:.4f}")

    out_path = (
        Path(args.output_json)
        if args.output_json
        else default_output_json_path(model_name, prompt_mode, ablate_group)
    )
    res_to_save = {
        "model_name":        model_name,
        "task":              "dyck_languages",
        "num_test_set":      args.num_test_set,
        "n_per_answer_bin":  n_per_bin,
        "seed":              args.seed,
        "max_new_tokens":    args.max_new_tokens,
        "prompt_mode":       prompt_mode,
        "ablate_group":      ablate_group,
        "ablation_modules":  patched,
        "pool_sizes_by_answer": pool_sizes,
        "overall": {
            "n":        results["n"],
            "accuracy": results["accuracy"],
        },
        "by_answer":  {k: v for k, v in by_ans.items()},
        "by_depth":   {str(k): v for k, v in by_depth.items()},
        "by_seq_len": {str(k): v for k, v in by_len.items()},
        "rows":       results["rows"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res_to_save, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    try:
        prompt_modes        = resolve_prompt_modes(args)
        ablate_group_values = resolve_ablate_groups(args)
        model_name          = resolve_model_name(args.model_name)
    except ValueError as e:
        sys.exit(str(e))

    num_runs = len(prompt_modes) * len(ablate_group_values)
    if args.output_json and num_runs != 1:
        sys.exit("--output_json can only be used when running a single configuration.")
    if args.output_json and args.sweep_layer_indices:
        sys.exit("--output_json cannot be used with --sweep_layer_indices.")

    print(
        f"Resolved model={model_name!r}; running {num_runs} configuration(s) "
        f"over prompt_modes={prompt_modes}, ablate_groups={ablate_group_values}"
    )

    cache_dir = Path(args.cache_dir)
    examples  = load_task_examples(cache_dir=cache_dir)
    print(f"Loaded {len(examples)} examples from BBH dyck_languages.")

    _DOL_INTERP = _ROOT.parents[2]
    if str(_DOL_INTERP) not in sys.path:
        sys.path.insert(0, str(_DOL_INTERP))
    from Models.model_util import (
        ablate_groups,
        ablate_single_attn_module,
        list_decoder_attention_layer_indices,
        load_model_and_tokenizer,
        restore_attention_ablation,
    )

    model, tokenizer = load_model_and_tokenizer(model_name)

    def apply_ablation(group: str) -> list[str]:
        return ablate_groups(model, group)

    failures: list[tuple[str, str, str]] = []

    if args.sweep_layer_indices:
        layer_indices = list_decoder_attention_layer_indices(model)
        print(f"Layerwise sweep: {len(layer_indices)} layers × {len(prompt_modes)} prompt modes")
        for prompt_mode in prompt_modes:
            for layer_idx in layer_indices:
                print("\n" + "=" * 80)
                print(f"Run config: prompt_mode={prompt_mode}, layer_idx={layer_idx}")
                out_path = default_layer_output_json_path(model_name, prompt_mode, layer_idx)
                try:
                    patched_names = ablate_single_attn_module(model, layer_idx)
                    layer_type = "self_attn" if any("self_attn" in n for n in patched_names) else "linear_attn"
                    result_path = run_single_config(
                        args=args,
                        examples=examples,
                        model_name=model_name,
                        model=model,
                        tokenizer=tokenizer,
                        apply_ablation=lambda _g: [],
                        prompt_mode=prompt_mode,
                        ablate_group="none",
                    )
                    restore_attention_ablation(model)
                    with open(result_path, encoding="utf-8") as f:
                        saved = json.load(f)
                    saved["ablate_mode"]      = "single_layer"
                    saved["layer_idx"]        = layer_idx
                    saved["layer_type"]       = layer_type
                    saved["ablation_modules"] = patched_names
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(saved, f, indent=2, ensure_ascii=False)
                    result_path.unlink(missing_ok=True)
                    print(f"Wrote {out_path}")
                except (ValueError, RuntimeError) as e:
                    restore_attention_ablation(model)
                    failures.append((prompt_mode, f"layer{layer_idx}", str(e)))
                    print(f"Failed: {e}")
    else:
        for prompt_mode, ablate_group in product(prompt_modes, ablate_group_values):
            print("\n" + "=" * 80)
            print(f"Run config: prompt_mode={prompt_mode}, ablate_group={ablate_group}")
            try:
                run_single_config(
                    args=args,
                    examples=examples,
                    model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    apply_ablation=apply_ablation,
                    prompt_mode=prompt_mode,
                    ablate_group=ablate_group,
                )
            except ValueError as e:
                failures.append((prompt_mode, ablate_group, str(e)))
                print(f"Failed: {e}")

    if failures:
        print("\nSome configurations failed:")
        for pm, ag, err in failures:
            print(f"  prompt_mode={pm}, ablate_group={ag}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
