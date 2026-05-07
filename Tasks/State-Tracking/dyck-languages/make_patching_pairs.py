#!/usr/bin/env python3
"""
Generate clean/corrupt paired prompts for activation patching on dyck_languages.

Corrupt variant: take a valid ("yes") sequence and flip one bracket near the
middle to create an unbalanced sequence ("no").  The structural change is
minimal — one character — making it ideal for residual-stream patching.

Usage:
    python make_patching_pairs.py --num_pairs 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent

_OPEN  = set("([{<")
_CLOSE = set(")]}>" )
_OPEN_LIST  = list("([{<")
_CLOSE_LIST = list(")]}>" )
_PAIR = {'(': ')', '[': ']', '{': '}', '<': '>',
         ')': '(', ']': '[', '}': '{', '>': '<'}


def _is_balanced(text: str) -> bool:
    stack: list[str] = []
    for ch in text:
        if ch in _OPEN:
            stack.append(ch)
        elif ch in _CLOSE:
            if not stack or stack[-1] != _PAIR[ch]:
                return False
            stack.pop()
    return len(stack) == 0


def _corrupt(text: str, rng: np.random.Generator) -> str | None:
    """
    Flip one bracket in `text` so the sequence becomes unbalanced.
    Try positions near the middle first; give up after 20 attempts.
    """
    chars = list(text)
    bracket_positions = [i for i, c in enumerate(chars) if c in _OPEN or c in _CLOSE]
    if not bracket_positions:
        return None

    # Prefer flipping a closing bracket in the second half of the sequence
    mid = len(bracket_positions) // 2
    candidates = bracket_positions[mid:] + bracket_positions[:mid]

    for pos in candidates:
        original = chars[pos]
        # Flip: open→wrong open, close→wrong close
        if original in _OPEN:
            alternates = [c for c in _OPEN_LIST if c != original]
        else:
            alternates = [c for c in _CLOSE_LIST if c != original]
        rng.shuffle(alternates := np.array(alternates))
        for alt in alternates:
            chars[pos] = str(alt)
            candidate = "".join(chars)
            if not _is_balanced(candidate):
                return candidate
            chars[pos] = original  # restore and try next

    return None


def make_pairs(
    examples: list[dict],
    rng: np.random.Generator,
    num_pairs: int,
) -> list[dict]:
    indices = np.arange(len(examples))
    rng.shuffle(indices)
    pairs: list[dict] = []
    for idx in indices:
        if len(pairs) >= num_pairs:
            break
        ex = examples[int(idx)]
        # Only corrupt "yes" examples (balanced → unbalanced is deterministic)
        if ex["gold_answer"] != "yes":
            continue
        corrupt_input = _corrupt(ex["input"], rng)
        if corrupt_input is None:
            continue
        pairs.append(
            {
                "clean_prompt":   ex["input"],
                "clean_answer":   "yes",
                "corrupt_prompt": corrupt_input,
                "corrupt_answer": "no",
                "seq_len":        ex["seq_len"],
                "max_depth":      ex["max_depth"],
            }
        )
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache_dir", type=str, default=str(_ROOT / "data"))
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _DOL_INTERP = _ROOT.parents[2]
    if str(_DOL_INTERP) not in sys.path:
        sys.path.insert(0, str(_DOL_INTERP))
    if str(_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(_ROOT / "src"))
    from dyck_languages_eval import load_task_examples

    examples = load_task_examples(cache_dir=Path(args.cache_dir))
    print(f"Loaded {len(examples)} examples.")

    rng   = np.random.default_rng(args.seed)
    pairs = make_pairs(examples, rng, args.num_pairs)
    if len(pairs) < args.num_pairs:
        print(f"Warning: only {len(pairs)} valid pairs from {len(examples)} examples.")

    out_path = (
        Path(args.output)
        if args.output
        else _ROOT / "output" / f"patching_pairs_{args.num_pairs}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"num_pairs": len(pairs), "seed": args.seed, "pairs": pairs}, f, indent=2)
    print(f"Wrote {len(pairs)} pairs to {out_path}")


if __name__ == "__main__":
    main()
