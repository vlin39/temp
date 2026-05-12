#!/usr/bin/env python3
"""
Generate clean/corrupt paired prompts for activation patching on dyck_languages.

The clean prompt is an unmodified BBH example whose target is its gold closing
sequence. The corrupt prompt is the same example with one bracket in the
*prefix* changed to a different bracket type, such that the required closing
sequence is provably different. The structural change is one character; the
information change is one bracket type — ideal for residual-stream patching.

Usage:
    python make_patching_pairs.py --num_pairs 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent

_OPEN_LIST  = list("([{<")
_CLOSE_LIST = list(")]}>" )
_OPEN  = set(_OPEN_LIST)
_CLOSE = set(_CLOSE_LIST)
_PAIR  = {'(': ')', '[': ']', '{': '}', '<': '>'}


def _bracket_tokens(text: str) -> list[str]:
    return [ch for ch in text if ch in _OPEN or ch in _CLOSE]


def _required_closing(tokens: list[str]) -> list[str] | None:
    """Given a prefix of bracket tokens, return the closing sequence that
    completes it into a valid Dyck word, or None if the prefix is already
    invalid (e.g., a close that doesn't match its open).
    """
    stack: list[str] = []
    for t in tokens:
        if t in _OPEN:
            stack.append(t)
        else:  # t in _CLOSE
            if not stack:
                return None
            if _PAIR[stack[-1]] != t:
                return None
            stack.pop()
    return [_PAIR[op] for op in reversed(stack)]


def _corrupt(text: str, gold_closing: list[str], rng: np.random.Generator) -> tuple[str, list[str]] | None:
    """Find one bracket position in `text` to change such that the required
    closing sequence becomes different from `gold_closing`. Try several
    candidate positions and bracket alternatives.

    Returns (corrupt_text, corrupt_closing) or None.
    """
    chars = list(text)
    bracket_positions = [i for i, c in enumerate(chars) if c in _OPEN or c in _CLOSE]
    if not bracket_positions:
        return None

    # Prefer positions in the first half (changes in the prefix tail are more
    # likely to flip the closing sequence than changes in the prefix head)
    rng.shuffle(positions := np.array(bracket_positions))
    for pos in positions:
        original = chars[pos]
        # Try alternates of the same direction (open→other open, close→other close)
        same_dir = _OPEN_LIST if original in _OPEN else _CLOSE_LIST
        alternates = [c for c in same_dir if c != original]
        rng.shuffle(alts := np.array(alternates))
        for alt in alts:
            chars[pos] = str(alt)
            new_text = "".join(chars)
            new_tokens = _bracket_tokens(new_text)
            new_closing = _required_closing(new_tokens)
            if new_closing is None:
                # corrupt prefix is now structurally invalid — not useful
                chars[pos] = original
                continue
            if new_closing != gold_closing:
                return new_text, new_closing
            chars[pos] = original  # closing unchanged, try next
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
        if not ex["gold_tokens"]:
            continue
        result = _corrupt(ex["input"], ex["gold_tokens"], rng)
        if result is None:
            continue
        corrupt_text, corrupt_closing = result
        pairs.append(
            {
                "clean_prompt":     ex["input"],
                "clean_closing":    " ".join(ex["gold_tokens"]),
                "corrupt_prompt":   corrupt_text,
                "corrupt_closing":  " ".join(corrupt_closing),
                "target_len":       ex["target_len"],
                "max_open_depth":   ex["max_open_depth"],
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
