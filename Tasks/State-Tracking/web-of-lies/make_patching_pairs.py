#!/usr/bin/env python3
"""
Generate clean/corrupt paired prompts for activation patching experiments.

Each pair has the same chain structure as a BBH web_of_lies example, but the
first statement's truth value is flipped, which deterministically flips the
final answer.  The structural similarity is high (same names, same chain) and
the information change is exactly one bit — the ideal setup for residual-stream
activation patching.

Usage:
    python make_patching_pairs.py --num_pairs 200 --seed 0
    python make_patching_pairs.py --num_pairs 50 --seed 1 --cache_dir ./data
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent

# Regex matching the very first statement in a web_of_lies prompt.
# The BBH dataset uses "X tells the truth." or "X lies." at the start.
_FIRST_STMT = re.compile(
    r"^(?P<name>[A-Za-z]+) (?P<verb>tells the truth|lies)\.",
    re.IGNORECASE,
)

# "tells the truth" ↔ "lies"
_FLIP = {"tells the truth": "lies", "lies": "tells the truth"}


def flip_first_statement(text: str) -> str | None:
    """Flip the first statement's truth verb, return None if unrecognised."""
    m = _FIRST_STMT.match(text)
    if not m:
        return None
    original_verb = m.group("verb").lower()
    new_verb = _FLIP[original_verb]
    # Preserve capitalisation: first letter of new_verb → capital as in original
    return text[: m.start("verb")] + new_verb + text[m.end("verb"):]


def derive_answer(text: str) -> str | None:
    """
    Re-derive the yes/no answer from a web_of_lies prompt by simulating the chain.

    Returns 'yes', 'no', or None if the prompt cannot be parsed.
    """
    # Each statement is:
    #   "<name> tells the truth." (initialiser)
    #   "<name> says <prev> tells the truth." / "<name> says <prev> lies."
    # The question at the end is "Does <name> tell the truth?"
    lines = [s.strip() for s in re.split(r"(?<=[.?])\s+", text.strip()) if s.strip()]
    if not lines:
        return None

    question = lines[-1]
    statements = lines[:-1]

    q_match = re.match(r"Does (?P<name>[A-Za-z]+) tell the truth\?", question, re.IGNORECASE)
    if not q_match:
        return None
    target_name = q_match.group("name").lower()

    # truth_state: True = tells the truth, False = lies
    truth_state: dict[str, bool] = {}

    for stmt in statements:
        # Initialiser: "X tells the truth." / "X lies."
        init = re.match(r"^(?P<name>[A-Za-z]+) (?P<verb>tells the truth|lies)\.$", stmt, re.IGNORECASE)
        if init:
            name = init.group("name").lower()
            truth_state[name] = init.group("verb").lower() == "tells the truth"
            continue
        # Reporter: "X says Y tells the truth." / "X says Y lies."
        rep = re.match(
            r"^(?P<speaker>[A-Za-z]+) says (?P<subject>[A-Za-z]+) (?P<verb>tells the truth|lies)\.$",
            stmt,
            re.IGNORECASE,
        )
        if rep:
            speaker = rep.group("speaker").lower()
            subject = rep.group("subject").lower()
            claimed_truth = rep.group("verb").lower() == "tells the truth"
            speaker_honest = truth_state.get(speaker)
            if speaker_honest is None:
                return None
            # If speaker lies, the claim is inverted.
            truth_state[subject] = claimed_truth if speaker_honest else not claimed_truth
            continue
        return None  # unrecognised statement form

    if target_name not in truth_state:
        return None
    return "yes" if truth_state[target_name] else "no"


def make_pairs(
    examples: list[dict],
    rng: np.random.Generator,
    num_pairs: int,
) -> list[dict]:
    rng.shuffle(arr := np.arange(len(examples)))
    pairs: list[dict] = []
    for idx in arr:
        if len(pairs) >= num_pairs:
            break
        ex = examples[int(idx)]
        corrupt_input = flip_first_statement(ex["input"])
        if corrupt_input is None:
            continue
        corrupt_answer = derive_answer(corrupt_input)
        if corrupt_answer is None:
            continue
        if corrupt_answer == ex["gold_answer"]:
            # flipping didn't change the answer (shouldn't happen for valid examples)
            continue
        pairs.append(
            {
                "clean_prompt": ex["input"],
                "clean_answer": ex["gold_answer"],
                "corrupt_prompt": corrupt_input,
                "corrupt_answer": corrupt_answer,
                "chain_depth": ex["chain_depth"],
            }
        )
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--cache_dir",
        type=str,
        default=str(_ROOT / "data"),
        help="Directory to cache the BBH dataset.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: output/patching_pairs_<num_pairs>.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _DOL_INTERP = _ROOT.parents[2]
    if str(_DOL_INTERP) not in sys.path:
        sys.path.insert(0, str(_DOL_INTERP))
    if str(_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(_ROOT / "src"))
    from web_of_lies_eval import load_task_examples

    examples = load_task_examples(cache_dir=Path(args.cache_dir))
    print(f"Loaded {len(examples)} examples.")

    rng = np.random.default_rng(args.seed)
    pairs = make_pairs(examples, rng, args.num_pairs)
    if len(pairs) < args.num_pairs:
        print(
            f"Warning: only {len(pairs)} valid pairs from {len(examples)} examples "
            f"(requested {args.num_pairs})."
        )

    out_path = (
        Path(args.output)
        if args.output
        else _ROOT / "output" / f"patching_pairs_{args.num_pairs}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"num_pairs": len(pairs), "seed": args.seed, "pairs": pairs}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(pairs)} pairs to {out_path}")


if __name__ == "__main__":
    main()
