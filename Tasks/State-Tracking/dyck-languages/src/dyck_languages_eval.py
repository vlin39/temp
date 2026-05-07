"""
Dyck-languages eval (BBH).

The BBH dyck_languages task is a **completion** task: given an incomplete
Dyck-4 word (e.g. "[ ( { < ( ) > }"), the model must produce the sequence
of closing brackets that completes it (e.g. "} ]"). The four bracket types
are () [] {} <>.

Loaded via `datasets.load_dataset("lukaemon/bbh", "dyck_languages")`.

Difficulty metrics stored per example:
  target_len      — number of closing-bracket tokens in the gold target
  max_open_depth  — peak nesting depth reached during the input prefix
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm import tqdm


_OPEN  = set("([{<")
_CLOSE = set(")]}>" )
_PAIR  = {'(': ')', '[': ']', '{': '}', '<': '>'}


def _bracket_tokens(text: str) -> list[str]:
    return [ch for ch in text if ch in _OPEN or ch in _CLOSE]


def _max_open_depth(tokens: list[str]) -> int:
    depth, peak = 0, 0
    for t in tokens:
        if t in _OPEN:
            depth += 1
            peak = max(peak, depth)
        else:
            depth -= 1
    return peak


def _normalize_target(target: str) -> list[str]:
    """Extract the bracket sequence from a gold target string."""
    return _bracket_tokens(target)


def load_task_examples(cache_dir: Path | None = None) -> list[dict]:
    """Load BBH dyck_languages from HuggingFace.

    Each returned dict has:
      input            — raw input text (instructions + bracket prefix)
      target           — raw target string (closing bracket sequence)
      gold_tokens      — list[str] of bracket chars from target (canonical answer)
      target_len       — len(gold_tokens)
      max_open_depth   — peak nesting depth in the input prefix
    """
    from datasets import load_dataset

    kwargs: dict = {"path": "lukaemon/bbh", "name": "dyck_languages"}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    ds = load_dataset(**kwargs)
    examples: list[dict] = []
    for split_name in ds:
        for row in ds[split_name]:
            input_tokens = _bracket_tokens(row["input"])
            gold_tokens  = _normalize_target(row["target"])
            examples.append(
                {
                    "input":           row["input"],
                    "target":          row["target"],
                    "gold_tokens":     gold_tokens,
                    "target_len":      len(gold_tokens),
                    "max_open_depth":  _max_open_depth(input_tokens),
                }
            )
    return examples


def sample_indices(
    examples: list[dict],
    rng: np.random.Generator,
    n_total: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Random sample (without replacement) up to n_total indices.

    Returns (indices, pool_sizes_by_target_len) for logging.
    """
    by_len: dict[int, list[int]] = {}
    for i, ex in enumerate(examples):
        by_len.setdefault(ex["target_len"], []).append(i)
    pool_sizes = {k: len(v) for k, v in sorted(by_len.items())}

    n_take = min(n_total, len(examples))
    arr    = rng.choice(len(examples), size=n_take, replace=False).astype(np.int64)
    rng.shuffle(arr)
    return arr, pool_sizes


def build_chat_conversation(ex: dict) -> list[dict]:
    return [{"role": "user", "content": ex["input"]}]


def build_chat_conversation_continual(ex: dict) -> list[dict]:
    """Pre-fill 'Answer:' as the assistant turn."""
    return [
        {"role": "user", "content": ex["input"]},
        {"role": "assistant", "content": "Answer:"},
    ]


def parse_model_completion(raw_suffix: str) -> list[str]:
    """Extract the closing-bracket sequence from a model completion.

    Strategy: strip any <think>…</think> reasoning, then take all bracket
    characters in their order of appearance.  Stops at first non-bracket
    non-whitespace character to avoid mixing brackets from later text
    (e.g., a follow-up sentence).
    """
    text = re.sub(r"(?is)<think>.*?</think>", " ", raw_suffix).strip()
    text = re.sub(r"(?is)^<think>\s*", "", text).strip()
    text = re.sub(r"(?is)^(answer|the answer is)\s*:?\s*", "", text)

    tokens: list[str] = []
    for ch in text:
        if ch in _OPEN or ch in _CLOSE:
            tokens.append(ch)
        elif ch.isspace():
            continue
        else:
            break  # stop at first non-bracket non-space char
    return tokens


def _exact_match(pred: list[str], gold: list[str]) -> bool:
    return pred == gold


def _prefix_match_count(pred: list[str], gold: list[str]) -> int:
    """Number of leading positions where pred matches gold."""
    n = 0
    for p, g in zip(pred, gold):
        if p == g:
            n += 1
        else:
            break
    return n


def evaluate_subset(
    examples: list[dict],
    model,
    tokenizer,
    indices: np.ndarray,
    *,
    max_new_tokens: int = 16,
    prompt_mode: Literal["continuation", "chat", "chat_continual"] = "continuation",
) -> dict:
    _DOL_INTERP = Path(__file__).resolve().parents[4]
    if str(_DOL_INTERP) not in sys.path:
        sys.path.insert(0, str(_DOL_INTERP))
    from Models.model_util import (
        prepare_model_inputs_conversation,
        prepare_model_inputs_conversation_continual,
        prepare_model_inputs_raw,
    )

    n_correct = 0
    sum_prefix_match = 0
    sum_target_len   = 0
    rows = []
    device = model.device
    desc = f"Evaluating [{prompt_mode}]: "
    for i in tqdm(indices, total=len(indices), desc=desc):
        ex = examples[int(i)]

        if prompt_mode == "continuation":
            model_inputs = prepare_model_inputs_raw(tokenizer, [ex["input"]], device)
        elif prompt_mode == "chat":
            conv = build_chat_conversation(ex)
            model_inputs = prepare_model_inputs_conversation(tokenizer, conv, device)
        elif prompt_mode == "chat_continual":
            conv = build_chat_conversation_continual(ex)
            model_inputs = prepare_model_inputs_conversation_continual(tokenizer, conv, device)
        else:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")

        input_ids = model_inputs["input_ids"]
        out = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        raw_suffix = tokenizer.decode(
            out[0, input_ids.shape[-1]:],
            skip_special_tokens=True,
        )
        pred_tokens = parse_model_completion(raw_suffix)
        gold_tokens = ex["gold_tokens"]

        is_exact   = _exact_match(pred_tokens, gold_tokens)
        prefix_n   = _prefix_match_count(pred_tokens, gold_tokens)
        n_correct += int(is_exact)
        sum_prefix_match += prefix_n
        sum_target_len   += len(gold_tokens)

        rows.append(
            {
                "i":              int(i),
                "target_len":     ex["target_len"],
                "max_open_depth": ex["max_open_depth"],
                "prompt_mode":    prompt_mode,
                "input":          ex["input"][:300],
                "gold":           " ".join(gold_tokens),
                "pred":           " ".join(pred_tokens),
                "raw_suffix":     raw_suffix[:200],
                "exact":          is_exact,
                "prefix_match":   prefix_n,
            }
        )
    n = len(indices)
    return {
        "n":                  n,
        "prompt_mode":        prompt_mode,
        "exact_accuracy":     n_correct / n if n else 0.0,
        "token_recall":       (sum_prefix_match / sum_target_len) if sum_target_len else 0.0,
        "rows":               rows,
    }


def accuracy_by_target_len(rows: list[dict]) -> dict[int, dict]:
    lens = sorted({int(r["target_len"]) for r in rows})
    out: dict[int, dict] = {}
    for tl in lens:
        sub = [r for r in rows if int(r["target_len"]) == tl]
        n   = len(sub)
        out[tl] = {
            "n":              n,
            "exact_accuracy": sum(1 for r in sub if r["exact"]) / n if n else 0.0,
            "token_recall":   (
                sum(r["prefix_match"] for r in sub) / sum(r["target_len"] for r in sub)
                if sub else 0.0
            ),
        }
    return out


def accuracy_by_depth(rows: list[dict]) -> dict[int, dict]:
    depths = sorted({int(r["max_open_depth"]) for r in rows})
    out: dict[int, dict] = {}
    for d in depths:
        sub = [r for r in rows if int(r["max_open_depth"]) == d]
        n   = len(sub)
        out[d] = {
            "n":              n,
            "exact_accuracy": sum(1 for r in sub if r["exact"]) / n if n else 0.0,
            "token_recall":   (
                sum(r["prefix_match"] for r in sub) / sum(r["target_len"] for r in sub)
                if sub else 0.0
            ),
        }
    return out
