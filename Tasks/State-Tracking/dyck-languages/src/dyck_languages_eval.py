"""
Dyck-languages eval for raw and chat-style prompting.

Loads the BBH dyck_languages task via
`datasets.load_dataset("lukaemon/bbh", "dyck_languages")`.
Each example is a bracket sequence; the model must answer "yes" (valid /
balanced) or "no" (invalid). The sequence may contain up to four bracket
types: ( ) [ ] { } < >.

Difficulty metrics stored per example:
  seq_len   — total number of bracket tokens in the input
  max_depth — maximum nesting depth reached during the sequence
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
_MATCH = {')': '(', ']': '[', '}': '{', '>': '<'}


def _tokenize_brackets(text: str) -> list[str]:
    """Return only the bracket characters from text."""
    return [ch for ch in text if ch in _OPEN or ch in _CLOSE]


def _seq_len(tokens: list[str]) -> int:
    return len(tokens)


def _max_depth(tokens: list[str]) -> int:
    depth, peak = 0, 0
    for t in tokens:
        if t in _OPEN:
            depth += 1
            peak = max(peak, depth)
        else:
            depth -= 1
    return peak


def load_task_examples(cache_dir: Path | None = None) -> list[dict]:
    """Load BBH dyck_languages from HuggingFace and return a list of dicts.

    Each dict has keys:
      input        — raw input text from the dataset
      target       — raw target ("Yes" / "No")
      gold_answer  — lowercased, stripped ("yes" / "no")
      seq_len      — number of bracket tokens in input
      max_depth    — maximum nesting depth in input
    """
    from datasets import load_dataset

    kwargs: dict = {"path": "lukaemon/bbh", "name": "dyck_languages"}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    ds = load_dataset(**kwargs)
    examples: list[dict] = []
    for split_name in ds:
        for row in ds[split_name]:
            tokens = _tokenize_brackets(row["input"])
            examples.append(
                {
                    "input":       row["input"],
                    "target":      row["target"],
                    "gold_answer": row["target"].strip().lower().rstrip("."),
                    "seq_len":     _seq_len(tokens),
                    "max_depth":   _max_depth(tokens),
                }
            )
    return examples


def stratified_indices_by_answer(
    examples: list[dict],
    rng: np.random.Generator,
    *,
    n_per_bin: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return (indices, pool_sizes) sampling n_per_bin from each answer class."""
    by_answer: dict[str, list[int]] = {"yes": [], "no": []}
    for i, ex in enumerate(examples):
        gold = ex["gold_answer"]
        if gold in by_answer:
            by_answer[gold].append(i)
    pool_sizes = {k: len(v) for k, v in by_answer.items()}
    chosen: list[int] = []
    for label in ("yes", "no"):
        pool = by_answer[label]
        if len(pool) < n_per_bin:
            raise ValueError(
                f"answer={label!r}: need {n_per_bin} samples but only {len(pool)} available"
            )
        pick = rng.choice(len(pool), size=n_per_bin, replace=False)
        chosen.extend(pool[int(j)] for j in pick)
    arr = np.array(chosen, dtype=np.int64)
    rng.shuffle(arr)
    return arr, pool_sizes


def build_chat_conversation(ex: dict) -> list[dict]:
    user_text = (
        f"{ex['input']}\n\n"
        "Answer with exactly one word: yes or no.\n"
        "One short answer only. No explanation."
    )
    return [{"role": "user", "content": user_text}]


def build_chat_conversation_continual(ex: dict) -> list[dict]:
    return [
        {"role": "user", "content": ex["input"]},
        {"role": "assistant", "content": "Answer:"},
    ]


_CHOICES = ("yes", "no")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _strip_reasoning(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", " ", text).strip()
    text = re.sub(r"(?is)^<think>\s*", "", text).strip()
    text = re.sub(
        r"(?is)^(thinking process|reasoning|analysis|scratchpad)\s*:\s*",
        "",
        text,
    ).strip()
    return text


def parse_model_completion(raw_suffix: str) -> str:
    """Return 'yes', 'no', or '' (unrecognized)."""
    text = _normalize(_strip_reasoning(raw_suffix))
    text = re.sub(r"^(answer|assistant)\s*:\s*", "", text)
    text = re.sub(r"^the answer is\s+", "", text)
    text = re.sub(r"^it is\s+", "", text)
    token = re.split(r"[\s.\,!\?:;]+", text, maxsplit=1)[0] if text else ""
    if token in _CHOICES:
        return token
    if text in _CHOICES:
        return text
    for choice in _CHOICES:
        if text.startswith(choice):
            return choice
    return ""


def evaluate_subset(
    examples: list[dict],
    model,
    tokenizer,
    indices: np.ndarray,
    *,
    max_new_tokens: int = 8,
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

    correct = 0
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
        pred = parse_model_completion(raw_suffix)
        ok = pred == ex["gold_answer"]
        correct += int(ok)
        rows.append(
            {
                "i":           int(i),
                "seq_len":     ex["seq_len"],
                "max_depth":   ex["max_depth"],
                "prompt_mode": prompt_mode,
                "input":       ex["input"][:300],
                "gold":        ex["gold_answer"],
                "pred":        pred,
                "raw_suffix":  raw_suffix[:200],
                "correct":     ok,
            }
        )
    n = len(indices)
    return {
        "n":           n,
        "prompt_mode": prompt_mode,
        "accuracy":    correct / n,
        "rows":        rows,
    }


def accuracy_by_answer(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for label in _CHOICES:
        sub = [r for r in rows if r["gold"] == label]
        n = len(sub)
        out[label] = {
            "n":        n,
            "accuracy": sum(1 for r in sub if r["correct"]) / n if n else 0.0,
        }
    return out


def accuracy_by_seq_len(rows: list[dict]) -> dict[int, dict]:
    lengths = sorted({int(r["seq_len"]) for r in rows})
    out: dict[int, dict] = {}
    for sl in lengths:
        sub = [r for r in rows if int(r["seq_len"]) == sl]
        n = len(sub)
        out[sl] = {
            "n":        n,
            "accuracy": sum(1 for r in sub if r["correct"]) / n if n else 0.0,
        }
    return out


def accuracy_by_depth(rows: list[dict]) -> dict[int, dict]:
    depths = sorted({int(r["max_depth"]) for r in rows})
    out: dict[int, dict] = {}
    for d in depths:
        sub = [r for r in rows if int(r["max_depth"]) == d]
        n = len(sub)
        out[d] = {
            "n":        n,
            "accuracy": sum(1 for r in sub if r["correct"]) / n if n else 0.0,
        }
    return out
