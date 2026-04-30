"""
Colored-objects eval for raw and chat-style prompting.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm import tqdm

def load_task_examples(task_json: str | Path) -> list[dict]:
    with open(task_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["examples"]


def correct_choices(example: dict) -> list[str]:
    return [k for k, v in example["target_scores"].items() if v == 1]


def count_objects_in_input(text: str) -> int:
    if "arranged in a row:" in text:
        items_text = text.split("arranged in a row:", 1)[1].split(". ", 1)[0].strip()
    else:
        m = re.search(
            r"(?:you see|I see|there is|there are)\s+(.*?)(?:\.\s|$)",
            text,
            flags=re.IGNORECASE,
        )
        items_text = m.group(1).strip() if m else text
    parts = re.split(r",\s+|\s+and\s+", items_text)
    return len([p for p in parts if p.strip()])


def build_chat_conversation(example: dict) -> tuple[list[dict[str, str]], list[str]]:
    choices = list(example["target_scores"].keys())
    user_text = (
        f"{example['input']}\n\n"
        "Answer with exactly one choice from this list:\n"
        f"{', '.join(choices)}\n"
        "One short answer only. No explanation."
    )
    return [{"role": "user", "content": user_text}], correct_choices(example)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def strip_reasoning_prefix(generated_suffix: str) -> str:
    text = generated_suffix.strip()
    if not text:
        return text

    # Remove full <think>...</think> blocks when present.
    text = re.sub(r"(?is)<think>.*?</think>", " ", text).strip()

    # If generation starts with a dangling thinking tag, drop the tag itself.
    text = re.sub(r"(?is)^<think>\s*", "", text).strip()

    # Remove common reasoning headers that some chat models emit.
    text = re.sub(
        r"(?is)^(thinking process|reasoning|analysis|scratchpad)\s*:\s*",
        "",
        text,
    ).strip()

    return text


def parse_model_completion(generated_suffix: str, choices: list[str]) -> str:
    text = normalize_text(strip_reasoning_prefix(generated_suffix))
    text = re.sub(r"^(answer|assistant)\s*:\s*", "", text)
    text = re.sub(r"^the answer is\s+", "", text)
    text = re.sub(r"^it is\s+", "", text)
    token = re.split(r"[\s\.\,\!\?\:\;]+", text, maxsplit=1)[0] if text else ""

    normalized_choices = {normalize_text(c): c for c in choices}
    if token in normalized_choices:
        return normalized_choices[token]
    if text in normalized_choices:
        return normalized_choices[text]
    for norm, original in sorted(normalized_choices.items(), key=lambda x: len(x[0]), reverse=True):
        if text.startswith(norm):
            return original
    return token


def stratified_indices_by_num_objects(
    examples: list[dict],
    rng: np.random.Generator,
    *,
    n_per_bin: int,
    allowed_bins: tuple[int, ...] | None = None,
    subtype: str | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    if allowed_bins is None:
        present = sorted(
            {
                count_objects_in_input(ex["input"])
                for ex in examples
                if subtype is None or ex.get("comment") == subtype
            }
        )
        allowed_bins = tuple(present)
    by_bin: dict[int, list[int]] = {k: [] for k in allowed_bins}
    for i, ex in enumerate(examples):
        if subtype is not None and ex.get("comment") != subtype:
            continue
        n = count_objects_in_input(ex["input"])
        if n in by_bin:
            by_bin[n].append(i)
    pool_sizes = {k: len(by_bin[k]) for k in allowed_bins}
    chosen: list[int] = []
    for k in allowed_bins:
        pool = by_bin[k]
        if len(pool) < n_per_bin:
            raise ValueError(
                f"num_objects={k}: need {n_per_bin} samples but only {len(pool)} in dataset"
            )
        pick = rng.choice(len(pool), size=n_per_bin, replace=False)
        chosen.extend(pool[int(j)] for j in pick)
    arr = np.array(chosen, dtype=np.int64)
    rng.shuffle(arr)
    return arr, pool_sizes


def evaluate_subset(
    examples: list[dict],
    model,
    tokenizer,
    indices: np.ndarray,
    max_new_tokens: int = 16,
    prompt_mode: Literal["continuation", "chat", "chat_continual"] = "continuation",
) -> dict:
    # Lazy import so `--help` does not need to initialize transformers/torch.
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
        choices = list(ex["target_scores"].keys())
        golds = correct_choices(ex)

        if prompt_mode == "continuation":
            model_inputs = prepare_model_inputs_raw(tokenizer, [ex["input"]], device)
        elif prompt_mode == "chat":
            conv, _ = build_chat_conversation(ex)
            model_inputs = prepare_model_inputs_conversation(tokenizer, conv, device)
        elif prompt_mode == "chat_continual":
            conv, _ = build_chat_conversation(ex)
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
            out[0, input_ids.shape[-1] :],
            skip_special_tokens=True,
        )
        pred = parse_model_completion(raw_suffix, choices)
        ok = pred in golds
        correct += int(ok)
        rows.append(
            {
                "i": int(i),
                "subtype": ex.get("comment", ""),
                "num_objects": count_objects_in_input(ex["input"]),
                "prompt_mode": prompt_mode,
                "input": ex["input"],
                "gold": golds,
                "pred": pred,
                "raw_suffix": raw_suffix[:200],
                "correct": ok,
            }
        )
    n = len(indices)
    return {
        "n": n,
        "prompt_mode": prompt_mode,
        "accuracy": correct / n,
        "rows": rows,
    }


def accuracy_by_num_objects(rows: list[dict]) -> dict[int, dict[str, float | int]]:
    bins = sorted({int(r["num_objects"]) for r in rows})
    out: dict[int, dict[str, float | int]] = {}
    for k in bins:
        sub = [r for r in rows if int(r["num_objects"]) == k]
        n = len(sub)
        out[k] = {
            "n": n,
            "accuracy": sum(1 for r in sub if r["correct"]) / n,
        }
    return out


def accuracy_by_subtype(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    bins = sorted({str(r["subtype"]) for r in rows})
    out: dict[str, dict[str, float | int]] = {}
    for k in bins:
        sub = [r for r in rows if str(r["subtype"]) == k]
        n = len(sub)
        out[k] = {
            "n": n,
            "accuracy": sum(1 for r in sub if r["correct"]) / n,
        }
    return out
