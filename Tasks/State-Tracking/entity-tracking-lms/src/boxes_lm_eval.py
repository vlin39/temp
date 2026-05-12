"""
Box entity-tracking eval (continuation vs chat), aligned with ``test.ipynb``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

# Repo layout: dol_interp/Tasks/State-Tracking/entity-tracking-lms/src/this_file.py
_DOL_INTERP = Path(__file__).resolve().parents[4]
if str(_DOL_INTERP) not in sys.path:
    sys.path.insert(0, str(_DOL_INTERP))

from Models.model_util import (  # noqa: E402
    load_model_and_tokenizer,
    prepare_model_inputs_conversation_continual,
    prepare_model_inputs_raw,
)

MASK_TOKEN = "<extra_id_0>"


def prompt_and_gold(row: dict) -> tuple[str, str]:
    ctx = row["sentence_masked"].split(MASK_TOKEN)[0].rstrip()
    gold = row["masked_content"].split(MASK_TOKEN)[1].lstrip()
    return ctx, gold


def target_box_from_prompt(context: str) -> int | None:
    m = re.search(r"Box (\d+) contains\s*$", context.rstrip())
    return int(m.group(1)) if m else None


def scenario_without_query(context: str) -> tuple[str, int]:
    box = target_box_from_prompt(context)
    if box is None:
        raise ValueError("continuation context must end with 'Box N contains'")
    scenario = re.sub(r"Box \d+ contains\s*$", "", context.rstrip(), count=1).rstrip()
    return scenario, box


def build_chat_conversation(row: dict) -> tuple[list[dict[str, str]], str, int]:
    context, gold = prompt_and_gold(row)
    scenario, box = scenario_without_query(context)
    user_text = (
        f"{scenario}\n\n"
        f"What does Box {box} contain? "
        "Answer with only the phrase that correctly completes "
        f'"Box {box} contains …" in this scenario '
        '(same style as elsewhere in the trace, e.g. "the apple and the key" or "nothing"). '
        "One short phrase only: no other boxes, no steps, no explanation."
    )

    assistant_text = (
        f"Box {box} contains"
    )

    return [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}], gold, box


_SPLIT_ITEMS = re.compile(r",?\s+and\s+")


def _strip_leading_article(item: str) -> str:
    w = item.strip().lower()
    for prefix in ("the ", "an ", "a "):
        if w.startswith(prefix):
            w = w[len(prefix) :].strip()
            break
    return w


def content_item_set(phrase: str) -> frozenset[str]:
    p = phrase.strip().lower()
    if p == "nothing" or p == "":
        return frozenset()
    parts = [x.strip() for x in _SPLIT_ITEMS.split(p) if x.strip()]
    return frozenset(_strip_leading_article(part) for part in parts)


def sets_match(pred: str, gold: str) -> bool:
    return content_item_set(pred) == content_item_set(gold)


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()


def strip_leading_contains(s: str) -> str:
    s = s.strip()
    low = s.lower()
    if low.startswith("contains "):
        s = s[9:].lstrip()
    return s


def parse_chat_completion(generated_suffix: str, target_box: int | None = None) -> str:
    t = generated_suffix.strip().split("\n", 1)[0].strip()
    if target_box is not None:
        t = re.sub(rf"(?i)^box\s+{target_box}\s+contains\s*", "", t)
    t = re.sub(r"(?i)^box\s+\d+\s+contains\s*", "", t)
    t = t.split(".", 1)[0].strip()
    return strip_leading_contains(t)


def parse_model_completion(generated_suffix: str, _context: str | None = None) -> str:
    t = generated_suffix.strip()
    t = t.split(".", 1)[0]
    return strip_leading_contains(t).rstrip(".").strip()


def stratified_indices_by_numops(
    ds,
    rng: np.random.Generator,
    *,
    n_per_bin: int,
    numops_bins: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
) -> tuple[np.ndarray, dict[int, int]]:
    by_bin: dict[int, list[int]] = {k: [] for k in numops_bins}
    for i in range(len(ds)):
        v = int(ds[i]["numops"])
        if v in by_bin:
            by_bin[v].append(i)
    pool_sizes = {k: len(by_bin[k]) for k in numops_bins}
    chosen: list[int] = []
    for k in numops_bins:
        pool = by_bin[k]
        if len(pool) < n_per_bin:
            raise ValueError(
                f"numops={k}: need {n_per_bin} samples but only {len(pool)} in dataset"
            )
        pick = rng.choice(len(pool), size=n_per_bin, replace=False)
        chosen.extend(pool[int(j)] for j in pick)
    arr = np.array(chosen, dtype=np.int64)
    rng.shuffle(arr)
    return arr, pool_sizes


def decode_new_tokens(tokenizer, input_ids, sequences) -> str:
    gen = sequences[0, input_ids.shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def _period_token_ids(tokenizer) -> list[int]:
    ids: set[int] = set()
    enc = tokenizer.encode(".", add_special_tokens=False)
    if len(enc) == 1:
        ids.add(enc[0])
    if ids:
        return list(ids)
    vs = getattr(tokenizer, "vocab_size", None)
    if vs is None:
        return []
    for i in range(vs):
        try:
            if tokenizer.decode([i]) == ".":
                ids.add(i)
        except Exception:
            pass
    return list(ids)


class StopAfterFirstPeriodToken(StoppingCriteria):
    def __init__(self, tokenizer):
        self.period_ids = torch.tensor(_period_token_ids(tokenizer), dtype=torch.long)

    def __call__(self, input_ids, scores, **kwargs) -> torch.Tensor:
        if self.period_ids.numel() == 0:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        last = input_ids[:, -1]
        ids = self.period_ids.to(device=last.device, dtype=last.dtype)
        return torch.isin(last, ids)


def evaluate_subset(
    ds,
    model,
    tokenizer,
    indices: np.ndarray,
    max_new_tokens: int = 64,
    prompt_mode: Literal["continuation", "chat"] = "continuation",
    eval_type: Literal["generation", "likelihood"] = "generation",
) -> dict:
    exact = 0
    set_ok = 0
    total_log_likelihood = 0.0
    total_probability = 0.0
    rows = []
    device = model.device
    continuation_stop = (
        StoppingCriteriaList([StopAfterFirstPeriodToken(tokenizer)])
        if prompt_mode == "continuation"
        else None
    )
    desc = f"Evaluating [{prompt_mode}, {eval_type}]: "
    for i in tqdm(indices, total=len(indices), desc=desc):
        row = ds[int(i)]
        if prompt_mode == "continuation":
            context, gold = prompt_and_gold(row)
            model_inputs = prepare_model_inputs_raw(tokenizer, context, device)

            def _parse(raw: str) -> str:
                return parse_model_completion(raw, context)

            parse = _parse
        elif prompt_mode == "chat":
            conv, gold, box = build_chat_conversation(row)
            model_inputs = prepare_model_inputs_conversation_continual(tokenizer, conv, device)

            def _parse_chat(raw: str) -> str:
                return parse_chat_completion(raw, target_box=box)

            parse = _parse_chat
        else:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")
        input_ids = model_inputs["input_ids"]
        if eval_type == "generation":
            gen_kw = dict(
                **model_inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            if prompt_mode == "continuation":
                gen_kw["stopping_criteria"] = continuation_stop
            out = model.generate(**gen_kw)
            raw_suffix = decode_new_tokens(tokenizer, input_ids, out)
            pred = parse(raw_suffix)
            e = exact_match(pred, gold)
            s = sets_match(pred, gold)
            exact += int(e)
            set_ok += int(s)
            out_row = {
                "i": int(i),
                "sample_id": row["sample_id"],
                "numops": int(row["numops"]),
                "prompt_mode": prompt_mode,
                "eval_type": eval_type,
                "gold": gold,
                "pred": pred,
                "raw_suffix": raw_suffix[:200],
                "exact": e,
                "set_match": s,
            }
        elif eval_type == "likelihood":
            gt_cont = " " + gold
            gt_ids = tokenizer(
                gt_cont,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(device)
            gt_len = int(gt_ids.shape[-1])
            if gt_len <= 0:
                raise ValueError("Gold continuation tokenized to 0 tokens in likelihood mode")

            full_input_ids = torch.cat([input_ids, gt_ids], dim=-1)
            full_inputs = {"input_ids": full_input_ids}
            if "attention_mask" in model_inputs:
                full_inputs["attention_mask"] = torch.cat(
                    [
                        model_inputs["attention_mask"],
                        torch.ones_like(gt_ids, device=model_inputs["attention_mask"].device),
                    ],
                    dim=-1,
                )

            with torch.no_grad():
                logits = model(**full_inputs).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            targets = full_input_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            prompt_len = int(input_ids.shape[-1])
            start = prompt_len - 1
            gt_token_log_probs = token_log_probs[:, start : start + gt_len]
            gt_log_likelihood = float(gt_token_log_probs.sum().item())
            gt_probability = float(np.exp(gt_log_likelihood))
            total_log_likelihood += gt_log_likelihood
            total_probability += gt_probability

            out_row = {
                "i": int(i),
                "sample_id": row["sample_id"],
                "numops": int(row["numops"]),
                "prompt_mode": prompt_mode,
                "eval_type": eval_type,
                "gold": gold,
                "gold_continuation_scored": gt_cont,
                "gold_num_tokens": gt_len,
                "gold_log_likelihood": gt_log_likelihood,
                "gold_probability": gt_probability,
            }
        else:
            raise ValueError(f"Unknown eval_type: {eval_type!r}")
        if "numops_by_op" in row:
            out_row["numops_by_op"] = {
                str(k): int(v) for k, v in dict(row["numops_by_op"]).items()
            }
        rows.append(out_row)
        print(out_row)
    n = len(indices)
    return {
        "n": n,
        "prompt_mode": prompt_mode,
        "eval_type": eval_type,
        "exact_accuracy": (exact / n) if eval_type == "generation" else None,
        "set_accuracy": (set_ok / n) if eval_type == "generation" else None,
        "mean_log_likelihood": (total_log_likelihood / n) if eval_type == "likelihood" else None,
        "mean_probability": (total_probability / n) if eval_type == "likelihood" else None,
        "rows": rows,
    }


def accuracy_by_numops(rows: list[dict]) -> dict[int, dict[str, float | int]]:
    bins = sorted({int(r["numops"]) for r in rows})
    out: dict[int, dict[str, float | int]] = {}
    for k in bins:
        sub = [r for r in rows if int(r["numops"]) == k]
        n = len(sub)
        out[k] = {
            "n": n,
            "exact_accuracy": sum(1 for r in sub if r["exact"]) / n,
            "set_accuracy": sum(1 for r in sub if r["set_match"]) / n,
        }
    return out


def likelihood_by_numops(rows: list[dict]) -> dict[int, dict[str, float | int]]:
    bins = sorted({int(r["numops"]) for r in rows})
    out: dict[int, dict[str, float | int]] = {}
    for k in bins:
        sub = [r for r in rows if int(r["numops"]) == k]
        n = len(sub)
        out[k] = {
            "n": n,
            "mean_log_likelihood": sum(float(r["gold_log_likelihood"]) for r in sub) / n,
            "mean_probability": sum(float(r["gold_probability"]) for r in sub) / n,
        }
    return out
