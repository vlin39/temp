#!/usr/bin/env python3
"""
Run layer-wise activation patching on paired box-tracking examples.

The expected input is the JSONL produced by ``generate_activation_patching_boxes.py``:
variants with the same ``skeleton_id`` share the same token-level structure but
use different object names. This script treats one variant as clean, another as
corrupt, patches clean decoder-layer outputs into the corrupt forward pass, and
measures the normalized clean-minus-corrupt-answer logit difference.

Example:

  python scripts/run_activation_patching_boxes.py \
    --model_name Qwen/Qwen3-8B \
    --data_file data/activation_patching_boxes6_1item_nops12/test-t5.jsonl \
    --layers 0,1,2,3 \
    --max_pairs 50
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

_DOL_INTERP = _ROOT.parents[2]
if str(_DOL_INTERP) not in sys.path:
    sys.path.insert(0, str(_DOL_INTERP))

from boxes_lm_eval import (  # noqa: E402
    build_chat_conversation,
    load_model_and_tokenizer,
    prompt_and_gold,
    scenario_without_query,
)
from Models.model_util import prepare_model_inputs_raw  # noqa: E402


LocationName = str
PromptMode = Literal["continuation", "chat"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def model_slug(model_name: str) -> str:
    slug = model_name.replace("/", "--")
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return slug.strip("._-") or "model"


def default_output_path(model_name: str, data_file: Path, prompt_mode: str) -> Path:
    return (
        _ROOT
        / "output"
        / "activation_patching"
        / f"{model_slug(model_name)}_{data_file.resolve().parent.name}_{prompt_mode}.jsonl"
    )


def get_input_device(model: torch.nn.Module) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cpu")


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList | list[torch.nn.Module]:
    candidates = (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for path in candidates:
        obj: Any = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, (torch.nn.ModuleList, list, tuple)) and len(obj) > 0:
            return obj
    raise AttributeError(
        "Could not find decoder layers. Tried model.layers, model.decoder.layers, "
        "transformer.h, and gpt_neox.layers."
    )


def parse_layers(raw: str | None, num_layers: int) -> list[int]:
    if raw is None or raw.strip().lower() == "all":
        return list(range(num_layers))
    layers: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx += num_layers
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {part!r} is out of range for {num_layers} layers")
        layers.append(idx)
    if not layers:
        raise ValueError("--layers did not contain any layer indices")
    return layers


def answer_token_ids(
    tokenizer,
    answer: str,
    *,
    prompt_mode: PromptMode,
) -> list[int]:
    if prompt_mode == "chat" and answer.strip().startswith("the "):
        text = " " + answer.strip()[len("the ") :]
    else:
        text = " " + answer.strip()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Answer tokenized to zero tokens: {answer!r}")
    if prompt_mode == "chat":
        return [int(ids[0])]
    return [int(i) for i in ids]


def chat_answers_have_object_prefix(*answers: str) -> bool:
    return all(answer.strip().startswith("the ") for answer in answers)


def add_chat_assistant_object_prefix(
    conversation: list[dict[str, str]],
) -> list[dict[str, str]]:
    out = [dict(message) for message in conversation]
    out[-1]["content"] = out[-1]["content"].rstrip() + " the"
    return out


def score_answer_from_logits(logits: torch.Tensor, token_ids: Sequence[int]) -> float:
    return float(logits[list(token_ids)].float().sum().item())


def answer_logit_diff(
    logits: torch.Tensor,
    clean_answer_ids: Sequence[int],
    corrupt_answer_ids: Sequence[int],
) -> float:
    return score_answer_from_logits(logits, clean_answer_ids) - score_answer_from_logits(
        logits, corrupt_answer_ids
    )


def forward_logits(
    model: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(**model_inputs, use_cache=False, return_dict=True)
    return outputs.logits[0, -1, :].detach().float().cpu()


def _hidden_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, list):
        return output[0]
    return output


def _replace_hidden_in_layer_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def collect_layer_activation(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    cache: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        cache["activation"] = _hidden_from_layer_output(output).detach()

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**model_inputs, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if "activation" not in cache:
        raise RuntimeError("Layer hook did not capture an activation")
    return cache["activation"]


def patched_forward_logits(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    clean_activation: torch.Tensor,
    corrupt_inputs: dict[str, torch.Tensor],
    positions: Sequence[int],
) -> torch.Tensor:
    if not positions:
        raise ValueError("Cannot patch an empty position set")
    position_tensor = torch.tensor(sorted(set(int(p) for p in positions)), dtype=torch.long)

    def hook(_module, _inputs, output):
        hidden = _hidden_from_layer_output(output)
        pos = position_tensor.to(hidden.device)
        patched = hidden.clone()
        clean = clean_activation.to(device=hidden.device, dtype=hidden.dtype)
        patched[:, pos, :] = clean[:, pos, :]
        return _replace_hidden_in_layer_output(output, patched)

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model(**corrupt_inputs, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    return outputs.logits[0, -1, :].detach().float().cpu()


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[^.]*\.", text):
        start, end = match.span()
        sent = text[start:end].strip()
        if sent:
            spans.append((start, end, sent))
    return spans


def context_statement_spans(text: str) -> list[tuple[int, int, str]]:
    """
    Return patchable context units.

    The generated context starts with one comma-separated initial-state sentence:
    ``Box 0 contains ..., Box 1 contains ..., ... .``. Treating that whole
    sentence as one unit would make every target-box initial-state patch include
    all boxes. We split only that initial inventory into per-box clauses and keep
    subsequent operation sentences intact.
    """
    spans = sentence_spans(text)
    if not spans:
        return []

    first_start, first_end, first_sent = spans[0]
    if not re.match(r"\s*Box \d+ contains\b", first_sent):
        return spans

    out: list[tuple[int, int, str]] = []
    first_text = text[first_start:first_end]
    clause_pattern = re.compile(
        r"Box \d+ contains .*?(?=,\s*Box \d+ contains |\.$)",
        flags=re.DOTALL,
    )
    for match in clause_pattern.finditer(first_text):
        start = first_start + match.start()
        end = first_start + match.end()
        if end < len(text) and text[end] in ",.":
            end += 1
        clause = text[start:end].strip()
        if clause:
            out.append((start, end, clause))

    if not out:
        out.append((first_start, first_end, first_sent))
    out.extend(spans[1:])
    return out


def character_spans_for_locations(
    scenario: str,
    target_box: int,
) -> dict[LocationName, list[tuple[int, int]]]:
    box_pattern = re.compile(rf"\bBox {target_box}\b")
    target_spans: list[tuple[int, int]] = []
    control_spans: list[tuple[int, int]] = []
    for start, end, sent in context_statement_spans(scenario):
        if box_pattern.search(sent):
            target_spans.append((start, end))
        else:
            control_spans.append((start, end))
    return {
        "all_context": [(0, len(scenario))],
        "target_box_sentences": target_spans,
        "last_arget_box_sentence": target_spans[-1:] if target_spans else [],
        "non_target_box_sentences": control_spans,
    }


def token_positions_for_spans(
    tokenizer,
    prompt: str,
    spans: Sequence[tuple[int, int]],
    *,
    add_special_tokens: bool,
) -> list[int]:
    if not spans:
        return []
    encoded = tokenizer(
        prompt,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
        return_token_type_ids=False,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer did not return offset mappings; a fast tokenizer is required")

    positions: list[int] = []
    for token_idx, offset in enumerate(offsets):
        start, end = int(offset[0]), int(offset[1])
        if start == end:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            positions.append(token_idx)
    return positions


def offset_spans(spans: Sequence[tuple[int, int]], offset: int) -> list[tuple[int, int]]:
    return [(start + offset, end + offset) for start, end in spans]


def post_scenario_box_spans(
    prompt: str,
    *,
    target_box: int,
    scenario_offset: int,
    scenario_length: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Find every ``Box {target_box}`` occurrence that appears after the scenario text.

    The last occurrence is the assistant-turn response prefix ("Box X contains …").
    All earlier occurrences are user-turn query mentions ("What does Box X contain?",
    the example '"Box X contains …"' in the instructions, etc.).

    Returns:
        query_spans:    all occurrences except the last (user-turn mentions)
        response_spans: the last occurrence only (assistant-turn prefix)
    """
    query_start = scenario_offset + scenario_length
    pattern = re.compile(rf"\bBox {target_box}\b")
    all_spans = [m.span() for m in pattern.finditer(prompt, pos=query_start)]
    if not all_spans:
        raise ValueError(
            f"Could not find 'Box {target_box}' after the scenario in the rendered prompt"
        )
    return all_spans[:-1], all_spans[-1:]


def render_chat_prompt(tokenizer, conversation: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )


def prepare_model_inputs_from_text(
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    add_special_tokens: bool,
) -> dict[str, torch.Tensor]:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        return_token_type_ids=False,
        add_special_tokens=add_special_tokens,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def prompts_and_span_inputs(
    row: dict[str, Any],
    tokenizer,
    device: torch.device,
    *,
    prompt_mode: PromptMode,
) -> dict[str, Any]:
    context, answer = prompt_and_gold(row)
    scenario, box = scenario_without_query(context)

    if prompt_mode == "continuation":
        prompt = context
        inputs = prepare_model_inputs_raw(tokenizer, prompt, device)
        scenario_offset = 0
        add_special_tokens_for_offsets = True
        query_box_spans: list[tuple[int, int]] = []
        response_target_box_spans: list[tuple[int, int]] = []
    elif prompt_mode == "chat":
        conversation, chat_answer, chat_box = build_chat_conversation(row)
        if chat_answer != answer or chat_box != box:
            raise ValueError("Chat prompt construction disagrees with continuation prompt")
        if chat_answers_have_object_prefix(answer):
            conversation = add_chat_assistant_object_prefix(conversation)
        prompt = render_chat_prompt(tokenizer, conversation)
        scenario_offset = prompt.find(scenario)
        if scenario_offset < 0:
            raise ValueError("Could not locate scenario text inside rendered chat prompt")
        inputs = prepare_model_inputs_from_text(
            tokenizer,
            prompt,
            device,
            add_special_tokens=False,
        )
        add_special_tokens_for_offsets = False
        query_box_spans, response_target_box_spans = post_scenario_box_spans(
            prompt,
            target_box=box,
            scenario_offset=scenario_offset,
            scenario_length=len(scenario),
        )
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")

    return {
        "prompt": prompt,
        "scenario": scenario,
        "scenario_offset": scenario_offset,
        "answer": answer,
        "target_box": box,
        "inputs": inputs,
        "add_special_tokens_for_offsets": add_special_tokens_for_offsets,
        "query_box_spans": query_box_spans,
        "response_target_box_spans": response_target_box_spans,
    }


def prepare_pair(
    clean_row: dict[str, Any],
    corrupt_row: dict[str, Any],
    tokenizer,
    device: torch.device,
    *,
    prompt_mode: PromptMode,
) -> dict[str, Any] | None:
    clean_prepared = prompts_and_span_inputs(
        clean_row,
        tokenizer,
        device,
        prompt_mode=prompt_mode,
    )
    corrupt_prepared = prompts_and_span_inputs(
        corrupt_row,
        tokenizer,
        device,
        prompt_mode=prompt_mode,
    )
    clean_box = int(clean_prepared["target_box"])
    corrupt_box = int(corrupt_prepared["target_box"])
    if clean_box != corrupt_box:
        raise ValueError(
            f"Paired variants query different boxes: clean={clean_box}, corrupt={corrupt_box}"
        )

    clean_inputs = clean_prepared["inputs"]
    corrupt_inputs = corrupt_prepared["inputs"]
    if clean_inputs["input_ids"].shape != corrupt_inputs["input_ids"].shape:
        raise ValueError(
            "Clean/corrupt prompts have different token shapes: "
            f"{tuple(clean_inputs['input_ids'].shape)} vs {tuple(corrupt_inputs['input_ids'].shape)}"
        )

    clean_locations = character_spans_for_locations(clean_prepared["scenario"], clean_box)
    corrupt_locations = character_spans_for_locations(corrupt_prepared["scenario"], corrupt_box)
    positions_by_location = {
        name: token_positions_for_spans(
            tokenizer,
            clean_prepared["prompt"],
            offset_spans(spans, int(clean_prepared["scenario_offset"])),
            add_special_tokens=bool(clean_prepared["add_special_tokens_for_offsets"]),
        )
        for name, spans in clean_locations.items()
    }
    corrupt_positions_by_location = {
        name: token_positions_for_spans(
            tokenizer,
            corrupt_prepared["prompt"],
            offset_spans(spans, int(corrupt_prepared["scenario_offset"])),
            add_special_tokens=bool(corrupt_prepared["add_special_tokens_for_offsets"]),
        )
        for name, spans in corrupt_locations.items()
    }
    if prompt_mode == "chat":
        for loc, key in [
            ("query_box", "query_box_spans"),
            ("response_target_box", "response_target_box_spans"),
        ]:
            positions_by_location[loc] = token_positions_for_spans(
                tokenizer,
                clean_prepared["prompt"],
                clean_prepared[key],
                add_special_tokens=bool(clean_prepared["add_special_tokens_for_offsets"]),
            )
            corrupt_positions_by_location[loc] = token_positions_for_spans(
                tokenizer,
                corrupt_prepared["prompt"],
                corrupt_prepared[key],
                add_special_tokens=bool(corrupt_prepared["add_special_tokens_for_offsets"]),
            )

    # last_token: sanity-check location — both inputs are guaranteed equal length
    n_tokens = int(clean_inputs["input_ids"].shape[1])
    positions_by_location["last_token"] = [n_tokens - 1]
    corrupt_positions_by_location["last_token"] = [n_tokens - 1]

    for name, positions in positions_by_location.items():
        other = corrupt_positions_by_location[name]
        if positions != other:
            raise ValueError(
                f"Clean/corrupt token positions differ for {name}: {positions} vs {other}"
            )

    return {
        "clean_prompt": clean_prepared["prompt"],
        "corrupt_prompt": corrupt_prepared["prompt"],
        "clean_answer": clean_prepared["answer"],
        "corrupt_answer": corrupt_prepared["answer"],
        "target_box": clean_box,
        "clean_inputs": clean_inputs,
        "corrupt_inputs": corrupt_inputs,
        "positions_by_location": positions_by_location,
    }


def build_pairs(
    rows: Sequence[dict[str, Any]],
    *,
    clean_variant_id: int,
    corrupt_variant_id: int,
    skip_same_answers: bool,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_skeleton: dict[Any, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if "skeleton_id" not in row or "variant_id" not in row:
            continue
        by_skeleton[row["skeleton_id"]][int(row["variant_id"])] = row

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for _skeleton_id, variants in sorted(by_skeleton.items(), key=lambda kv: kv[0]):
        if clean_variant_id not in variants or corrupt_variant_id not in variants:
            continue
        clean = variants[clean_variant_id]
        corrupt = variants[corrupt_variant_id]
        if skip_same_answers and prompt_and_gold(clean)[1] == prompt_and_gold(corrupt)[1]:
            continue
        pairs.append((clean, corrupt))
    return pairs


def normalized_logit_diff(
    patched_diff: float,
    clean_diff: float,
    corrupt_diff: float,
) -> float | None:
    denom = clean_diff - corrupt_diff
    if math.isclose(denom, 0.0, abs_tol=1e-12):
        return None
    return (patched_diff - corrupt_diff) / denom


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    buckets: dict[tuple[int, str], list[float]] = defaultdict(list)
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in rows:
        key = (int(row["layer_idx"]), str(row["location"]))
        counts[key] += 1
        value = row.get("normalized_logit_diff")
        if value is not None:
            buckets[key].append(float(value))
    out: dict[str, dict[str, float | int]] = {}
    for key in sorted(counts):
        vals = buckets.get(key, [])
        layer_idx, location = key
        out[f"layer_{layer_idx}/{location}"] = {
            "n": counts[key],
            "n_finite": len(vals),
            "mean_normalized_logit_diff": sum(vals) / len(vals) if vals else float("nan"),
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument(
        "--data_file",
        type=Path,
        default=_ROOT / "data" / "activation_patching_boxes6_1item_nops12" / "test-t5.jsonl",
    )
    p.add_argument("--output_file", type=Path, default=None)
    p.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated decoder layer indices, or 'all' (default). Negative indices allowed.",
    )
    p.add_argument(
        "--prompt_mode",
        type=str,
        choices=("continuation", "chat"),
        default="chat",
        help=(
            "continuation: raw prefix ending in 'Box k contains'; "
            "chat: render the same chat prompt style as boxes_lm_eval.py."
        ),
    )
    p.add_argument("--clean_variant_id", type=int, default=0)
    p.add_argument("--corrupt_variant_id", type=int, default=1)
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--skip_same_answers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip pairs where clean and corrupt answers are identical (default: true).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_file.is_file():
        sys.exit(f"Data file not found: {args.data_file}")
    if args.clean_variant_id == args.corrupt_variant_id:
        sys.exit("--clean_variant_id and --corrupt_variant_id must differ")

    rows = read_jsonl(args.data_file)
    pairs = build_pairs(
        rows,
        clean_variant_id=args.clean_variant_id,
        corrupt_variant_id=args.corrupt_variant_id,
        skip_same_answers=args.skip_same_answers,
    )
    if args.max_pairs is not None:
        rng = random.Random(args.seed)
        rng.shuffle(pairs)
        pairs = pairs[: args.max_pairs]
    if not pairs:
        sys.exit("No clean/corrupt variant pairs found")

    print(f"Data: {args.data_file} ({len(rows)} rows, {len(pairs)} pairs)")
    print(f"Prompt mode: {args.prompt_mode}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    device = get_input_device(model)
    layers = get_decoder_layers(model)
    layer_indices = parse_layers(args.layers, len(layers))
    print(f"Model: {args.model_name}")
    print(f"Patching decoder layers: {layer_indices}")

    output_file = args.output_file or default_output_path(
        args.model_name,
        args.data_file,
        args.prompt_mode,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    with open(output_file, "w", encoding="utf-8") as f:
        iterator = tqdm(pairs, total=len(pairs), desc="Activation patching")
        for pair_idx, (clean_row, corrupt_row) in enumerate(iterator):
            prepared = prepare_pair(
                clean_row,
                corrupt_row,
                tokenizer,
                device,
                prompt_mode=args.prompt_mode,
            )
            if prepared is None:
                continue

            clean_answer_ids = answer_token_ids(
                tokenizer,
                prepared["clean_answer"],
                prompt_mode=args.prompt_mode,
            )
            corrupt_answer_ids = answer_token_ids(
                tokenizer,
                prepared["corrupt_answer"],
                prompt_mode=args.prompt_mode,
            )

            clean_logits = forward_logits(model, prepared["clean_inputs"])
            corrupt_logits = forward_logits(model, prepared["corrupt_inputs"])
            clean_diff = answer_logit_diff(clean_logits, clean_answer_ids, corrupt_answer_ids)
            corrupt_diff = answer_logit_diff(corrupt_logits, clean_answer_ids, corrupt_answer_ids)

            for layer_idx in layer_indices:
                layer = layers[layer_idx]
                clean_activation = collect_layer_activation(model, layer, prepared["clean_inputs"])
                for location, positions in prepared["positions_by_location"].items():
                    base_row = {
                        "pair_idx": pair_idx,
                        "skeleton_id": clean_row.get("skeleton_id"),
                        "clean_variant_id": clean_row.get("variant_id"),
                        "corrupt_variant_id": corrupt_row.get("variant_id"),
                        "prompt_mode": args.prompt_mode,
                        "target_box": prepared["target_box"],
                        "numops": int(clean_row["numops"]) if "numops" in clean_row else None,
                        "numops_by_op": clean_row.get("numops_by_op"),
                        "layer_idx": layer_idx,
                        "location": location,
                        "n_positions": len(positions),
                        "positions": positions,
                        "clean_answer": prepared["clean_answer"],
                        "corrupt_answer": prepared["corrupt_answer"],
                        "clean_answer_token_ids": clean_answer_ids,
                        "corrupt_answer_token_ids": corrupt_answer_ids,
                        "clean_logit_diff": clean_diff,
                        "corrupt_logit_diff": corrupt_diff,
                    }
                    if not positions:
                        out_row = {
                            **base_row,
                            "patched_logit_diff": None,
                            "normalized_logit_diff": None,
                        }
                    else:
                        patched_logits = patched_forward_logits(
                            model,
                            layer,
                            clean_activation,
                            prepared["corrupt_inputs"],
                            positions,
                        )
                        patched_diff = answer_logit_diff(
                            patched_logits, clean_answer_ids, corrupt_answer_ids
                        )
                        out_row = {
                            **base_row,
                            "patched_logit_diff": patched_diff,
                            "normalized_logit_diff": normalized_logit_diff(
                                patched_diff, clean_diff, corrupt_diff
                            ),
                        }
                    f.write(json.dumps(out_row) + "\n")
                    f.flush()
                    all_results.append(out_row)

    summary = summarize(all_results)
    summary_file = output_file.with_suffix(output_file.suffix + ".summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "data_file": str(args.data_file),
                "output_file": str(output_file),
                "prompt_mode": args.prompt_mode,
                "num_pairs": len(pairs),
                "layers": layer_indices,
                "summary": summary,
            },
            f,
            indent=2,
        )

    print(f"Wrote rows: {output_file}")
    print(f"Wrote summary: {summary_file}")


if __name__ == "__main__":
    main()
