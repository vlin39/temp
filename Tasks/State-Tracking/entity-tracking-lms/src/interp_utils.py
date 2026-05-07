"""
Interpretability helpers for entity-tracking language-model analyses.

The functions here are intentionally lightweight wrappers around HuggingFace
causal LMs so notebooks can inspect residual-stream activations without
duplicating model-specific boilerplate.
"""
from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch


def first_real_device(model: torch.nn.Module) -> torch.device:
    """Return the first non-meta parameter/buffer device for a possibly sharded model."""

    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cpu")


def lm_head_device(model: torch.nn.Module) -> torch.device:
    """Device where ``model.lm_head`` expects hidden states."""

    head = getattr(model, "lm_head", None)
    if head is None:
        return first_real_device(model)
    for tensor in list(head.parameters()) + list(head.buffers()):
        if tensor.device.type != "meta":
            return tensor.device
    return first_real_device(model)


def final_norm_module(model: torch.nn.Module) -> torch.nn.Module | None:
    """
    Best-effort lookup for the final decoder normalization used before ``lm_head``.

    Common HuggingFace causal LMs expose this as ``model.model.norm`` (Llama/Qwen),
    ``model.transformer.ln_f`` (GPT-style), or a decoder final layer norm.
    """

    candidates = (
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
        ("model", "decoder", "final_layer_norm"),
        ("model", "final_layer_norm"),
    )
    for path in candidates:
        obj: Any = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, torch.nn.Module):
            return obj
    return None


def last_non_pad_positions(
    attention_mask: torch.Tensor | None,
    sequence_length: int,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Return one final-content token position per batch row."""

    if attention_mask is None:
        return torch.full((batch_size,), sequence_length - 1, dtype=torch.long)
    return attention_mask.to(torch.long).sum(dim=-1).sub(1).clamp_min(0).cpu()


def normalize_positions(
    positions: int | Sequence[int] | torch.Tensor | None,
    *,
    attention_mask: torch.Tensor | None,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    """Normalize user-specified token positions to a CPU tensor of shape ``(batch,)``."""

    if positions is None:
        return last_non_pad_positions(
            attention_mask,
            sequence_length,
            batch_size=batch_size,
        )

    if isinstance(positions, int):
        pos = positions if positions >= 0 else sequence_length + positions
        return torch.full((batch_size,), pos, dtype=torch.long)

    pos_tensor = torch.as_tensor(positions, dtype=torch.long).flatten().cpu()
    if pos_tensor.numel() != batch_size:
        raise ValueError(
            f"positions must have one entry per batch row ({batch_size}), "
            f"got {pos_tensor.numel()}"
        )
    pos_tensor = torch.where(pos_tensor < 0, pos_tensor + sequence_length, pos_tensor)
    return pos_tensor


def decode_token(tokenizer, token_id: int) -> str:
    """Decode a single token without cleanup so whitespace markers stay visible."""

    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def load_object_vocabulary(csv_path: str | Path) -> list[str]:
    """Load object names from the dataset generator's ``object_name`` CSV column."""

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row["object_name"].strip() for row in reader]


def objects_mentioned_in_text(text: str, object_vocabulary: Sequence[str]) -> list[str]:
    """
    Return object vocabulary entries that occur in ``text``, preserving first mention order.

    The generated box data names objects as noun phrases like ``the book``. Matching by
    vocabulary avoids accidentally treating box/container/task words as objects.
    """

    hits: list[tuple[int, str]] = []
    for obj in object_vocabulary:
        pattern = rf"(?<![A-Za-z]){re.escape(obj)}(?![A-Za-z])"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            hits.append((match.start(), obj))
    return [obj for _, obj in sorted(hits, key=lambda x: (x[0], x[1]))]


def token_id_for_label(
    tokenizer,
    label: str,
    *,
    prefer_leading_space: bool = True,
) -> dict[str, Any]:
    """
    Pick the token id used to track a label in next-token logits.

    Since ``lm_head`` predicts one next token at a time, multi-token labels are tracked
    by their first token. For these box objects, the leading-space form is usually the
    relevant next token after phrases like ``the`` or ``contains``.
    """

    variants = [f" {label}", label] if prefer_leading_space else [label, f" {label}"]
    tokenizations: list[tuple[str, list[int]]] = []
    for text in variants:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            tokenizations.append((text, [int(i) for i in ids]))
    if not tokenizations:
        raise ValueError(f"Could not tokenize label: {label!r}")

    token_text, token_ids = next(
        ((text, ids) for text, ids in tokenizations if len(ids) == 1),
        tokenizations[0],
    )
    token_id = int(token_ids[0])
    return {
        "label": label,
        "token_text": token_text,
        "token_ids": token_ids,
        "token_id": token_id,
        "token": decode_token(tokenizer, token_id),
        "is_single_token": len(token_ids) == 1,
    }


def token_components_for_tracked_label(
    tokenizer,
    label: str,
    *,
    object_prefix: str = "the",
    unprefixed_labels: Sequence[str] = ("nothing",),
    prefer_leading_space: bool = True,
) -> dict[str, Any]:
    """
    Build the token components used for task-specific scoring.

    Box objects are scored as ``" the" + " <object>"`` so their row logit is the
    sum of the determiner-token logit and object-token logit. Labels like
    ``nothing`` remain unprefixed and are scored as a single token.
    """

    normalized = label.strip()
    if normalized in set(unprefixed_labels):
        info = token_id_for_label(
            tokenizer,
            normalized,
            prefer_leading_space=prefer_leading_space,
        )
        return {
            **info,
            "phrase_text": info["token_text"],
            "component_names": [normalized],
            "component_token_texts": [info["token_text"]],
            "component_token_ids": [info["token_id"]],
            "is_prefixed_object": False,
        }

    prefix_info = token_id_for_label(
        tokenizer,
        object_prefix,
        prefer_leading_space=prefer_leading_space,
    )
    object_info = token_id_for_label(
        tokenizer,
        normalized,
        prefer_leading_space=prefer_leading_space,
    )
    component_token_ids = [prefix_info["token_id"], object_info["token_id"]]
    return {
        "label": normalized,
        "phrase_text": f" {object_prefix} {normalized}",
        "token_text": object_info["token_text"],
        "token_ids": component_token_ids,
        "token_id": object_info["token_id"],
        "token": object_info["token"],
        "is_single_token": object_info["is_single_token"],
        "component_names": [object_prefix, normalized],
        "component_token_texts": [prefix_info["token_text"], object_info["token_text"]],
        "component_token_ids": component_token_ids,
        "is_prefixed_object": True,
    }


def token_context(
    tokenizer,
    input_ids: torch.Tensor,
    *,
    batch_idx: int = 0,
    position: int = -1,
    radius: int = 8,
) -> str:
    """Decode a small token window around a position for orientation."""

    seq_len = int(input_ids.shape[-1])
    pos = position if position >= 0 else seq_len + position
    start = max(0, pos - radius)
    end = min(seq_len, pos + radius + 1)
    ids = input_ids[batch_idx, start:end].detach().cpu()
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def forward_hidden_states(model: torch.nn.Module, model_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Run a no-grad forward pass and return embedding plus per-layer hidden states."""

    forward_kwargs = {
        k: v
        for k, v in model_inputs.items()
        if k in {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
    }
    with torch.no_grad():
        outputs = model(
            **forward_kwargs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states; check output_hidden_states support.")
    return tuple(hidden_states)


def layer_labels(num_hidden_states: int, *, include_embedding: bool = True) -> list[str]:
    """Human-readable names for hidden-state tuple entries."""

    start = 0 if include_embedding else 1
    labels: list[str] = []
    for idx in range(start, num_hidden_states):
        labels.append(hidden_state_label(idx))
    return labels


def hidden_state_label(hidden_state_index: int) -> str:
    """Map HF hidden-state tuple index to a readable residual-stream label."""

    return "embed" if hidden_state_index == 0 else f"layer_{hidden_state_index - 1}"


def _iter_hidden_state_indices(
    num_hidden_states: int,
    layer_indices: Iterable[int] | None,
    *,
    include_embedding: bool,
) -> list[int]:
    if layer_indices is None:
        return list(range(0 if include_embedding else 1, num_hidden_states))
    out: list[int] = []
    for idx in layer_indices:
        resolved = idx if idx >= 0 else num_hidden_states + idx
        if resolved < 0 or resolved >= num_hidden_states:
            raise IndexError(f"hidden-state index {idx} is out of range for {num_hidden_states} states")
        if resolved == 0 and not include_embedding:
            continue
        out.append(resolved)
    return out


def logit_lens_topk(
    model: torch.nn.Module,
    tokenizer,
    hidden_states: tuple[torch.Tensor, ...],
    *,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    positions: int | Sequence[int] | torch.Tensor | None = None,
    top_k: int = 10,
    layer_indices: Iterable[int] | None = None,
    include_embedding: bool = True,
    apply_final_norm: bool = True,
) -> list[dict[str, Any]]:
    """
    Apply ``lm_head`` to intermediate hidden states and return top-k decoded tokens.

    Parameters
    ----------
    hidden_states
        Tuple returned by a HuggingFace forward pass with ``output_hidden_states=True``.
        Entry 0 is the embedding output; entry N is the residual stream after
        decoder layer N - 1.
    positions
        Token position(s) to inspect. ``None`` means the final non-padding token.
    apply_final_norm
        If true, pass each intermediate vector through the model's final decoder norm
        before ``lm_head``. This usually matches how final logits are produced for
        Qwen/Llama-style models while still using the same residual activations.
    """

    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    batch_size, sequence_length = hidden_states[0].shape[:2]
    if input_ids is not None:
        batch_size, sequence_length = input_ids.shape[:2]
    position_tensor = normalize_positions(
        positions,
        attention_mask=attention_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )

    norm = final_norm_module(model) if apply_final_norm else None
    head = getattr(model, "lm_head", None)
    if head is None:
        raise AttributeError("model does not expose an lm_head module")

    rows: list[dict[str, Any]] = []
    head_device = lm_head_device(model)
    for hs_idx in _iter_hidden_state_indices(
        len(hidden_states),
        layer_indices,
        include_embedding=include_embedding,
    ):
        layer_name = hidden_state_label(hs_idx)
        hs = hidden_states[hs_idx]
        for batch_idx in range(batch_size):
            pos = int(position_tensor[batch_idx])
            vec = hs[batch_idx, pos : pos + 1, :]
            if norm is not None:
                norm_device = first_real_device(norm)
                vec = norm(vec.to(norm_device))
            logits = head(vec.to(head_device)).squeeze(0).float()
            probs = torch.softmax(logits, dim=-1)
            vals, ids = torch.topk(logits, k=top_k, dim=-1)
            for rank, (token_id, logit) in enumerate(zip(ids.tolist(), vals.tolist()), start=1):
                token_prob = float(probs[int(token_id)].item())
                rows.append(
                    {
                        "hidden_state_index": hs_idx,
                        "layer": layer_name,
                        "batch_idx": batch_idx,
                        "position": pos,
                        "rank": rank,
                        "token_id": int(token_id),
                        "token": decode_token(tokenizer, int(token_id)),
                        "logit": float(logit),
                        "probability": token_prob,
                    }
                )
    return rows


def tracked_token_logits(
    model: torch.nn.Module,
    tokenizer,
    hidden_states: tuple[torch.Tensor, ...],
    target_labels: Sequence[str],
    *,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    positions: int | Sequence[int] | torch.Tensor | None = None,
    layer_indices: Iterable[int] | None = None,
    include_embedding: bool = True,
    apply_final_norm: bool = True,
    object_prefix: str = "the",
    unprefixed_labels: Sequence[str] = ("nothing",),
    prefer_leading_space: bool = True,
    include_rank: bool = True,
) -> list[dict[str, Any]]:
    """
    Score selected object/nothing labels across layers.

    Object labels are scored as ``" the" + " <object>"`` by default, summing the
    component logits. Labels in ``unprefixed_labels`` (``nothing`` by default) are
    scored as a single token.
    """

    if not target_labels:
        return []

    seen: set[str] = set()
    targets = []
    for label in target_labels:
        label = str(label).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        targets.append(
            token_components_for_tracked_label(
                tokenizer,
                label,
                object_prefix=object_prefix,
                unprefixed_labels=unprefixed_labels,
                prefer_leading_space=prefer_leading_space,
            )
        )

    batch_size, sequence_length = hidden_states[0].shape[:2]
    if input_ids is not None:
        batch_size, sequence_length = input_ids.shape[:2]
    position_tensor = normalize_positions(
        positions,
        attention_mask=attention_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )

    norm = final_norm_module(model) if apply_final_norm else None
    head = getattr(model, "lm_head", None)
    if head is None:
        raise AttributeError("model does not expose an lm_head module")

    rows: list[dict[str, Any]] = []
    head_device = lm_head_device(model)
    for hs_idx in _iter_hidden_state_indices(
        len(hidden_states),
        layer_indices,
        include_embedding=include_embedding,
    ):
        layer_name = hidden_state_label(hs_idx)
        hs = hidden_states[hs_idx]
        for batch_idx in range(batch_size):
            pos = int(position_tensor[batch_idx])
            vec = hs[batch_idx, pos : pos + 1, :]
            if norm is not None:
                norm_device = first_real_device(norm)
                vec = norm(vec.to(norm_device))
            logits = head(vec.to(head_device)).squeeze(0).float()
            probs = torch.softmax(logits, dim=-1)
            for target in targets:
                component_ids = [int(tid) for tid in target["component_token_ids"]]
                component_logits = [float(logits[tid].item()) for tid in component_ids]
                component_probs = [float(probs[tid].item()) for tid in component_ids]
                component_ranks = None
                if include_rank:
                    component_ranks = [
                        int((logits > logits[tid]).sum().item() + 1)
                        for tid in component_ids
                    ]
                phrase_probability = 1.0
                for prob in component_probs:
                    phrase_probability *= prob
                row = {
                    "hidden_state_index": hs_idx,
                    "layer": layer_name,
                    "batch_idx": batch_idx,
                    "position": pos,
                    "label": target["label"],
                    "phrase_text": target["phrase_text"],
                    "token_text": target["token_text"],
                    "token_id": target["token_id"],
                    "token": target["token"],
                    "token_ids": target["token_ids"],
                    "is_single_token": target["is_single_token"],
                    "is_prefixed_object": target["is_prefixed_object"],
                    "component_names": target["component_names"],
                    "component_token_texts": target["component_token_texts"],
                    "component_token_ids": component_ids,
                    "component_logits": component_logits,
                    "component_probabilities": component_probs,
                    "the_logit": component_logits[0] if target["is_prefixed_object"] else None,
                    "object_logit": component_logits[-1] if target["is_prefixed_object"] else None,
                    "logit": float(sum(component_logits)),
                    "probability": float(phrase_probability),
                }
                if component_ranks is not None:
                    row["component_ranks"] = component_ranks
                    row["rank"] = None if target["is_prefixed_object"] else component_ranks[0]
                rows.append(row)
    return rows


def run_logit_lens(
    model: torch.nn.Module,
    tokenizer,
    model_inputs: dict[str, torch.Tensor],
    *,
    positions: int | Sequence[int] | torch.Tensor | None = None,
    top_k: int = 10,
    layer_indices: Iterable[int] | None = None,
    include_embedding: bool = True,
    apply_final_norm: bool = True,
) -> list[dict[str, Any]]:
    """Convenience wrapper: forward pass plus :func:`logit_lens_topk`."""

    hidden_states = forward_hidden_states(model, model_inputs)
    return logit_lens_topk(
        model,
        tokenizer,
        hidden_states,
        input_ids=model_inputs.get("input_ids"),
        attention_mask=model_inputs.get("attention_mask"),
        positions=positions,
        top_k=top_k,
        layer_indices=layer_indices,
        include_embedding=include_embedding,
        apply_final_norm=apply_final_norm,
    )


def run_tracked_token_lens(
    model: torch.nn.Module,
    tokenizer,
    model_inputs: dict[str, torch.Tensor],
    target_labels: Sequence[str],
    *,
    positions: int | Sequence[int] | torch.Tensor | None = None,
    layer_indices: Iterable[int] | None = None,
    include_embedding: bool = True,
    apply_final_norm: bool = True,
    object_prefix: str = "the",
    unprefixed_labels: Sequence[str] = ("nothing",),
    prefer_leading_space: bool = True,
    include_rank: bool = True,
) -> list[dict[str, Any]]:
    """Convenience wrapper for task-specific object/nothing logit trajectories."""

    hidden_states = forward_hidden_states(model, model_inputs)
    return tracked_token_logits(
        model,
        tokenizer,
        hidden_states,
        target_labels,
        input_ids=model_inputs.get("input_ids"),
        attention_mask=model_inputs.get("attention_mask"),
        positions=positions,
        layer_indices=layer_indices,
        include_embedding=include_embedding,
        apply_final_norm=apply_final_norm,
        object_prefix=object_prefix,
        unprefixed_labels=unprefixed_labels,
        prefer_leading_space=prefer_leading_space,
        include_rank=include_rank,
    )
