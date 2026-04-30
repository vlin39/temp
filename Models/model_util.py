from __future__ import annotations

import re
import types
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

AttentionAblateGroup = Literal["none", "self_attn", "linear_attn"]

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

MODEL_IDS = {
    "olmo3_7b_instruct": "allenai/Olmo-3-7B-Instruct-SFT", # transformer counterpart
    "olmo_hybrid_7b_instruct": "allenai/Olmo-Hybrid-Instruct-SFT-7B", # our hybrid model to study
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_id: str,
    trust_remote_code: bool = True,
):
    """Load model + tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    
    return model, tokenizer


# ---------------------------------------------------------------------------
# Hybrid attention ablation (``self_attn`` vs ``linear_attn``)
#
# Matches Olmo Hybrid (``olmo_hybrid``) and Qwen3Next / Qwen3.5-style hybrids
# (``qwen3_next``): each decoder layer exposes *one* of ``self_attn`` or
# ``linear_attn``; we patch by the last segment of ``named_modules()`` names.
# ---------------------------------------------------------------------------

def restore_attention_ablation(model: nn.Module) -> None:
    """Undo :func:`ablate_groups` by restoring saved ``forward`` methods."""
    pairs = getattr(model, "_attention_ablation_backup", None)
    if not pairs:
        return
    for mod, fwd in pairs:
        mod.forward = fwd
    model._attention_ablation_backup = []


def _identity_hybrid_self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings=None,
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    **kwargs,
):
    """Skip MHA: return input as attention output (``(hidden_states, None)``).

    Compatible with OlmoHybridAttention, Qwen3NextAttention, and similar
    modules that return ``(tensor, attn_weights_or_none)``.
    """
    return hidden_states, None


def _identity_hybrid_linear_attn_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
):
    """Skip linear / GatedDeltaNet: return input tensor unchanged."""

    return hidden_states


def ablate_groups(
    model: nn.Module,
    group: AttentionAblateGroup = "none",
) -> list[str]:
    """
    Replace selected attention blocks with identity-style forwards (output = input).

    Intended for hybrid stacks that name full-attention blocks ``self_attn`` and
    linear / recurrent blocks ``linear_attn`` (e.g. Olmo Hybrid, Transformers
    ``qwen3_next`` / Qwen3.5 hybrid checkpoints). Matching is by **submodule
    name** (last path segment ``self_attn`` or ``linear_attn``), so layers that
    only instantiate one of the two are still handled correctly.

    Parameters
    ----------
    model
        Loaded ``AutoModelForCausalLM`` (or inner ``model.model`` if you prefer).
    group
        - ``"none"``: remove any prior ablation.
        - ``"self_attn"``: ablate standard self-attention modules only.
        - ``"linear_attn"``: ablate linear / GatedDeltaNet modules only.

    Returns
    -------
    list[str]
        Dotted names of patched modules (for logging).

    Notes
    -----
    Call :func:`restore_attention_ablation` (or ``ablate_groups(..., group="none")``)
    before ``torch.compile`` / saving checkpoints. Identity here means the submodule
    returns the same ``hidden_states`` tensor it received, skipping learned computation.
    """
    restore_attention_ablation(model)

    if group == "none":
        return []

    if group not in ("self_attn", "linear_attn"):
        raise ValueError(f"group must be 'none', 'self_attn', or 'linear_attn', got {group!r}")

    suffix = group  # "self_attn" | "linear_attn"
    patched: list[str] = []
    backup: list[tuple[nn.Module, Callable[..., object]]] = []

    for name, module in model.named_modules():
        if name == "":
            continue
        local = name.rsplit(".", 1)[-1]
        if local != suffix:
            continue

        backup.append((module, module.forward))
        if suffix == "self_attn":
            module.forward = types.MethodType(
                _identity_hybrid_self_attn_forward, module
            )
        else:
            module.forward = types.MethodType(
                _identity_hybrid_linear_attn_forward, module
            )
        patched.append(name)

    model._attention_ablation_backup = backup
    return patched


_LAYER_INDEX_IN_NAME = re.compile(r"\.(?:layers|h)\.(\d+)\.")


def _decoder_layer_index_from_module_name(name: str) -> int | None:
    m = _LAYER_INDEX_IN_NAME.search(name)
    if not m:
        return None
    return int(m.group(1))


def list_decoder_attention_layer_indices(model: nn.Module) -> list[int]:
    """Layer indices under ``…layers.N…`` / ``…h.N…`` that have ``self_attn`` or ``linear_attn``."""
    found: set[int] = set()
    for name, _mod in model.named_modules():
        if not name:
            continue
        local = name.rsplit(".", 1)[-1]
        if local not in ("self_attn", "linear_attn"):
            continue
        idx = _decoder_layer_index_from_module_name(name)
        if idx is not None:
            found.add(idx)
    return sorted(found)


def ablate_single_attn_module(model: nn.Module, layer_idx: int) -> list[str]:
    """
    Replace **one** attention submodule (``self_attn`` or ``linear_attn``) for decoder
    index ``layer_idx`` with the same identity-style forwards as :func:`ablate_groups`.

    Layer index ``N`` matches module paths like ``….layers.N.self_attn`` or
    ``….h.N.self_attn`` (and the ``linear_attn`` variant). This is **not** the
    group-wide ablation from :func:`ablate_groups`; it isolates a single block by
    **decoder index**.

    Parameters
    ----------
    model
        Loaded ``AutoModelForCausalLM`` (full wrapper; uses ``named_modules()`` on it).
    layer_idx
        Non-negative index matching ``….layers.{layer_idx}.…`` (or ``h.{layer_idx}``).

    Returns
    -------
    list[str]
        Dotted names of patched modules (length 1 for typical hybrid stacks).

    Raises
    ------
    ValueError
        If no ``self_attn`` / ``linear_attn`` submodule is found for that index, or
        ``layer_idx`` is negative.
    """
    restore_attention_ablation(model)

    if layer_idx < 0:
        raise ValueError(f"layer_idx must be >= 0, got {layer_idx}")

    patched: list[str] = []
    backup: list[tuple[nn.Module, Callable[..., object]]] = []

    for name, module in model.named_modules():
        if name == "":
            continue
        local = name.rsplit(".", 1)[-1]
        if local not in ("self_attn", "linear_attn"):
            continue
        idx = _decoder_layer_index_from_module_name(name)
        if idx != layer_idx:
            continue
        backup.append((module, module.forward))
        if local == "self_attn":
            module.forward = types.MethodType(
                _identity_hybrid_self_attn_forward, module
            )
        else:
            module.forward = types.MethodType(
                _identity_hybrid_linear_attn_forward, module
            )
        patched.append(name)

    if not patched:
        valid = list_decoder_attention_layer_indices(model)
        raise ValueError(
            f"No self_attn/linear_attn found for decoder layer index {layer_idx}. "
            f"Known indices from paths '.layers.N.' / '.h.N.': {valid}"
        )

    model._attention_ablation_backup = backup
    return patched


# ---------------------------------------------------------------------------
# Inference related
# ---------------------------------------------------------------------------

def prepare_model_inputs_raw(tokenizer, messages, device):
    # messages: list[str]
    inputs = tokenizer(
        messages,
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_token_type_ids=False,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

def prepare_model_inputs_conversation(tokenizer, conversation, device):
    try:
        inputs = tokenizer.apply_chat_template(
            conversation,
            return_tensors='pt',
            return_token_type_ids=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        inputs = tokenizer.apply_chat_template(
            conversation,
            return_tensors='pt',
            return_token_type_ids=False,
            add_generation_prompt=True,
        )

    if hasattr(inputs, "items"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = {"input_ids": inputs.to(device)}
    return inputs



def prepare_model_inputs_conversation_continual(tokenizer, conversation, device):

    inputs = tokenizer.apply_chat_template(
        conversation, 
        return_tensors='pt', 
        return_token_type_ids=False, 
        add_generation_prompt=False,
        continue_final_message=True,
        enable_thinking=False
    )

    if hasattr(inputs, "items"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = {"input_ids": inputs.to(device)}
    
    return inputs
