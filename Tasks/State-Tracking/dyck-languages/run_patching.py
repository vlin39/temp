#!/usr/bin/env python3
"""
Residual-stream activation patching for dyck_languages.

For each decoder layer L, patches the *clean* residual stream (post-layer-L output)
into the *corrupt* forward pass and measures whether the model recovers the clean answer.

A high mean_logit_diff at layer L → layer L carries answer-relevant computation.

Usage:
    python run_patching.py \
        --model_name allenai/Olmo-3-7B-Instruct-SFT \
        --pairs_file output/patching_pairs_200.json \
        --prompt_mode chat

    python run_patching.py \
        --model_name allenai/Olmo-Hybrid-Instruct-SFT-7B \
        --pairs_file output/patching_pairs_200.json \
        --prompt_mode chat
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent


def model_slug(model_name: str) -> str:
    slug = model_name.replace("/", "--")
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
    return slug.strip("._-") or "model"


def first_target_token_id(tokenizer, target_text: str) -> int | None:
    """First sub-token id of `target_text` after tokenisation.

    Tries with and without a leading space (different tokenisers behave
    differently). Returns None if encoding is empty.
    """
    for variant in (target_text, " " + target_text.lstrip()):
        enc = tokenizer.encode(variant, add_special_tokens=False)
        if enc:
            return enc[0]
    return None


def build_prompt_input_ids(tokenizer, prompt: str, prompt_mode: str, device) -> torch.Tensor:
    if str(_ROOT.parents[2]) not in sys.path:
        sys.path.insert(0, str(_ROOT.parents[2]))
    from Models.model_util import (
        prepare_model_inputs_conversation,
        prepare_model_inputs_conversation_continual,
        prepare_model_inputs_raw,
    )

    if prompt_mode == "continuation":
        inputs = prepare_model_inputs_raw(tokenizer, [prompt], device)
    elif prompt_mode == "chat":
        conv = [{"role": "user", "content": prompt}]
        inputs = prepare_model_inputs_conversation(tokenizer, conv, device)
    elif prompt_mode == "chat_continual":
        conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Answer:"},
        ]
        inputs = prepare_model_inputs_conversation_continual(tokenizer, conv, device)
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")
    return inputs["input_ids"]


@torch.no_grad()
def collect_layer_outputs(
    model,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Forward pass capturing the output hidden state of every decoder layer.

    Returns a list of length num_decoder_layers where element i is a
    (1, seq_len, hidden_dim) tensor (on CPU to save GPU memory).
    """
    layer_outputs: list[torch.Tensor] = []
    hooks: list = []

    def make_hook(store: list[torch.Tensor]):
        def hook(_module, _input, output):
            # Decoder layers may return a tuple; first element is hidden states.
            hs = output[0] if isinstance(output, tuple) else output
            store.append(hs.detach().cpu())
        return hook

    decoder_layers = _find_decoder_layers(model)
    for layer in decoder_layers:
        hooks.append(layer.register_forward_hook(make_hook(layer_outputs)))

    try:
        model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return layer_outputs


def _find_decoder_layers(model) -> list:
    """Return list of decoder layer modules in index order."""
    # Try common attribute names: model.layers, model.model.layers, model.transformer.h
    inner = getattr(model, "model", model)
    for attr in ("layers", "h", "blocks"):
        layers = getattr(inner, attr, None)
        if layers is not None:
            return list(layers)
    # Fallback: collect via named_modules matching .layers.N or .h.N
    import re as _re
    pattern = _re.compile(r"^(?:model\.)?(?:model\.)?(?:layers|h)\.(\d+)$")
    found: dict[int, object] = {}
    for name, mod in model.named_modules():
        m = pattern.match(name)
        if m:
            found[int(m.group(1))] = mod
    return [found[k] for k in sorted(found)]


@torch.no_grad()
def patched_forward_logits(
    model,
    corrupt_input_ids: torch.Tensor,
    clean_layer_outputs: list[torch.Tensor],
    patch_layer_idx: int,
) -> torch.Tensor:
    """Run model on corrupt_input_ids, patching layer patch_layer_idx's output.

    The clean residual is injected only for the sequence positions that exist in both
    prompts (min of the two lengths).  Returns logits for the last token.
    """
    decoder_layers = _find_decoder_layers(model)
    if patch_layer_idx >= len(decoder_layers):
        raise ValueError(
            f"patch_layer_idx={patch_layer_idx} out of range (num_layers={len(decoder_layers)})"
        )

    clean_hs = clean_layer_outputs[patch_layer_idx].to(corrupt_input_ids.device)
    hooks: list = []

    def patch_hook(_module, _input, output):
        hs = output[0] if isinstance(output, tuple) else output
        patch_len = min(hs.shape[1], clean_hs.shape[1])
        hs = hs.clone()
        hs[:, :patch_len, :] = clean_hs[:, :patch_len, :].to(hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    hooks.append(decoder_layers[patch_layer_idx].register_forward_hook(patch_hook))
    try:
        out = model(corrupt_input_ids)
    finally:
        for h in hooks:
            h.remove()

    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits[0, -1, :]  # (vocab_size,)


def run_patching(
    *,
    model,
    tokenizer,
    pairs: list[dict],
    prompt_mode: str,
    device,
) -> list[dict]:
    decoder_layers = _find_decoder_layers(model)
    num_layers = len(decoder_layers)

    print(f"Running patching: {len(pairs)} pairs × {num_layers} layers")

    # Accumulate per-layer:
    #   sum_logit_diff:     (patched_logit[clean_first_tok] - base_corrupt_logit[clean_first_tok])
    #   sum_correct:        argmax(patched_logits) == clean_first_tok
    per_layer: list[dict[str, float]] = [
        {"sum_logit_diff": 0.0, "sum_correct": 0.0, "n": 0}
        for _ in range(num_layers)
    ]

    skipped = 0
    for pair_idx, pair in enumerate(pairs):
        clean_first_id = first_target_token_id(tokenizer, pair["clean_closing"])
        if clean_first_id is None:
            skipped += 1
            continue

        clean_ids   = build_prompt_input_ids(tokenizer, pair["clean_prompt"], prompt_mode, device)
        corrupt_ids = build_prompt_input_ids(tokenizer, pair["corrupt_prompt"], prompt_mode, device)

        # Collect clean activations once.
        clean_outputs = collect_layer_outputs(model, clean_ids)

        with torch.no_grad():
            base_out = model(corrupt_ids)
            base_logits = (base_out.logits if hasattr(base_out, "logits") else base_out[0])[0, -1, :]

        for layer_idx in range(num_layers):
            if layer_idx >= len(clean_outputs):
                break
            try:
                patched_logits = patched_forward_logits(
                    model, corrupt_ids, clean_outputs, layer_idx
                )
            except Exception as e:
                print(f"  Pair {pair_idx}, layer {layer_idx}: patching error: {e}")
                continue

            logit_diff = float(
                patched_logits[clean_first_id] - base_logits[clean_first_id]
            )
            patched_pred_id = int(patched_logits.argmax())
            patched_correct = patched_pred_id == clean_first_id

            per_layer[layer_idx]["sum_logit_diff"] += logit_diff
            per_layer[layer_idx]["sum_correct"] += float(patched_correct)
            per_layer[layer_idx]["n"] += 1

        if (pair_idx + 1) % 10 == 0:
            print(f"  Processed {pair_idx + 1}/{len(pairs)} pairs")

    if skipped:
        print(f"  Skipped {skipped} pair(s) where the gold target could not be tokenised.")

    # Determine layer type using model's module names.
    layer_types = _get_layer_types(model, num_layers)

    results = []
    for layer_idx, stats in enumerate(per_layer):
        n = stats["n"] or 1
        results.append(
            {
                "layer_idx": layer_idx,
                "layer_type": layer_types.get(layer_idx, "unknown"),
                "n_pairs": stats["n"],
                "mean_logit_diff": stats["sum_logit_diff"] / n,
                "patched_accuracy": stats["sum_correct"] / n,
            }
        )
    return results


def _get_layer_types(model, num_layers: int) -> dict[int, str]:
    import re as _re
    _pat = _re.compile(r"\.(?:layers|h)\.(\d+)\.")
    types: dict[int, str] = {}
    for name, _ in model.named_modules():
        local = name.rsplit(".", 1)[-1]
        if local not in ("self_attn", "linear_attn"):
            continue
        m = _pat.search(name)
        if m:
            types[int(m.group(1))] = local
    return types


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", type=str, required=True,
                   help="HuggingFace model ID or key from Models.model_util.MODEL_IDS.")
    p.add_argument("--pairs_file", type=str, default=None,
                   help="Path to pairs JSON (default: output/patching_pairs_200.json).")
    p.add_argument("--prompt_mode", type=str, choices=("continuation", "chat", "chat_continual"),
                   default="chat")
    p.add_argument("--cache_dir", type=str, default=str(_ROOT / "data"))
    p.add_argument("--output_json", type=str, default=None)
    return p.parse_args()


def resolve_model_name(model_name: str) -> str:
    _DOL_INTERP = _ROOT.parents[2]
    if str(_DOL_INTERP) not in sys.path:
        sys.path.insert(0, str(_DOL_INTERP))
    from Models.model_util import MODEL_IDS
    return MODEL_IDS.get(model_name, model_name)


def main() -> None:
    args = parse_args()
    model_name = resolve_model_name(args.model_name)

    pairs_path = Path(args.pairs_file) if args.pairs_file else _ROOT / "output" / "patching_pairs_200.json"
    if not pairs_path.exists():
        sys.exit(
            f"Pairs file not found: {pairs_path}\n"
            f"Run: python make_patching_pairs.py --num_pairs 200"
        )

    with open(pairs_path, encoding="utf-8") as f:
        pairs_data = json.load(f)
    pairs = pairs_data["pairs"]
    print(f"Loaded {len(pairs)} pairs from {pairs_path}")

    _DOL_INTERP = _ROOT.parents[2]
    if str(_DOL_INTERP) not in sys.path:
        sys.path.insert(0, str(_DOL_INTERP))
    from Models.model_util import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(model_name)
    device = model.device

    patching_results = run_patching(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        prompt_mode=args.prompt_mode,
        device=device,
    )

    out = {
        "model_name": model_name,
        "task": "dyck_languages",
        "prompt_mode": args.prompt_mode,
        "pairs_file": str(pairs_path),
        "num_pairs": len(pairs),
        "patching_results": patching_results,
    }

    slug = model_slug(model_name)
    out_path = (
        Path(args.output_json)
        if args.output_json
        else _ROOT / "output" / f"{slug}_{args.prompt_mode}_patching.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
