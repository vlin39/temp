#!/usr/bin/env python3
"""
Generate paired box-tracking examples for activation patching.

Each skeleton has:
- one item in each box initially,
- a fixed sequence of box operations,
- multiple variants that differ only in object names,
- the same JSONL fields as the existing T5-style box data.

When a tokenizer is provided, variants for the same skeleton are required to have
identical token counts for ``sentence``, ``sentence_masked``, and
``masked_content``. This makes clean/corrupt activation patching less confounded
by sequence length or formatting.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[1]
_GEN_SRC = _ROOT / "src" / "dataset_generation"
if str(_GEN_SRC) not in sys.path:
    sys.path.insert(0, str(_GEN_SRC))

from generate_boxes_data import example_to_t5  # noqa: E402

OperationKind = Literal["move", "remove", "put"]


@dataclass(frozen=True)
class Operation:
    kind: OperationKind
    box1: int
    box2: int | None
    slot: str


@dataclass(frozen=True)
class Skeleton:
    skeleton_id: int
    initial_slots: tuple[str, ...]
    operations: tuple[Operation, ...]
    target_box: int
    final_slots: tuple[str | None, ...]
    numops_by_box: tuple[int, ...]
    numops_by_type_by_box: tuple[dict[str, int], ...]


def load_object_names(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row["object_name"].strip() for row in reader if row.get("object_name")]


def state_sentence(box_slots: list[str | None], assignment: dict[str, str]) -> str:
    parts = []
    for box, slot in enumerate(box_slots):
        if slot is None:
            parts.append(f"Box {box} contains nothing")
        else:
            parts.append(f"Box {box} contains the {assignment[slot]}")
    return ", ".join(parts) + "."


def operation_sentence(op: Operation, assignment: dict[str, str]) -> str:
    obj = assignment[op.slot]
    if op.kind == "move":
        if op.box2 is None:
            raise ValueError("move operation requires box2")
        return f"Move the {obj} from Box {op.box1} to Box {op.box2}."
    if op.kind == "remove":
        return f"Remove the {obj} from Box {op.box1}."
    if op.kind == "put":
        return f"Put the {obj} into Box {op.box1}."
    raise ValueError(f"unknown operation kind: {op.kind}")


def apply_operation(
    state: list[str | None],
    void: list[str],
    op: Operation,
) -> None:
    if op.kind == "move":
        if op.box2 is None:
            raise ValueError("move operation requires box2")
        state[op.box1] = None
        state[op.box2] = op.slot
    elif op.kind == "remove":
        state[op.box1] = None
        void.append(op.slot)
    elif op.kind == "put":
        state[op.box1] = op.slot
        if op.slot in void:
            void.remove(op.slot)
    else:
        raise ValueError(f"unknown operation kind: {op.kind}")


def render_example(skeleton: Skeleton, assignment: dict[str, str], *, variant_id: int) -> dict:
    state = list(skeleton.initial_slots)
    void = [slot for op in skeleton.operations if op.kind == "put" for slot in [op.slot]]
    sentences = [state_sentence(state, assignment)]
    for op in skeleton.operations:
        sentences.append(operation_sentence(op, assignment))
        apply_operation(state, void, op)

    target_slot = state[skeleton.target_box]
    if target_slot is None:
        sentences.append(f"Box {skeleton.target_box} contains nothing.")
    else:
        sentences.append(f"Box {skeleton.target_box} contains the {assignment[target_slot]}.")

    row = example_to_t5(" ".join(sentences), zero_shot=True)
    row["sample_id"] = skeleton.skeleton_id
    row["skeleton_id"] = skeleton.skeleton_id
    row["variant_id"] = variant_id
    row["target_box"] = skeleton.target_box
    row["numops"] = int(skeleton.numops_by_box[skeleton.target_box])
    row["numops_by_op"] = dict(skeleton.numops_by_type_by_box[skeleton.target_box])
    row["object_assignment"] = assignment
    row["operation_skeleton"] = [
        {"kind": op.kind, "box1": op.box1, "box2": op.box2, "slot": op.slot}
        for op in skeleton.operations
    ]
    return row


def count_box_change(before: str | None, after: str | None) -> bool:
    return before != after


def make_skeleton(
    rng: random.Random,
    *,
    skeleton_id: int,
    num_boxes: int,
    num_operations: int,
    allow_empty_target: bool,
) -> Skeleton:
    initial_slots = [f"s{i}" for i in range(num_boxes)]
    state: list[str | None] = list(initial_slots)
    void = [f"s{num_boxes + i}" for i in range(num_operations)]
    operations: list[Operation] = []
    numops_by_box = [0 for _ in range(num_boxes)]
    numops_by_type_by_box = [
        {"move": 0, "remove": 0, "put": 0} for _ in range(num_boxes)
    ]

    for _step in range(num_operations):
        nonempty = [i for i, slot in enumerate(state) if slot is not None]
        empty = [i for i, slot in enumerate(state) if slot is None]
        valid: list[OperationKind] = []
        if nonempty and empty:
            valid.append("move")
        if nonempty:
            valid.append("remove")
        if empty and void:
            valid.append("put")
        if not valid:
            break

        # Prefer moves once an empty box exists, but force a remove if all boxes are full.
        if not empty:
            kind: OperationKind = "remove"
        else:
            kind = rng.choices(
                valid,
                weights=[3 if k == "move" else 2 if k == "put" else 1 for k in valid],
                k=1,
            )[0]

        before = list(state)
        if kind == "move":
            box1 = rng.choice(nonempty)
            box2 = rng.choice([i for i in empty if i != box1])
            slot = state[box1]
            if slot is None:
                raise AssertionError("chosen move source was empty")
            op = Operation(kind="move", box1=box1, box2=box2, slot=slot)
        elif kind == "remove":
            box1 = rng.choice(nonempty)
            slot = state[box1]
            if slot is None:
                raise AssertionError("chosen remove source was empty")
            op = Operation(kind="remove", box1=box1, box2=None, slot=slot)
        else:
            box1 = rng.choice(empty)
            slot = void.pop(0)
            op = Operation(kind="put", box1=box1, box2=None, slot=slot)

        apply_operation(state, void, op)
        operations.append(op)
        for box, (old, new) in enumerate(zip(before, state)):
            if count_box_change(old, new):
                numops_by_box[box] += 1
                numops_by_type_by_box[box][kind] += 1

    candidate_targets = list(range(num_boxes)) if allow_empty_target else [
        i for i, slot in enumerate(state) if slot is not None
    ]
    if not candidate_targets:
        candidate_targets = list(range(num_boxes))
    target_box = rng.choice(candidate_targets)
    return Skeleton(
        skeleton_id=skeleton_id,
        initial_slots=tuple(initial_slots),
        operations=tuple(operations),
        target_box=target_box,
        final_slots=tuple(state),
        numops_by_box=tuple(numops_by_box),
        numops_by_type_by_box=tuple(numops_by_type_by_box),
    )


def slots_used(skeleton: Skeleton) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for slot in skeleton.initial_slots:
        if slot not in seen:
            out.append(slot)
            seen.add(slot)
    for op in skeleton.operations:
        if op.slot not in seen:
            out.append(op.slot)
            seen.add(op.slot)
    return out


def load_tokenizer(tokenizer_name: str | None, trust_remote_code: bool):
    if tokenizer_name is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )


def object_token_length(tokenizer, obj: str) -> int:
    if tokenizer is None:
        return 1
    return len(tokenizer(f" {obj}", add_special_tokens=False)["input_ids"])


def token_counts(tokenizer, row: dict) -> dict[str, int]:
    if tokenizer is None:
        return {}
    fields = ("sentence", "sentence_masked", "masked_content")
    return {
        field: len(tokenizer(row[field], add_special_tokens=False)["input_ids"])
        for field in fields
    }


def same_token_counts(rows: list[dict]) -> bool:
    if not rows or "token_counts" not in rows[0]:
        return True
    ref = rows[0]["token_counts"]
    return all(row["token_counts"] == ref for row in rows[1:])


def make_variant_assignments(
    rng: random.Random,
    *,
    slot_ids: list[str],
    object_names: list[str],
    num_variants: int,
    tokenizer,
    max_attempts: int,
) -> list[dict[str, str]]:
    by_len: dict[int, list[str]] = {}
    for obj in object_names:
        by_len.setdefault(object_token_length(tokenizer, obj), []).append(obj)

    needed = len(slot_ids) * num_variants
    candidate_buckets = [objs for objs in by_len.values() if len(objs) >= needed]
    if not candidate_buckets:
        raise ValueError(
            f"No object token-length bucket has {needed} names. "
            "Use fewer variants/operations or disable tokenizer checking."
        )

    for _ in range(max_attempts):
        bucket = list(rng.choice(candidate_buckets))
        rng.shuffle(bucket)
        assignments = []
        for variant_id in range(num_variants):
            start = variant_id * len(slot_ids)
            chunk = bucket[start : start + len(slot_ids)]
            assignments.append(dict(zip(slot_ids, chunk)))
        return assignments

    raise RuntimeError("failed to sample object assignments")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_file", type=Path, default=None)
    p.add_argument("--num_skeletons", type=int, default=200)
    p.add_argument("--num_variants", type=int, default=3)
    p.add_argument("--num_boxes", type=int, default=6)
    p.add_argument("--num_operations", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--object_csv",
        type=Path,
        default=_ROOT / "data" / "objects_with_bnc_frequency.csv",
    )
    p.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Tokenizer used to enforce equal token counts. Use --no_tokenizer_check to skip.",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--no_tokenizer_check", action="store_true")
    p.add_argument(
        "--allow_empty_target",
        action="store_true",
        help="Allow target boxes whose final answer is 'nothing'.",
    )
    p.add_argument("--max_assignment_attempts", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_variants < 2:
        raise ValueError("--num_variants must be at least 2 for patching pairs")
    if args.num_boxes < 2:
        raise ValueError("--num_boxes must be at least 2")
    if args.num_operations < 0:
        raise ValueError("--num_operations must be non-negative")

    output_file = args.output_file
    if output_file is None:
        output_file = (
            _ROOT
            / "data"
            / f"activation_patching_boxes{args.num_boxes}_1item_nops{args.num_operations}"
            / "test-t5.jsonl"
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    object_names = load_object_names(args.object_csv)
    tokenizer = None if args.no_tokenizer_check else load_tokenizer(
        args.tokenizer_name,
        args.trust_remote_code,
    )

    rows: list[dict] = []
    skeleton_id = 0
    attempts = 0
    while skeleton_id < args.num_skeletons:
        attempts += 1
        skeleton = make_skeleton(
            rng,
            skeleton_id=skeleton_id,
            num_boxes=args.num_boxes,
            num_operations=args.num_operations,
            allow_empty_target=args.allow_empty_target,
        )
        slot_ids = slots_used(skeleton)
        assignments = make_variant_assignments(
            rng,
            slot_ids=slot_ids,
            object_names=object_names,
            num_variants=args.num_variants,
            tokenizer=tokenizer,
            max_attempts=args.max_assignment_attempts,
        )
        candidate_rows = [
            render_example(skeleton, assignment, variant_id=variant_id)
            for variant_id, assignment in enumerate(assignments)
        ]
        if tokenizer is not None:
            for row in candidate_rows:
                row["token_counts"] = token_counts(tokenizer, row)
            if not same_token_counts(candidate_rows):
                continue
        rows.extend(candidate_rows)
        skeleton_id += 1

    with open(output_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows ({args.num_skeletons} skeletons x {args.num_variants} variants)")
    print(f"Output: {output_file}")
    print(f"Skeleton attempts: {attempts}")
    if tokenizer is not None:
        print(f"Tokenizer: {args.tokenizer_name}")


if __name__ == "__main__":
    main()
