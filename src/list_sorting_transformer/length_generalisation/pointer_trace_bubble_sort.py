"""Compose a pointer comparison checkpoint into externally controlled sorting."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from list_sorting_transformer.core.positions import sample_position_offsets
from list_sorting_transformer.core.tokens import (
    BOS,
    COMMA,
    SEP,
    VALUE_OFFSET,
    PointerCompareVocabulary,
)
from list_sorting_transformer.length_generalisation.sparse_attention_adam import (
    SparseAttentionAdamConfig,
    SparseAttentionPointerTransformer,
    generate_comparison_trace,
)


TracePolicy = Callable[[Tensor, int], Tensor]


@dataclass(frozen=True)
class BubbleSortCompositionResult:
    """Aggregate metrics from fixed-schedule bubble-sort composition."""

    length: int
    examples: int
    comparisons_per_example: int
    total_comparisons: int
    final_sorted_accuracy: float
    valid_execution_accuracy: float
    perfect_action_trace_accuracy: float
    perfect_retrieval_trace_accuracy: float
    perfect_complete_trace_accuracy: float
    action_accuracy: float
    marked_value_accuracy: float
    following_value_accuracy: float
    retrieval_accuracy: float
    action_accuracy_given_retrieval: float
    invalid_action_rate: float
    false_keep_rate: float
    false_swap_rate: float
    remaining_inversion_fraction: float
    first_action_error_step_min: int | None
    first_action_error_step_median: float | None
    first_action_error_step_mean: float | None
    comparisons_per_second: float
    elapsed_seconds: float
    passes: tuple[dict[str, float], ...]


def make_pointer_prompt(
    values: Tensor,
    pointer: int,
    vocabulary: PointerCompareVocabulary,
) -> Tensor:
    """Vectorize the checkpoint's full-list prompt with one pointer marker."""

    if values.ndim != 2:
        raise ValueError("values must have shape [batch, length]")
    batch_size, length = values.shape
    if length < 2:
        raise ValueError("pointer comparison requires length at least two")
    if not 0 <= pointer < length - 1:
        raise ValueError("pointer must have a following value")
    base = torch.empty(
        batch_size,
        2 * length + 1,
        dtype=torch.long,
        device=values.device,
    )
    base[:, 0] = BOS
    base[:, 1 : 2 * length : 2] = values + VALUE_OFFSET
    if length > 1:
        base[:, 2 : 2 * length - 1 : 2] = COMMA
    base[:, -1] = SEP

    insertion = 1 + 2 * pointer
    prompt = torch.empty(
        batch_size,
        base.shape[1] + 1,
        dtype=torch.long,
        device=values.device,
    )
    prompt[:, :insertion] = base[:, :insertion]
    prompt[:, insertion] = vocabulary.marker_token("PTR")
    prompt[:, insertion + 1 :] = base[:, insertion:]
    return prompt


def inversion_counts(values: Tensor) -> Tensor:
    """Count strict inversions in each row; duplicate values are ordered."""

    if values.ndim != 2:
        raise ValueError("values must have shape [batch, length]")
    comparisons = values[:, :, None] > values[:, None, :]
    upper_triangle = torch.ones(
        values.shape[1],
        values.shape[1],
        dtype=torch.bool,
        device=values.device,
    ).triu(diagonal=1)
    return (comparisons & upper_triangle).sum(dim=(1, 2))


@torch.inference_mode()
def compose_bubble_sort(
    values: Tensor,
    policy: TracePolicy,
    vocabulary: PointerCompareVocabulary,
) -> tuple[BubbleSortCompositionResult, Tensor]:
    """Run fixed-pass bubble sort using fresh-context model decisions."""

    if values.ndim != 2:
        raise ValueError("values must have shape [batch, length]")
    examples, length = values.shape
    if examples < 1 or length < 2:
        raise ValueError("bubble-sort evaluation requires a non-empty batch")
    if int(values.min()) < 0 or int(values.max()) >= vocabulary.symbol_count:
        raise ValueError("values fall outside the checkpoint vocabulary")

    started_at = time.monotonic()
    current = values.clone()
    target = values.sort(dim=1).values
    initial_inversions = inversion_counts(values)

    perfect_action = torch.ones(
        examples, dtype=torch.bool, device=values.device
    )
    perfect_retrieval = perfect_action.clone()
    perfect_trace = perfect_action.clone()
    valid_execution = perfect_action.clone()
    first_action_error = torch.full(
        (examples,), -1, dtype=torch.long, device=values.device
    )

    action_correct_total = values.new_zeros((), dtype=torch.long)
    marked_correct_total = values.new_zeros((), dtype=torch.long)
    following_correct_total = values.new_zeros((), dtype=torch.long)
    retrieval_correct_total = values.new_zeros((), dtype=torch.long)
    action_given_retrieval_total = values.new_zeros((), dtype=torch.long)
    invalid_action_total = values.new_zeros((), dtype=torch.long)
    false_keep_total = values.new_zeros((), dtype=torch.long)
    false_swap_total = values.new_zeros((), dtype=torch.long)
    comparison_step = 0
    pass_summaries: list[dict[str, float]] = []

    keep_token = vocabulary.action_token("KEEP")
    swap_token = vocabulary.action_token("SWAP")

    for pass_index in range(length - 1):
        active_end = length - pass_index - 1
        pass_action_correct = values.new_zeros((), dtype=torch.long)
        for pointer in range(active_end):
            predicted = policy(current, pointer)
            if predicted.shape != (examples, 3):
                raise ValueError(
                    "trace policy must return shape [batch, 3]"
                )

            left = current[:, pointer].clone()
            right = current[:, pointer + 1].clone()
            expected_action = torch.where(
                left > right,
                torch.full_like(left, swap_token),
                torch.full_like(left, keep_token),
            )
            marked_correct = predicted[:, 0].eq(left + VALUE_OFFSET)
            following_correct = predicted[:, 1].eq(right + VALUE_OFFSET)
            retrieval_correct = marked_correct & following_correct
            action_valid = (
                predicted[:, 2].eq(keep_token)
                | predicted[:, 2].eq(swap_token)
            )
            action_correct = predicted[:, 2].eq(expected_action)
            complete_correct = retrieval_correct & action_correct

            newly_wrong = first_action_error.lt(0) & ~action_correct
            first_action_error[newly_wrong] = comparison_step
            perfect_action &= action_correct
            perfect_retrieval &= retrieval_correct
            perfect_trace &= complete_correct
            valid_execution &= action_valid

            action_correct_total += action_correct.sum()
            marked_correct_total += marked_correct.sum()
            following_correct_total += following_correct.sum()
            retrieval_correct_total += retrieval_correct.sum()
            action_given_retrieval_total += (
                retrieval_correct & action_correct
            ).sum()
            invalid_action_total += (~action_valid).sum()
            false_keep_total += (
                predicted[:, 2].eq(keep_token)
                & expected_action.eq(swap_token)
            ).sum()
            false_swap_total += (
                predicted[:, 2].eq(swap_token)
                & expected_action.eq(keep_token)
            ).sum()
            pass_action_correct += action_correct.sum()

            predicted_swap = predicted[:, 2].eq(swap_token)
            current[:, pointer] = torch.where(
                predicted_swap, right, left
            )
            current[:, pointer + 1] = torch.where(
                predicted_swap, left, right
            )
            comparison_step += 1

        pass_summaries.append(
            {
                "pass": float(pass_index),
                "comparisons_per_example": float(active_end),
                "action_accuracy": int(pass_action_correct)
                / (examples * active_end),
            }
        )

    elapsed_seconds = time.monotonic() - started_at
    total_comparisons = examples * comparison_step
    retrieval_denominator = int(retrieval_correct_total)
    remaining_inversions = inversion_counts(current)
    inversion_denominator = initial_inversions.sum().clamp_min(1)
    error_steps = first_action_error[first_action_error.ge(0)].float()

    result = BubbleSortCompositionResult(
        length=length,
        examples=examples,
        comparisons_per_example=comparison_step,
        total_comparisons=total_comparisons,
        final_sorted_accuracy=float(current.eq(target).all(dim=1).float().mean()),
        valid_execution_accuracy=float(valid_execution.float().mean()),
        perfect_action_trace_accuracy=float(perfect_action.float().mean()),
        perfect_retrieval_trace_accuracy=float(
            perfect_retrieval.float().mean()
        ),
        perfect_complete_trace_accuracy=float(perfect_trace.float().mean()),
        action_accuracy=float(
            action_correct_total.float() / total_comparisons
        ),
        marked_value_accuracy=float(
            marked_correct_total.float() / total_comparisons
        ),
        following_value_accuracy=float(
            following_correct_total.float() / total_comparisons
        ),
        retrieval_accuracy=float(
            retrieval_correct_total.float() / total_comparisons
        ),
        action_accuracy_given_retrieval=(
            float(
                action_given_retrieval_total.float()
                / retrieval_denominator
            )
            if retrieval_denominator
            else 0.0
        ),
        invalid_action_rate=float(
            invalid_action_total.float() / total_comparisons
        ),
        false_keep_rate=float(false_keep_total.float() / total_comparisons),
        false_swap_rate=float(false_swap_total.float() / total_comparisons),
        remaining_inversion_fraction=float(
            remaining_inversions.sum().float() / inversion_denominator
        ),
        first_action_error_step_min=(
            int(error_steps.min()) if error_steps.numel() else None
        ),
        first_action_error_step_median=(
            float(error_steps.median()) if error_steps.numel() else None
        ),
        first_action_error_step_mean=(
            float(error_steps.mean()) if error_steps.numel() else None
        ),
        comparisons_per_second=total_comparisons / elapsed_seconds,
        elapsed_seconds=elapsed_seconds,
        passes=tuple(pass_summaries),
    )
    return result, current


def parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not lengths or any(length < 2 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be integers >= 2")
    return lengths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    SparseAttentionPointerTransformer,
    SparseAttentionAdamConfig,
    int,
]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = SparseAttentionAdamConfig(**checkpoint["config"])
    if config.task != "pointer_compare_trace":
        raise ValueError(
            "bubble-sort composition requires a pointer_compare_trace checkpoint"
        )
    model = SparseAttentionPointerTransformer(config).to(device)
    if config.precision == "bfloat16-true":
        if device.type != "cuda":
            raise ValueError("bfloat16 checkpoint evaluation requires CUDA")
        model = model.to(dtype=torch.bfloat16)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config, int(checkpoint["step"])


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    lengths: Sequence[int],
    examples: int,
    long_examples: int,
    long_min_length: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model, config, checkpoint_step = load_model(checkpoint_path, device)
    vocabulary = model.vocabulary
    if not isinstance(vocabulary, PointerCompareVocabulary):
        raise TypeError("comparison checkpoint has an unexpected vocabulary")
    generator = torch.Generator().manual_seed(seed)
    results: dict[str, object] = {}

    for length in lengths:
        length_examples = (
            long_examples if length >= long_min_length else examples
        )
        values = torch.randint(
            0,
            vocabulary.symbol_count,
            (length_examples, length),
            generator=generator,
        ).to(device)
        offsets = sample_position_offsets(
            length_examples,
            minimum=config.position_offset_min,
            maximum=config.position_offset_max,
            generator=generator,
            device=device,
        )

        def policy(current: Tensor, pointer: int) -> Tensor:
            prompt = make_pointer_prompt(current, pointer, vocabulary)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(
                    config.precision == "bfloat16" and device.type == "cuda"
                ),
            ):
                return generate_comparison_trace(
                    model,
                    prompt,
                    offsets=offsets,
                    steps=3,
                )

        result, _ = compose_bubble_sort(values, policy, vocabulary)
        results[str(length)] = asdict(result)
        print(json.dumps(asdict(result)), flush=True)

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "model_config": asdict(config),
        "evaluation": {
            "controller": "fixed_pass_bubble_sort",
            "fresh_full_context_each_comparison": True,
            "lengths": list(lengths),
            "examples": examples,
            "long_examples": long_examples,
            "long_min_length": long_min_length,
            "seed": seed,
        },
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose a pointer trace checkpoint into bubble sort."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/sparse_attention_adam/results/"
            "pointer_trace_bubble_sort_summary.json"
        ),
    )
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=(2, 3, 5, 10, 20, 40, 100),
    )
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--long-examples", type=int, default=16)
    parser.add_argument("--long-min-length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.examples, args.long_examples, args.long_min_length) < 1:
        raise ValueError("example counts and long_min_length must be positive")
    device = torch.device(args.device)
    summary = evaluate_checkpoint(
        args.checkpoint,
        lengths=args.lengths,
        examples=args.examples,
        long_examples=args.long_examples,
        long_min_length=args.long_min_length,
        seed=args.seed,
        device=device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
