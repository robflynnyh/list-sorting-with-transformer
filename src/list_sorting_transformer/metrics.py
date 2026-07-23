"""Strict metrics for generated sorted lists."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .data import IGNORE_INDEX
from .quicksort import QuicksortTrace, split_generated_events
from .tokens import EOS, PAD, QuicksortTraceVocabulary, SymbolVocabulary


def masked_token_accuracy(logits: Tensor, labels: Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    included = labels.ne(IGNORE_INDEX)
    correct = predictions.eq(labels) & included
    return float(correct.sum().item() / max(included.sum().item(), 1))


def generated_sorting_metrics(
    values: Tensor,
    generated_tokens: Tensor,
    vocabulary: SymbolVocabulary,
) -> dict[str, float]:
    """Measure syntax, ordering, conservation, and exact sequence success."""

    if values.ndim != 2 or generated_tokens.ndim != 2:
        raise ValueError("values and generated_tokens must both be rank two")
    if values.shape[0] != generated_tokens.shape[0]:
        raise ValueError("values and generated_tokens must have the same batch size")

    totals = {
        "valid_syntax": 0,
        "correct_length": 0,
        "sorted": 0,
        "multiset_preserved": 0,
        "exact_match": 0,
    }
    token_correct = 0
    token_total = 0
    for input_row, generated_row in zip(values.tolist(), generated_tokens.tolist()):
        expected_values = sorted(int(value) for value in input_row)
        expected_tokens = vocabulary.encode_target(expected_values)
        for index, expected_token in enumerate(expected_tokens):
            if index < len(generated_row) and generated_row[index] == expected_token:
                token_correct += 1
            token_total += 1

        decoded = vocabulary.decode_list(generated_row)
        if decoded is None:
            continue
        totals["valid_syntax"] += 1
        totals["correct_length"] += int(len(decoded) == len(expected_values))
        totals["sorted"] += int(
            all(left <= right for left, right in zip(decoded, decoded[1:]))
        )
        totals["multiset_preserved"] += int(sorted(decoded) == expected_values)
        totals["exact_match"] += int(decoded == expected_values)

    batch_size = values.shape[0]
    metrics = {
        name: count / batch_size
        for name, count in totals.items()
    }
    metrics["target_token_accuracy"] = token_correct / max(token_total, 1)
    return metrics


def generated_quicksort_metrics(
    values: Tensor,
    generated_tokens: Tensor,
    vocabulary: QuicksortTraceVocabulary,
    traces: Sequence[QuicksortTrace],
) -> dict[str, float]:
    """Score final answers and the deterministic valid-operation prefix."""

    if values.ndim != 2 or generated_tokens.ndim != 2:
        raise ValueError("values and generated_tokens must both be rank two")
    if values.shape[0] != generated_tokens.shape[0]:
        raise ValueError("values and generated_tokens must have the same batch size")
    if len(traces) != values.shape[0]:
        raise ValueError("one reference trace is required per generated row")

    totals = {
        "valid_syntax": 0.0,
        "correct_length": 0.0,
        "sorted": 0.0,
        "multiset_preserved": 0.0,
        "exact_match": 0.0,
        "trace_syntax_valid": 0.0,
        "trace_exact_match": 0.0,
        "full_exact_match": 0.0,
        "operation_prefix_fraction": 0.0,
    }
    answer_token_correct = 0
    answer_token_total = 0
    full_token_correct = 0
    full_token_total = 0

    for input_row, generated_row, trace in zip(
        values.tolist(),
        generated_tokens.tolist(),
        traces,
    ):
        expected_values = sorted(int(value) for value in input_row)
        expected_answer = vocabulary.encode_target(expected_values)
        expected_events = [event.tokens for event in trace.events]

        generated_events, answer_tokens, trace_syntax_valid = split_generated_events(
            generated_row,
            vocabulary,
        )
        totals["trace_syntax_valid"] += float(trace_syntax_valid)
        valid_prefix = 0
        for generated_event, expected_event in zip(
            generated_events,
            expected_events,
        ):
            if generated_event != expected_event:
                break
            valid_prefix += 1
        totals["operation_prefix_fraction"] += valid_prefix / max(
            len(expected_events),
            1,
        )
        trace_exact = trace_syntax_valid and generated_events == expected_events
        totals["trace_exact_match"] += float(trace_exact)

        answer_row = answer_tokens if answer_tokens is not None else []
        for index, expected_token in enumerate(expected_answer):
            if index < len(answer_row) and answer_row[index] == expected_token:
                answer_token_correct += 1
            answer_token_total += 1
        decoded = vocabulary.decode_list(answer_row)
        if decoded is not None:
            totals["valid_syntax"] += 1
            totals["correct_length"] += int(len(decoded) == len(expected_values))
            totals["sorted"] += int(
                all(left <= right for left, right in zip(decoded, decoded[1:]))
            )
            totals["multiset_preserved"] += int(
                sorted(decoded) == expected_values
            )
            totals["exact_match"] += int(decoded == expected_values)

        trimmed_generation = []
        for token in generated_row:
            if token == PAD:
                break
            trimmed_generation.append(token)
            if token == EOS:
                break
        expected_target = list(trace.target_tokens)
        for index, expected_token in enumerate(expected_target):
            if (
                index < len(trimmed_generation)
                and trimmed_generation[index] == expected_token
            ):
                full_token_correct += 1
            full_token_total += 1
        totals["full_exact_match"] += float(
            trimmed_generation == expected_target
        )

    batch_size = values.shape[0]
    metrics = {name: value / batch_size for name, value in totals.items()}
    metrics["target_token_accuracy"] = answer_token_correct / max(
        answer_token_total,
        1,
    )
    metrics["full_target_token_accuracy"] = full_token_correct / max(
        full_token_total,
        1,
    )
    return metrics


def mean_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average an empty metric collection")
    keys = rows[0].keys()
    if any(row.keys() != rows[0].keys() for row in rows[1:]):
        raise ValueError("metric rows must contain the same keys")
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in keys
    }
