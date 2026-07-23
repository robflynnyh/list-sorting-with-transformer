"""Strict metrics for generated sorted lists."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .data import IGNORE_INDEX
from .tokens import SymbolVocabulary


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
