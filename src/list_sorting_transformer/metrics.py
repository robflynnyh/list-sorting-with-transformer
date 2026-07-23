"""Strict metrics for generated sorted lists."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from .adjacent_sort import (
    AdjacentSortRollout,
    AdjacentSortTrace,
    replay_adjacent_sort_transcript,
)
from .data import IGNORE_INDEX
from .pointer_quicksort import (
    PointerQuicksortRollout,
    PointerQuicksortTrace,
    replay_pointer_quicksort_transcript,
)
from .quicksort import QuicksortTrace, split_generated_events
from .tokens import (
    EOS,
    PAD,
    AdjacentSortVocabulary,
    PointerQuicksortVocabulary,
    QuicksortTraceVocabulary,
    SymbolVocabulary,
)


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


def _generated_executor_machine_metrics(
    values: Tensor,
    rollouts: Sequence[Any],
    action_token_vocabulary: Sequence[int],
    traces: Sequence[Any],
) -> dict[str, float]:
    """Score executed state and canonical action-sequence fidelity."""

    if values.ndim != 2:
        raise ValueError("values must be rank two")
    if len(rollouts) != values.shape[0] or len(traces) != values.shape[0]:
        raise ValueError("one rollout and reference trace are required per row")

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
        "execution_completed": 0.0,
        "timed_out": 0.0,
    }
    action_correct = 0
    action_total = 0
    valid_action_tokens = frozenset(action_token_vocabulary)

    for input_row, rollout, trace in zip(values.tolist(), rollouts, traces):
        expected_values = sorted(int(value) for value in input_row)
        generated_actions = rollout.action_tokens
        expected_actions = trace.action_tokens
        valid_action_stream = all(
            action in valid_action_tokens for action in generated_actions
        )
        completed = rollout.completed and rollout.valid_execution
        final_values = list(rollout.final_values)

        totals["valid_syntax"] += float(completed)
        totals["trace_syntax_valid"] += float(
            valid_action_stream and rollout.valid_execution
        )
        totals["execution_completed"] += float(completed)
        totals["timed_out"] += float(rollout.timed_out)
        totals["correct_length"] += float(len(final_values) == len(input_row))
        totals["sorted"] += float(
            all(
                left <= right
                for left, right in zip(final_values, final_values[1:])
            )
        )
        totals["multiset_preserved"] += float(
            sorted(final_values) == expected_values
        )
        exact = completed and final_values == expected_values
        totals["exact_match"] += float(exact)

        valid_prefix = 0
        for generated_action, expected_action in zip(
            generated_actions,
            expected_actions,
        ):
            if generated_action != expected_action:
                break
            valid_prefix += 1
        totals["operation_prefix_fraction"] += valid_prefix / max(
            len(expected_actions),
            1,
        )
        trace_exact = completed and generated_actions == expected_actions
        totals["trace_exact_match"] += float(trace_exact)
        totals["full_exact_match"] += float(trace_exact and exact)

        for index, expected_action in enumerate(expected_actions):
            if (
                index < len(generated_actions)
                and generated_actions[index] == expected_action
            ):
                action_correct += 1
            action_total += 1

    batch_size = values.shape[0]
    metrics = {name: value / batch_size for name, value in totals.items()}
    metrics["target_token_accuracy"] = action_correct / max(action_total, 1)
    metrics["full_target_token_accuracy"] = metrics["target_token_accuracy"]
    return metrics


def generated_pointer_quicksort_metrics(
    values: Tensor,
    rollouts: Sequence[PointerQuicksortRollout],
    vocabulary: PointerQuicksortVocabulary,
    traces: Sequence[PointerQuicksortTrace],
) -> dict[str, float]:
    """Score interactively executed pointer quicksort actions."""

    return _generated_executor_machine_metrics(
        values,
        rollouts,
        vocabulary.action_tokens,
        traces,
    )


def generated_adjacent_sort_metrics(
    values: Tensor,
    rollouts: Sequence[AdjacentSortRollout],
    vocabulary: AdjacentSortVocabulary,
    traces: Sequence[AdjacentSortTrace],
) -> dict[str, float]:
    """Score interactively executed adjacent-sort actions."""

    return _generated_executor_machine_metrics(
        values,
        rollouts,
        vocabulary.action_tokens,
        traces,
    )


def _generated_no_tool_machine_metrics(
    values: Tensor,
    generated_tokens: Tensor,
    vocabulary: Any,
    traces: Sequence[Any],
    replay_transcript: Callable[[Sequence[int], Sequence[int], Any], Any],
) -> dict[str, float]:
    """Score complete machine transcripts generated without a live tool."""

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
        "execution_completed": 0.0,
        "observation_exact_match": 0.0,
        "timed_out": 0.0,
    }
    action_correct = 0
    action_total = 0
    observation_correct = 0
    observation_total = 0
    full_token_correct = 0
    full_token_total = 0

    for input_row, generated_row, trace in zip(
        values.tolist(),
        generated_tokens.tolist(),
        traces,
    ):
        expected_values = sorted(int(value) for value in input_row)
        rollout = replay_transcript(
            input_row,
            generated_row,
            vocabulary,
        )
        generated_actions = rollout.action_tokens
        expected_actions = trace.action_tokens
        final_values = list(rollout.final_values)
        execution_completed = rollout.completed and rollout.valid_execution
        observation_exact = execution_completed and rollout.observations_valid

        totals["valid_syntax"] += float(rollout.syntax_valid)
        totals["trace_syntax_valid"] += float(rollout.syntax_valid)
        totals["execution_completed"] += float(execution_completed)
        totals["observation_exact_match"] += float(observation_exact)
        totals["timed_out"] += float(rollout.timed_out)
        totals["correct_length"] += float(len(final_values) == len(input_row))
        totals["sorted"] += float(
            all(
                left <= right
                for left, right in zip(final_values, final_values[1:])
            )
        )
        totals["multiset_preserved"] += float(
            sorted(final_values) == expected_values
        )
        exact = (
            observation_exact
            and rollout.syntax_valid
            and final_values == expected_values
        )
        totals["exact_match"] += float(exact)

        valid_prefix = 0
        for generated_action, expected_action in zip(
            generated_actions,
            expected_actions,
        ):
            if generated_action != expected_action:
                break
            valid_prefix += 1
        totals["operation_prefix_fraction"] += valid_prefix / max(
            len(expected_actions),
            1,
        )
        trace_exact = rollout.generated_tokens == trace.target_tokens
        totals["trace_exact_match"] += float(trace_exact)
        totals["full_exact_match"] += float(trace_exact and exact)

        for index, expected_action in enumerate(expected_actions):
            if (
                index < len(generated_actions)
                and generated_actions[index] == expected_action
            ):
                action_correct += 1
            action_total += 1
        observation_correct += rollout.observation_correct
        observation_total += len(trace.target_tokens) - len(expected_actions)
        for index, expected_token in enumerate(trace.target_tokens):
            if (
                index < len(rollout.generated_tokens)
                and rollout.generated_tokens[index] == expected_token
            ):
                full_token_correct += 1
            full_token_total += 1

    batch_size = values.shape[0]
    metrics = {name: value / batch_size for name, value in totals.items()}
    metrics["target_token_accuracy"] = action_correct / max(action_total, 1)
    metrics["observation_token_accuracy"] = observation_correct / max(
        observation_total,
        1,
    )
    metrics["full_target_token_accuracy"] = full_token_correct / max(
        full_token_total,
        1,
    )
    return metrics


def generated_pointer_no_tool_metrics(
    values: Tensor,
    generated_tokens: Tensor,
    vocabulary: PointerQuicksortVocabulary,
    traces: Sequence[PointerQuicksortTrace],
) -> dict[str, float]:
    """Score pointer transcripts generated without live executor observations."""

    return _generated_no_tool_machine_metrics(
        values,
        generated_tokens,
        vocabulary,
        traces,
        replay_pointer_quicksort_transcript,
    )


def generated_adjacent_no_tool_metrics(
    values: Tensor,
    generated_tokens: Tensor,
    vocabulary: AdjacentSortVocabulary,
    traces: Sequence[AdjacentSortTrace],
) -> dict[str, float]:
    """Score adjacent transcripts generated without live tool observations."""

    return _generated_no_tool_machine_metrics(
        values,
        generated_tokens,
        vocabulary,
        traces,
        replay_adjacent_sort_transcript,
    )


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
