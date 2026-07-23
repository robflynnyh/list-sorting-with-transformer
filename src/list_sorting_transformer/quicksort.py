"""Deterministic instruction-level traces for three-way quicksort."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .tokens import EOS, QuicksortTraceVocabulary


SnapshotMode = Literal["none", "partition", "swap"]
SNAPSHOT_MODES = ("none", "partition", "swap")


@dataclass(frozen=True)
class TraceEvent:
    operation: str
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class QuicksortTrace:
    input_values: tuple[int, ...]
    events: tuple[TraceEvent, ...]
    target_tokens: tuple[int, ...]


def _validate_snapshot_mode(snapshot_mode: str) -> None:
    if snapshot_mode not in SNAPSHOT_MODES:
        raise ValueError(
            f"snapshot_mode must be one of {', '.join(SNAPSHOT_MODES)}"
        )


def generate_quicksort_trace(
    values: list[int] | tuple[int, ...],
    vocabulary: QuicksortTraceVocabulary,
    *,
    snapshot_mode: SnapshotMode = "partition",
) -> QuicksortTrace:
    """Execute deterministic three-way quicksort and serialize every operation."""

    _validate_snapshot_mode(snapshot_mode)
    if not values:
        raise ValueError("quicksort traces require a non-empty input")
    array = [int(value) for value in values]
    for value in array:
        vocabulary.value_token(value)

    events: list[TraceEvent] = []

    def emit(operation: str, payload: list[int] | None = None) -> None:
        tokens = [vocabulary.trace_token(operation)]
        if payload is not None:
            tokens.extend(payload)
        events.append(TraceEvent(operation, tuple(tokens)))

    def index(index_value: int) -> list[int]:
        return vocabulary.encode_index(index_value)

    def value(value_item: int) -> list[int]:
        return [vocabulary.value_token(value_item)]

    def range_payload(lo: int, hi: int) -> list[int]:
        return [*index(lo), *index(hi)]

    def check_and_push(stack: list[tuple[int, int]], lo: int, hi: int) -> None:
        active = lo < hi
        emit(
            "CHECK_RANGE",
            [
                *range_payload(lo, hi),
                vocabulary.trace_token("ACTIVE" if active else "SKIP"),
            ],
        )
        if active:
            emit("PUSH", range_payload(lo, hi))
            stack.append((lo, hi))

    stack: list[tuple[int, int]] = []
    check_and_push(stack, 0, len(array) - 1)
    while stack:
        lo, hi = stack.pop()
        emit("POP", range_payload(lo, hi))

        pivot_index = (lo + hi) // 2
        pivot = array[pivot_index]
        emit(
            "LOAD_PIVOT",
            [*index(pivot_index), *value(pivot)],
        )

        lower = lo
        scan = lo
        upper = hi
        emit("SET_LT", index(lower))
        emit("SET_SCAN", index(scan))
        emit("SET_GT", index(upper))

        while scan <= upper:
            scanned_value = array[scan]
            if scanned_value < pivot:
                comparison = "LESS"
            elif scanned_value > pivot:
                comparison = "GREATER"
            else:
                comparison = "EQUAL"
            emit(
                "COMPARE",
                [
                    *index(scan),
                    *value(scanned_value),
                    *value(pivot),
                    vocabulary.trace_token(comparison),
                ],
            )

            if comparison == "LESS":
                emit("SWAP", [*index(lower), *index(scan)])
                array[lower], array[scan] = array[scan], array[lower]
                if snapshot_mode == "swap":
                    emit("ARRAY", [vocabulary.value_token(item) for item in array])
                lower += 1
                emit("INC_LT", index(lower))
                scan += 1
                emit("INC_SCAN", index(scan))
            elif comparison == "GREATER":
                emit("SWAP", [*index(scan), *index(upper)])
                array[scan], array[upper] = array[upper], array[scan]
                if snapshot_mode == "swap":
                    emit("ARRAY", [vocabulary.value_token(item) for item in array])
                upper -= 1
                emit("DEC_GT", index(upper))
            else:
                scan += 1
                emit("INC_SCAN", index(scan))

        emit("PARTITION_DONE", [*index(lower), *index(upper)])
        if snapshot_mode == "partition":
            emit("ARRAY", [vocabulary.value_token(item) for item in array])

        # Push right first so the left range is processed next by the LIFO stack.
        check_and_push(stack, upper + 1, hi)
        check_and_push(stack, lo, lower - 1)

    answer = sorted(int(value) for value in values)
    if array != answer:
        raise RuntimeError("reference quicksort trace did not sort the input")
    emit("DONE")
    answer_tokens = [
        vocabulary.trace_token("ANSWER"),
        *vocabulary.encode_list(answer),
        EOS,
    ]
    target_tokens = tuple(
        token
        for event in events
        for token in event.tokens
    ) + tuple(answer_tokens)
    return QuicksortTrace(
        input_values=tuple(int(value) for value in values),
        events=tuple(events),
        target_tokens=target_tokens,
    )


def split_generated_events(
    tokens: list[int] | tuple[int, ...],
    vocabulary: QuicksortTraceVocabulary,
) -> tuple[list[tuple[int, ...]], list[int] | None, bool]:
    """Split generated tokens into operation events and an answer suffix."""

    operation_tokens = vocabulary.operation_tokens
    answer_token = vocabulary.trace_token("ANSWER")
    events: list[list[int]] = []
    answer: list[int] | None = None
    syntax_valid = True

    for position, raw_token in enumerate(tokens):
        token = int(raw_token)
        if token == answer_token:
            answer = [int(item) for item in tokens[position + 1 :]]
            break
        if token == EOS:
            syntax_valid = False
            break
        if token in operation_tokens:
            events.append([token])
        elif not events:
            syntax_valid = False
            break
        else:
            events[-1].append(token)
    if answer is None:
        syntax_valid = False
    return [tuple(event) for event in events], answer, syntax_valid
