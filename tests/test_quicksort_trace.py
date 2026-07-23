from __future__ import annotations

import torch

from list_sorting_transformer.data import (
    IGNORE_INDEX,
    make_quicksort_trace_batch,
)
from list_sorting_transformer.metrics import generated_quicksort_metrics
from list_sorting_transformer.quicksort import generate_quicksort_trace
from list_sorting_transformer.tokens import (
    COMMA,
    EOS,
    PAD,
    VALUE_OFFSET,
    QuicksortTraceVocabulary,
)


def test_trace_vocabulary_separates_values_indices_and_operations() -> None:
    vocabulary = QuicksortTraceVocabulary("numbers", 10)

    assert vocabulary.value_token(2) != vocabulary.index_digit_token(2)
    assert vocabulary.trace_token("SWAP") in vocabulary.operation_tokens
    assert vocabulary.trace_token("LESS") not in vocabulary.operation_tokens
    assert vocabulary.encode_index(25) == [
        vocabulary.trace_token("IDX"),
        vocabulary.index_digit_token(2),
        vocabulary.index_digit_token(5),
    ]
    assert vocabulary.encode_index(-1) == [
        vocabulary.trace_token("IDX"),
        vocabulary.trace_token("NEG"),
        vocabulary.index_digit_token(1),
    ]
    assert vocabulary.size > VALUE_OFFSET + vocabulary.symbol_count


def test_quicksort_trace_is_deterministic_and_fully_sorts() -> None:
    vocabulary = QuicksortTraceVocabulary()
    trace = generate_quicksort_trace([3, 1, 2], vocabulary)

    assert [event.operation for event in trace.events] == [
        "CHECK_RANGE",
        "PUSH",
        "POP",
        "LOAD_PIVOT",
        "SET_LT",
        "SET_SCAN",
        "SET_GT",
        "COMPARE",
        "SWAP",
        "DEC_GT",
        "COMPARE",
        "SWAP",
        "DEC_GT",
        "COMPARE",
        "INC_SCAN",
        "PARTITION_DONE",
        "ARRAY",
        "CHECK_RANGE",
        "PUSH",
        "CHECK_RANGE",
        "POP",
        "LOAD_PIVOT",
        "SET_LT",
        "SET_SCAN",
        "SET_GT",
        "COMPARE",
        "INC_SCAN",
        "COMPARE",
        "SWAP",
        "DEC_GT",
        "PARTITION_DONE",
        "ARRAY",
        "CHECK_RANGE",
        "CHECK_RANGE",
        "DONE",
    ]
    assert trace.target_tokens[-7:] == tuple(
        [
            vocabulary.trace_token("ANSWER"),
            vocabulary.value_token(1),
            COMMA,
            vocabulary.value_token(2),
            COMMA,
            vocabulary.value_token(3),
            EOS,
        ]
    )
    array_events = [
        event for event in trace.events if event.operation == "ARRAY"
    ]
    assert list(array_events[-1].tokens[1:]) == [
        vocabulary.value_token(1),
        vocabulary.value_token(2),
        vocabulary.value_token(3),
    ]


def test_snapshot_mode_controls_array_repetition() -> None:
    vocabulary = QuicksortTraceVocabulary()
    values = [3, 1, 2]

    no_snapshots = generate_quicksort_trace(
        values,
        vocabulary,
        snapshot_mode="none",
    )
    partition_snapshots = generate_quicksort_trace(
        values,
        vocabulary,
        snapshot_mode="partition",
    )
    swap_snapshots = generate_quicksort_trace(
        values,
        vocabulary,
        snapshot_mode="swap",
    )

    assert not any(event.operation == "ARRAY" for event in no_snapshots.events)
    assert sum(
        event.operation == "ARRAY" for event in partition_snapshots.events
    ) == 2
    assert sum(
        event.operation == "ARRAY" for event in swap_snapshots.events
    ) == 3


def test_reference_quicksort_sorts_random_duplicate_heavy_lists() -> None:
    vocabulary = QuicksortTraceVocabulary()
    generator = torch.Generator().manual_seed(29)

    for length in range(2, 21):
        values = torch.randint(0, 4, (length,), generator=generator).tolist()
        trace = generate_quicksort_trace(values, vocabulary)
        final_array = [
            event for event in trace.events if event.operation == "ARRAY"
        ][-1]
        assert list(final_array.tokens[1:]) == [
            vocabulary.value_token(value) for value in sorted(values)
        ]


def test_trace_batch_pads_targets_and_masks_prompt_and_padding() -> None:
    vocabulary = QuicksortTraceVocabulary()
    batch = make_quicksort_trace_batch(
        4,
        5,
        generator=torch.Generator().manual_seed(3),
        vocabulary=vocabulary,
    )

    assert batch.token_ids.shape[0] == 4
    assert batch.prompt_length == 11
    assert batch.prompt_ids.shape == (4, 11)
    assert torch.all(batch.labels[:, : batch.prompt_length - 1].eq(IGNORE_INDEX))
    assert torch.all(batch.labels[batch.token_ids[:, 1:].eq(PAD)].eq(IGNORE_INDEX))
    for row, trace in zip(batch.token_ids.tolist(), batch.traces):
        target_start = batch.prompt_length
        assert tuple(
            row[target_start : target_start + len(trace.target_tokens)]
        ) == trace.target_tokens


def test_trace_metrics_separate_answer_from_valid_execution() -> None:
    vocabulary = QuicksortTraceVocabulary()
    values = torch.tensor([[3, 1, 2]])
    trace = generate_quicksort_trace(values[0].tolist(), vocabulary)
    perfect = torch.tensor([trace.target_tokens])

    metrics = generated_quicksort_metrics(
        values,
        perfect,
        vocabulary,
        [trace],
    )
    assert metrics["exact_match"] == 1.0
    assert metrics["trace_exact_match"] == 1.0
    assert metrics["full_exact_match"] == 1.0
    assert metrics["operation_prefix_fraction"] == 1.0

    corrupted = perfect.clone()
    corrupted[0, 0] = vocabulary.trace_token("POP")
    metrics = generated_quicksort_metrics(
        values,
        corrupted,
        vocabulary,
        [trace],
    )
    assert metrics["exact_match"] == 1.0
    assert metrics["trace_exact_match"] == 0.0
    assert metrics["full_exact_match"] == 0.0
    assert metrics["operation_prefix_fraction"] == 0.0


def test_trace_operation_in_answer_is_invalid_instead_of_raising() -> None:
    vocabulary = QuicksortTraceVocabulary()
    values = torch.tensor([[2, 0]])
    trace = generate_quicksort_trace(values[0].tolist(), vocabulary)
    generated = list(trace.target_tokens)
    answer_index = generated.index(vocabulary.trace_token("ANSWER"))
    generated[answer_index + 1] = vocabulary.trace_token("POP")

    metrics = generated_quicksort_metrics(
        values,
        torch.tensor([generated]),
        vocabulary,
        [trace],
    )

    assert metrics["valid_syntax"] == 0.0
    assert metrics["exact_match"] == 0.0
