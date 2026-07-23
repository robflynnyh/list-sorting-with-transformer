"""Online generation of digit-list sorting examples."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .adjacent_sort import (
    AdjacentSortTrace,
    generate_adjacent_sort_trace,
    generate_auto_advance_sort_trace,
)
from .local_window_sort import (
    WINDOW_TOOL_EVENTS,
    LocalWindowSortTrace,
    generate_local_window_sort_trace,
)
from .pointer_quicksort import (
    PointerQuicksortTrace,
    generate_pointer_quicksort_trace,
)
from .quicksort import QuicksortTrace, SnapshotMode, generate_quicksort_trace
from .tokens import (
    BOS,
    COMMA,
    EOS,
    PAD,
    SEP,
    VALUE_OFFSET,
    AdjacentSortVocabulary,
    AutoAdvanceSortVocabulary,
    LocalWindowSortVocabulary,
    PointerNextVocabulary,
    PointerQuicksortVocabulary,
    QuicksortTraceVocabulary,
)


IGNORE_INDEX = -100


@dataclass(frozen=True)
class SortingBatch:
    token_ids: Tensor
    labels: Tensor
    values: Tensor
    length: int

    @property
    def model_inputs(self) -> Tensor:
        return self.token_ids[:, :-1]

    @property
    def prompt_ids(self) -> Tensor:
        return self.token_ids[:, : 2 * self.length + 1]


@dataclass(frozen=True)
class QuicksortTraceBatch:
    token_ids: Tensor
    labels: Tensor
    values: Tensor
    length: int
    prompt_length: int
    traces: tuple[QuicksortTrace, ...]

    @property
    def model_inputs(self) -> Tensor:
        return self.token_ids[:, :-1]

    @property
    def prompt_ids(self) -> Tensor:
        return self.token_ids[:, : self.prompt_length]


@dataclass(frozen=True)
class PointerQuicksortBatch:
    token_ids: Tensor
    labels: Tensor
    values: Tensor
    length: int
    prompt_length: int
    traces: tuple[PointerQuicksortTrace, ...]

    @property
    def model_inputs(self) -> Tensor:
        return self.token_ids[:, :-1]

    @property
    def prompt_ids(self) -> Tensor:
        return self.token_ids[:, : self.prompt_length]


@dataclass(frozen=True)
class AdjacentSortBatch:
    token_ids: Tensor
    labels: Tensor
    values: Tensor
    length: int
    prompt_length: int
    traces: tuple[AdjacentSortTrace, ...]

    @property
    def model_inputs(self) -> Tensor:
        return self.token_ids[:, :-1]

    @property
    def prompt_ids(self) -> Tensor:
        return self.token_ids[:, : self.prompt_length]


@dataclass(frozen=True)
class LocalWindowSortBatch:
    token_ids: Tensor
    labels: Tensor
    values: Tensor
    length: int
    prompt_length: int
    traces: tuple[LocalWindowSortTrace, ...]

    @property
    def model_inputs(self) -> Tensor:
        return self.token_ids[:, :-1]

    @property
    def prompt_ids(self) -> Tensor:
        return self.token_ids[:, : self.prompt_length]


@dataclass(frozen=True)
class PointerNextBatch:
    token_ids: Tensor
    labels: Tensor
    values: Tensor
    pointers: Tensor
    length: int
    prompt_length: int

    @property
    def model_inputs(self) -> Tensor:
        return self.token_ids[:, :-1]

    @property
    def prompt_ids(self) -> Tensor:
        return self.token_ids[:, : self.prompt_length]


def make_sorting_batch(
    batch_size: int,
    length: int,
    *,
    generator: torch.Generator,
    symbol_count: int = 10,
    device: torch.device | str | None = None,
) -> SortingBatch:
    """Generate a same-length batch without Python loops over examples."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if length < 1:
        raise ValueError("length must be positive")
    if symbol_count < 2:
        raise ValueError("symbol_count must be at least two")
    values = torch.randint(
        0,
        symbol_count,
        (batch_size, length),
        generator=generator,
    )
    sorted_values = values.sort(dim=1).values
    sequence_length = 4 * length + 1
    token_ids = torch.empty(batch_size, sequence_length, dtype=torch.long)
    token_ids[:, 0] = BOS

    digit_offsets = 1 + 2 * torch.arange(length)
    token_ids[:, digit_offsets] = values + VALUE_OFFSET
    if length > 1:
        token_ids[:, 2 : 2 * length : 2] = COMMA
    separator_index = 2 * length
    token_ids[:, separator_index] = SEP

    output_offsets = separator_index + 1 + 2 * torch.arange(length)
    token_ids[:, output_offsets] = sorted_values + VALUE_OFFSET
    if length > 1:
        token_ids[:, separator_index + 2 : 4 * length : 2] = COMMA
    token_ids[:, -1] = EOS

    labels = token_ids[:, 1:].clone()
    labels[:, :separator_index] = IGNORE_INDEX
    if device is not None:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        values = values.to(device)
    return SortingBatch(
        token_ids=token_ids,
        labels=labels,
        values=values,
        length=length,
    )


def make_pointer_next_batch(
    batch_size: int,
    length: int,
    *,
    generator: torch.Generator,
    vocabulary: PointerNextVocabulary,
    device: torch.device | str | None = None,
) -> PointerNextBatch:
    """Generate examples that ask for the value after a marked pointer."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if length < 2:
        raise ValueError("pointer-next length must be at least two")
    values = torch.randint(
        0,
        vocabulary.symbol_count,
        (batch_size, length),
        generator=generator,
    )
    pointers = torch.randint(
        0,
        length - 1,
        (batch_size,),
        generator=generator,
    )
    examples = tuple(
        vocabulary.encode_example_with_pointer(row, int(pointer))
        for row, pointer in zip(values.tolist(), pointers.tolist())
    )
    sequence_length = len(examples[0])
    prompt_length = sequence_length - 2
    token_ids = torch.empty(batch_size, sequence_length, dtype=torch.long)
    for row_index, example in enumerate(examples):
        token_ids[row_index] = torch.tensor(example)

    labels = token_ids[:, 1:].clone()
    labels[:, : prompt_length - 1] = IGNORE_INDEX
    if device is not None:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        values = values.to(device)
        pointers = pointers.to(device)
    return PointerNextBatch(
        token_ids=token_ids,
        labels=labels,
        values=values,
        pointers=pointers,
        length=length,
        prompt_length=prompt_length,
    )


def make_quicksort_trace_batch(
    batch_size: int,
    length: int,
    *,
    generator: torch.Generator,
    vocabulary: QuicksortTraceVocabulary,
    snapshot_mode: SnapshotMode = "partition",
    device: torch.device | str | None = None,
) -> QuicksortTraceBatch:
    """Generate and pad instruction-level quicksort traces."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if length < 1:
        raise ValueError("length must be positive")
    values = torch.randint(
        0,
        vocabulary.symbol_count,
        (batch_size, length),
        generator=generator,
    )
    traces = tuple(
        generate_quicksort_trace(row, vocabulary, snapshot_mode=snapshot_mode)
        for row in values.tolist()
    )
    prompts = tuple(vocabulary.encode_prompt(row) for row in values.tolist())
    prompt_length = len(prompts[0])
    examples = tuple(
        [*prompt, *trace.target_tokens]
        for prompt, trace in zip(prompts, traces)
    )
    sequence_length = max(len(example) for example in examples)
    token_ids = torch.full(
        (batch_size, sequence_length),
        PAD,
        dtype=torch.long,
    )
    for row_index, example in enumerate(examples):
        token_ids[row_index, : len(example)] = torch.tensor(example)

    labels = token_ids[:, 1:].clone()
    labels[:, : prompt_length - 1] = IGNORE_INDEX
    labels[labels.eq(PAD)] = IGNORE_INDEX
    if device is not None:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        values = values.to(device)
    return QuicksortTraceBatch(
        token_ids=token_ids,
        labels=labels,
        values=values,
        length=length,
        prompt_length=prompt_length,
        traces=traces,
    )


def make_pointer_quicksort_batch(
    batch_size: int,
    length: int,
    *,
    generator: torch.Generator,
    vocabulary: PointerQuicksortVocabulary,
    supervise_observations: bool = False,
    device: torch.device | str | None = None,
) -> PointerQuicksortBatch:
    """Generate pointer-machine transcripts and mask executor observations."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if length < 1:
        raise ValueError("length must be positive")
    values = torch.randint(
        0,
        vocabulary.symbol_count,
        (batch_size, length),
        generator=generator,
    )
    value_rows = values.tolist()
    traces = tuple(
        generate_pointer_quicksort_trace(row, vocabulary)
        for row in value_rows
    )
    prompts = tuple(vocabulary.encode_prompt(row) for row in value_rows)
    prompt_length = len(prompts[0])
    sequence_length = max(
        prompt_length + len(trace.target_tokens)
        for trace in traces
    )
    token_ids = torch.full(
        (batch_size, sequence_length),
        PAD,
        dtype=torch.long,
    )
    prediction_mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.bool,
    )
    for row_index, (prompt, trace) in enumerate(zip(prompts, traces)):
        example = [*prompt, *trace.target_tokens]
        token_ids[row_index, : len(example)] = torch.tensor(example)
        target_end = prompt_length + len(trace.target_tokens)
        if supervise_observations:
            prediction_mask[row_index, prompt_length:target_end] = True
        else:
            prediction_mask[row_index, prompt_length:target_end] = torch.tensor(
                trace.target_prediction_mask,
                dtype=torch.bool,
            )

    labels = token_ids[:, 1:].clone()
    labels[~prediction_mask[:, 1:]] = IGNORE_INDEX
    if device is not None:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        values = values.to(device)
    return PointerQuicksortBatch(
        token_ids=token_ids,
        labels=labels,
        values=values,
        length=length,
        prompt_length=prompt_length,
        traces=traces,
    )


def make_adjacent_sort_batch(
    batch_size: int,
    length: int,
    *,
    generator: torch.Generator,
    vocabulary: AdjacentSortVocabulary,
    supervise_observations: bool = False,
    device: torch.device | str | None = None,
) -> AdjacentSortBatch:
    """Generate adjacent-pair transcripts and mask tool observations."""

    return _make_adjacent_sort_batch(
        batch_size,
        length,
        generator=generator,
        vocabulary=vocabulary,
        trace_generator=generate_adjacent_sort_trace,
        supervise_observations=supervise_observations,
        device=device,
    )


def make_auto_advance_sort_batch(
    batch_size: int,
    length: int,
    *,
    generator: torch.Generator,
    vocabulary: AutoAdvanceSortVocabulary,
    supervise_observations: bool = False,
    device: torch.device | str | None = None,
) -> AdjacentSortBatch:
    """Generate transcripts whose executor controls cursor advancement."""

    return _make_adjacent_sort_batch(
        batch_size,
        length,
        generator=generator,
        vocabulary=vocabulary,
        trace_generator=generate_auto_advance_sort_trace,
        supervise_observations=supervise_observations,
        device=device,
    )


def _make_adjacent_sort_batch(
    batch_size: int,
    length: int,
    *,
    generator: torch.Generator,
    vocabulary: AdjacentSortVocabulary | AutoAdvanceSortVocabulary,
    trace_generator: Callable[..., AdjacentSortTrace],
    supervise_observations: bool,
    device: torch.device | str | None,
) -> AdjacentSortBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if length < 1:
        raise ValueError("length must be positive")
    values = torch.randint(
        0,
        vocabulary.symbol_count,
        (batch_size, length),
        generator=generator,
    )
    value_rows = values.tolist()
    traces = tuple(
        trace_generator(row, vocabulary)
        for row in value_rows
    )
    prompts = tuple(vocabulary.encode_prompt(row) for row in value_rows)
    prompt_length = len(prompts[0])
    sequence_length = max(
        prompt_length + len(trace.target_tokens)
        for trace in traces
    )
    token_ids = torch.full(
        (batch_size, sequence_length),
        PAD,
        dtype=torch.long,
    )
    prediction_mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.bool,
    )
    for row_index, (prompt, trace) in enumerate(zip(prompts, traces)):
        example = [*prompt, *trace.target_tokens]
        token_ids[row_index, : len(example)] = torch.tensor(example)
        target_end = prompt_length + len(trace.target_tokens)
        if supervise_observations:
            prediction_mask[row_index, prompt_length:target_end] = True
        else:
            prediction_mask[row_index, prompt_length:target_end] = torch.tensor(
                trace.target_prediction_mask,
                dtype=torch.bool,
            )

    labels = token_ids[:, 1:].clone()
    labels[~prediction_mask[:, 1:]] = IGNORE_INDEX
    if device is not None:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        values = values.to(device)
    return AdjacentSortBatch(
        token_ids=token_ids,
        labels=labels,
        values=values,
        length=length,
        prompt_length=prompt_length,
        traces=traces,
    )


def make_local_window_sort_batch(
    batch_size: int,
    length: int,
    *,
    generator: torch.Generator,
    vocabulary: LocalWindowSortVocabulary,
    tool_events: Sequence[str] = WINDOW_TOOL_EVENTS,
    device: torch.device | str | None = None,
) -> LocalWindowSortBatch:
    """Generate continuous action/fixed-window autoregressive transcripts."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if length < 1:
        raise ValueError("length must be positive")
    tool_event_names = frozenset(str(name).upper() for name in tool_events)
    if not tool_event_names <= set(WINDOW_TOOL_EVENTS):
        allowed = ", ".join(WINDOW_TOOL_EVENTS)
        raise ValueError(f"tool_events may only contain {allowed}")

    values = torch.randint(
        0,
        vocabulary.symbol_count,
        (batch_size, length),
        generator=generator,
    )
    traces = tuple(
        generate_local_window_sort_trace(row, vocabulary)
        for row in values.tolist()
    )
    prompts = tuple(
        (
            *vocabulary.encode_prompt(row),
            *trace.initial_window_tokens,
        )
        for row, trace in zip(values.tolist(), traces)
    )
    prompt_length = len(prompts[0])
    targets: list[tuple[int, ...]] = []
    target_masks: list[tuple[bool, ...]] = []
    for trace in traces:
        target_tokens: list[int] = []
        prediction_mask: list[bool] = []
        for transition in trace.transitions:
            target_tokens.append(transition.action_token)
            prediction_mask.append(True)
            if transition.response_tokens:
                target_tokens.extend(transition.response_tokens)
                supervise_window = (
                    transition.response_event not in tool_event_names
                )
                prediction_mask.extend(
                    supervise_window
                    for _ in transition.response_tokens
                )
        targets.append(tuple(target_tokens))
        target_masks.append(tuple(prediction_mask))

    sequence_length = max(
        prompt_length + len(target)
        for target in targets
    )
    token_ids = torch.full(
        (batch_size, sequence_length),
        PAD,
        dtype=torch.long,
    )
    prediction_mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.bool,
    )

    for row_index, (prompt, target, target_mask) in enumerate(
        zip(prompts, targets, target_masks)
    ):
        if len(prompt) != prompt_length:
            raise RuntimeError("local-window prompt has an unexpected length")
        example = [*prompt, *target]
        token_ids[row_index, : len(example)] = torch.tensor(example)
        target_end = prompt_length + len(target)
        prediction_mask[row_index, prompt_length:target_end] = torch.tensor(
            target_mask,
            dtype=torch.bool,
        )

    labels = token_ids[:, 1:].clone()
    labels[~prediction_mask[:, 1:]] = IGNORE_INDEX
    if device is not None:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        values = values.to(device)
    return LocalWindowSortBatch(
        token_ids=token_ids,
        labels=labels,
        values=values,
        length=length,
        prompt_length=prompt_length,
        traces=traces,
    )


def sample_length(
    minimum: int,
    maximum: int,
    *,
    generator: torch.Generator,
) -> int:
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid length range")
    return int(torch.randint(minimum, maximum + 1, (), generator=generator))
