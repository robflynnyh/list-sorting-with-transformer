"""Online generation of digit-list sorting examples."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

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


def sample_length(
    minimum: int,
    maximum: int,
    *,
    generator: torch.Generator,
) -> int:
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid length range")
    return int(torch.randint(minimum, maximum + 1, (), generator=generator))
