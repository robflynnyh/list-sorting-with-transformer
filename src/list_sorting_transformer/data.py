"""Online generation of digit-list sorting examples."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .tokens import BOS, COMMA, EOS, SEP, VALUE_OFFSET


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


def sample_length(
    minimum: int,
    maximum: int,
    *,
    generator: torch.Generator,
) -> int:
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid length range")
    return int(torch.randint(minimum, maximum + 1, (), generator=generator))
