from __future__ import annotations

import torch

from list_sorting_transformer.core.tokens import (
    VALUE_OFFSET,
    PointerCompareVocabulary,
)
from list_sorting_transformer.length_generalisation.pointer_trace_bubble_sort import (
    compose_bubble_sort,
    make_pointer_prompt,
)


def test_vectorized_pointer_prompt_matches_vocabulary_encoding() -> None:
    vocabulary = PointerCompareVocabulary(symbol_count=6)
    values = torch.tensor([[3, 1, 4, 1], [5, 2, 0, 3]])

    for pointer in range(values.shape[1] - 1):
        actual = make_pointer_prompt(values, pointer, vocabulary)
        expected = torch.tensor(
            [
                vocabulary.encode_prompt_with_pointer(row, pointer)
                for row in values.tolist()
            ]
        )
        torch.testing.assert_close(actual, expected)


def test_perfect_local_policy_composes_into_exact_sorting() -> None:
    vocabulary = PointerCompareVocabulary(symbol_count=5)
    values = torch.randint(
        0,
        vocabulary.symbol_count,
        (32, 20),
        generator=torch.Generator().manual_seed(17),
    )

    def perfect_policy(current: torch.Tensor, pointer: int) -> torch.Tensor:
        left = current[:, pointer]
        right = current[:, pointer + 1]
        action = torch.where(
            left > right,
            torch.full_like(left, vocabulary.action_token("SWAP")),
            torch.full_like(left, vocabulary.action_token("KEEP")),
        )
        return torch.stack(
            (left + VALUE_OFFSET, right + VALUE_OFFSET, action),
            dim=1,
        )

    result, final_values = compose_bubble_sort(
        values,
        perfect_policy,
        vocabulary,
    )

    torch.testing.assert_close(final_values, values.sort(dim=1).values)
    assert result.comparisons_per_example == 190
    assert result.final_sorted_accuracy == 1.0
    assert result.perfect_action_trace_accuracy == 1.0
    assert result.perfect_retrieval_trace_accuracy == 1.0
    assert result.perfect_complete_trace_accuracy == 1.0
    assert result.action_accuracy == 1.0
    assert result.remaining_inversion_fraction == 0.0
    assert result.first_action_error_step_min is None


def test_incorrect_keep_is_reported_and_leaves_inversion() -> None:
    vocabulary = PointerCompareVocabulary(symbol_count=4)
    values = torch.tensor([[2, 1]])

    def keep_policy(current: torch.Tensor, pointer: int) -> torch.Tensor:
        left = current[:, pointer]
        right = current[:, pointer + 1]
        action = torch.full_like(left, vocabulary.action_token("KEEP"))
        return torch.stack(
            (left + VALUE_OFFSET, right + VALUE_OFFSET, action),
            dim=1,
        )

    result, final_values = compose_bubble_sort(
        values,
        keep_policy,
        vocabulary,
    )

    torch.testing.assert_close(final_values, values)
    assert result.final_sorted_accuracy == 0.0
    assert result.perfect_action_trace_accuracy == 0.0
    assert result.action_accuracy == 0.0
    assert result.false_keep_rate == 1.0
    assert result.false_swap_rate == 0.0
    assert result.remaining_inversion_fraction == 1.0
    assert result.first_action_error_step_min == 0
