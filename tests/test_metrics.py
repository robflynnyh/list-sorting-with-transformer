from __future__ import annotations

import torch

from list_sorting_transformer.metrics import generated_sorting_metrics
from list_sorting_transformer.tokens import SymbolVocabulary


def test_strict_metrics_separate_order_from_multiset_preservation() -> None:
    vocabulary = SymbolVocabulary()
    values = torch.tensor([[2, 0, 1], [3, 1, 3]])
    generated = torch.tensor(
        [
            vocabulary.encode_target([0, 2, 1]),
            vocabulary.encode_target([1, 3, 3]),
        ]
    )

    metrics = generated_sorting_metrics(values, generated, vocabulary)

    assert metrics["valid_syntax"] == 1.0
    assert metrics["correct_length"] == 1.0
    assert metrics["sorted"] == 0.5
    assert metrics["multiset_preserved"] == 1.0
    assert metrics["exact_match"] == 0.5


def test_invalid_generation_gets_no_structural_credit() -> None:
    vocabulary = SymbolVocabulary()
    values = torch.tensor([[2, 0]])
    generated = torch.tensor(
        [[vocabulary.value_token(0), vocabulary.value_token(2), 3]]
    )

    metrics = generated_sorting_metrics(values, generated, vocabulary)

    assert metrics["valid_syntax"] == 0.0
    assert metrics["correct_length"] == 0.0
    assert metrics["exact_match"] == 0.0
