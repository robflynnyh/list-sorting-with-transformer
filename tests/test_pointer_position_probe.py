from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from list_sorting_transformer.pointer_position_probe import (
    PointerPositionProbe,
    pointer_position_metrics,
)
from list_sorting_transformer.model import ModelConfig
from list_sorting_transformer.tokens import PointerNextVocabulary


def small_probe() -> PointerPositionProbe:
    vocabulary = PointerNextVocabulary("numbers", 10)
    return PointerPositionProbe(
        ModelConfig(
            vocab_size=vocabulary.size,
            representation="numbers",
            symbol_count=10,
            d_model=32,
            n_layers=2,
            n_heads=4,
            ffn_multiplier=2.0,
            position_pattern="none",
            rotate_values_with_rope=False,
        )
    )


def test_candidate_classes_use_pointer_token_offsets() -> None:
    model = small_probe()

    positions = model.candidate_positions(5, device=torch.device("cpu"))

    assert positions.tolist() == [1, 3, 5, 7]


def test_target_positions_are_the_actual_ptr_token_offsets() -> None:
    model = small_probe()
    pointers = torch.tensor([0, 2, 3])

    positions = model.target_positions(pointers)

    assert positions.tolist() == [1, 5, 7]


def test_pointer_position_probe_regresses_position_vector() -> None:
    torch.manual_seed(9)
    vocabulary = PointerNextVocabulary("numbers", 10)
    model = small_probe()
    prompt = torch.tensor(
        [
            vocabulary.encode_prompt_with_pointer([3, 1, 4, 1, 5], 2),
            vocabulary.encode_prompt_with_pointer([2, 7, 1, 8, 2], 0),
        ]
    )

    emitted_vectors = model(prompt)
    targets = model.target_embeddings(torch.tensor([2, 0]))
    loss = F.mse_loss(emitted_vectors, targets)
    loss.backward()

    assert emitted_vectors.shape == (2, 32)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_pointer_position_metrics_report_argmax_and_unseen_slices() -> None:
    model = small_probe()
    pointers = torch.tensor([0, 3, 2])
    emitted_vectors = model.target_embeddings(torch.tensor([0, 2, 1]))

    metrics = pointer_position_metrics(
        emitted_vectors,
        pointers,
        model=model,
        length=5,
        train_max_length=3,
    )

    assert metrics["argmax_accuracy"] == pytest.approx(1 / 3)
    assert metrics["argmax_token_mae"] == pytest.approx(4 / 3)
    assert metrics["seen_argmax_accuracy"] == 1.0
    assert metrics["unseen_argmax_accuracy"] == 0.0
    assert metrics["unseen_pointer_fraction"] == pytest.approx(2 / 3)
