from __future__ import annotations

import pytest
import torch

from list_sorting_transformer.model import ModelConfig
from list_sorting_transformer.pointer_position_sequence import (
    ModularPositionSequenceModel,
    PositionSequenceConfig,
    generated_metrics,
    gradient_noise_std,
    sequence_loss_and_metrics,
)
from list_sorting_transformer.tokens import PointerNextVocabulary


def small_model() -> ModularPositionSequenceModel:
    vocabulary = PointerNextVocabulary("numbers", 10)
    return ModularPositionSequenceModel(
        ModelConfig(
            vocab_size=vocabulary.size,
            representation="numbers",
            symbol_count=10,
            d_model=32,
            n_layers=2,
            n_heads=4,
            ffn_multiplier=2.0,
            position_pattern="none",
        ),
        (3, 5, 7, 11),
    )


def test_target_sequence_contains_pointer_then_successor_residues() -> None:
    model = small_model()

    targets = model.target_sequence(
        pointers=torch.tensor([0, 2]),
        offsets=torch.tensor([20, -10]),
    )

    assert targets.tolist() == [
        [[0, 1, 0, 10], [1, 2, 1, 0]],
        [[1, 0, 2, 6], [2, 1, 3, 7]],
    ]


def test_history_embeddings_put_each_residue_in_its_product_key_slice() -> None:
    model = small_model()
    history = torch.tensor([[[1, 2, 3, 4]]])

    embeddings = model.history_embeddings(history)

    assert embeddings.shape == (1, 1, 16)
    assert torch.allclose(
        embeddings[0, 0, :4],
        model.position_embedding.codebooks[0].weight[1],
    )
    assert torch.allclose(
        embeddings[0, 0, 12:],
        model.position_embedding.codebooks[3].weight[4],
    )


def test_teacher_forced_sequence_has_eight_categorical_predictions() -> None:
    torch.manual_seed(4)
    model = small_model()
    vocabulary = PointerNextVocabulary("numbers", 10)
    prompt = torch.tensor(
        [
            vocabulary.encode_prompt_with_pointer([3, 1, 4, 1, 5], 2),
            vocabulary.encode_prompt_with_pointer([2, 7, 1, 8, 2], 0),
        ]
    )
    pointers = torch.tensor([2, 0])
    offsets = torch.tensor([-12, 30])
    targets = model.target_sequence(pointers, offsets)

    logits = model.teacher_forced_logits(prompt, targets, offsets=offsets)

    assert [
        [component.shape for component in position_logits]
        for position_logits in logits
    ] == [
        [(2, 3), (2, 5), (2, 7), (2, 11)],
        [(2, 3), (2, 5), (2, 7), (2, 11)],
    ]


def test_attention_supervision_covers_every_layer_and_head() -> None:
    torch.manual_seed(5)
    model = small_model()
    vocabulary = PointerNextVocabulary("numbers", 10)
    prompt = torch.tensor(
        [
            vocabulary.encode_prompt_with_pointer([3, 1, 4], 1),
            vocabulary.encode_prompt_with_pointer([2, 7, 1], 0),
        ]
    )
    pointers = torch.tensor([1, 0])
    offsets = torch.tensor([-12, 30])
    targets = model.target_sequence(pointers, offsets)

    logits, attention_logits = (
        model.teacher_forced_logits_with_successor_attention(
            prompt,
            targets,
            offsets=offsets,
        )
    )

    assert len(logits) == 2
    assert attention_logits.shape == (
        2,
        model.encoder.config.n_layers,
        model.encoder.config.n_heads,
        prompt.shape[1] + 1,
    )


def test_attention_supervision_adds_routing_metrics_and_gradients() -> None:
    torch.manual_seed(6)
    model = small_model()
    vocabulary = PointerNextVocabulary("numbers", 10)
    prompt = torch.tensor(
        [
            vocabulary.encode_prompt_with_pointer([3, 1, 4], 1),
            vocabulary.encode_prompt_with_pointer([2, 7, 1], 0),
        ]
    )

    loss, metrics = sequence_loss_and_metrics(
        model,
        prompt,
        torch.tensor([1, 0]),
        torch.tensor([-12, 30]),
        successor_attention_supervision_weight=0.1,
    )
    loss.backward()

    assert metrics["loss"] > metrics["position_loss"]
    assert metrics["successor_attention_supervision_loss"] > 0
    assert 0 <= metrics["successor_attention_target_accuracy"] <= 1
    assert 0 <= metrics["successor_attention_target_probability"] <= 1
    assert model.encoder.blocks[0].attention.qkv.weight.grad is not None


def test_successor_attention_isolation_only_changes_final_query_row() -> None:
    mask = ModularPositionSequenceModel.successor_attention_mask(
        batch_size=2,
        stream_length=6,
        history_length=1,
        isolate_successor=torch.tensor([True, False]),
        device=torch.device("cpu"),
    )

    assert mask is not None
    assert mask[0, :-1].all()
    assert mask[0, -1].tolist() == [False, False, False, False, False, True]
    assert mask[1].all()


def test_gradient_noise_std_uses_configured_decay() -> None:
    config = PositionSequenceConfig(
        gradient_noise_scale=0.01,
        gradient_noise_decay=0.5,
    )

    assert gradient_noise_std(config, 1) == pytest.approx(0.01)
    assert gradient_noise_std(config, 100) == pytest.approx(0.001)


def test_attention_supervision_replaces_successor_isolation() -> None:
    with pytest.raises(
        ValueError,
        match="supervision replaces successor isolation",
    ):
        PositionSequenceConfig(
            successor_attention_supervision_weight=0.1,
            successor_attention_isolation_probability=0.5,
        )


def test_generated_metrics_separate_accuracy_from_successor_consistency() -> None:
    targets = torch.tensor([[[1, 2, 3, 4], [2, 3, 4, 5]]])
    consistently_wrong = torch.tensor([[[0, 1, 2, 3], [1, 2, 3, 4]]])

    metrics = generated_metrics(
        consistently_wrong,
        targets,
        moduli=(3, 5, 7, 11),
    )

    assert metrics["both_positions_accuracy"] == 0.0
    assert metrics["successor_consistency"] == 1.0
