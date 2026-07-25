from __future__ import annotations

import pytest
import torch

from list_sorting_transformer.data import make_pointer_pair_batch
from list_sorting_transformer.model import ModelConfig
from list_sorting_transformer.pointer_next_value_from_position import (
    ModularNextValueFromPositionModel,
    NextValueFromPositionConfig,
    generated_stage_five_metrics,
    load_stage_four_checkpoint,
    next_value_token_loss_and_metrics,
    target_next_value_token_ids,
)
from list_sorting_transformer.pointer_next_value_position import (
    ModularNextValuePositionModel,
)
from list_sorting_transformer.pointer_value_from_position import (
    target_token_ids,
)
from list_sorting_transformer.tokens import VALUE_OFFSET, PointerNextVocabulary


def small_model() -> ModularNextValueFromPositionModel:
    vocabulary = PointerNextVocabulary("numbers", 10)
    return ModularNextValueFromPositionModel(
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


def small_config() -> NextValueFromPositionConfig:
    return NextValueFromPositionConfig(
        eval_max_length=40,
        position_moduli=(3, 5, 7, 11),
        position_offset_min=-50,
        position_offset_max=50,
        successor_attention_isolation_probability=0.5,
        next_value_position_attention_isolation_probability=0.5,
        next_value_position_consistency_weight=1.0,
        stage_three_distillation_weight=0.0,
        next_value_token_loss_weight=1.0,
        stage_four_distillation_weight=1.0,
    )


def test_target_is_value_after_pointer_and_pointer_is_never_last() -> None:
    vocabulary = PointerNextVocabulary("numbers", 10)
    batch = make_pointer_pair_batch(
        64,
        5,
        generator=torch.Generator().manual_seed(12),
        vocabulary=vocabulary,
    )

    targets = target_next_value_token_ids(batch)
    expected_values = batch.values[
        torch.arange(batch.values.shape[0]),
        batch.pointers + 1,
    ]

    assert bool((batch.pointers < batch.length - 1).all())
    torch.testing.assert_close(targets, expected_values + VALUE_OFFSET)


def test_stage_five_history_appends_final_address_after_marked_token() -> None:
    torch.manual_seed(2)
    model = small_model()
    vocabulary = PointerNextVocabulary("numbers", 10)
    batch = make_pointer_pair_batch(
        2,
        4,
        generator=torch.Generator().manual_seed(3),
        vocabulary=vocabulary,
    )
    offsets = torch.tensor([-12, 30])
    positions = model.target_sequence(batch.pointers, offsets)
    marked_tokens = target_token_ids(batch)
    next_positions = model.target_next_value_position(
        batch.pointers,
        offsets,
    )

    hidden = model.stage_five_hidden_states(
        batch.prompt_ids,
        positions,
        marked_tokens[:, None],
        next_positions[:, None],
        offsets=offsets,
    )
    logits = model.teacher_forced_next_value_token_logits(
        batch.prompt_ids,
        positions,
        marked_tokens,
        next_positions,
        offsets=offsets,
    )

    assert hidden.shape == (2, batch.prompt_length + 4, 32)
    assert logits.shape == (2, vocabulary.size)


def test_stage_four_checkpoint_transfers_without_new_parameters(
    tmp_path,
) -> None:
    torch.manual_seed(3)
    target_model = small_model()
    source_model = ModularNextValuePositionModel(
        target_model.encoder.config,
        target_model.moduli,
    )
    checkpoint_path = tmp_path / "stage_four.pt"
    torch.save(
        {
            "probe": "pointer_next_value_position",
            "model_state": source_model.state_dict(),
            "step": 20_000,
        },
        checkpoint_path,
    )

    metadata = load_stage_four_checkpoint(target_model, checkpoint_path)

    assert metadata["stage_four_step"] == 20_000
    assert metadata["transferred_tensors"] == len(source_model.state_dict())
    torch.testing.assert_close(
        target_model.token_query_projection.weight,
        source_model.token_query_projection.weight,
    )


def test_stage_five_loss_trains_retrieval_and_preserves_frozen_teacher() -> None:
    torch.manual_seed(4)
    model = small_model()
    teacher = ModularNextValuePositionModel(
        model.encoder.config,
        model.moduli,
    )
    teacher.load_state_dict(model.state_dict())
    teacher.requires_grad_(False)
    teacher.eval()
    vocabulary = PointerNextVocabulary("numbers", 10)
    batch = make_pointer_pair_batch(
        4,
        5,
        generator=torch.Generator().manual_seed(5),
        vocabulary=vocabulary,
    )

    loss, metrics = next_value_token_loss_and_metrics(
        model,
        batch,
        torch.tensor([-20, -3, 17, 40]),
        config=small_config(),
        isolate_successor=torch.tensor([True, False, True, False]),
        isolate_next_value_position=torch.tensor(
            [False, True, False, True]
        ),
        stage_four_teacher=teacher,
    )
    loss.backward()

    assert metrics["stage_four_loss"] > 0
    assert metrics["next_value_token_loss"] > 0
    assert metrics["stage_four_distillation_loss"] == pytest.approx(
        0.0,
        abs=1e-8,
    )
    assert 0 <= metrics["teacher_forced_next_value_token_accuracy"] <= 1
    assert model.token_query_projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_complete_trace_includes_retrieved_next_value() -> None:
    target_positions = torch.tensor(
        [
            [[1, 2, 3, 4], [2, 3, 4, 5]],
            [[0, 1, 2, 3], [1, 2, 3, 4]],
        ]
    )
    target_marked_tokens = torch.tensor([8, 7])
    target_next_positions = torch.tensor(
        [[1, 0, 6, 7], [0, 4, 5, 6]]
    )
    generated_next_tokens = torch.tensor([9, 6])
    target_next_tokens = torch.tensor([9, 5])

    metrics = generated_stage_five_metrics(
        target_positions,
        target_marked_tokens,
        target_next_positions,
        generated_next_tokens,
        target_positions,
        target_marked_tokens,
        target_next_positions,
        target_next_tokens,
        moduli=(3, 5, 7, 11),
    )

    assert metrics["stage_four_complete_trace_accuracy"] == 1.0
    assert metrics["next_value_token_accuracy"] == 0.5
    assert metrics["next_value_token_accuracy_given_correct_position"] == 0.5
    assert metrics["complete_trace_accuracy"] == 0.5
