from __future__ import annotations

import pytest
import torch

from list_sorting_transformer.core.data import make_pointer_value_batch
from list_sorting_transformer.core.model import ModelConfig
from list_sorting_transformer.length_generalisation.pointer_next_value_position import (
    ModularNextValuePositionModel,
    NextValuePositionConfig,
    generated_stage_four_metrics,
    load_stage_three_checkpoint,
    next_value_position_loss_and_metrics,
    relative_logit_distillation_loss,
)
from list_sorting_transformer.length_generalisation.pointer_value_from_position import (
    ModularPositionValueModel,
)
from list_sorting_transformer.core.tokens import PointerNextVocabulary


def small_model() -> ModularNextValuePositionModel:
    vocabulary = PointerNextVocabulary("numbers", 10)
    return ModularNextValuePositionModel(
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


def small_config() -> NextValuePositionConfig:
    return NextValuePositionConfig(
        eval_max_length=40,
        position_moduli=(3, 5, 7, 11),
        position_offset_min=-50,
        position_offset_max=50,
        successor_attention_isolation_probability=0.5,
        next_value_position_attention_isolation_probability=0.5,
        next_value_position_consistency_weight=1.0,
        stage_three_distillation_weight=1.0,
    )


def test_target_is_position_of_list_value_after_marked_value() -> None:
    model = small_model()

    targets = model.target_next_value_position(
        pointers=torch.tensor([0, 2]),
        offsets=torch.tensor([20, -10]),
    )

    assert targets.tolist() == [
        [0, 4, 3, 2],
        [1, 3, 5, 9],
    ]


def test_mixed_history_appends_two_addresses_then_marked_token() -> None:
    torch.manual_seed(2)
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
    positions = model.target_sequence(pointers, offsets)
    marked_tokens = torch.tensor(
        [vocabulary.value_token(1), vocabulary.value_token(2)]
    )

    hidden = model.mixed_hidden_states(
        prompt,
        positions,
        marked_tokens[:, None],
        offsets=offsets,
    )

    assert hidden.shape == (2, prompt.shape[1] + 3, 32)


def test_position_query_has_the_modular_embedding_dimension() -> None:
    torch.manual_seed(2)
    model = small_model()
    vocabulary = PointerNextVocabulary("numbers", 10)
    prompt = torch.tensor(
        [vocabulary.encode_prompt_with_pointer([3, 1, 4], 1)]
    )
    pointers = torch.tensor([1])
    offsets = torch.tensor([-12])
    positions = model.target_sequence(pointers, offsets)
    marked_tokens = torch.tensor([vocabulary.value_token(1)])

    query = model.teacher_forced_next_value_position_query(
        prompt,
        positions,
        marked_tokens,
        offsets=offsets,
    )

    assert query.shape == (1, model.position_embedding.dim)


def test_stage_three_checkpoint_transfers_without_new_parameters(tmp_path) -> None:
    torch.manual_seed(3)
    target_model = small_model()
    source_model = ModularPositionValueModel(
        target_model.encoder.config,
        target_model.moduli,
    )
    checkpoint_path = tmp_path / "stage_three.pt"
    torch.save(
        {
            "probe": "pointer_value_from_position",
            "model_state": source_model.state_dict(),
            "step": 20_000,
        },
        checkpoint_path,
    )

    metadata = load_stage_three_checkpoint(target_model, checkpoint_path)

    assert metadata["stage_three_step"] == 20_000
    assert metadata["transferred_tensors"] == len(source_model.state_dict())
    torch.testing.assert_close(
        target_model.token_query_projection.weight,
        source_model.token_query_projection.weight,
    )


def test_stage_four_loss_trains_address_head_after_marked_token() -> None:
    torch.manual_seed(4)
    model = small_model()
    torch.manual_seed(44)
    teacher = ModularPositionValueModel(
        model.encoder.config,
        model.moduli,
    )
    teacher.requires_grad_(False)
    teacher.eval()
    config = small_config()
    vocabulary = PointerNextVocabulary("numbers", 10)
    batch = make_pointer_value_batch(
        4,
        5,
        generator=torch.Generator().manual_seed(5),
        vocabulary=vocabulary,
    )

    loss, metrics = next_value_position_loss_and_metrics(
        model,
        batch,
        torch.tensor([-20, -3, 17, 40]),
        config=config,
        isolate_successor=torch.tensor([True, False, True, False]),
        isolate_next_value_position=torch.tensor(
            [False, True, False, True]
        ),
        stage_three_teacher=teacher,
    )
    loss.backward()

    assert metrics["stage_three_loss"] > 0
    assert metrics["next_value_position_loss"] > 0
    assert (
        metrics["next_value_position_attention_isolation_fraction"] == 0.5
    )
    assert metrics["next_value_position_consistency_loss"] > 0
    assert metrics["unrestricted_next_value_position_loss"] > 0
    assert metrics["teacher_branch_next_value_position_loss"] > 0
    assert metrics["stage_three_distillation_loss"] > 0
    assert 0 <= metrics["teacher_branch_next_value_position_accuracy"] <= 1
    assert 0 <= metrics["teacher_forced_next_value_position_accuracy"] <= 1
    assert model.query_projection.weight.grad is not None
    assert model.token_query_projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_relative_logit_distillation_ignores_additive_offsets() -> None:
    teacher = torch.tensor([[1.0, 2.0, 4.0], [-3.0, 0.0, 2.0]])

    identical = relative_logit_distillation_loss(teacher + 10.0, teacher)
    changed = relative_logit_distillation_loss(teacher * 1.5, teacher)

    assert identical.item() == pytest.approx(0.0, abs=1e-12)
    assert changed.item() > 0


def test_consistency_weight_must_be_nonnegative() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        NextValuePositionConfig(
            next_value_position_consistency_weight=-1.0
        )
    with pytest.raises(ValueError, match="must be nonnegative"):
        NextValuePositionConfig(stage_three_distillation_weight=-1.0)


def test_stage_four_isolation_exposes_only_preceding_address() -> None:
    mask = ModularNextValuePositionModel.next_value_position_attention_mask(
        batch_size=2,
        stream_length=9,
        isolate_next_value_position=torch.tensor([True, False]),
        device=torch.device("cpu"),
    )

    assert mask is not None
    assert mask[0, :-1].all()
    assert mask[0, -1].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    assert mask[1].all()


def test_complete_trace_includes_the_next_value_position() -> None:
    target_positions = torch.tensor(
        [
            [[1, 2, 3, 4], [2, 3, 4, 5]],
            [[0, 1, 2, 3], [1, 2, 3, 4]],
        ]
    )
    target_next = torch.tensor([[1, 0, 6, 7], [0, 4, 5, 6]])
    generated_next = target_next.clone()
    generated_next[1, 0] = 1

    metrics = generated_stage_four_metrics(
        target_positions,
        torch.tensor([8, 7]),
        generated_next,
        target_positions,
        torch.tensor([8, 7]),
        target_next,
        moduli=(3, 5, 7, 11),
    )

    assert metrics["stage_three_complete_trace_accuracy"] == 1.0
    assert metrics["next_value_position_accuracy"] == 0.5
    assert metrics["complete_trace_accuracy"] == 0.5
