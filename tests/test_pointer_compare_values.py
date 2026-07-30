from __future__ import annotations

import pytest
import torch

from list_sorting_transformer.core.data import make_pointer_pair_batch
from list_sorting_transformer.core.model import ModelConfig
from list_sorting_transformer.length_generalisation.pointer_compare_values import (
    ModularPointerCompareModel,
    PointerCompareConfig,
    action_logit_distillation_scale_at_step,
    generated_stage_six_metrics,
    load_stage_five_checkpoint,
    pointer_compare_loss_and_metrics,
    stage_five_parameter_anchor_loss,
    target_action_classes,
)
from list_sorting_transformer.length_generalisation.pointer_next_value_from_position import (
    ModularNextValueFromPositionModel,
)
from list_sorting_transformer.core.tokens import (
    PointerCompareVocabulary,
    PointerNextVocabulary,
)


MODULI = (3, 5, 7, 11)


def model_config(vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        representation="numbers",
        symbol_count=10,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ffn_multiplier=2.0,
        dropout=0.0,
        position_pattern="none",
    )


def stage_six_model() -> ModularPointerCompareModel:
    vocabulary = PointerCompareVocabulary("numbers", 10)
    return ModularPointerCompareModel(
        model_config(vocabulary.size),
        MODULI,
        action_token_offset=vocabulary.action_token_offset,
    )


def stage_six_config() -> PointerCompareConfig:
    return PointerCompareConfig(
        eval_max_length=40,
        position_moduli=MODULI,
        position_offset_min=-50,
        position_offset_max=50,
        successor_attention_isolation_probability=0.5,
        next_value_position_attention_isolation_probability=0.5,
        next_value_position_consistency_weight=1.0,
        stage_three_distillation_weight=0.0,
        next_value_token_loss_weight=1.0,
        stage_four_distillation_weight=0.0,
        action_attention_isolation_probability=0.5,
        action_consistency_weight=1.0,
        action_logit_distillation_weight=1.0,
        stage_five_distillation_weight=1.0,
    )


def test_compare_vocabulary_appends_keep_and_swap() -> None:
    vocabulary = PointerCompareVocabulary("numbers", 10)

    assert vocabulary.action_tokens == (
        vocabulary.action_token("KEEP"),
        vocabulary.action_token("SWAP"),
    )
    assert vocabulary.size == PointerNextVocabulary("numbers", 10).size + 2
    assert vocabulary.render_tokens(vocabulary.action_tokens) == (
        "<KEEP><SWAP>"
    )


def test_action_targets_keep_ties_and_swap_descending_pairs() -> None:
    vocabulary = PointerCompareVocabulary("numbers", 10)
    batch = make_pointer_pair_batch(
        3,
        3,
        generator=torch.Generator().manual_seed(3),
        vocabulary=vocabulary,
    )
    batch.values[:] = torch.tensor(
        [
            [1, 2, 9],
            [7, 7, 0],
            [8, 3, 4],
        ]
    )
    batch.pointers[:] = 0

    torch.testing.assert_close(
        target_action_classes(batch),
        torch.tensor([0, 0, 1]),
    )


def test_isolated_action_can_only_use_the_two_retrieved_values() -> None:
    mask = ModularPointerCompareModel.action_attention_mask(
        batch_size=2,
        prompt_length=8,
        stream_length=13,
        isolate_action=torch.tensor([True, False]),
        device=torch.device("cpu"),
    )

    assert mask is not None
    assert mask[0, -1].nonzero().flatten().tolist() == [10, 12]
    assert bool(mask[1].all())


def test_stage_five_checkpoint_expands_only_the_action_embedding_rows(
    tmp_path,
) -> None:
    torch.manual_seed(4)
    old_vocabulary = PointerNextVocabulary("numbers", 10)
    source = ModularNextValueFromPositionModel(
        model_config(old_vocabulary.size),
        MODULI,
    )
    target = stage_six_model()
    initial_action_rows = (
        target.encoder.token_embedding.weight[
            old_vocabulary.size :
        ]
        .detach()
        .clone()
    )
    checkpoint_path = tmp_path / "stage_five.pt"
    torch.save(
        {
            "probe": "pointer_next_value_from_position",
            "model_state": source.state_dict(),
            "step": 10_000,
        },
        checkpoint_path,
    )

    metadata = load_stage_five_checkpoint(target, checkpoint_path)

    assert metadata["stage_five_step"] == 10_000
    torch.testing.assert_close(
        target.encoder.token_embedding.weight[: old_vocabulary.size],
        source.encoder.token_embedding.weight,
    )
    torch.testing.assert_close(
        target.encoder.token_embedding.weight[old_vocabulary.size :],
        initial_action_rows,
    )


def test_stage_six_loss_trains_actions_and_keeps_teacher_frozen() -> None:
    torch.manual_seed(5)
    vocabulary = PointerCompareVocabulary("numbers", 10)
    model = stage_six_model()
    teacher = ModularNextValueFromPositionModel(
        model_config(vocabulary.action_token_offset),
        MODULI,
    )
    source_state = teacher.state_dict()
    target_state = model.state_dict()
    for name, source in source_state.items():
        if name == "encoder.token_embedding.weight":
            target_state[name][: source.shape[0]].copy_(source)
        else:
            target_state[name].copy_(source)
    model.load_state_dict(target_state)
    teacher.requires_grad_(False)
    teacher.eval()
    batch = make_pointer_pair_batch(
        4,
        5,
        generator=torch.Generator().manual_seed(6),
        vocabulary=vocabulary,
    )

    loss, metrics = pointer_compare_loss_and_metrics(
        model,
        batch,
        torch.tensor([-20, -3, 17, 40]),
        config=stage_six_config(),
        isolate_successor=torch.tensor([True, False, True, False]),
        isolate_next_value_position=torch.tensor(
            [False, True, False, True]
        ),
        isolate_action=torch.tensor([True, False, True, False]),
        stage_five_teacher=teacher,
    )
    loss.backward()

    assert metrics["action_loss"] > 0
    assert metrics["action_logit_distillation_loss"] > 0
    assert metrics["action_attention_isolation_fraction"] == pytest.approx(
        0.5
    )
    assert 0 <= metrics["teacher_forced_action_accuracy"] <= 1
    assert 0 <= metrics["masked_action_accuracy"] <= 1
    assert (
        model.encoder.token_embedding.weight.grad[
            vocabulary.action_token_offset :
        ]
        .abs()
        .sum()
        .item()
        > 0
    )
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_parameter_anchor_excludes_only_new_action_rows() -> None:
    torch.manual_seed(7)
    vocabulary = PointerCompareVocabulary("numbers", 10)
    model = stage_six_model()
    teacher = ModularNextValueFromPositionModel(
        model_config(vocabulary.action_token_offset),
        MODULI,
    )
    target_state = model.state_dict()
    for name, source in teacher.state_dict().items():
        if name == "encoder.token_embedding.weight":
            target_state[name][: source.shape[0]].copy_(source)
        else:
            target_state[name].copy_(source)
    model.load_state_dict(target_state)

    assert stage_five_parameter_anchor_loss(model, teacher).item() == 0
    with torch.no_grad():
        model.encoder.token_embedding.weight[
            vocabulary.action_token_offset :
        ].add_(1.0)
    assert stage_five_parameter_anchor_loss(model, teacher).item() == 0
    with torch.no_grad():
        model.token_query_projection.weight[0, 0].add_(0.25)
    assert stage_five_parameter_anchor_loss(model, teacher).item() == (
        pytest.approx(0.25**2)
    )


def test_action_logit_distillation_has_a_predetermined_ramp() -> None:
    config = PointerCompareConfig(
        eval_max_length=40,
        position_moduli=MODULI,
        position_offset_min=-50,
        position_offset_max=50,
        action_logit_distillation_start_step=2_000,
        action_logit_distillation_ramp_steps=1_000,
    )

    assert action_logit_distillation_scale_at_step(config, 2_000) == 0
    assert action_logit_distillation_scale_at_step(config, 2_500) == 0.5
    assert action_logit_distillation_scale_at_step(config, 3_000) == 1
    assert action_logit_distillation_scale_at_step(config, 8_000) == 1


def test_complete_trace_requires_the_comparison_action() -> None:
    positions = torch.tensor(
        [
            [[1, 2, 3, 4], [2, 3, 4, 5]],
            [[0, 1, 2, 3], [1, 2, 3, 4]],
        ]
    )
    marked = torch.tensor([8, 7])
    next_positions = torch.tensor([[1, 0, 6, 7], [0, 4, 5, 6]])
    next_tokens = torch.tensor([9, 5])
    target_actions = torch.tensor([16, 17])

    metrics = generated_stage_six_metrics(
        positions,
        marked,
        next_positions,
        next_tokens,
        torch.tensor([16, 16]),
        positions,
        marked,
        next_positions,
        next_tokens,
        target_actions,
        moduli=MODULI,
    )

    assert metrics["stage_five_complete_trace_accuracy"] == 1.0
    assert metrics["action_accuracy"] == 0.5
    assert metrics["complete_trace_accuracy"] == 0.5
