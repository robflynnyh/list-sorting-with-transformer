from __future__ import annotations

import torch

from list_sorting_transformer.data import make_pointer_value_batch
from list_sorting_transformer.model import ModelConfig
from list_sorting_transformer.pointer_next_value_position import (
    ModularNextValuePositionModel,
    NextValuePositionConfig,
    generated_stage_four_metrics,
    load_stage_three_checkpoint,
    next_value_position_loss_and_metrics,
)
from list_sorting_transformer.pointer_value_from_position import (
    ModularPositionValueModel,
)
from list_sorting_transformer.tokens import PointerNextVocabulary


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
    )
    loss.backward()

    assert metrics["stage_three_loss"] > 0
    assert metrics["next_value_position_loss"] > 0
    assert 0 <= metrics["teacher_forced_next_value_position_accuracy"] <= 1
    assert model.query_projection.weight.grad is not None
    assert model.token_query_projection.weight.grad is not None


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
