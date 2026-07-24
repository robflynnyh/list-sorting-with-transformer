from __future__ import annotations

import torch

from list_sorting_transformer.data import make_pointer_value_batch
from list_sorting_transformer.model import ModelConfig
from list_sorting_transformer.pointer_position_sequence import (
    ModularPositionSequenceModel,
)
from list_sorting_transformer.pointer_value_from_position import (
    ModularPositionValueModel,
    PositionValueConfig,
    generated_trace_metrics,
    load_stage_two_checkpoint,
    position_value_loss_and_metrics,
    target_token_ids,
)
from list_sorting_transformer.tokens import VALUE_OFFSET, PointerNextVocabulary


def small_model() -> ModularPositionValueModel:
    vocabulary = PointerNextVocabulary("numbers", 10)
    return ModularPositionValueModel(
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


def small_config() -> PositionValueConfig:
    return PositionValueConfig(
        eval_max_length=40,
        position_moduli=(3, 5, 7, 11),
        position_offset_min=-50,
        position_offset_max=50,
        successor_attention_isolation_probability=0.5,
    )


def test_target_token_is_the_marked_value_immediately_after_ptr() -> None:
    vocabulary = PointerNextVocabulary("numbers", 10)
    batch = make_pointer_value_batch(
        16,
        5,
        generator=torch.Generator().manual_seed(12),
        vocabulary=vocabulary,
    )

    targets = target_token_ids(batch)
    expected_values = batch.values[
        torch.arange(batch.values.shape[0]),
        batch.pointers,
    ]

    torch.testing.assert_close(targets, expected_values + VALUE_OFFSET)


def test_token_head_scores_the_existing_vocabulary_embeddings() -> None:
    torch.manual_seed(2)
    model = small_model()
    hidden = torch.randn(3, model.encoder.config.d_model)

    logits = model.token_logits(hidden)

    assert logits.shape == (3, model.encoder.config.vocab_size)
    expected = model.token_query_projection(hidden) @ (
        model.encoder.token_embedding.weight.T
    )
    torch.testing.assert_close(logits, expected)


def test_two_position_latents_can_condition_token_prediction() -> None:
    torch.manual_seed(3)
    model = small_model()
    vocabulary = PointerNextVocabulary("numbers", 10)
    batch = make_pointer_value_batch(
        2,
        4,
        generator=torch.Generator().manual_seed(4),
        vocabulary=vocabulary,
    )
    offsets = torch.tensor([-12, 30])
    position_targets = model.target_sequence(batch.pointers, offsets)

    logits = model.teacher_forced_token_logits(
        batch.prompt_ids,
        position_targets,
        offsets=offsets,
    )

    assert logits.shape == (2, vocabulary.size)


def test_stage_two_transfer_only_initializes_the_new_token_head(
    tmp_path,
) -> None:
    torch.manual_seed(5)
    target_model = small_model()
    source_model = ModularPositionSequenceModel(
        target_model.encoder.config,
        target_model.moduli,
    )
    checkpoint_path = tmp_path / "stage_two.pt"
    torch.save(
        {
            "model_state": source_model.state_dict(),
            "step": 20_000,
        },
        checkpoint_path,
    )

    metadata = load_stage_two_checkpoint(target_model, checkpoint_path)

    assert metadata["stage_two_step"] == 20_000
    assert metadata["transferred_tensors"] == len(source_model.state_dict())
    torch.testing.assert_close(
        target_model.encoder.final_norm.weight,
        source_model.encoder.final_norm.weight,
    )


def test_position_value_loss_trains_token_head_with_masked_successor() -> None:
    torch.manual_seed(6)
    model = small_model()
    config = small_config()
    vocabulary = PointerNextVocabulary("numbers", 10)
    batch = make_pointer_value_batch(
        4,
        5,
        generator=torch.Generator().manual_seed(7),
        vocabulary=vocabulary,
    )

    loss, metrics = position_value_loss_and_metrics(
        model,
        batch,
        torch.tensor([-20, -3, 17, 40]),
        config=config,
        isolate_successor=torch.tensor([True, False, True, False]),
    )
    loss.backward()

    assert metrics["successor_attention_isolation_fraction"] == 0.5
    assert metrics["token_loss"] > 0
    assert 0 <= metrics["teacher_forced_token_accuracy"] <= 1
    assert model.token_query_projection.weight.grad is not None


def test_complete_trace_requires_positions_and_token_to_be_correct() -> None:
    targets = torch.tensor(
        [
            [[1, 2, 3, 4], [2, 3, 4, 5]],
            [[0, 1, 2, 3], [1, 2, 3, 4]],
        ]
    )
    generated_positions = targets.clone()
    generated_positions[1, 1, 0] = 2

    metrics = generated_trace_metrics(
        generated_positions,
        torch.tensor([8, 7]),
        targets,
        torch.tensor([8, 7]),
        moduli=(3, 5, 7, 11),
    )

    assert metrics["token_accuracy"] == 1.0
    assert metrics["both_positions_accuracy"] == 0.5
    assert metrics["complete_trace_accuracy"] == 0.5
    assert metrics["token_accuracy_given_correct_positions"] == 1.0
