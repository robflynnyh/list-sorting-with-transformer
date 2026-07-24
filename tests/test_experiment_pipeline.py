from __future__ import annotations

import torch

from list_sorting_transformer.experiment import (
    TrainConfig,
    curriculum_max_length_at_step,
    initialize_from_checkpoint,
    initialize_from_pointer_position_checkpoint,
    learning_rate_at_step,
    sample_training_length,
)
from list_sorting_transformer.model import DecoderTransformer, ModelConfig
from list_sorting_transformer.pointer_position_probe import PointerPositionProbe
from list_sorting_transformer.tokens import PointerNextVocabulary


def small_transformer() -> DecoderTransformer:
    vocabulary = PointerNextVocabulary("numbers", 10)
    return DecoderTransformer(
        ModelConfig(
            vocab_size=vocabulary.size,
            representation="numbers",
            symbol_count=10,
            d_model=32,
            n_layers=2,
            n_heads=4,
            ffn_multiplier=2.0,
            position_pattern="none",
        )
    )


def test_main_trainer_strict_curriculum_caps_lengths() -> None:
    config = TrainConfig(
        task="pointer_value",
        train_min_length=2,
        train_max_length=20,
        curriculum=True,
        curriculum_review_probability=0.0,
    )
    generator = torch.Generator().manual_seed(19)

    lengths = [
        sample_training_length(config, current_max_length=6, generator=generator)
        for _ in range(64)
    ]

    assert min(lengths) >= 2
    assert max(lengths) <= 6


def test_linear_curriculum_spreads_caps_across_training() -> None:
    config = TrainConfig(
        task="pointer_value",
        train_min_length=2,
        train_max_length=20,
        steps=100_000,
        curriculum=True,
        curriculum_mode="linear",
        curriculum_start_length=2,
        curriculum_linear_end_step=80_000,
    )

    assert curriculum_max_length_at_step(config, 1) == 2
    assert curriculum_max_length_at_step(config, 4_000) == 2
    assert curriculum_max_length_at_step(config, 50_000) == 13
    assert curriculum_max_length_at_step(config, 80_000) == 20
    assert curriculum_max_length_at_step(config, 95_000) == 20


def test_main_trainer_constant_lr_schedule() -> None:
    config = TrainConfig(
        learning_rate=0.003,
        lr_schedule="constant",
        warmup_steps=10,
        steps=100,
    )

    assert learning_rate_at_step(config, 5) == 0.0015
    assert learning_rate_at_step(config, 10) == 0.003
    assert learning_rate_at_step(config, 100) == 0.003


def test_pointer_position_checkpoint_initializes_decoder_body(tmp_path) -> None:
    source = PointerPositionProbe(small_transformer().config)
    target = small_transformer()
    with torch.no_grad():
        source.encoder.token_embedding.weight.fill_(0.123)
        source.query_projection.weight.fill_(0.987)
    checkpoint_path = tmp_path / "pointer_position.pt"
    torch.save({"model_state": source.state_dict()}, checkpoint_path)

    metadata = initialize_from_pointer_position_checkpoint(target, checkpoint_path)

    assert metadata["transferred_tensors"] > 0
    assert torch.allclose(
        target.token_embedding.weight,
        torch.full_like(target.token_embedding.weight, 0.123),
    )
    assert "query_projection.weight" not in target.state_dict()


def test_model_checkpoint_initializes_matching_decoder_weights(tmp_path) -> None:
    source = small_transformer()
    target = small_transformer()
    with torch.no_grad():
        source.token_embedding.weight.fill_(0.456)
    checkpoint_path = tmp_path / "pointer_value.pt"
    torch.save({"model_state": source.state_dict()}, checkpoint_path)

    metadata = initialize_from_checkpoint(target, checkpoint_path)

    assert metadata["transferred_tensors"] > 0
    assert torch.allclose(
        target.token_embedding.weight,
        torch.full_like(target.token_embedding.weight, 0.456),
    )
