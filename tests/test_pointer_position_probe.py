from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from list_sorting_transformer.pointer_position_probe import (
    PointerPositionConfig,
    PointerPositionProbe,
    gradient_noise_std,
    learning_rate_at_step,
    modular_position_metrics,
    pointer_position_ce_metrics,
    pointer_position_metrics,
    sample_position_offsets,
    sample_training_length,
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


def small_modular_probe() -> PointerPositionProbe:
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
        ),
        position_moduli=(3, 5, 7, 11),
    )


def small_split_modular_probe() -> PointerPositionProbe:
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
        ),
        position_moduli=(3, 5, 7, 11),
        split_input=True,
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


def test_target_positions_include_per_example_offsets() -> None:
    model = small_probe()
    pointers = torch.tensor([0, 2, 3])
    offsets = torch.tensor([-10, 100, 1_000])

    positions = model.target_positions(pointers, offsets)

    assert positions.tolist() == [-9, 105, 1007]


def test_sample_position_offsets_uses_configured_range() -> None:
    config = PointerPositionConfig(
        position_offset_min=-3,
        position_offset_max=4,
    )
    generator = torch.Generator().manual_seed(3)

    offsets = sample_position_offsets(
        128,
        config=config,
        generator=generator,
        device=torch.device("cpu"),
    )

    assert int(offsets.min()) >= -3
    assert int(offsets.max()) <= 4


def test_curriculum_training_length_caps_main_samples() -> None:
    config = PointerPositionConfig(
        train_min_length=2,
        train_max_length=20,
        curriculum=True,
        curriculum_review_probability=0.0,
    )
    generator = torch.Generator().manual_seed(4)

    lengths = [
        sample_training_length(config, current_max_length=5, generator=generator)
        for _ in range(64)
    ]

    assert min(lengths) >= 2
    assert max(lengths) <= 5


def test_gradient_noise_std_decays_from_configured_scale() -> None:
    config = PointerPositionConfig(
        gradient_noise_scale=0.01,
        gradient_noise_decay=0.5,
    )

    assert gradient_noise_std(config, 1) == pytest.approx(0.01)
    assert gradient_noise_std(config, 100) == pytest.approx(0.001)


def test_constant_lr_schedule_keeps_base_rate_after_warmup() -> None:
    config = PointerPositionConfig(
        learning_rate=0.003,
        lr_schedule="constant",
        warmup_steps=10,
        steps=100,
    )

    assert learning_rate_at_step(config, 5) == pytest.approx(0.0015)
    assert learning_rate_at_step(config, 10) == pytest.approx(0.003)
    assert learning_rate_at_step(config, 100) == pytest.approx(0.003)


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
    offsets = torch.tensor([-12, 30])

    emitted_vectors = model(prompt, offsets=offsets)
    targets = model.target_embeddings(torch.tensor([2, 0]), offsets)
    loss = F.mse_loss(emitted_vectors, targets)
    loss.backward()

    assert emitted_vectors.shape == (2, 32)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_pointer_position_probe_scores_candidate_pointer_slots() -> None:
    torch.manual_seed(10)
    vocabulary = PointerNextVocabulary("numbers", 10)
    model = small_probe()
    prompt = torch.tensor(
        [
            vocabulary.encode_prompt_with_pointer([3, 1, 4, 1, 5], 2),
            vocabulary.encode_prompt_with_pointer([2, 7, 1, 8, 2], 0),
        ]
    )
    offsets = torch.tensor([-12, 30])

    logits = model.pointer_logits(prompt, length=5, offsets=offsets)
    loss = F.cross_entropy(logits, torch.tensor([2, 0]))
    loss.backward()

    assert logits.shape == (2, 4)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_modular_pointer_probe_predicts_each_residue_category() -> None:
    torch.manual_seed(11)
    vocabulary = PointerNextVocabulary("numbers", 10)
    model = small_modular_probe()
    prompt = torch.tensor(
        [
            vocabulary.encode_prompt_with_pointer([3, 1, 4, 1, 5], 2),
            vocabulary.encode_prompt_with_pointer([2, 7, 1, 8, 2], 0),
        ]
    )
    offsets = torch.tensor([-12, 30])

    logits = model.residue_logits(prompt, offsets=offsets)
    targets = model.target_residues(torch.tensor([2, 0]), offsets)
    loss = torch.stack(
        [
            F.cross_entropy(component_logits, target)
            for component_logits, target in zip(logits, targets)
        ]
    ).mean()
    loss.backward()

    assert [component.shape for component in logits] == [
        (2, 3),
        (2, 5),
        (2, 7),
        (2, 11),
    ]
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_split_modular_probe_concatenates_content_and_position_channels() -> None:
    torch.manual_seed(12)
    vocabulary = PointerNextVocabulary("numbers", 10)
    model = small_split_modular_probe()
    prompt = torch.tensor(
        [vocabulary.encode_prompt_with_pointer([3, 1, 4, 1, 5], 2)]
    )
    offsets = torch.tensor([100])

    hidden = model.hidden_states(prompt, offsets=offsets)
    logits = model.residue_logits(prompt, offsets=offsets)

    assert model.encoder.content_dim == 16
    assert model.encoder.position_dim == 16
    assert model.position_embedding.dim == 16
    assert hidden.shape == (1, prompt.shape[1], 32)
    assert [component.shape for component in logits] == [
        (1, 3),
        (1, 5),
        (1, 7),
        (1, 11),
    ]


def test_modular_metrics_require_every_residue_for_exact_position() -> None:
    model = small_modular_probe()
    pointers = torch.tensor([0, 2])
    offsets = torch.tensor([20, -10])
    targets = model.target_residues(pointers, offsets)
    logits = []
    for modulus, target in zip((3, 5, 7, 11), targets):
        component = torch.full((2, modulus), -10.0)
        component.scatter_(1, target[:, None], 10.0)
        logits.append(component)
    logits[0][1].fill_(-10.0)
    logits[0][1, (int(targets[0][1]) + 1) % 3] = 10.0

    metrics = modular_position_metrics(
        tuple(logits),
        pointers,
        model=model,
        length=5,
        offsets=offsets,
        train_max_length=3,
    )

    assert metrics["exact_position_accuracy"] == 0.5
    assert metrics["mean_residue_accuracy"] == 0.875
    assert metrics["candidate_pointer_accuracy"] == 1.0


def test_pointer_position_metrics_report_argmax_and_unseen_slices() -> None:
    model = small_probe()
    pointers = torch.tensor([0, 3, 2])
    offsets = torch.tensor([100, -50, 700])
    emitted_vectors = model.target_embeddings(torch.tensor([0, 2, 1]), offsets)

    metrics = pointer_position_metrics(
        emitted_vectors,
        pointers,
        model=model,
        length=5,
        offsets=offsets,
        train_max_length=3,
    )

    assert metrics["argmax_accuracy"] == pytest.approx(1 / 3)
    assert metrics["argmax_token_mae"] == pytest.approx(4 / 3)
    assert metrics["seen_argmax_accuracy"] == 1.0
    assert metrics["unseen_argmax_accuracy"] == 0.0
    assert metrics["unseen_pointer_fraction"] == pytest.approx(2 / 3)


def test_pointer_position_ce_metrics_report_candidate_argmax() -> None:
    model = small_probe()
    pointers = torch.tensor([0, 3, 2])
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
        ]
    )

    metrics = pointer_position_ce_metrics(
        logits,
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
