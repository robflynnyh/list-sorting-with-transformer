from __future__ import annotations

from copy import deepcopy

import torch

from list_sorting_transformer.shortcut_credit import (
    BackwardRuleConfig,
    LearnedBackwardRule,
    ShortcutDecoderTransformer,
    ShortcutPointerVocabulary,
    apply_eggroll_direction,
    clone_center_parameters,
    make_fitness_batches,
    make_forward_model_config,
    make_shortcut_batch,
    paper_eggroll_update,
    sample_eggroll_direction,
    shortcut_loss,
)


def small_vocabulary() -> ShortcutPointerVocabulary:
    return ShortcutPointerVocabulary("numbers", 10)


def small_model() -> ShortcutDecoderTransformer:
    vocabulary = small_vocabulary()
    return ShortcutDecoderTransformer(
        make_forward_model_config(
            vocabulary,
            d_model=32,
            n_layers=2,
            n_heads=4,
        )
    )


def small_rule() -> LearnedBackwardRule:
    return LearnedBackwardRule(
        BackwardRuleConfig(
            d_model=32,
            forward_d_model=32,
            n_layers=1,
            n_heads=4,
            forward_layers=2,
            ffn_multiplier=2.0,
        )
    )


def test_shortcut_batch_places_correct_masked_and_incorrect_hints() -> None:
    vocabulary = small_vocabulary()
    batches = {
        mode: make_shortcut_batch(
            8,
            6,
            leak_mode=mode,
            generator=torch.Generator().manual_seed(12),
            vocabulary=vocabulary,
        )
        for mode in ("correct", "masked", "incorrect")
    }

    for batch in batches.values():
        assert batch.input_ids.shape == (8, 17)
        assert batch.input_ids[:, -1].eq(vocabulary.query_token).all()
        assert batch.input_ids[:, -3].eq(vocabulary.leak_token).all()
    assert batches["correct"].input_ids[:, -2].equal(
        batches["correct"].targets
    )
    assert batches["masked"].input_ids[:, -2].eq(
        vocabulary.mask_token
    ).all()
    assert not batches["incorrect"].input_ids[:, -2].eq(
        batches["incorrect"].targets
    ).any()


def test_fitness_data_is_exactly_balanced_and_has_requested_size() -> None:
    batches = make_fitness_batches(
        32,
        min_length=8,
        max_length=12,
        batch_size=4,
        generator=torch.Generator().manual_seed(3),
        vocabulary=small_vocabulary(),
    )

    counts = {"masked": 0, "incorrect": 0}
    for batch in batches:
        counts[batch.leak_mode] += batch.batch_size
        assert 8 <= batch.length <= 12
    assert counts == {"masked": 16, "incorrect": 16}


def test_zero_gate_is_exactly_ordinary_backpropagation() -> None:
    torch.manual_seed(5)
    ordinary = small_model()
    modified = deepcopy(ordinary)
    rule = small_rule()
    batch = make_shortcut_batch(
        4,
        8,
        leak_mode="correct",
        generator=torch.Generator().manual_seed(9),
        vocabulary=small_vocabulary(),
    )

    ordinary_loss = shortcut_loss(ordinary, batch)
    ordinary_loss.backward()
    modified_loss = shortcut_loss(modified, batch, rule)
    modified_loss.backward()

    torch.testing.assert_close(modified_loss, ordinary_loss, rtol=0, atol=0)
    for ordinary_parameter, modified_parameter in zip(
        ordinary.parameters(),
        modified.parameters(),
    ):
        assert ordinary_parameter.grad is not None
        assert modified_parameter.grad is not None
        torch.testing.assert_close(
            modified_parameter.grad,
            ordinary_parameter.grad,
            rtol=0,
            atol=0,
        )


def test_modified_gradient_preserves_per_example_rms() -> None:
    torch.manual_seed(8)
    rule = small_rule()
    with torch.no_grad():
        rule.gates.fill_(0.2)
    gradient = torch.randn(3, 9, 32)
    activation = torch.randn(3, 9, 32)

    modified = rule.transform(gradient, activation, 0)
    original_rms = gradient.square().mean(dim=(-2, -1)).sqrt()
    modified_rms = modified.square().mean(dim=(-2, -1)).sqrt()

    torch.testing.assert_close(modified_rms, original_rms, atol=1e-6, rtol=1e-5)
    assert not torch.equal(modified, gradient)


def test_eggroll_matrix_perturbations_are_rank_one_and_antithetic() -> None:
    rule = small_rule()
    center = clone_center_parameters(rule)
    direction = sample_eggroll_direction(
        rule,
        generator=torch.Generator().manual_seed(21),
    )
    matrix_name = next(
        name
        for name, parameter in rule.named_parameters()
        if parameter.ndim == 2
    )
    assert torch.linalg.matrix_rank(direction.tensors[matrix_name]) <= 1

    positive = deepcopy(rule)
    negative = deepcopy(rule)
    apply_eggroll_direction(
        positive,
        center,
        direction,
        sigma=0.1,
        sign=1,
    )
    apply_eggroll_direction(
        negative,
        center,
        direction,
        sigma=0.1,
        sign=-1,
    )
    positive_parameter = dict(positive.named_parameters())[matrix_name]
    negative_parameter = dict(negative.named_parameters())[matrix_name]
    torch.testing.assert_close(
        (positive_parameter + negative_parameter) / 2,
        center[matrix_name],
    )


def test_paper_update_moves_toward_the_fitter_antithetic_candidate() -> None:
    rule = small_rule()
    direction = sample_eggroll_direction(
        rule,
        generator=torch.Generator().manual_seed(22),
    )
    center = clone_center_parameters(rule)

    standardized = paper_eggroll_update(
        rule,
        [direction],
        torch.tensor([2.0, 0.0]),
        sigma=0.1,
        learning_rate=0.5,
    )

    assert standardized[0] > 0
    assert standardized[1] < 0
    for name, parameter in rule.named_parameters():
        displacement = parameter - center[name]
        alignment = (displacement * direction.tensors[name]).sum()
        assert alignment > 0
