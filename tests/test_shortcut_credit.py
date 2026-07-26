from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from list_sorting_transformer.tokens import SEP
from list_sorting_transformer.shortcut_credit import (
    AttentionRoutingRule,
    AttentionRoutingRuleConfig,
    BackwardRuleConfig,
    LearnedBackwardRule,
    ShortcutDecoderTransformer,
    ShortcutMetrics,
    ShortcutPointerVocabulary,
    apply_eggroll_direction,
    clone_center_parameters,
    evaluate_shortcut_batches,
    make_fitness_batches,
    make_forward_model_config,
    make_shortcut_batch,
    paper_eggroll_update,
    sample_eggroll_direction,
    shortcut_loss,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    PlateauState,
    ShortcutCreditExperimentConfig,
    candidate_fitness,
    candidate_summary,
    center_rule_summary,
    center_routing_summary,
    initialize_fresh_backward_rule,
    load_checkpoint,
    make_inner_batches,
    parse_candidate_devices,
    routing_population_summary,
    save_checkpoint,
    shard_candidate_specs,
    trajectory_summary,
    update_plateau_state,
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


def small_routing_rule(
    *,
    route_output_projection: bool = False,
    shared_routing_map: bool = False,
) -> AttentionRoutingRule:
    return AttentionRoutingRule(
        AttentionRoutingRuleConfig(
            vocab_size=small_vocabulary().size,
            d_model=32,
            n_heads=4,
            forward_layers=2,
            ffn_multiplier=2.0,
            route_output_projection=route_output_projection,
            shared_routing_map=shared_routing_map,
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


def test_shortcut_batch_can_randomize_leak_within_the_list() -> None:
    vocabulary = small_vocabulary()
    batch = make_shortcut_batch(
        64,
        8,
        leak_mode="correct",
        leak_placement="random_list",
        generator=torch.Generator().manual_seed(12),
        vocabulary=vocabulary,
    )

    leak_positions = (
        batch.input_ids.eq(vocabulary.leak_token)
        .nonzero(as_tuple=False)[:, 1]
    )
    rows = torch.arange(batch.batch_size)
    assert leak_positions.unique().numel() > 1
    assert batch.input_ids[rows, leak_positions + 1].equal(batch.targets)
    assert batch.input_ids[:, -1].eq(vocabulary.query_token).all()
    assert batch.input_ids[:, -2].eq(SEP).all()
    assert batch.leak_placement == "random_list"


def test_fitness_data_uses_requested_leak_placement() -> None:
    batches = make_fitness_batches(
        32,
        min_length=8,
        max_length=12,
        batch_size=4,
        leak_placement="random_list",
        generator=torch.Generator().manual_seed(3),
        vocabulary=small_vocabulary(),
    )

    assert all(batch.leak_placement == "random_list" for batch in batches)
    assert all(
        not batch.input_ids[:, -3].eq(small_vocabulary().leak_token).all()
        for batch in batches
    )


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


def test_evaluation_reports_prediction_diversity() -> None:
    model = small_model()
    batches = make_fitness_batches(
        32,
        min_length=8,
        max_length=12,
        batch_size=4,
        generator=torch.Generator().manual_seed(31),
        vocabulary=small_vocabulary(),
    )

    metrics = evaluate_shortcut_batches(model, batches)

    assert 1 <= metrics.unique_prediction_count <= model.config.vocab_size
    assert 0 <= metrics.unique_value_prediction_count <= 10
    assert 1 / 32 <= metrics.prediction_mode_fraction <= 1


def test_candidate_summary_reports_the_fittest_candidate_metrics() -> None:
    clean = [
        ShortcutMetrics(
            2.4,
            0.2,
            {"masked": 0.3, "incorrect": 0.1},
            {"masked": 2.3, "incorrect": 2.5},
            4,
            3,
            0.6,
        ),
        ShortcutMetrics(
            2.1,
            0.6,
            {"masked": 0.7, "incorrect": 0.5},
            {"masked": 2.0, "incorrect": 2.2},
            8,
            7,
            0.3,
        ),
        ShortcutMetrics(
            2.2,
            0.6,
            {"masked": 0.6, "incorrect": 0.6},
            {"masked": 2.2, "incorrect": 2.2},
            8,
            8,
            0.2,
        ),
        ShortcutMetrics(
            2.5,
            0.1,
            {"masked": 0.1, "incorrect": 0.1},
            {"masked": 2.5, "incorrect": 2.5},
            2,
            2,
            0.8,
        ),
    ]
    correct = [
        ShortcutMetrics(2.0, 0.4, {"correct": 0.4}, {"correct": 2.0}, 5, 4, 0.5),
        ShortcutMetrics(1.0, 0.9, {"correct": 0.9}, {"correct": 1.0}, 9, 8, 0.2),
        ShortcutMetrics(1.5, 0.7, {"correct": 0.7}, {"correct": 1.5}, 8, 8, 0.2),
        ShortcutMetrics(2.5, 0.1, {"correct": 0.1}, {"correct": 2.5}, 2, 2, 0.8),
    ]

    summary = candidate_summary(
        torch.tensor([0.1, 0.8, 0.2, 0.0]),
        clean,
        correct,
    )

    assert summary["best/candidate_index"] == 1
    assert summary["best/fitness"] == torch.tensor(0.8).item()
    assert summary["best/clean_accuracy"] == 0.6
    assert summary["best/masked_accuracy"] == 0.7
    assert summary["best/incorrect_accuracy"] == 0.5
    assert summary["best/correct_leak_accuracy"] == 0.9
    assert summary["robust/candidate_index"] == 2
    assert summary["robust/min_mode_accuracy"] == 0.6
    assert summary["robust/masked_accuracy"] == 0.6
    assert summary["robust/incorrect_accuracy"] == 0.6
    assert summary["robust/correct_leak_accuracy"] == 0.7


def test_routing_population_summary_links_selectivity_to_fitness() -> None:
    clean = [
        ShortcutMetrics(
            2.0,
            0.5,
            {"masked": 0.6, "incorrect": 0.4},
            {"masked": 1.9, "incorrect": 2.1},
            8,
            8,
            0.2,
        ),
        ShortcutMetrics(
            2.5,
            0.2,
            {"masked": 0.3, "incorrect": 0.1},
            {"masked": 2.4, "incorrect": 2.6},
            6,
            5,
            0.4,
        ),
    ]
    statistics = [
        [
            {
                "routing_leak_relative_gate": 0.8,
                "routing_hint_source_relative_gate": 0.7,
            }
        ],
        [
            {
                "routing_leak_relative_gate": 1.1,
                "routing_hint_source_relative_gate": 1.2,
            }
        ],
    ]

    summary = routing_population_summary(
        torch.tensor([2.0, 0.0]),
        clean,
        statistics,
    )

    assert abs(summary["backward/population_leak_relative_gate_min"] - 0.8) < 1e-6
    assert summary["backward/population_selective_fraction"] == 0.5
    assert summary["backward/selectivity_fitness_correlation"] > 0.99
    assert abs(
        summary["backward/best_fitness_leak_relative_gate"] - 0.8
    ) < 1e-6
    assert (
        summary["backward/population_hint_source_selective_fraction"]
        == 0.5
    )
    assert (
        summary[
            "backward/hint_source_selectivity_fitness_correlation"
        ]
        > 0.99
    )
    assert abs(
        summary[
            "backward/best_fitness_hint_source_relative_gate"
        ]
        - 0.7
    ) < 1e-6


def test_center_rule_summary_reports_unperturbed_training_metrics() -> None:
    clean = ShortcutMetrics(
        2.0,
        0.5,
        {"masked": 0.6, "incorrect": 0.4},
        {"masked": 1.9, "incorrect": 2.1},
        8,
        7,
        0.2,
    )
    correct = ShortcutMetrics(
        1.2,
        0.8,
        {"correct": 0.8},
        {"correct": 1.2},
        9,
        8,
        0.2,
    )

    summary = center_rule_summary(0.7, clean, correct)

    assert summary["center_rule/fitness"] == 0.7
    assert summary["center_rule/masked_accuracy"] == 0.6
    assert summary["center_rule/incorrect_accuracy"] == 0.4
    assert summary["center_rule/min_mode_accuracy"] == 0.4
    assert summary["center_rule/correct_leak_accuracy"] == 0.8


def test_trajectory_summary_supports_matched_baseline_prefixes() -> None:
    clean = ShortcutMetrics(
        2.0,
        0.5,
        {"masked": 0.6, "incorrect": 0.4},
        {"masked": 1.9, "incorrect": 2.1},
        8,
        7,
        0.2,
    )
    correct = ShortcutMetrics(
        1.2,
        0.8,
        {"correct": 0.8},
        {"correct": 1.2},
        9,
        8,
        0.2,
    )

    summary = trajectory_summary("ordinary_rule", 0.7, clean, correct)

    assert summary["ordinary_rule/fitness"] == 0.7
    assert summary["ordinary_rule/min_mode_accuracy"] == 0.4
    assert summary["ordinary_rule/correct_leak_accuracy"] == 0.8


def test_masked_inner_batches_match_correct_batch_content_and_positions() -> None:
    config = ShortcutCreditExperimentConfig(
        population_size=2,
        horizon=3,
        max_horizon=3,
        batch_size=8,
        min_length=4,
        max_length=7,
        leak_placement="random_list",
    )
    vocabulary = small_vocabulary()
    correct = make_inner_batches(
        config,
        horizon=3,
        vocabulary=vocabulary,
        generator=torch.Generator().manual_seed(91),
        device=torch.device("cpu"),
    )
    masked = make_inner_batches(
        config,
        horizon=3,
        vocabulary=vocabulary,
        generator=torch.Generator().manual_seed(91),
        device=torch.device("cpu"),
        leak_mode="masked",
    )

    for correct_batch, masked_batch in zip(correct, masked):
        assert correct_batch.length == masked_batch.length
        assert correct_batch.targets.equal(masked_batch.targets)
        leak_positions = (
            correct_batch.input_ids.eq(vocabulary.leak_token)
            .nonzero(as_tuple=False)[:, 1]
        )
        assert leak_positions.equal(
            masked_batch.input_ids.eq(vocabulary.leak_token)
            .nonzero(as_tuple=False)[:, 1]
        )
        reconstructed = masked_batch.input_ids.clone()
        rows = torch.arange(masked_batch.batch_size)
        reconstructed[rows, leak_positions + 1] = correct_batch.targets
        assert reconstructed.equal(correct_batch.input_ids)


def test_worst_mode_fitness_uses_the_weaker_clean_split() -> None:
    initial = ShortcutMetrics(
        3.0,
        0.1,
        {"masked": 0.1, "incorrect": 0.1},
        {"masked": 2.8, "incorrect": 3.2},
        4,
        3,
        0.6,
    )
    trained = ShortcutMetrics(
        2.2,
        0.2,
        {"masked": 0.2, "incorrect": 0.2},
        {"masked": 2.0, "incorrect": 2.5},
        5,
        4,
        0.5,
    )

    assert abs(
        candidate_fitness("mean_clean_ce", initial, trained) - 0.8
    ) < 1e-12
    assert abs(
        candidate_fitness("worst_mode_ce", initial, trained) - 0.7
    ) < 1e-12


def test_right_padded_evaluation_preserves_query_logits() -> None:
    torch.manual_seed(34)
    model = small_model().eval()
    vocabulary = small_vocabulary()
    short = make_shortcut_batch(
        3,
        8,
        leak_mode="masked",
        generator=torch.Generator().manual_seed(35),
        vocabulary=vocabulary,
    )
    long = make_shortcut_batch(
        2,
        12,
        leak_mode="incorrect",
        generator=torch.Generator().manual_seed(36),
        vocabulary=vocabulary,
    )
    expected_short = model(short.input_ids)[:, -1]
    expected_long = model(long.input_ids)[:, -1]

    rows = [*short.input_ids, *long.input_ids]
    padded = torch.nn.utils.rnn.pad_sequence(
        rows,
        batch_first=True,
        padding_value=0,
    )
    positions = torch.tensor([row.shape[0] - 1 for row in rows])
    packed_logits = model(padded)[torch.arange(len(rows)), positions]

    torch.testing.assert_close(
        packed_logits,
        torch.cat((expected_short, expected_long)),
        atol=1e-6,
        rtol=1e-5,
    )


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


def test_zero_routing_gate_preserves_forward_and_ordinary_gradients() -> None:
    torch.manual_seed(51)
    ordinary = small_model()
    modified = deepcopy(ordinary)
    rule = small_routing_rule(route_output_projection=True)
    batch = make_shortcut_batch(
        4,
        8,
        leak_mode="correct",
        generator=torch.Generator().manual_seed(52),
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
            rtol=1e-5,
            atol=1e-7,
        )


def test_attention_router_only_suppresses_existing_routes() -> None:
    torch.manual_seed(53)
    ordinary = small_model()
    modified = deepcopy(ordinary)
    rule = small_routing_rule()
    with torch.no_grad():
        rule.gates.fill_(0.2)
    batch = make_shortcut_batch(
        4,
        8,
        leak_mode="correct",
        generator=torch.Generator().manual_seed(54),
        vocabulary=small_vocabulary(),
    )

    attention_gates = rule.attention_gates(batch.input_ids)
    for gate in attention_gates:
        assert bool((gate > 0).all())
        assert bool((gate <= 1).all())
        assert bool((gate < 1).any())

    rule.capture_statistics = True
    rule.clear_statistics()
    rule.attention_gates(batch.input_ids)
    assert abs(rule.statistics[-1]["routing_strength"] - 1.6) < 1e-6
    assert 0 < rule.statistics[-1]["routing_leak_relative_gate"]

    ordinary_loss = shortcut_loss(ordinary, batch)
    ordinary_loss.backward()
    modified_loss = shortcut_loss(modified, batch, rule)
    modified_loss.backward()

    torch.testing.assert_close(modified_loss, ordinary_loss, rtol=0, atol=0)
    assert any(
        not torch.allclose(
            ordinary_parameter.grad,
            modified_parameter.grad,
        )
        for ordinary_parameter, modified_parameter in zip(
            ordinary.parameters(),
            modified.parameters(),
        )
    )


def test_attention_router_statistics_follow_random_leak_positions() -> None:
    torch.manual_seed(61)
    vocabulary = small_vocabulary()
    rule = small_routing_rule(shared_routing_map=True)
    with torch.no_grad():
        rule.gates.fill_(0.2)
    batch = make_shortcut_batch(
        16,
        8,
        leak_mode="correct",
        leak_placement="random_list",
        generator=torch.Generator().manual_seed(62),
        vocabulary=vocabulary,
    )
    leak_positions = (
        batch.input_ids.eq(vocabulary.leak_token)
        .nonzero(as_tuple=False)[:, 1]
    )
    assert leak_positions.unique().numel() > 1

    gates = rule.attention_gates(batch.input_ids)[0]
    rows = torch.arange(batch.batch_size)
    expected_leak_gate = gates[
        rows,
        :,
        -1,
        leak_positions + 1,
    ].mean()
    other_mask = torch.ones(
        batch.batch_size,
        gates.shape[-1],
        dtype=torch.bool,
    )
    other_mask.scatter_(1, (leak_positions + 1).unsqueeze(1), False)
    expected_other_gate = gates[:, :, -1].masked_select(
        other_mask[:, None, :].expand_as(gates[:, :, -1])
    ).mean()
    expected_hint_source = torch.cat(
        [
            gates[
                row,
                :,
                hint_position:,
                hint_position,
            ].flatten()
            for row, hint_position in enumerate(
                (leak_positions + 1).tolist()
            )
        ]
    ).mean()
    expected_other_sources = torch.cat(
        [
            gates[row, :, destination, : destination + 1][
                :,
                torch.arange(destination + 1) != hint_position,
            ].flatten()
            for row, hint_position in enumerate(
                (leak_positions + 1).tolist()
            )
            for destination in range(gates.shape[-1])
        ]
    ).mean()

    rule.capture_statistics = True
    rule.clear_statistics()
    rule.attention_gates(batch.input_ids)
    statistics = rule.statistics[-1]

    assert statistics["routing_leak_gate"] == pytest.approx(
        float(expected_leak_gate)
    )
    assert statistics["routing_query_other_gate"] == pytest.approx(
        float(expected_other_gate)
    )
    assert statistics["routing_leak_relative_gate"] == pytest.approx(
        float(expected_leak_gate / expected_other_gate)
    )
    assert statistics["routing_hint_source_gate"] == pytest.approx(
        float(expected_hint_source)
    )
    assert statistics[
        "routing_hint_source_other_gate"
    ] == pytest.approx(float(expected_other_sources))
    assert statistics[
        "routing_hint_source_relative_gate"
    ] == pytest.approx(
        float(expected_hint_source / expected_other_sources)
    )


def test_shared_attention_router_reuses_one_map_everywhere() -> None:
    torch.manual_seed(59)
    rule = small_routing_rule(shared_routing_map=True)
    with torch.no_grad():
        rule.gates.fill_(0.2)
    batch = make_shortcut_batch(
        4,
        8,
        leak_mode="correct",
        generator=torch.Generator().manual_seed(60),
        vocabulary=small_vocabulary(),
    )

    attention_gates = rule.attention_gates(batch.input_ids)

    assert len(attention_gates) == 2
    sequence_length = batch.input_ids.shape[1]
    assert attention_gates[0].shape == (
        4,
        4,
        sequence_length,
        sequence_length,
    )
    torch.testing.assert_close(attention_gates[0], attention_gates[1])
    for head_index in range(1, 4):
        torch.testing.assert_close(
            attention_gates[0][:, 0],
            attention_gates[0][:, head_index],
        )


def test_attention_router_projects_suppression_strength_nonnegative() -> None:
    rule = small_routing_rule(shared_routing_map=True)
    with torch.no_grad():
        rule.gates.fill_(-0.2)

    rule.project_parameters_()

    torch.testing.assert_close(rule.gates, torch.zeros_like(rule.gates))


def test_attention_router_can_route_output_projection_credit() -> None:
    torch.manual_seed(55)
    ordinary = small_model()
    qkv_only = deepcopy(ordinary)
    projection_routed = deepcopy(ordinary)
    qkv_rule = small_routing_rule()
    projection_rule = small_routing_rule(route_output_projection=True)
    projection_rule.load_state_dict(qkv_rule.state_dict())
    with torch.no_grad():
        qkv_rule.gates.fill_(0.2)
        projection_rule.gates.fill_(0.2)
    batch = make_shortcut_batch(
        4,
        8,
        leak_mode="correct",
        generator=torch.Generator().manual_seed(56),
        vocabulary=small_vocabulary(),
    )

    ordinary_loss = shortcut_loss(ordinary, batch)
    ordinary_loss.backward()
    qkv_loss = shortcut_loss(qkv_only, batch, qkv_rule)
    qkv_loss.backward()
    projection_loss = shortcut_loss(
        projection_routed,
        batch,
        projection_rule,
    )
    projection_loss.backward()

    torch.testing.assert_close(qkv_loss, ordinary_loss, rtol=0, atol=0)
    torch.testing.assert_close(projection_loss, ordinary_loss, rtol=0, atol=0)
    ordinary_gradient = ordinary.blocks[-1].attention.output.weight.grad
    qkv_gradient = qkv_only.blocks[-1].attention.output.weight.grad
    projection_gradient = (
        projection_routed.blocks[-1].attention.output.weight.grad
    )
    assert ordinary_gradient is not None
    assert qkv_gradient is not None
    assert projection_gradient is not None
    torch.testing.assert_close(
        qkv_gradient,
        ordinary_gradient,
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(projection_gradient, ordinary_gradient)


def test_center_routing_summary_reports_the_unperturbed_rule() -> None:
    rule = small_routing_rule()
    with torch.no_grad():
        rule.gates.fill_(0.2)
    batch = make_shortcut_batch(
        4,
        8,
        leak_mode="correct",
        generator=torch.Generator().manual_seed(57),
        vocabulary=small_vocabulary(),
    )

    summary = center_routing_summary(rule, batch.input_ids)

    assert 0 < summary["backward/center_routing_gate"] <= 1
    assert 0 < summary["backward/center_routing_leak_relative_gate"]
    assert rule.statistics == []


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


def test_plateau_state_tracks_a_comparable_objective() -> None:
    config = ShortcutCreditExperimentConfig(
        plateau_patience=2,
        plateau_min_delta=0.01,
        plateau_ema_decay=0.0,
    )
    state = PlateauState()

    assert not update_plateau_state(state, objective=-2.0, config=config)
    assert not update_plateau_state(state, objective=-2.0, config=config)
    assert update_plateau_state(state, objective=-2.0, config=config)
    assert not update_plateau_state(state, objective=-1.9, config=config)
    assert state.stale_generations == 0


def test_candidate_device_parser_preserves_explicit_shard_order() -> None:
    devices = parse_candidate_devices(
        "cuda:2, cuda:0, cuda:1",
        torch.device("cuda:0"),
    )

    assert devices == (
        torch.device("cuda:2"),
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    )


def test_fresh_backward_rule_initialization_uses_experiment_seed() -> None:
    config = ShortcutCreditExperimentConfig(
        backward_rule_type="attention_router",
        backward_d_model=32,
        forward_layers=2,
        heads=4,
        seed=61,
    )
    first = initialize_fresh_backward_rule(
        config,
        small_vocabulary(),
        device=torch.device("cpu"),
    )
    torch.manual_seed(999)
    _ = torch.randn(100)
    second = initialize_fresh_backward_rule(
        config,
        small_vocabulary(),
        device=torch.device("cpu"),
    )

    for first_parameter, second_parameter in zip(
        first.parameters(),
        second.parameters(),
    ):
        torch.testing.assert_close(
            first_parameter,
            second_parameter,
            rtol=0,
            atol=0,
        )


def test_attention_router_checkpoint_round_trip(tmp_path: Path) -> None:
    config = ShortcutCreditExperimentConfig(
        backward_rule_type="attention_router",
        backward_d_model=32,
        forward_layers=2,
        heads=4,
    )
    rule = initialize_fresh_backward_rule(
        config,
        small_vocabulary(),
        device=torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "router.pt"
    save_checkpoint(
        checkpoint_path,
        backward_rule=rule,
        config=config,
        generation=4,
        horizon=20,
        plateau_state=PlateauState(stale_generations=3),
    )

    loaded, generation, horizon, plateau = load_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded, AttentionRoutingRule)
    assert generation == 5
    assert horizon == 20
    assert plateau.stale_generations == 3
    for expected, actual in zip(rule.parameters(), loaded.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_legacy_gradient_checkpoint_defaults_to_transformer_rule(
    tmp_path: Path,
) -> None:
    config = ShortcutCreditExperimentConfig(
        backward_d_model=32,
        forward_layers=2,
        backward_layers=1,
        heads=4,
    )
    rule = small_rule()
    checkpoint_path = tmp_path / "current.pt"
    legacy_path = tmp_path / "legacy.pt"
    save_checkpoint(
        checkpoint_path,
        backward_rule=rule,
        config=config,
        generation=7,
        horizon=40,
        plateau_state=PlateauState(stale_generations=6),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.pop("backward_rule_type")
    torch.save(checkpoint, legacy_path)

    loaded, generation, horizon, plateau = load_checkpoint(
        legacy_path,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded, LearnedBackwardRule)
    assert generation == 8
    assert horizon == 40
    assert plateau.stale_generations == 6


def test_candidate_shards_keep_antithetic_pairs_together() -> None:
    specs = tuple(
        (2 * direction + sign_index, direction, sign)
        for direction in range(4)
        for sign_index, sign in enumerate((1, -1))
    )

    shards = shard_candidate_specs(specs, 2)

    assert shards == (
        ((0, 0, 1), (1, 0, -1), (4, 2, 1), (5, 2, -1)),
        ((2, 1, 1), (3, 1, -1), (6, 3, 1), (7, 3, -1)),
    )
