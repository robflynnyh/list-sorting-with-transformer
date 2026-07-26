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
    apply_resume_horizon,
    elite_acceptance_seed,
    elite_centroid_update,
    elite_proposal_mean_improvement,
    function_delta_alignment_summary,
    heldout_candidate_summary,
    initialize_fresh_backward_rule,
    load_checkpoint,
    make_inner_batches,
    parse_candidate_devices,
    restore_center_parameters,
    resolve_resume_horizon,
    routing_population_summary,
    save_checkpoint,
    shard_candidate_specs,
    train_forward_trajectory,
    trajectory_summary,
    update_elite_search_state,
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
                "routing_input_conditioned_rms": 0.4,
                "routing_position_profile_std": 0.2,
            }
        ],
        [
            {
                "routing_leak_relative_gate": 1.1,
                "routing_hint_source_relative_gate": 1.2,
                "routing_input_conditioned_rms": 0.1,
                "routing_position_profile_std": 0.3,
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
    assert (
        summary["backward/input_conditioning_fitness_correlation"]
        > 0.99
    )
    assert (
        summary["backward/robust_candidate_input_conditioned_rms"]
        == pytest.approx(0.4)
    )
    assert (
        summary["backward/position_profile_fitness_correlation"]
        < -0.99
    )


def test_function_delta_alignment_detects_agreeing_top_candidates() -> None:
    deltas = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    summary = function_delta_alignment_summary(
        deltas,
        torch.tensor([4.0, 3.0, 2.0, 1.0]),
        robust_index=1,
        top_k=2,
    )

    assert summary["backward/top_function_pairwise_cosine_mean"] == (
        pytest.approx(1.0)
    )
    assert summary["backward/top_function_centroid_rms"] > 0
    assert summary["backward/fitness_weighted_cosine_with_best"] > 0


def test_elite_centroid_update_averages_selected_candidates() -> None:
    rule = small_routing_rule()
    center = clone_center_parameters(rule)
    directions = (
        sample_eggroll_direction(
            rule,
            generator=torch.Generator().manual_seed(91),
        ),
        sample_eggroll_direction(
            rule,
            generator=torch.Generator().manual_seed(92),
        ),
    )
    fitnesses = torch.tensor([4.0, 0.0, 3.0, 1.0])

    elite_indices = elite_centroid_update(
        rule,
        directions,
        fitnesses,
        sigma=0.2,
        elite_count=2,
        interpolation=0.5,
    )

    assert elite_indices.tolist() == [0, 2]
    for name, parameter in rule.named_parameters():
        expected_delta = 0.5 * 0.2 * (
            directions[0].tensors[name] + directions[1].tensors[name]
        ) / 2
        torch.testing.assert_close(
            parameter,
            center[name] + expected_delta,
        )


def test_restore_center_parameters_reverts_elite_update() -> None:
    rule = small_rule()
    center = clone_center_parameters(rule)
    with torch.no_grad():
        for parameter in rule.parameters():
            parameter.add_(1.0)

    restore_center_parameters(rule, center)

    for name, parameter in rule.named_parameters():
        torch.testing.assert_close(parameter, center[name], rtol=0, atol=0)


def test_elite_search_sigma_shrinks_on_rejection() -> None:
    config = ShortcutCreditExperimentConfig(
        sigma=0.2,
        elite_min_sigma=0.025,
        elite_rejection_sigma_decay=0.5,
    )
    state = PlateauState(
        search_sigma=0.1,
        consecutive_accepted_updates=2,
    )

    update_elite_search_state(state, accepted=False, config=config)

    assert state.search_sigma == pytest.approx(0.05)
    assert state.consecutive_accepted_updates == 0


def test_elite_search_sigma_grows_after_success_streak() -> None:
    config = ShortcutCreditExperimentConfig(
        sigma=0.2,
        elite_acceptance_patience=3,
        elite_acceptance_sigma_growth=2.0,
    )
    state = PlateauState(search_sigma=0.05)

    update_elite_search_state(state, accepted=True, config=config)
    update_elite_search_state(state, accepted=True, config=config)
    assert state.search_sigma == pytest.approx(0.05)
    assert state.consecutive_accepted_updates == 2

    update_elite_search_state(state, accepted=True, config=config)
    assert state.search_sigma == pytest.approx(0.1)
    assert state.consecutive_accepted_updates == 0

    state.search_sigma = 0.15
    state.consecutive_accepted_updates = 2
    update_elite_search_state(state, accepted=True, config=config)
    assert state.search_sigma == pytest.approx(0.2)


def test_elite_proposal_mean_improvement_uses_matched_trajectories() -> None:
    improvement = elite_proposal_mean_improvement(
        [1.0, 2.0, 3.0],
        [1.2, 1.9, 3.5],
    )

    assert improvement == pytest.approx(0.2)
    with pytest.raises(ValueError, match="at least one"):
        elite_proposal_mean_improvement([], [])
    with pytest.raises(ValueError, match="counts must match"):
        elite_proposal_mean_improvement([1.0], [1.0, 2.0])


def test_elite_acceptance_seed_separates_extra_trajectories() -> None:
    assert elite_acceptance_seed(123, 1) != elite_acceptance_seed(123, 2)
    with pytest.raises(ValueError, match="must be positive"):
        elite_acceptance_seed(123, 0)


def test_resume_horizon_override_is_explicit() -> None:
    default_config = ShortcutCreditExperimentConfig()
    assert resolve_resume_horizon(default_config, 160) == 160

    override_config = ShortcutCreditExperimentConfig(
        resume="checkpoint.pt",
        resume_horizon=320,
        max_horizon=640,
    )
    assert resolve_resume_horizon(override_config, 160) == 320
    state = PlateauState(consecutive_accepted_updates=2)
    assert apply_resume_horizon(override_config, 160, state) == 320
    assert state.consecutive_accepted_updates == 0

    unchanged_state = PlateauState(consecutive_accepted_updates=2)
    assert apply_resume_horizon(
        default_config,
        160,
        unchanged_state,
    ) == 160
    assert unchanged_state.consecutive_accepted_updates == 2

    with pytest.raises(ValueError, match="requires a resume"):
        ShortcutCreditExperimentConfig(resume_horizon=20)
    with pytest.raises(ValueError, match="resume_horizon"):
        ShortcutCreditExperimentConfig(
            resume="checkpoint.pt",
            resume_horizon=641,
            max_horizon=640,
        )


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


def test_trajectory_summary_can_omit_fitness_for_heldout_data() -> None:
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

    summary = trajectory_summary(
        "heldout_center_rule",
        None,
        clean,
        correct,
    )

    assert "heldout_center_rule/fitness" not in summary
    assert summary["heldout_center_rule/min_mode_accuracy"] == 0.4


def test_heldout_candidate_summary_keeps_outer_selection_indices() -> None:
    outer_clean = [
        ShortcutMetrics(
            2.0,
            0.2,
            {"masked": 0.2, "incorrect": 0.1},
            {"masked": 1.8, "incorrect": 2.0},
            8,
            7,
            0.2,
        ),
        ShortcutMetrics(
            1.0,
            0.7,
            {"masked": 0.8, "incorrect": 0.7},
            {"masked": 0.9, "incorrect": 1.0},
            10,
            10,
            0.15,
        ),
    ]
    heldout_clean = [
        ShortcutMetrics(
            1.8,
            0.3,
            {"masked": 0.3, "incorrect": 0.2},
            {"masked": 1.7, "incorrect": 2.0},
            9,
            8,
            0.2,
        ),
        ShortcutMetrics(
            1.0,
            0.7,
            {"masked": 0.7, "incorrect": 0.6},
            {"masked": 0.9, "incorrect": 1.0},
            10,
            10,
            0.15,
        ),
    ]
    heldout_correct = [
        ShortcutMetrics(
            0.8,
            0.8,
            {"correct": 0.8},
            {"correct": 0.8},
            9,
            9,
            0.2,
        ),
        ShortcutMetrics(
            0.7,
            0.9,
            {"correct": 0.9},
            {"correct": 0.7},
            10,
            10,
            0.15,
        ),
    ]

    summary = heldout_candidate_summary(
        torch.tensor([0.9, 0.1]),
        outer_clean,
        heldout_clean,
        heldout_correct,
    )

    assert summary["best/heldout_min_mode_accuracy"] == pytest.approx(0.2)
    assert summary["best/heldout_correct_leak_accuracy"] == pytest.approx(0.8)
    assert summary["robust/heldout_min_mode_accuracy"] == pytest.approx(0.6)
    assert summary["robust/heldout_correct_leak_accuracy"] == pytest.approx(
        0.9
    )
    assert summary["heldout_candidates/outer_fitness_correlation"] < -0.99


def test_forward_trajectory_evaluates_outer_loop_unseen_batches() -> None:
    config = ShortcutCreditExperimentConfig(
        population_size=2,
        horizon=1,
        max_horizon=1,
        batch_size=4,
        fitness_examples=8,
        fitness_batch_size=4,
        correct_eval_examples=4,
        min_length=4,
        max_length=4,
        d_model=32,
        backward_d_model=32,
        forward_layers=1,
        backward_layers=1,
        heads=4,
    )
    vocabulary = small_vocabulary()
    model = ShortcutDecoderTransformer(
        make_forward_model_config(
            vocabulary,
            d_model=32,
            n_layers=1,
            n_heads=4,
        )
    )
    base_state = deepcopy(model.state_dict())
    inner = (
        make_shortcut_batch(
            4,
            4,
            leak_mode="correct",
            generator=torch.Generator().manual_seed(1),
            vocabulary=vocabulary,
        ),
    )
    fitness = make_fitness_batches(
        8,
        min_length=4,
        max_length=4,
        batch_size=4,
        generator=torch.Generator().manual_seed(2),
        vocabulary=vocabulary,
    )
    correct = (
        make_shortcut_batch(
            4,
            4,
            leak_mode="correct",
            generator=torch.Generator().manual_seed(3),
            vocabulary=vocabulary,
        ),
    )
    heldout_fitness = make_fitness_batches(
        8,
        min_length=4,
        max_length=4,
        batch_size=4,
        generator=torch.Generator().manual_seed(4),
        vocabulary=vocabulary,
    )
    heldout_correct = (
        make_shortcut_batch(
            4,
            4,
            leak_mode="correct",
            generator=torch.Generator().manual_seed(5),
            vocabulary=vocabulary,
        ),
    )

    trajectory = train_forward_trajectory(
        config,
        base_state=base_state,
        backward_rule=None,
        inner_batches=inner,
        fitness_batches=fitness,
        correct_batches=correct,
        heldout_fitness_batches=heldout_fitness,
        heldout_correct_batches=heldout_correct,
        device=torch.device("cpu"),
    )

    assert trajectory.heldout_clean is not None
    assert trajectory.heldout_correct is not None
    assert set(trajectory.heldout_clean.mode_accuracy) == {
        "masked",
        "incorrect",
    }
    assert set(trajectory.heldout_correct.mode_accuracy) == {"correct"}


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
    assert statistics["routing_gate_std"] > 0
    assert statistics["routing_position_profile_std"] > 0
    assert statistics["routing_input_conditioned_rms"] > 0


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
        plateau_state=PlateauState(
            stale_generations=3,
            search_sigma=0.025,
            consecutive_accepted_updates=2,
        ),
    )

    loaded, generation, horizon, plateau = load_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded, AttentionRoutingRule)
    assert generation == 5
    assert horizon == 20
    assert plateau.stale_generations == 3
    assert plateau.search_sigma == 0.025
    assert plateau.consecutive_accepted_updates == 2
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
