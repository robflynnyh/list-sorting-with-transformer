from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from list_sorting_transformer.shortcut_credit import (
    AttentionRoutingRule,
    ShortcutPointerVocabulary,
    apply_eggroll_direction,
    clone_center_parameters,
    evaluate_shortcut_batches,
    make_fitness_batches,
    make_shortcut_batch,
    sample_eggroll_direction,
    shortcut_loss,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_forward_model,
    initialize_fresh_backward_rule,
    parse_adaptive_elite_counts,
    parse_successive_halving_rungs,
    probe_longer_horizon,
    run,
)
from list_sorting_transformer.vectorized_routing_population import (
    stack_candidate_rule_parameters,
    train_vectorized_routing_population,
)


def test_vectorized_population_rejects_unsupported_rules() -> None:
    with pytest.raises(
        ValueError,
        match="require an unconditioned attention router",
    ):
        ShortcutCreditExperimentConfig(vectorized_population=True)
    with pytest.raises(
        ValueError,
        match="require an unconditioned attention router",
    ):
        ShortcutCreditExperimentConfig(
            backward_rule_type="attention_router",
            vectorized_population=True,
            route_output_projection=True,
        )


def test_adaptive_elite_counts_require_sorted_unique_positives() -> None:
    assert parse_adaptive_elite_counts("1,2,4,8") == (1, 2, 4, 8)
    for invalid in ("", "2,1", "1,1", "0,1", "one"):
        with pytest.raises(ValueError):
            parse_adaptive_elite_counts(invalid)


def test_successive_halving_rungs_are_opt_in_and_validated() -> None:
    assert parse_successive_halving_rungs(
        "80:16,160:8,320:8"
    ) == ((80, 16), (160, 8), (320, 8))
    for invalid in ("", "80", "80:16,40:8", "80:8,160:16"):
        with pytest.raises(ValueError):
            parse_successive_halving_rungs(invalid)

    default = ShortcutCreditExperimentConfig()
    assert default.successive_halving_rungs is None


def test_successive_halving_controller_keeps_separate_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = run(
        ShortcutCreditExperimentConfig(
            run_name="halving-smoke",
            output_dir=str(tmp_path),
            generations=1,
            population_size=4,
            horizon=2,
            max_horizon=2,
            horizon_promotion_mode="fixed",
            batch_size=4,
            fitness_examples=4,
            acceptance_fitness_examples=4,
            fitness_batch_size=4,
            correct_eval_examples=4,
            heldout_examples=4,
            report_interval=1,
            task_variant="pointer_next_length",
            min_length=2,
            max_length=4,
            fitness_length=6,
            heldout_length=8,
            d_model=16,
            backward_d_model=16,
            forward_layers=1,
            backward_layers=1,
            heads=2,
            backward_rule_type="attention_router",
            outer_update_rule="elite_centroid",
            elite_backtracking=True,
            adaptive_elite_counts="1,2",
            elite_acceptance_trajectories=2,
            vectorized_population=True,
            vectorized_chunk_size=2,
            successive_halving_rungs="1:2,2:2",
            checkpoint_interval=1,
            device="cpu",
        )
    )
    metrics = json.loads((output_dir / "metrics.jsonl").read_text())

    assert output_dir.name == "halving-smoke"
    assert metrics["halving/rung_0/candidates"] == 4
    assert metrics["halving/rung_0/survivors"] == 2
    assert metrics["halving/rung_1/candidates"] == 2
    assert metrics["halving/rung_1/survivors"] == 2


def test_adaptive_controller_replays_identically(tmp_path: Path) -> None:
    common = dict(
        output_dir=str(tmp_path),
        generations=1,
        population_size=4,
        horizon=2,
        max_horizon=2,
        batch_size=4,
        fitness_examples=8,
        fitness_batch_size=4,
        correct_eval_examples=4,
        min_length=4,
        max_length=4,
        d_model=16,
        backward_d_model=16,
        forward_layers=1,
        backward_layers=1,
        heads=2,
        forward_learning_rate=1e-4,
        backward_rule_type="attention_router",
        shared_routing_map=True,
        leak_placement="random_list",
        outer_update_rule="elite_centroid",
        elite_backtracking=True,
        adaptive_elite_counts="1,2",
        adaptive_commit_scale=0.05,
        elite_acceptance_trajectories=1,
        vectorized_population=True,
        vectorized_chunk_size=4,
        horizon_promotion_mode="rejection_probe",
        checkpoint_interval=1,
        device="cpu",
    )
    first_dir = run(
        ShortcutCreditExperimentConfig(
            run_name="first",
            **common,
        )
    )
    second_dir = run(
        ShortcutCreditExperimentConfig(
            run_name="second",
            **common,
        )
    )
    first_metrics = json.loads(
        (first_dir / "metrics.jsonl").read_text()
    )
    second_metrics = json.loads(
        (second_dir / "metrics.jsonl").read_text()
    )
    for key in (
        "outer/selected_elite_count",
        "outer/update_accepted",
        "outer/adaptive_elite_1_acceptance_fitness",
        "outer/adaptive_elite_2_acceptance_fitness",
        "outer/commit_scale",
        "outer/next_commit_scale",
        "outer/commit_search_0_scale",
        "outer/commit_search_5_selection_fitness",
    ):
        assert first_metrics[key] == second_metrics[key]

    first_state = torch.load(
        first_dir / "latest.pt",
        map_location="cpu",
    )["backward_rule_state"]
    second_state = torch.load(
        second_dir / "latest.pt",
        map_location="cpu",
    )["backward_rule_state"]
    for name in first_state:
        torch.testing.assert_close(
            first_state[name],
            second_state[name],
            rtol=0,
            atol=0,
        )


def test_fixed_horizon_sparse_reporting_skips_only_reporting_metrics(
    tmp_path: Path,
) -> None:
    output_dir = run(
        ShortcutCreditExperimentConfig(
            run_name="fixed-sparse",
            output_dir=str(tmp_path),
            generations=3,
            population_size=2,
            horizon=1,
            max_horizon=1,
            horizon_promotion_mode="fixed",
            report_interval=2,
            batch_size=2,
            fitness_examples=4,
            acceptance_fitness_examples=4,
            fitness_batch_size=2,
            correct_eval_examples=2,
            heldout_examples=2,
            task_variant="pointer_next_length",
            min_length=2,
            max_length=3,
            fitness_length=4,
            heldout_length=6,
            d_model=8,
            backward_d_model=8,
            forward_layers=1,
            backward_layers=1,
            heads=1,
            backward_rule_type="attention_router",
            outer_update_rule="elite_centroid",
            elite_backtracking=True,
            adaptive_elite_counts="1",
            vectorized_population=True,
            vectorized_chunk_size=2,
            checkpoint_interval=1,
            device="cpu",
        )
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["report/full_generation"] for row in rows] == [
        1.0,
        0.0,
        1.0,
    ]
    assert "length_6/center_accuracy" in rows[0]
    assert "length_6/center_accuracy" not in rows[1]
    assert "length_6/center_accuracy" in rows[2]
    assert all(
        not any(
            term in key
            for term in ("masked", "incorrect", "correct_leak")
        )
        for row in rows
        for key in row
    )


def test_performance_curriculum_runs_to_max_horizon_and_stops(
    tmp_path: Path,
) -> None:
    output_dir = run(
        ShortcutCreditExperimentConfig(
            run_name="performance-curriculum",
            output_dir=str(tmp_path),
            generations=10,
            population_size=4,
            horizon=1,
            max_horizon=4,
            horizon_multiplier=2,
            horizon_promotion_mode="performance_plateau",
            horizon_score_window=1,
            horizon_min_generations=1,
            horizon_max_generations=1,
            horizon_failed_extension_limit=10,
            batch_size=4,
            fitness_examples=8,
            fitness_batch_size=4,
            correct_eval_examples=4,
            min_length=4,
            max_length=4,
            d_model=16,
            backward_d_model=16,
            forward_layers=1,
            backward_layers=1,
            heads=2,
            forward_learning_rate=1e-4,
            backward_rule_type="attention_router",
            shared_routing_map=True,
            leak_placement="random_list",
            outer_update_rule="elite_centroid",
            elite_backtracking=True,
            adaptive_elite_counts="1,2",
            elite_acceptance_trajectories=1,
            vectorized_population=True,
            vectorized_chunk_size=4,
            checkpoint_interval=10,
            device="cpu",
        )
    )
    metrics = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]

    assert [row["horizon"] for row in metrics] == [1, 2, 4]
    assert [row["curriculum/promoted"] for row in metrics] == [
        1.0,
        1.0,
        0.0,
    ]
    assert metrics[-1]["curriculum/stop_triggered"] == 1.0
    assert (
        metrics[-1]["curriculum/stop_reason"]
        == "max_horizon_plateau"
    )
    checkpoint = torch.load(output_dir / "latest.pt", map_location="cpu")
    assert checkpoint["generation"] == 2
    assert checkpoint["horizon"] == 4


def test_horizon_probe_replays_identically() -> None:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    config = ShortcutCreditExperimentConfig(
        population_size=4,
        horizon=2,
        max_horizon=4,
        batch_size=4,
        fitness_examples=8,
        fitness_batch_size=4,
        correct_eval_examples=4,
        min_length=4,
        max_length=4,
        d_model=16,
        backward_d_model=16,
        forward_layers=1,
        backward_layers=1,
        heads=2,
        backward_rule_type="attention_router",
        shared_routing_map=True,
        leak_placement="random_list",
        device="cpu",
    )
    rule = initialize_fresh_backward_rule(
        config,
        vocabulary,
        device=torch.device("cpu"),
    )
    fitness_batches = make_fitness_batches(
        8,
        min_length=4,
        max_length=4,
        batch_size=4,
        generator=torch.Generator().manual_seed(31),
        vocabulary=vocabulary,
        leak_placement="random_list",
    )
    correct_batches = (
        make_shortcut_batch(
            4,
            4,
            leak_mode="correct",
            leak_placement="random_list",
            generator=torch.Generator().manual_seed(33),
            vocabulary=vocabulary,
        ),
    )

    first = probe_longer_horizon(
        config,
        center_rule=rule,
        generation_seed=91,
        horizon=2,
        vocabulary=vocabulary,
        fitness_batches=fitness_batches,
        correct_batches=correct_batches,
        device=torch.device("cpu"),
    )
    second = probe_longer_horizon(
        config,
        center_rule=rule,
        generation_seed=91,
        horizon=2,
        vocabulary=vocabulary,
        fitness_batches=fitness_batches,
        correct_batches=correct_batches,
        device=torch.device("cpu"),
    )

    assert first == second
    assert first.next_horizon == 4


@pytest.mark.parametrize(
    "routing_credit_mode",
    ("suppress_renorm", "signed"),
)
def test_vectorized_routing_population_matches_serial_candidates(
    routing_credit_mode: str,
) -> None:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    config = ShortcutCreditExperimentConfig(
        horizon=2,
        max_horizon=2,
        batch_size=4,
        fitness_examples=8,
        fitness_batch_size=4,
        correct_eval_examples=4,
        min_length=4,
        max_length=4,
        d_model=16,
        backward_d_model=16,
        forward_layers=1,
        backward_layers=1,
        heads=2,
        forward_learning_rate=1e-4,
        backward_rule_type="attention_router",
        routing_credit_mode=routing_credit_mode,
        shared_routing_map=True,
        leak_placement="random_list",
        device="cpu",
    )
    base = initialize_forward_model(
        config,
        vocabulary,
        initialization_seed=17,
        device=torch.device("cpu"),
    )
    base_state = deepcopy(base.state_dict())
    center_rule = initialize_fresh_backward_rule(
        config,
        vocabulary,
        device=torch.device("cpu"),
    )
    assert isinstance(center_rule, AttentionRoutingRule)
    with torch.no_grad():
        center_rule.gates.fill_(0.1)
    center_parameters = clone_center_parameters(center_rule)
    directions = tuple(
        sample_eggroll_direction(
            center_rule,
            generator=torch.Generator().manual_seed(seed),
        )
        for seed in (41, 42)
    )
    candidate_specs = ((0, 0, 1), (1, 1, -1))
    rule_parameters = stack_candidate_rule_parameters(
        center_parameters,
        directions,
        candidate_specs,
        sigma=0.01,
        device=torch.device("cpu"),
    )
    inner_batches = tuple(
        make_shortcut_batch(
            4,
            4,
            leak_mode="correct",
            leak_placement="random_list",
            generator=torch.Generator().manual_seed(seed),
            vocabulary=vocabulary,
        )
        for seed in (21, 22)
    )
    fitness_batches = make_fitness_batches(
        8,
        min_length=4,
        max_length=4,
        batch_size=4,
        generator=torch.Generator().manual_seed(31),
        vocabulary=vocabulary,
        leak_placement="random_list",
    )
    heldout_batches = make_fitness_batches(
        8,
        min_length=4,
        max_length=4,
        batch_size=4,
        generator=torch.Generator().manual_seed(32),
        vocabulary=vocabulary,
        leak_placement="random_list",
    )
    correct_batches = (
        make_shortcut_batch(
            4,
            4,
            leak_mode="correct",
            leak_placement="random_list",
            generator=torch.Generator().manual_seed(33),
            vocabulary=vocabulary,
        ),
    )

    vectorized = train_vectorized_routing_population(
        config=config,
        base_state=base_state,
        center_rule=center_rule,
        rule_parameters=rule_parameters,
        inner_batches=inner_batches,
        fitness_batches=fitness_batches,
        correct_batches=correct_batches,
        heldout_fitness_batches=heldout_batches,
        heldout_correct_batches=correct_batches,
        device=torch.device("cpu"),
    )

    for local_index, (_, direction_index, sign) in enumerate(
        candidate_specs
    ):
        model = initialize_forward_model(
            config,
            vocabulary,
            initialization_seed=None,
            device=torch.device("cpu"),
        )
        model.load_state_dict(base_state)
        rule = AttentionRoutingRule(center_rule.config)
        apply_eggroll_direction(
            rule,
            center_parameters,
            directions[direction_index],
            sigma=0.01,
            sign=sign,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.forward_learning_rate,
        )
        for batch in inner_batches:
            optimizer.zero_grad(set_to_none=True)
            shortcut_loss(model, batch, rule).backward()
            optimizer.step()

        expected = evaluate_shortcut_batches(model, fitness_batches)
        actual = vectorized.trajectories[local_index].clean
        assert actual.loss == pytest.approx(expected.loss, abs=2e-6)
        assert actual.accuracy == expected.accuracy
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                vectorized.forward_parameters[
                    f"forward_model.{name}"
                ][local_index],
                parameter,
                rtol=2e-5,
                atol=2e-7,
            )

    first_segment = train_vectorized_routing_population(
        config=config,
        base_state=base_state,
        center_rule=center_rule,
        rule_parameters=rule_parameters,
        inner_batches=inner_batches[:1],
        fitness_batches=fitness_batches,
        correct_batches=correct_batches,
        heldout_fitness_batches=None,
        heldout_correct_batches=None,
        device=torch.device("cpu"),
    )
    resumed = train_vectorized_routing_population(
        config=config,
        base_state=base_state,
        center_rule=center_rule,
        rule_parameters=rule_parameters,
        inner_batches=inner_batches[1:],
        fitness_batches=fitness_batches,
        correct_batches=correct_batches,
        heldout_fitness_batches=heldout_batches,
        heldout_correct_batches=correct_batches,
        device=torch.device("cpu"),
        initial_state=first_segment,
    )
    assert resumed.step == 2
    for name in vectorized.forward_parameters:
        torch.testing.assert_close(
            resumed.forward_parameters[name],
            vectorized.forward_parameters[name],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            resumed.first_moments[name],
            vectorized.first_moments[name],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            resumed.second_moments[name],
            vectorized.second_moments[name],
            rtol=0,
            atol=0,
        )
