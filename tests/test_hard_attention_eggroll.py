from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from list_sorting_transformer.hard_attention_eggroll import (
    AntitheticRankOneNoise,
    CurriculumState,
    HardAttentionEggrollConfig,
    RankOneFactors,
    curriculum_check_due,
    curriculum_is_complete,
    estimate_elite_centroid_directions,
    estimate_reward_gradients,
    evaluation_batch_size,
    evaluate_population,
    initialize_curriculum_state,
    make_model,
    pointer_targets,
    population_forward,
    run,
    sample_antithetic_rank_one_noise,
    shape_fitness,
    update_curriculum,
)
from list_sorting_transformer.data import make_pointer_next_batch
from list_sorting_transformer.positions import sample_position_offsets


def small_config(**overrides: object) -> HardAttentionEggrollConfig:
    values = {
        "run_name": "test",
        "generations": 1,
        "population_size": 4,
        "population_chunk_size": 4,
        "batch_size": 3,
        "train_min_length": 2,
        "train_max_length": 4,
        "eval_lengths": (2, 4),
        "eval_examples": 4,
        "d_model": 64,
        "layers": 2,
        "heads": 4,
        "position_moduli": (3, 5, 7, 11),
        "position_offset_min": -100,
        "position_offset_max": 100,
        "log_interval": 1,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "device": "cpu",
    }
    values.update(overrides)
    return HardAttentionEggrollConfig(**values)


def materialize_candidate(
    model: torch.nn.Module,
    noise: object,
    candidate_index: int,
    sigma: float,
) -> torch.nn.Module:
    candidate = deepcopy(model)
    parameters = dict(candidate.named_parameters())
    with torch.no_grad():
        for name, factors in noise.matrices.items():
            parameters[name].add_(
                sigma
                * torch.outer(
                    factors.left[candidate_index],
                    factors.right[candidate_index],
                )
            )
        for name, values in noise.vectors.items():
            parameters[name].add_(sigma * values[candidate_index])
    return candidate


def test_model_uses_fixed_modular_positions_and_exact_top1() -> None:
    model = make_model(small_config(), device=torch.device("cpu"))

    assert model.encoder.layer_position_modes == ("none", "none")
    assert tuple(
        block.attention.top_k for block in model.encoder.blocks
    ) == (1, 1)
    assert not any(
        parameter.requires_grad
        for parameter in model.position_embedding.parameters()
    )
    assert model.position_embedding.period == 3 * 5 * 7 * 11


def test_evaluation_batch_size_respects_attention_budget() -> None:
    assert (
        evaluation_batch_size(
            configured_batch_size=128,
            attention_element_budget=4 * 100 * 100,
            heads=4,
            prompt_length=10,
        )
        == 100
    )
    assert (
        evaluation_batch_size(
            configured_batch_size=128,
            attention_element_budget=4 * 100 * 100,
            heads=4,
            prompt_length=100,
        )
        == 1
    )


def assert_factorized_population_matches_materialized_candidates(
    top_k: int | None,
) -> None:
    config = small_config()
    model = make_model(config, device=torch.device("cpu"))
    model.set_attention_top_k(top_k)
    data_generator = torch.Generator().manual_seed(31)
    batch = make_pointer_next_batch(
        config.batch_size,
        4,
        generator=data_generator,
        vocabulary=model.vocabulary,
    )
    offsets = sample_position_offsets(
        config.batch_size,
        minimum=config.position_offset_min,
        maximum=config.position_offset_max,
        generator=data_generator,
        device=torch.device("cpu"),
    )
    noise = sample_antithetic_rank_one_noise(
        model,
        config.population_size,
        generator=torch.Generator().manual_seed(37),
    ).pair_chunk(0, config.population_size // 2)

    population_logits, _ = population_forward(
        model,
        batch.prompt_ids,
        offsets,
        noise,
        config.sigma,
    )
    materialized_logits = torch.stack(
        [
            materialize_candidate(
                model,
                noise,
                candidate_index,
                config.sigma,
            )(batch.prompt_ids, offsets=offsets)
            for candidate_index in range(config.population_size)
        ]
    )

    torch.testing.assert_close(
        population_logits,
        materialized_logits,
        rtol=1e-5,
        atol=2e-6,
    )


def test_factorized_population_matches_materialized_top1_candidates() -> None:
    assert_factorized_population_matches_materialized_candidates(1)


def test_factorized_population_matches_materialized_top2_candidates() -> None:
    assert_factorized_population_matches_materialized_candidates(2)


def test_factorized_population_matches_materialized_dense_candidates() -> None:
    assert_factorized_population_matches_materialized_candidates(None)


def test_grouped_population_matches_candidate_specific_materialization() -> None:
    config = small_config(batch_size=2, population_size=4)
    model = make_model(config, device=torch.device("cpu"))
    data_generator = torch.Generator().manual_seed(41)
    batch = make_pointer_next_batch(
        config.batch_size,
        4,
        generator=data_generator,
        vocabulary=model.vocabulary,
    )
    offsets = sample_position_offsets(
        config.batch_size,
        minimum=config.position_offset_min,
        maximum=config.position_offset_max,
        generator=data_generator,
        device=torch.device("cpu"),
    )
    antithetic_noise = sample_antithetic_rank_one_noise(
        model,
        config.population_size,
        generator=torch.Generator().manual_seed(43),
    )
    signed_noise = antithetic_noise.pair_chunk(
        0,
        antithetic_noise.pair_count,
    )
    example_indices = torch.tensor([0, 1, 0, 1])

    for top_k in (None, 2, 1):
        model.set_attention_top_k(top_k)
        grouped_logits, _ = population_forward(
            model,
            batch.prompt_ids[example_indices],
            offsets[example_indices],
            signed_noise,
            config.sigma,
            candidate_inputs=True,
        )
        materialized_logits = torch.stack(
            [
                materialize_candidate(
                    model,
                    signed_noise,
                    candidate_index,
                    config.sigma,
                )(
                    batch.prompt_ids[example_index : example_index + 1],
                    offsets=offsets[example_index : example_index + 1],
                )[0]
                for candidate_index, example_index in enumerate(
                    example_indices.tolist()
                )
            ]
        )
        torch.testing.assert_close(
            grouped_logits,
            materialized_logits,
            rtol=1e-5,
            atol=2e-6,
        )

    population = evaluate_population(
        model,
        batch,
        offsets,
        antithetic_noise,
        sigma=config.sigma,
        population_chunk_size=2,
        data_mode="grouped",
    )
    targets = pointer_targets(batch)[example_indices]
    expected_losses = F.cross_entropy(
        materialized_logits,
        targets,
        reduction="none",
    )
    torch.testing.assert_close(population.losses, expected_losses)


def test_grouped_population_preserves_global_pair_assignment_across_chunks() -> None:
    config = small_config(batch_size=2, population_size=16)
    model = make_model(config, device=torch.device("cpu"))
    data_generator = torch.Generator().manual_seed(47)
    batch = make_pointer_next_batch(
        config.batch_size,
        4,
        generator=data_generator,
        vocabulary=model.vocabulary,
    )
    offsets = torch.tensor([-91, 73])
    antithetic_noise = sample_antithetic_rank_one_noise(
        model,
        config.population_size,
        generator=torch.Generator().manual_seed(53),
    )
    signed_noise = antithetic_noise.pair_chunk(
        0,
        antithetic_noise.pair_count,
    )
    pair_examples = torch.arange(antithetic_noise.pair_count) % len(offsets)
    signed_examples = torch.cat((pair_examples, pair_examples))

    materialized_logits = torch.stack(
        [
            materialize_candidate(
                model,
                signed_noise,
                candidate_index,
                config.sigma,
            )(
                batch.prompt_ids[example_index : example_index + 1],
                offsets=offsets[example_index : example_index + 1],
            )[0]
            for candidate_index, example_index in enumerate(
                signed_examples.tolist()
            )
        ]
    )
    expected_losses = F.cross_entropy(
        materialized_logits,
        pointer_targets(batch)[signed_examples],
        reduction="none",
    )

    chunked = evaluate_population(
        model,
        batch,
        offsets,
        antithetic_noise,
        sigma=config.sigma,
        population_chunk_size=6,
        data_mode="grouped",
    )
    unchunked = evaluate_population(
        model,
        batch,
        offsets,
        antithetic_noise,
        sigma=config.sigma,
        population_chunk_size=config.population_size,
        data_mode="grouped",
    )

    torch.testing.assert_close(chunked.losses, expected_losses)
    torch.testing.assert_close(chunked.losses, unchunked.losses)


def test_curriculum_requires_repeated_success_and_advances_in_order() -> None:
    config = small_config(
        curriculum=True,
        curriculum_accuracy_threshold=0.70,
        curriculum_success_checks=2,
        curriculum_initial_top_k=3,
    )
    state = initialize_curriculum_state(config)

    assert state == CurriculumState(
        current_max_length=2,
        attention_top_k=None,
    )
    assert update_curriculum(state, config, criterion_accuracy=0.69) is None
    assert state.success_streak == 0

    expected_promotions = (
        ("length", 3, None),
        ("length", 4, None),
        ("start_sparsity", 4, 3),
        ("increase_sparsity", 4, 2),
        ("increase_sparsity", 4, 1),
    )
    for promotion, expected_length, expected_top_k in expected_promotions:
        assert update_curriculum(
            state,
            config,
            criterion_accuracy=0.70,
        ) is None
        assert update_curriculum(
            state,
            config,
            criterion_accuracy=0.80,
        ) == promotion
        assert state.current_max_length == expected_length
        assert state.attention_top_k == expected_top_k

    assert curriculum_is_complete(state, config)
    assert state.promotion_count == len(expected_promotions)
    assert update_curriculum(state, config, criterion_accuracy=1.0) is None


def test_training_streak_only_checks_current_maximum_length() -> None:
    config = small_config(
        curriculum=True,
        curriculum_progress_mode="training_streak",
    )
    state = initialize_curriculum_state(config)

    assert curriculum_check_due(
        config,
        state,
        generation=1,
        batch_length=2,
    )
    state.current_max_length = 4
    assert not curriculum_check_due(
        config,
        state,
        generation=2,
        batch_length=3,
    )
    assert curriculum_check_due(
        config,
        state,
        generation=3,
        batch_length=4,
    )


def test_elite_centroid_selects_one_sign_per_antithetic_pair() -> None:
    noise = AntitheticRankOneNoise(
        matrices={
            "matrix": RankOneFactors(
                left=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                right=torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
            )
        },
        vectors={
            "vector": torch.tensor([[6.0, 7.0], [8.0, 9.0]])
        },
    )
    directions, selected = estimate_elite_centroid_directions(
        noise,
        torch.tensor([3.0, 1.0, 0.0, 2.0]),
        elite_count=1,
    )

    assert selected.tolist() == [2]
    torch.testing.assert_close(
        directions["matrix"],
        -torch.tensor([[2.0, 3.0], [0.0, 0.0]]),
    )
    torch.testing.assert_close(
        directions["vector"],
        -torch.tensor([6.0, 7.0]),
    )


def test_averaged_local_estimators_equal_global_population_estimator() -> None:
    worker_noises = (
        AntitheticRankOneNoise(
            matrices={
                "matrix": RankOneFactors(
                    left=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    right=torch.tensor([[0.5, 1.5], [2.5, 3.5]]),
                )
            },
            vectors={"vector": torch.tensor([[1.0, -1.0], [2.0, -2.0]])},
        ),
        AntitheticRankOneNoise(
            matrices={
                "matrix": RankOneFactors(
                    left=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                    right=torch.tensor([[4.5, 5.5], [6.5, 7.5]]),
                )
            },
            vectors={"vector": torch.tensor([[3.0, -3.0], [4.0, -4.0]])},
        ),
    )
    worker_losses = (
        torch.tensor([1.0, 1.5, 2.0, 2.5]),
        torch.tensor([0.5, 1.25, 1.75, 3.0]),
    )
    block_fitness = shape_fitness(torch.cat(worker_losses), "zscore")
    local_gradients = [
        estimate_reward_gradients(
            noise,
            block_fitness[index * 4 : (index + 1) * 4],
        )
        for index, noise in enumerate(worker_noises)
    ]
    averaged = {
        name: (local_gradients[0][name] + local_gradients[1][name]) / 2
        for name in local_gradients[0]
    }

    global_noise = AntitheticRankOneNoise(
        matrices={
            "matrix": RankOneFactors(
                left=torch.cat(
                    tuple(
                        noise.matrices["matrix"].left
                        for noise in worker_noises
                    )
                ),
                right=torch.cat(
                    tuple(
                        noise.matrices["matrix"].right
                        for noise in worker_noises
                    )
                ),
            )
        },
        vectors={
            "vector": torch.cat(
                tuple(noise.vectors["vector"] for noise in worker_noises)
            )
        },
    )
    global_order = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7])
    global_gradients = estimate_reward_gradients(
        global_noise,
        block_fitness[global_order],
    )

    for name in averaged:
        torch.testing.assert_close(averaged[name], global_gradients[name])


def test_tiny_run_writes_metrics_and_checkpoint(tmp_path: Path) -> None:
    output_dir = run(
        small_config(
            run_name="tiny",
            output_dir=str(tmp_path),
            population_size=2,
            population_chunk_size=2,
            batch_size=2,
            eval_examples=2,
            eval_batch_size=1,
        )
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    checkpoint = torch.load(output_dir / "latest.pt", map_location="cpu")

    assert [row["generation"] for row in rows] == [0.0, 1.0]
    assert rows[1]["optimization/parameter_update_rms"] > 0
    assert rows[1]["optimization/update_to_parameter_rms_ratio"] > 0
    assert checkpoint["experiment"] == "hard_attention_forward_eggroll"
    assert checkpoint["generation"] == 1
    assert checkpoint["curriculum_state"] == {
        "current_max_length": 4,
        "attention_top_k": 1,
        "success_streak": 0,
        "promotion_count": 0,
    }
    assert len(checkpoint["noise_generator_states"]) == 1
    assert checkpoint["wandb_run_id"] is None
    assert "curriculum_generator_state" in checkpoint


def test_tiny_training_streak_run_uses_training_batch_criterion(
    tmp_path: Path,
) -> None:
    output_dir = run(
        small_config(
            run_name="tiny-training-streak",
            output_dir=str(tmp_path),
            population_size=2,
            population_chunk_size=2,
            batch_size=2,
            eval_examples=2,
            eval_batch_size=1,
            curriculum=True,
            curriculum_progress_mode="training_streak",
            curriculum_success_checks=10,
        )
    )

    final = json.loads(
        (output_dir / "metrics.jsonl").read_text().splitlines()[-1]
    )
    assert final["curriculum/criterion_is_training_batch"] == 1
    assert final["curriculum/criterion_accuracy"] == (
        final["train/center_accuracy_after_update"]
    )
    assert "curriculum/probe_accuracy" not in final


def test_tiny_elite_run_reports_selected_centroid(tmp_path: Path) -> None:
    output_dir = run(
        small_config(
            run_name="tiny-elite",
            output_dir=str(tmp_path),
            population_size=2,
            population_chunk_size=2,
            batch_size=2,
            eval_examples=2,
            eval_batch_size=1,
            update_rule="elite_centroid",
            elite_count=1,
        )
    )

    final = json.loads(
        (output_dir / "metrics.jsonl").read_text().splitlines()[-1]
    )
    assert final["optimization/update_rule_elite_centroid"] == 1
    assert final["optimization/elite_count"] == 1


def test_tiny_grouped_run_reports_data_allocation(tmp_path: Path) -> None:
    output_dir = run(
        small_config(
            run_name="tiny-grouped",
            output_dir=str(tmp_path),
            population_size=4,
            population_chunk_size=2,
            batch_size=2,
            eval_examples=2,
            eval_batch_size=1,
            population_data_mode="grouped",
        )
    )

    final = json.loads(
        (output_dir / "metrics.jsonl").read_text().splitlines()[-1]
    )
    assert final["population/grouped_data"] == 1
    assert final["population/unique_examples"] == 2
    assert final["population/candidates_per_example"] == 2
    assert final["population/candidate_example_evaluations"] == 4


def test_tiny_grouped_run_resumes_generator_states(tmp_path: Path) -> None:
    first = run(
        small_config(
            run_name="tiny-grouped-resume",
            output_dir=str(tmp_path),
            population_size=4,
            population_chunk_size=2,
            batch_size=2,
            eval_examples=2,
            eval_batch_size=1,
            population_data_mode="grouped",
        )
    )
    resumed = run(
        small_config(
            run_name="tiny-grouped-resume",
            output_dir=str(tmp_path),
            generations=2,
            population_size=4,
            population_chunk_size=2,
            batch_size=2,
            eval_examples=2,
            eval_batch_size=1,
            population_data_mode="grouped",
            resume=str(first / "latest.pt"),
        )
    )

    rows = [
        json.loads(line)
        for line in (resumed / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["generation"] for row in rows] == [0.0, 1.0, 2.0]
    checkpoint = torch.load(resumed / "latest.pt", map_location="cpu")
    assert checkpoint["generation"] == 2
