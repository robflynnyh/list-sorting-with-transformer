from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from list_sorting_transformer.direction_sampling import (
    sample_function_diverse_directions,
    select_diverse_signatures,
)
from list_sorting_transformer.shortcut_credit import (
    AttentionRoutingRule,
    ShortcutPointerVocabulary,
    make_shortcut_batch,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_fresh_backward_rule,
    run,
)


def make_router_and_probe() -> tuple[AttentionRoutingRule, torch.Tensor]:
    config = ShortcutCreditExperimentConfig(
        population_size=8,
        backward_rule_type="attention_router",
        routing_credit_mode="signed",
        shared_routing_map=True,
        d_model=16,
        backward_d_model=16,
        forward_layers=1,
        backward_layers=1,
        heads=2,
    )
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    router = initialize_fresh_backward_rule(
        config,
        vocabulary,
        device=torch.device("cpu"),
    )
    assert isinstance(router, AttentionRoutingRule)
    batch = make_shortcut_batch(
        4,
        6,
        leak_mode="correct",
        leak_placement="random_list",
        generator=torch.Generator().manual_seed(31),
        vocabulary=vocabulary,
    )
    return router, batch.input_ids


def test_diverse_signature_selection_treats_opposites_as_duplicates() -> None:
    signatures = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    selected = select_diverse_signatures(signatures, 2)

    assert selected[0] == 0
    assert selected[1] == 2


def test_function_diverse_sampler_is_deterministic_and_restores_router() -> None:
    router, probe = make_router_and_probe()
    original = deepcopy(router.state_dict())
    router.capture_statistics = True

    first = sample_function_diverse_directions(
        router,
        generator=torch.Generator().manual_seed(41),
        count=4,
        candidate_multiplier=4,
        sigma=0.21,
        probe_token_ids=probe,
        signature_size=128,
    )
    second = sample_function_diverse_directions(
        router,
        generator=torch.Generator().manual_seed(41),
        count=4,
        candidate_multiplier=4,
        sigma=0.21,
        probe_token_ids=probe,
        signature_size=128,
    )

    assert router.capture_statistics
    for name, value in router.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert len(first.directions) == 4
    assert first.metrics == second.metrics
    assert (
        first.metrics["direction_sampling/selected_abs_cosine_mean"]
        <= first.metrics["direction_sampling/pool_abs_cosine_mean"]
    )
    for first_direction, second_direction in zip(
        first.directions,
        second.directions,
    ):
        for name in first_direction.tensors:
            torch.testing.assert_close(
                first_direction.tensors[name],
                second_direction.tensors[name],
                rtol=0,
                atol=0,
            )
            if first_direction.tensors[name].ndim == 2:
                assert (
                    torch.linalg.matrix_rank(
                        first_direction.tensors[name]
                    )
                    <= 1
                )


def test_random_direction_sampler_remains_default() -> None:
    config = ShortcutCreditExperimentConfig()

    assert config.direction_sampler == "random"
    assert config.direction_candidate_multiplier == 4


def test_function_diverse_sampler_requires_attention_router() -> None:
    with pytest.raises(
        ValueError,
        match="require an attention router",
    ):
        ShortcutCreditExperimentConfig(
            direction_sampler="function_diverse",
        )


def test_controller_reports_function_diverse_sampling(
    tmp_path: Path,
) -> None:
    output_dir = run(
        ShortcutCreditExperimentConfig(
            run_name="function-diverse-smoke",
            output_dir=str(tmp_path),
            generations=1,
            population_size=4,
            horizon=1,
            max_horizon=1,
            horizon_promotion_mode="fixed",
            batch_size=4,
            fitness_examples=4,
            acceptance_fitness_examples=4,
            fitness_batch_size=4,
            correct_eval_examples=4,
            heldout_examples=4,
            min_length=4,
            max_length=4,
            d_model=16,
            backward_d_model=16,
            forward_layers=1,
            backward_layers=1,
            heads=2,
            backward_rule_type="attention_router",
            routing_credit_mode="signed",
            outer_update_rule="elite_centroid",
            elite_backtracking=True,
            adaptive_elite_counts="1,2",
            vectorized_population=True,
            vectorized_chunk_size=4,
            direction_sampler="function_diverse",
            direction_candidate_multiplier=2,
            direction_probe_examples=2,
            direction_signature_size=32,
            checkpoint_interval=1,
            device="cpu",
        )
    )
    metrics = json.loads((output_dir / "metrics.jsonl").read_text())

    assert metrics["direction_sampling/method_function_diverse"] == 1.0
    assert metrics["direction_sampling/pool_size"] == 4.0
    assert (
        metrics["direction_sampling/selected_abs_cosine_mean"]
        <= metrics["direction_sampling/pool_abs_cosine_mean"]
    )
