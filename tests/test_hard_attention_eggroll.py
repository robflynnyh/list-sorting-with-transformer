from copy import deepcopy
import json
from pathlib import Path

import torch

from list_sorting_transformer.hard_attention_eggroll import (
    HardAttentionEggrollConfig,
    make_model,
    population_forward,
    run,
    sample_antithetic_rank_one_noise,
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


def test_factorized_population_matches_materialized_candidates() -> None:
    config = small_config()
    model = make_model(config, device=torch.device("cpu"))
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
