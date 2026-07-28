from __future__ import annotations

import json
from pathlib import Path

import torch

from list_sorting_transformer.maml_shortcut_experiment import (
    MAMLShortcutConfig,
    make_fitness_pairs,
    make_model,
    one_step_router_objective,
    run,
)
from list_sorting_transformer.maml_length_generalization import make_router
from list_sorting_transformer.shortcut_credit import (
    ShortcutPointerVocabulary,
    make_shortcut_batch,
)


def small_config(**overrides: object) -> MAMLShortcutConfig:
    values = {
        "steps": 1,
        "batch_size": 4,
        "min_length": 3,
        "max_length": 4,
        "fitness_examples": 8,
        "fitness_batch_size": 2,
        "eval_examples": 4,
        "eval_batch_size": 2,
        "d_model": 16,
        "layers": 1,
        "heads": 1,
        "router_d_model": 16,
        "router_heads": 1,
        "log_interval": 1,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "device": "cpu",
    }
    values.update(overrides)
    return MAMLShortcutConfig(**values)


def test_fixed_fitness_pairs_are_balanced_and_deterministic() -> None:
    config = small_config()
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    first = make_fitness_pairs(
        config,
        vocabulary=vocabulary,
        device=torch.device("cpu"),
        seed_offset=10_000,
    )
    second = make_fitness_pairs(
        config,
        vocabulary=vocabulary,
        device=torch.device("cpu"),
        seed_offset=10_000,
    )

    assert [batch.leak_mode for pair in first for batch in pair] == [
        "masked",
        "incorrect",
        "masked",
        "incorrect",
    ]
    for first_pair, second_pair in zip(first, second):
        for first_batch, second_batch in zip(first_pair, second_pair):
            torch.testing.assert_close(
                first_batch.input_ids,
                second_batch.input_ids,
            )
            torch.testing.assert_close(
                first_batch.targets,
                second_batch.targets,
            )


def test_shortcut_router_receives_meta_gradient() -> None:
    config = small_config()
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    model = make_model(config, vocabulary, device=torch.device("cpu"))
    router = make_router(config, vocabulary, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(41)
    biased_batch = make_shortcut_batch(
        4,
        4,
        leak_mode="correct",
        leak_placement="random_list",
        generator=generator,
        vocabulary=vocabulary,
    )
    fitness_pair = make_fitness_pairs(
        config,
        vocabulary=vocabulary,
        device=torch.device("cpu"),
        seed_offset=10_000,
    )[0]

    objective = one_step_router_objective(
        model,
        router,
        biased_batch,
        fitness_pair,
        inner_learning_rate=config.inner_learning_rate,
    )
    gradients = torch.autograd.grad(
        objective.meta_loss,
        tuple(router.parameters()),
    )

    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.square().sum()) for gradient in gradients) > 0


def test_shortcut_router_run_persists_both_networks(tmp_path: Path) -> None:
    output_dir = run(
        small_config(
            run_name="shortcut-router-smoke",
            output_dir=str(tmp_path),
        )
    )
    checkpoint = torch.load(output_dir / "latest.pt", map_location="cpu")
    row = json.loads(
        (output_dir / "metrics.jsonl").read_text().splitlines()[-1]
    )

    assert checkpoint["step"] == 1
    assert checkpoint["router"] is not None
    assert checkpoint["router_optimizer"] is not None
    assert "fitness_fixed/masked_accuracy" in row
    assert "fitness_heldout/incorrect_accuracy" in row
    assert "correct_leak/accuracy" in row
