import importlib.util
from pathlib import Path

import pytest
import torch

from list_sorting_transformer.shortcut_learning.shortcut_credit import EggrollDirection


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "learned_backward_shortcuts"
    / "collapse_window_population.py"
)
SPEC = importlib.util.spec_from_file_location(
    "collapse_window_population",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_collapse_fitness_can_target_literal_accuracy() -> None:
    accuracy_fitness = MODULE.collapse_fitness(
        "minimum_mode_accuracy",
        center_worst_mode_loss=2.0,
        candidate_worst_mode_loss=1.0,
        center_minimum_mode_accuracy=0.7,
        candidate_minimum_mode_accuracy=0.8,
    )
    ce_fitness = MODULE.collapse_fitness(
        "worst_mode_ce",
        center_worst_mode_loss=2.0,
        candidate_worst_mode_loss=1.0,
        center_minimum_mode_accuracy=0.7,
        candidate_minimum_mode_accuracy=0.6,
    )
    assert accuracy_fitness == pytest.approx(0.1)
    assert ce_fitness == pytest.approx(1.0)


def test_collapse_fitness_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown collapse fitness"):
        MODULE.collapse_fitness(
            "unknown",
            center_worst_mode_loss=2.0,
            candidate_worst_mode_loss=1.0,
            center_minimum_mode_accuracy=0.7,
            candidate_minimum_mode_accuracy=0.8,
        )


def test_window_fitness_can_require_every_window_to_improve() -> None:
    assert MODULE.aggregate_window_fitness(
        [0.1, -0.02, 0.3],
        aggregation="mean",
    ) == pytest.approx(0.1266666667)
    assert MODULE.aggregate_window_fitness(
        [0.1, -0.02, 0.3],
        aggregation="minimum",
    ) == pytest.approx(-0.02)


def test_window_fitness_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MODULE.aggregate_window_fitness([], aggregation="minimum")


def test_direction_can_be_restricted_to_parameter_prefix() -> None:
    direction = EggrollDirection(
        {
            "token_embedding.weight": torch.ones(2, 3),
            "forward_state_projection.weight": torch.ones(3, 4),
            "gates": torch.ones(1),
        }
    )

    restricted = MODULE.restrict_direction(
        direction,
        parameter_prefixes=("forward_state_projection",),
    )

    assert not bool(restricted.tensors["token_embedding.weight"].any())
    assert bool(restricted.tensors["forward_state_projection.weight"].all())
    assert not bool(restricted.tensors["gates"].any())


def test_direction_prefix_must_match_parameter() -> None:
    direction = EggrollDirection({"gates": torch.ones(1)})
    with pytest.raises(ValueError, match="did not match"):
        MODULE.restrict_direction(
            direction,
            parameter_prefixes=("missing",),
        )


def test_parse_parameter_prefixes() -> None:
    assert MODULE.parse_parameter_prefixes(
        "state_projection, gates"
    ) == ("state_projection", "gates")
