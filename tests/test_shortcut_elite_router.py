from __future__ import annotations

import pytest
import torch

from experiments.learned_backward_shortcuts.elite_router_checkpoint import (
    checkpoint_search_sigma,
    elite_parameter_delta,
    parse_indices,
)
from list_sorting_transformer.shortcut_credit import EggrollDirection
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
)


def test_parse_indices_requires_unique_nonnegative_values() -> None:
    assert parse_indices("2,5,7") == (2, 5, 7)
    with pytest.raises(Exception):
        parse_indices("2,2")
    with pytest.raises(Exception):
        parse_indices("-1,2")


def test_elite_parameter_delta_respects_antithetic_signs() -> None:
    directions = (
        EggrollDirection({"weight": torch.tensor([1.0, 2.0])}),
        EggrollDirection({"weight": torch.tensor([3.0, 4.0])}),
    )

    delta = elite_parameter_delta(
        directions,
        (0, 3),
        parameter_name="weight",
        sigma=0.5,
    )

    torch.testing.assert_close(
        delta,
        torch.tensor([-0.5, -0.5]),
    )


def test_checkpoint_search_sigma_prefers_adaptive_state() -> None:
    config = ShortcutCreditExperimentConfig(sigma=0.2)

    assert checkpoint_search_sigma({}, config) == pytest.approx(0.2)
    assert checkpoint_search_sigma(
        {"plateau_state": {"search_sigma": 0.05}},
        config,
    ) == pytest.approx(0.05)


def test_backtracking_config_rejects_invalid_sigma_bounds() -> None:
    with pytest.raises(ValueError, match="sigma_decay"):
        ShortcutCreditExperimentConfig(
            elite_rejection_sigma_decay=1.0,
        )
    with pytest.raises(ValueError, match="elite_min_sigma"):
        ShortcutCreditExperimentConfig(
            sigma=0.1,
            elite_min_sigma=0.2,
        )
    with pytest.raises(ValueError, match="sigma_growth"):
        ShortcutCreditExperimentConfig(
            elite_acceptance_sigma_growth=1.0,
        )
