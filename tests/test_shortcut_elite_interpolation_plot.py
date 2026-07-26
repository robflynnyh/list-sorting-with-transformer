from __future__ import annotations

import json

import pytest

from experiments.learned_backward_shortcuts.elite_interpolation_plot import (
    load_point,
    parse_summary,
)


def test_load_point_recovers_matched_ordinary_accuracy(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "mean/center_rule/min_mode_accuracy": 0.8,
                "mean/comparison/center_minus_ordinary_min_accuracy": 0.6,
                "mean/center_rule/correct_leak_accuracy": 0.9,
                "mean/comparison/center_clean_loss_improvement_over_ordinary": 1.2,
                "comparison/center_accuracy_win_fraction": 1.0,
            }
        )
    )

    point = load_point(0.5, path, split="screen")

    assert point["ordinary_min_accuracy"] == pytest.approx(0.2)
    assert parse_summary(f"0.5={path}") == (0.5, path)
