from __future__ import annotations

import pytest

from experiments.learned_backward_shortcuts.checkpoint_replication_diagnostic import (
    aggregate_rows,
)


def test_aggregate_rows_reports_paired_control_advantages() -> None:
    rows = [
        {
            "center_rule/min_mode_accuracy": 0.4,
            "center_rule/correct_leak_accuracy": 0.8,
            "center_rule/clean_loss": 1.2,
            "ordinary_rule/min_mode_accuracy": 0.1,
            "ordinary_rule/correct_leak_accuracy": 1.0,
            "ordinary_rule/clean_loss": 2.0,
            "masked_training/min_mode_accuracy": 0.5,
            "masked_training/correct_leak_accuracy": 0.7,
            "masked_training/clean_loss": 1.0,
            "comparison/center_minus_ordinary_min_accuracy": 0.3,
            "comparison/center_clean_loss_improvement_over_ordinary": 0.8,
        },
        {
            "center_rule/min_mode_accuracy": 0.0,
            "center_rule/correct_leak_accuracy": 1.0,
            "center_rule/clean_loss": 2.5,
            "ordinary_rule/min_mode_accuracy": 0.1,
            "ordinary_rule/correct_leak_accuracy": 1.0,
            "ordinary_rule/clean_loss": 2.2,
            "masked_training/min_mode_accuracy": 0.4,
            "masked_training/correct_leak_accuracy": 0.6,
            "masked_training/clean_loss": 1.1,
            "comparison/center_minus_ordinary_min_accuracy": -0.1,
            "comparison/center_clean_loss_improvement_over_ordinary": -0.3,
        },
    ]

    summary = aggregate_rows(rows)

    assert summary["replicates"] == 2
    assert summary[
        "mean/comparison/center_minus_ordinary_min_accuracy"
    ] == pytest.approx(0.1)
    assert summary[
        "mean/comparison/center_clean_loss_improvement_over_ordinary"
    ] == pytest.approx(0.25)
    assert summary["comparison/center_accuracy_win_fraction"] == 0.5
    assert summary["comparison/center_loss_win_fraction"] == 0.5


def test_aggregate_rows_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        aggregate_rows([])
