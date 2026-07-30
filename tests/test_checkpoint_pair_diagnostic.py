import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "learned_backward_shortcuts"
    / "checkpoint_pair_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location(
    "checkpoint_pair_diagnostic",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aggregate_rows_reports_paired_wins_and_ties() -> None:
    rows = [
        {
            "checkpoint_a/min_mode_accuracy": 0.8,
            "checkpoint_a/clean_loss": 0.4,
            "checkpoint_b/min_mode_accuracy": 0.9,
            "checkpoint_b/clean_loss": 0.3,
            "comparison/b_minus_a_min_accuracy": 0.1,
            "comparison/a_minus_b_clean_loss": 0.1,
        },
        {
            "checkpoint_a/min_mode_accuracy": 0.9,
            "checkpoint_a/clean_loss": 0.2,
            "checkpoint_b/min_mode_accuracy": 0.9,
            "checkpoint_b/clean_loss": 0.25,
            "comparison/b_minus_a_min_accuracy": 0.0,
            "comparison/a_minus_b_clean_loss": -0.05,
        },
    ]

    summary = MODULE.aggregate_rows(rows)

    assert summary["replicates"] == 2.0
    assert summary["mean/checkpoint_a/min_mode_accuracy"] == pytest.approx(
        0.85
    )
    assert summary["mean/checkpoint_b/min_mode_accuracy"] == pytest.approx(
        0.9
    )
    assert summary["mean/comparison/b_minus_a_min_accuracy"] == pytest.approx(
        0.05
    )
    assert summary["comparison/b_accuracy_win_fraction"] == 0.5
    assert summary["comparison/a_accuracy_win_fraction"] == 0.0
    assert summary["comparison/accuracy_tie_fraction"] == 0.5
    assert summary["comparison/b_loss_win_fraction"] == 0.5


def test_aggregate_rows_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one comparison row"):
        MODULE.aggregate_rows([])


def test_parse_generation_seeds_requires_unique_nonnegative_values() -> None:
    assert MODULE.parse_generation_seeds("7,11,13") == (7, 11, 13)
    with pytest.raises(
        MODULE.argparse.ArgumentTypeError,
        match="unique nonnegative",
    ):
        MODULE.parse_generation_seeds("7,7")
    with pytest.raises(
        MODULE.argparse.ArgumentTypeError,
        match="unique nonnegative",
    ):
        MODULE.parse_generation_seeds("-1,7")
