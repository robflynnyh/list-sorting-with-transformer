from __future__ import annotations

import pytest

from experiments.learned_backward_shortcuts.masked_horizon_sweep import (
    aggregate_horizon_rows,
    parse_horizons,
)


def test_parse_horizons_sorts_and_deduplicates() -> None:
    assert parse_horizons("640,160,320,320") == (160, 320, 640)
    with pytest.raises(Exception):
        parse_horizons("0,160")


def test_aggregate_horizon_rows_groups_matched_replicates() -> None:
    rows = []
    for replicate, offset in ((0, 0.0), (1, 0.2)):
        for horizon, accuracy in ((160, 0.5), (320, 0.8)):
            rows.append(
                {
                    "replicate": float(replicate),
                    "horizon": float(horizon),
                    "clean_loss": 1.0 - accuracy + offset,
                    "min_mode_accuracy": accuracy + offset,
                    "masked_accuracy": accuracy + offset,
                    "incorrect_accuracy": accuracy + offset,
                    "correct_leak_accuracy": accuracy + offset,
                }
            )

    summary = aggregate_horizon_rows(rows)

    assert summary["replicates"] == 2
    horizons = summary["horizons"]
    assert isinstance(horizons, dict)
    assert horizons["160"]["mean/min_mode_accuracy"] == pytest.approx(0.6)
    assert horizons["320"]["mean/min_mode_accuracy"] == pytest.approx(0.9)
