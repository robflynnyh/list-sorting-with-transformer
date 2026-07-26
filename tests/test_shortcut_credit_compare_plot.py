from __future__ import annotations

import json
from pathlib import Path

import pytest

from list_sorting_transformer.shortcut_credit_compare_plot import (
    load_metrics,
    parse_series,
    plot_center_comparison,
    plot_matched_controls,
)


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_load_metrics_sorts_generations(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    write_rows(
        metrics,
        [
            {"generation": 2, "center_rule/clean_loss": 2.4},
            {"generation": 1, "center_rule/clean_loss": 2.5},
        ],
    )

    rows = load_metrics(metrics)

    assert [row["generation"] for row in rows] == [1, 2]


def test_parse_series_requires_label_and_path() -> None:
    assert parse_series("medium=metrics.jsonl") == (
        "medium",
        Path("metrics.jsonl"),
    )
    with pytest.raises(Exception):
        parse_series("metrics.jsonl")


def test_plot_center_comparison(tmp_path: Path) -> None:
    rows = [
        {
            "generation": 80,
            "center_rule/clean_loss": 2.9,
            "center_rule/masked_accuracy": 0.2,
            "center_rule/incorrect_accuracy": 0.01,
            "center_rule/correct_leak_accuracy": 0.98,
            "center_rule/min_mode_accuracy": 0.01,
            "robust/min_mode_accuracy": 0.12,
            "outer/update_to_center_rms": 0.02,
            "backward/center_routing_leak_relative_gate": 0.4,
        }
    ]
    output = tmp_path / "comparison.png"

    plot_center_comparison(
        [("conservative", rows), ("medium", rows)],
        output,
    )

    assert output.exists()
    assert output.stat().st_size > 1_000


def test_plot_matched_controls(tmp_path: Path) -> None:
    rows = [
        {
            "generation": 3,
            "center_rule/min_mode_accuracy": 0.2,
            "center_rule/clean_loss": 2.1,
            "center_rule/correct_leak_accuracy": 0.9,
            "ordinary_rule/min_mode_accuracy": 0.1,
            "ordinary_rule/clean_loss": 2.4,
            "ordinary_rule/correct_leak_accuracy": 1.0,
            "masked_training/min_mode_accuracy": 0.4,
            "masked_training/clean_loss": 1.8,
            "masked_training/correct_leak_accuracy": 0.6,
            "robust/min_mode_accuracy": 0.8,
            "robust/clean_loss": 0.7,
            "robust/correct_leak_accuracy": 1.0,
            "comparison/center_minus_ordinary_min_accuracy": 0.1,
            "comparison/center_clean_loss_improvement_over_ordinary": 0.3,
        }
    ]
    output = tmp_path / "matched.png"

    plot_matched_controls(rows, output)

    assert output.exists()
    assert output.stat().st_size > 1_000
