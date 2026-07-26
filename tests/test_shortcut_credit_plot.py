from __future__ import annotations

import json
from pathlib import Path

from list_sorting_transformer.shortcut_credit_plot import (
    load_chained_metrics,
    plot_chained_metrics,
)


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_chained_metrics_use_later_resumed_generation(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    resumed = tmp_path / "resumed.jsonl"
    write_rows(
        first,
        [
            {"generation": 0, "clean/loss_mean": 2.8},
            {"generation": 1, "clean/loss_mean": 2.7},
        ],
    )
    write_rows(
        resumed,
        [
            {"generation": 1, "clean/loss_mean": 2.6},
            {"generation": 2, "clean/loss_mean": 2.5},
        ],
    )

    rows = load_chained_metrics([first, resumed])

    assert [row["generation"] for row in rows] == [0, 1, 2]
    assert rows[1]["clean/loss_mean"] == 2.6


def test_plot_accepts_metrics_added_in_later_segments(tmp_path: Path) -> None:
    rows = [
        {
            "generation": 0,
            "horizon": 10,
            "clean/loss_mean": 2.8,
            "fitness/mean": 0.2,
            "correct_leak/accuracy_mean": 0.1,
            "clean/masked_accuracy_mean": 0.1,
            "clean/incorrect_accuracy_mean": 0.1,
        },
        {
            "generation": 1,
            "horizon": 20,
            "clean/loss_mean": 2.6,
            "fitness/mean": 0.4,
            "fitness/pair_delta_rms": 0.08,
            "backward/center_gate_abs_mean": 0.01,
            "correct_leak/accuracy_mean": 0.2,
            "clean/masked_accuracy_mean": 0.12,
            "clean/incorrect_accuracy_mean": 0.11,
            "clean/unique_value_predictions_mean": 3.0,
            "clean/prediction_mode_fraction_mean": 0.7,
            "search/sigma": 0.01,
        },
    ]
    output = tmp_path / "progress.png"

    plot_chained_metrics(rows, output)

    assert output.exists()
    assert output.stat().st_size > 1_000
