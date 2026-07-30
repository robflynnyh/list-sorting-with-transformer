"""Plot centre and locally evolved rule trajectories around two collapses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_accuracy(path: Path) -> tuple[list[int], list[float]]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    steps = []
    accuracies = []
    for row in rows:
        if "absolute_step" in row:
            steps.append(int(row["absolute_step"]))
            accuracies.append(float(row["min_mode_accuracy"]))
        elif row.get("horizon", 0):
            steps.append(int(row["horizon"]))
            accuracies.append(float(row["fixed/min_mode_accuracy"]))
    return steps, accuracies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/learned_backward_shortcuts/results"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panels = (
        (
            "Seed 57,155,105",
            args.results_dir
            / "collapse_window_seed57155105_center.jsonl",
            args.results_dir
            / "collapse_window_seed57155105_candidate63_full.jsonl",
            (2960, 3050),
        ),
        (
            "Seed 7,700,511",
            args.results_dir
            / "collapse_window_seed7700511_center.jsonl",
            args.results_dir
            / "collapse_window_seed7700511_candidate63_full.jsonl",
            (2880, 3050),
        ),
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.2),
        sharey=True,
    )
    for axis, (title, center_path, candidate_path, x_limits) in zip(
        axes,
        panels,
    ):
        center_steps, center_accuracy = load_accuracy(center_path)
        candidate_steps, candidate_accuracy = load_accuracy(candidate_path)
        axis.plot(
            center_steps,
            [100 * value for value in center_accuracy],
            color="#b33a3a",
            linewidth=2.0,
            marker="o",
            markersize=2.8,
            label="Accepted centre",
        )
        axis.plot(
            candidate_steps,
            [100 * value for value in candidate_accuracy],
            color="#19745d",
            linewidth=2.0,
            marker="o",
            markersize=2.8,
            label="Collapse-window candidate 63",
        )
        axis.set_title(title)
        axis.set_xlabel("Forward optimizer update")
        axis.set_xlim(*x_limits)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Minimum clean-mode accuracy (%)")
    axes[0].legend(loc="lower left", frameon=False)
    figure.suptitle(
        "A shared locally evolved backward rule softens both collapse events"
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
