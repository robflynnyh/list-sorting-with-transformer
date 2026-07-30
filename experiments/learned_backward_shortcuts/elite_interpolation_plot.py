"""Plot matched replication results for elite-centroid interpolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_summary(value: str) -> tuple[float, Path]:
    try:
        raw_alpha, raw_path = value.split("=", maxsplit=1)
        alpha = float(raw_alpha)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "summary must use ALPHA=PATH"
        ) from error
    if not 0 <= alpha <= 1 or not raw_path:
        raise argparse.ArgumentTypeError(
            "summary alpha must be in [0, 1] and path nonempty"
        )
    return alpha, Path(raw_path)


def load_point(
    alpha: float,
    path: Path,
    *,
    split: str,
) -> dict[str, float | str]:
    summary = json.loads(path.read_text())
    center = summary["mean/center_rule/min_mode_accuracy"]
    advantage = summary[
        "mean/comparison/center_minus_ordinary_min_accuracy"
    ]
    return {
        "alpha": alpha,
        "split": split,
        "center_min_accuracy": center,
        "ordinary_min_accuracy": center - advantage,
        "correct_hint_accuracy": summary[
            "mean/center_rule/correct_leak_accuracy"
        ],
        "clean_ce_improvement": summary[
            "mean/comparison/center_clean_loss_improvement_over_ordinary"
        ],
        "accuracy_win_fraction": summary[
            "comparison/center_accuracy_win_fraction"
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen-summary",
        action="append",
        type=parse_summary,
        default=[],
    )
    parser.add_argument(
        "--validation-summary",
        action="append",
        type=parse_summary,
        default=[],
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.screen_summary:
        raise ValueError("at least one screen summary is required")

    points = [
        load_point(alpha, path, split="five-seed screen")
        for alpha, path in args.screen_summary
    ]
    points.extend(
        load_point(alpha, path, split="20-seed validation")
        for alpha, path in args.validation_summary
    )
    points.sort(key=lambda point: (point["split"], point["alpha"]))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(points, indent=2) + "\n")

    screen = [
        point for point in points if point["split"] == "five-seed screen"
    ]
    validation = [
        point for point in points if point["split"] == "20-seed validation"
    ]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        constrained_layout=True,
    )
    axes[0].plot(
        [point["alpha"] for point in screen],
        [point["center_min_accuracy"] for point in screen],
        marker="o",
        label="elite interpolation",
    )
    axes[0].plot(
        [point["alpha"] for point in screen],
        [point["ordinary_min_accuracy"] for point in screen],
        linestyle="--",
        color="#666666",
        label="ordinary training",
    )
    axes[1].plot(
        [point["alpha"] for point in screen],
        [point["clean_ce_improvement"] for point in screen],
        marker="o",
    )
    for axis, metric in zip(
        axes,
        ("center_min_accuracy", "clean_ce_improvement"),
    ):
        if validation:
            axis.scatter(
                [point["alpha"] for point in validation],
                [point[metric] for point in validation],
                marker="*",
                s=150,
                color="#d62728",
                label="20-seed validation",
                zorder=3,
            )
        axis.axhline(0, color="#999999", linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Interpolation toward top-eight centroid")
    axes[0].set_ylabel("Fresh weaker-split accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Shortcut-resistant learning")
    axes[0].legend()
    axes[1].set_ylabel("Clean CE improvement over ordinary")
    axes[1].set_title("Matched objective improvement")
    axes[1].legend()
    figure.savefig(args.output_plot, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
