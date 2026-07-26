"""Compare unperturbed-centre learning across shortcut-credit runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path: Path) -> list[dict[str, Any]]:
    """Load generation-indexed JSONL metrics."""

    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSON"
                ) from error
            if "generation" not in row:
                raise ValueError(
                    f"{path}:{line_number} has no generation"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no metric rows")
    return sorted(rows, key=lambda row: int(row["generation"]))


def parse_series(value: str) -> tuple[str, Path]:
    """Parse a ``LABEL=PATH`` command-line series."""

    try:
        label, raw_path = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "series must use LABEL=PATH"
        ) from error
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("series must use LABEL=PATH")
    return label, Path(raw_path)


def _plot(
    axis: plt.Axes,
    rows: Sequence[dict[str, Any]],
    key: str,
    *,
    label: str,
    **kwargs: Any,
) -> None:
    points = [
        (int(row["generation"]), float(row[key]))
        for row in rows
        if key in row
    ]
    if points:
        if len(points) == 1 and "marker" not in kwargs:
            kwargs["marker"] = "o"
        axis.plot(
            [generation for generation, _ in points],
            [value for _, value in points],
            label=label,
            **kwargs,
        )


def plot_center_comparison(
    series: Sequence[tuple[str, Sequence[dict[str, Any]]]],
    output_path: Path,
) -> None:
    """Plot centre behavior separately from sampled-candidate behavior."""

    if not series:
        raise ValueError("at least one series is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
        constrained_layout=True,
    )
    ce_axis, split_axis, candidate_axis, update_axis = axes.flat
    colors = plt.get_cmap("tab10").colors

    for index, (label, rows) in enumerate(series):
        color = colors[index % len(colors)]
        _plot(
            ce_axis,
            rows,
            "center_rule/clean_loss",
            label=label,
            color=color,
        )
        for key, split, linestyle in (
            ("center_rule/masked_accuracy", "masked", "--"),
            ("center_rule/incorrect_accuracy", "wrong hint", "-"),
            ("center_rule/correct_leak_accuracy", "correct leak", ":"),
        ):
            _plot(
                split_axis,
                rows,
                key,
                label=f"{label}: {split}",
                color=color,
                linestyle=linestyle,
            )
        _plot(
            candidate_axis,
            rows,
            "center_rule/min_mode_accuracy",
            label=f"{label}: centre",
            color=color,
        )
        _plot(
            candidate_axis,
            rows,
            "robust/min_mode_accuracy",
            label=f"{label}: best candidate",
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        _plot(
            update_axis,
            rows,
            "outer/update_to_center_rms",
            label=f"{label}: update",
            color=color,
        )

    ce_axis.axhline(
        2.302585,
        color="#777777",
        linestyle=":",
        linewidth=1.2,
        label="uniform-digit CE",
    )
    ce_axis.set_title("Unperturbed-centre clean objective")
    ce_axis.set_ylabel("Cross-entropy")
    ce_axis.legend(fontsize=8)

    split_axis.axhline(
        0.1,
        color="#777777",
        linestyle="-.",
        linewidth=1.0,
        label="chance",
    )
    split_axis.set_title("Unperturbed-centre clean splits")
    split_axis.set_ylabel("Accuracy")
    split_axis.set_ylim(0, 1)
    split_axis.legend(fontsize=8, ncols=2)

    candidate_axis.axhline(
        0.1,
        color="#777777",
        linestyle=":",
        linewidth=1.0,
        label="chance",
    )
    candidate_axis.set_title("Learned centre vs sampled candidates")
    candidate_axis.set_ylabel("Weaker clean-split accuracy")
    candidate_axis.set_ylim(0, 1)
    candidate_axis.legend(fontsize=8)

    route_axis = update_axis.twinx()
    for index, (label, rows) in enumerate(series):
        color = colors[index % len(colors)]
        _plot(
            route_axis,
            rows,
            "backward/center_routing_leak_relative_gate",
            label=f"{label}: leak/other gate",
            color=color,
            linestyle="--",
            alpha=0.75,
        )
    update_axis.set_title("Outer update and routing selectivity")
    update_axis.set_ylabel("Update / centre RMS")
    route_axis.set_ylabel("Leak / other routing gate")
    update_handles, update_labels = update_axis.get_legend_handles_labels()
    route_handles, route_labels = route_axis.get_legend_handles_labels()
    update_axis.legend(
        update_handles + route_handles,
        update_labels + route_labels,
        fontsize=8,
    )

    for axis in axes.flat:
        axis.set_xlabel("EGGROLL generation")
        axis.grid(alpha=0.2)
    figure.suptitle("Horizon-80 outer-step comparison")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_matched_controls(
    rows: Sequence[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot the learned centre against same-generation training controls."""

    if not rows:
        raise ValueError("at least one metric row is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
        constrained_layout=True,
    )
    accuracy_axis, loss_axis, correct_axis, delta_axis = axes.flat
    use_heldout = any(
        "heldout_center_rule/min_mode_accuracy" in row
        for row in rows
    )
    heldout_prefix = "heldout_" if use_heldout else ""
    trajectories = (
        (
            "Evolved centre",
            f"{heldout_prefix}center_rule",
            "#1f77b4",
            "-",
        ),
        (
            "Ordinary leak training",
            f"{heldout_prefix}ordinary_rule",
            "#555555",
            "--",
        ),
        (
            "Masked training",
            f"{heldout_prefix}masked_training",
            "#2ca02c",
            "-.",
        ),
        (
            (
                "Most robust sampled rule (outer set)"
                if use_heldout
                else "Most robust sampled rule"
            ),
            "robust",
            "#d62728",
            ":",
        ),
    )

    for label, prefix, color, linestyle in trajectories:
        _plot(
            accuracy_axis,
            rows,
            f"{prefix}/min_mode_accuracy",
            label=label,
            color=color,
            linestyle=linestyle,
        )
        _plot(
            loss_axis,
            rows,
            f"{prefix}/clean_loss",
            label=label,
            color=color,
            linestyle=linestyle,
        )
        _plot(
            correct_axis,
            rows,
            f"{prefix}/correct_leak_accuracy",
            label=label,
            color=color,
            linestyle=linestyle,
        )

    for axis, title, ylabel in (
        (
            accuracy_axis,
            "Generalization without a reliable hint",
            "Weaker clean-split accuracy",
        ),
        (loss_axis, "Clean objective", "Cross-entropy"),
        (
            correct_axis,
            "Shortcut-present behavior",
            "Correct-hint accuracy",
        ),
    ):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
    accuracy_axis.set_ylim(0, 1)
    correct_axis.set_ylim(0, 1)
    accuracy_axis.axhline(
        0.1,
        color="#999999",
        linestyle=":",
        linewidth=1,
        label="chance",
    )

    _plot(
        delta_axis,
        rows,
        (
            "heldout_comparison/center_minus_ordinary_min_accuracy"
            if use_heldout
            else "comparison/center_minus_ordinary_min_accuracy"
        ),
        label="Accuracy advantage",
        color="#1f77b4",
    )
    loss_delta_axis = delta_axis.twinx()
    _plot(
        loss_delta_axis,
        rows,
        (
            "heldout_comparison/center_clean_loss_improvement_over_ordinary"
            if use_heldout
            else "comparison/center_clean_loss_improvement_over_ordinary"
        ),
        label="Clean-loss improvement",
        color="#ff7f0e",
        linestyle="--",
    )
    delta_axis.axhline(0, color="#777777", linewidth=1)
    delta_axis.set_title("Evolved centre minus ordinary training")
    delta_axis.set_ylabel("Weak-split accuracy difference")
    loss_delta_axis.set_ylabel("Clean CE reduction")
    delta_handles, delta_labels = delta_axis.get_legend_handles_labels()
    loss_handles, loss_labels = loss_delta_axis.get_legend_handles_labels()
    delta_axis.legend(
        delta_handles + loss_handles,
        delta_labels + loss_labels,
        fontsize=8,
    )

    for axis in axes.flat:
        axis.set_xlabel("EGGROLL generation")
        axis.grid(alpha=0.2)
    figure.suptitle(
        (
            "Matched shortcut-learning controls: fresh held-out evaluation"
            if use_heldout
            else "Matched shortcut-learning controls"
        )
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--series",
        action="append",
        type=parse_series,
        metavar="LABEL=PATH",
    )
    inputs.add_argument(
        "--matched-controls",
        type=Path,
        metavar="PATH",
        help="plot one run containing same-generation control metrics",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.matched_controls is not None:
        plot_matched_controls(
            load_metrics(args.matched_controls),
            args.output,
        )
        print(args.output)
        return
    series = [
        (label, load_metrics(path))
        for label, path in args.series or []
    ]
    plot_center_comparison(series, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
