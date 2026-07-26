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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        type=parse_series,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    series = [
        (label, load_metrics(path))
        for label, path in args.series
    ]
    plot_center_comparison(series, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
