"""Plot a chained learned-backward shortcut experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_chained_metrics(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load JSONL segments, letting later paths replace resumed generations."""

    by_generation: dict[int, dict[str, Any]] = {}
    for path in paths:
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
                by_generation[int(row["generation"])] = row
    if not by_generation:
        raise ValueError("no metric rows were loaded")
    return [by_generation[generation] for generation in sorted(by_generation)]


def _series(
    rows: Sequence[dict[str, Any]],
    key: str,
) -> tuple[list[int], list[float]]:
    points = [
        (int(row["generation"]), float(row[key]))
        for row in rows
        if key in row
    ]
    return [point[0] for point in points], [point[1] for point in points]


def _plot_series(
    axis: plt.Axes,
    rows: Sequence[dict[str, Any]],
    key: str,
    *,
    label: str,
    **kwargs: Any,
) -> None:
    generations, values = _series(rows, key)
    if generations:
        axis.plot(generations, values, label=label, **kwargs)


def plot_chained_metrics(
    rows: Sequence[dict[str, Any]],
    output_path: Path,
) -> None:
    """Render the main optimization and shortcut-resistance diagnostics."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    loss_axis, accuracy_axis, diversity_axis, eggroll_axis = axes.flat

    _plot_series(
        loss_axis,
        rows,
        "clean/loss_mean",
        label="post-training clean CE",
        color="#16697a",
    )
    _plot_series(
        loss_axis,
        rows,
        "center_rule/clean_loss",
        label="unperturbed centre clean CE",
        color="#003049",
        linestyle="--",
    )
    loss_axis.axhline(
        2.302585,
        color="#777777",
        linestyle=":",
        linewidth=1.2,
        label="uniform-digit CE",
    )
    fitness_axis = loss_axis.twinx()
    _plot_series(
        fitness_axis,
        rows,
        "fitness/mean",
        label="fitness",
        color="#d1495b",
        alpha=0.7,
    )
    loss_axis.set_title("Clean objective")
    loss_axis.set_ylabel("Cross-entropy")
    fitness_axis.set_ylabel("Initial CE - final CE")
    loss_handles, loss_labels = loss_axis.get_legend_handles_labels()
    fit_handles, fit_labels = fitness_axis.get_legend_handles_labels()
    loss_axis.legend(
        loss_handles + fit_handles,
        loss_labels + fit_labels,
        fontsize=8,
        loc="best",
    )

    for key, label, color in (
        ("correct_leak/accuracy_mean", "correct leak", "#2a9d8f"),
        ("clean/masked_accuracy_mean", "masked leak", "#e9c46a"),
        ("clean/incorrect_accuracy_mean", "incorrect leak", "#e76f51"),
    ):
        _plot_series(
            accuracy_axis,
            rows,
            key,
            label=label,
            color=color,
        )
    _plot_series(
        accuracy_axis,
        rows,
        "best/clean_accuracy",
        label="fittest candidate clean",
        color="#222222",
        linestyle="--",
    )
    _plot_series(
        accuracy_axis,
        rows,
        "robust/min_mode_accuracy",
        label="best candidate's weaker clean split",
        color="#7b2cbf",
        linestyle=":",
    )
    _plot_series(
        accuracy_axis,
        rows,
        "center_rule/min_mode_accuracy",
        label="unperturbed centre weaker clean split",
        color="#264653",
        linestyle="-.",
    )
    accuracy_axis.axhline(
        0.1,
        color="#555555",
        linestyle=":",
        linewidth=1.2,
        label="chance",
    )
    accuracy_axis.axhline(
        1 / 9,
        color="#9c6644",
        linestyle="--",
        linewidth=1.0,
        label="exclude-wrong-hint",
    )
    accuracy_axis.set_ylim(0, 1)
    accuracy_axis.set_title("Shortcut diagnostics")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.legend(fontsize=8, loc="upper left")

    _plot_series(
        diversity_axis,
        rows,
        "clean/unique_value_predictions_mean",
        label="distinct digit predictions",
        color="#6a4c93",
    )
    diversity_axis.set_ylim(0, 10.2)
    diversity_axis.set_ylabel("Distinct digits")
    mode_axis = diversity_axis.twinx()
    _plot_series(
        mode_axis,
        rows,
        "clean/prediction_mode_fraction_mean",
        label="modal fraction",
        color="#f77f00",
    )
    mode_axis.set_ylim(0, 1.02)
    mode_axis.set_ylabel("Modal prediction fraction")
    diversity_axis.set_title("Prediction diversity")
    div_handles, div_labels = diversity_axis.get_legend_handles_labels()
    mode_handles, mode_labels = mode_axis.get_legend_handles_labels()
    diversity_axis.legend(
        div_handles + mode_handles,
        div_labels + mode_labels,
        fontsize=8,
        loc="center right",
    )

    _plot_series(
        eggroll_axis,
        rows,
        "fitness/pair_delta_rms",
        label="antithetic pair delta RMS",
        color="#277da1",
    )
    _plot_series(
        eggroll_axis,
        rows,
        "backward/center_gate_abs_mean",
        label="center gate magnitude",
        color="#f94144",
    )
    _plot_series(
        eggroll_axis,
        rows,
        "backward/center_routing_leak_relative_gate",
        label="leak/other routing gate",
        color="#8c564b",
    )
    _plot_series(
        eggroll_axis,
        rows,
        "backward/best_fitness_leak_relative_gate",
        label="fittest candidate leak/other gate",
        color="#17becf",
        linestyle=":",
    )
    horizon_axis = eggroll_axis.twinx()
    _plot_series(
        horizon_axis,
        rows,
        "horizon",
        label="horizon",
        color="#43aa8b",
        drawstyle="steps-post",
        alpha=0.65,
    )
    sigma_generations, sigma_values = _series(rows, "search/sigma")
    previous_sigma: float | None = None
    for generation, sigma in zip(sigma_generations, sigma_values):
        if sigma == previous_sigma:
            continue
        eggroll_axis.axvline(
            generation,
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            label=f"sigma={sigma:g}",
        )
        previous_sigma = sigma
    eggroll_axis.set_title("Evolution and curriculum")
    eggroll_axis.set_ylabel("Magnitude")
    horizon_axis.set_ylabel("Inner updates")
    egg_handles, egg_labels = eggroll_axis.get_legend_handles_labels()
    horizon_handles, horizon_labels = horizon_axis.get_legend_handles_labels()
    eggroll_axis.legend(
        egg_handles + horizon_handles,
        egg_labels + horizon_labels,
        fontsize=8,
        loc="best",
    )

    for axis in axes.flat:
        axis.set_xlabel("EGGROLL generation")
        axis.grid(alpha=0.2)
    figure.suptitle("Learned backward rule for shortcut resistance")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="+",
        type=Path,
        help="ordered metrics.jsonl segments",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = load_chained_metrics(args.metrics)
    plot_chained_metrics(rows, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
