"""Small plotting helpers for reproducible experiment artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_history(
    history: Sequence[dict[str, float]],
    output_path: Path,
) -> None:
    if not history:
        return
    steps = [row["step"] for row in history]
    accuracy_key = (
        "token_accuracy"
        if "token_accuracy" in history[0]
        else "argmax_accuracy"
        if "argmax_accuracy" in history[0]
        else None
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(steps, [row["loss"] for row in history], color="#1f5f8b")
    axes[0].set(title="Training loss", xlabel="Step", ylabel="Loss")
    if accuracy_key is not None:
        axes[1].plot(
            steps,
            [row[accuracy_key] for row in history],
            color="#b24726",
        )
        axes[1].set(
            title="Training accuracy",
            xlabel="Step",
            ylabel=accuracy_key.replace("_", " ").title(),
            ylim=(0.0, 1.01),
        )
    else:
        axes[1].axis("off")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_length_generalization(
    per_length: dict[int, dict[str, float]],
    output_path: Path,
    *,
    train_max_length: int,
) -> None:
    lengths = sorted(per_length)
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    first_metrics = per_length[lengths[0]]
    title = "Sorting performance by length"
    if "execution_completed" in first_metrics:
        series = (
            ("exact_match", "Exact executed sort", "#1f5f8b"),
            (
                "operation_prefix_fraction",
                "Valid operation prefix",
                "#20854e",
            ),
            ("target_token_accuracy", "Generated-action accuracy", "#b24726"),
        )
    elif "next_value_accuracy" in first_metrics:
        series = (
            ("exact_match", "Exact next value", "#1f5f8b"),
            ("next_value_accuracy", "Next-value token", "#20854e"),
            ("target_token_accuracy", "Generated-token accuracy", "#b24726"),
        )
    elif "value_accuracy" in first_metrics:
        title = "Pointer-value performance by length"
        series = (
            ("exact_match", "Exact marked value", "#1f5f8b"),
            ("value_accuracy", "Marked-value token", "#20854e"),
            ("target_token_accuracy", "Generated-token accuracy", "#b24726"),
        )
    elif "argmax_accuracy" in first_metrics:
        title = "Pointer-position performance by length"
        series = (
            ("argmax_accuracy", "Pointer argmax accuracy", "#1f5f8b"),
            ("seen_argmax_accuracy", "Seen pointer positions", "#20854e"),
            ("unseen_argmax_accuracy", "Unseen pointer positions", "#b24726"),
        )
    else:
        series = (
            ("exact_match", "Exact match", "#1f5f8b"),
            ("multiset_preserved", "Same multiset", "#20854e"),
            ("target_token_accuracy", "Generated-token accuracy", "#b24726"),
        )
    for key, label, color in series:
        axis.plot(
            lengths,
            [per_length[length][key] for length in lengths],
            marker="o",
            markersize=2.5,
            linewidth=1.5,
            label=label,
            color=color,
        )
    axis.axvline(
        train_max_length + 0.5,
        linestyle="--",
        linewidth=1.2,
        color="#555555",
        label="Training-length boundary",
    )
    axis.set(
        xlabel="Input list length",
        ylabel="Fraction",
        ylim=(-0.02, 1.02),
        title=title,
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_representation_comparison(
    runs: Sequence[tuple[str, dict[int, dict[str, float]]]],
    output_path: Path,
    *,
    train_max_length: int,
) -> None:
    colors = ("#237a57", "#b05a2a", "#456a9d", "#755181")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
    panels = (
        ("exact_match", "Exact sequence accuracy"),
        ("valid_syntax", "Valid output syntax"),
    )
    for axis, (metric, title) in zip(axes, panels):
        for (label, per_length), color in zip(runs, colors):
            lengths = sorted(per_length)
            axis.plot(
                lengths,
                [per_length[length][metric] for length in lengths],
                marker="o",
                markersize=2.5,
                linewidth=1.6,
                label=label,
                color=color,
            )
        axis.axvline(
            train_max_length + 0.5,
            linestyle="--",
            linewidth=1.2,
            color="#555555",
        )
        axis.set(
            title=title,
            xlabel="Input list length",
            ylabel="Fraction",
            ylim=(-0.02, 1.02),
        )
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="lower left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_learning_dynamics(
    runs: Sequence[
        tuple[str, Sequence[tuple[int, dict[int, dict[str, float]]]]]
    ],
    output_path: Path,
    *,
    lengths: tuple[int, ...] = (20, 25),
) -> None:
    colors = ("#237a57", "#b05a2a", "#456a9d", "#755181")
    figure, axes = plt.subplots(1, len(lengths), figsize=(10.5, 4.0), sharey=True)
    if len(lengths) == 1:
        axes = [axes]
    for axis, length in zip(axes, lengths):
        for (label, history), color in zip(runs, colors):
            selected = [
                (step, per_length[length]["exact_match"])
                for step, per_length in history
                if length in per_length
            ]
            axis.plot(
                [step for step, _ in selected],
                [value for _, value in selected],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=label,
                color=color,
            )
        axis.set(
            title=f"Exact accuracy at length {length}",
            xlabel="Training step",
            ylabel="Fraction",
            ylim=(-0.02, 1.02),
        )
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
