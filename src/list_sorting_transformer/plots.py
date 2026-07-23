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
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(steps, [row["loss"] for row in history], color="#1f5f8b")
    axes[0].set(title="Training loss", xlabel="Step", ylabel="Cross-entropy")
    axes[1].plot(
        steps,
        [row["token_accuracy"] for row in history],
        color="#b24726",
    )
    axes[1].set(
        title="Teacher-forced training accuracy",
        xlabel="Step",
        ylabel="Token accuracy",
        ylim=(0.0, 1.01),
    )
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
        title="Sorting performance by length",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
