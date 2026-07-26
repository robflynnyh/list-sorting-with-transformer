"""Compare two learned backward-rule checkpoints on matched trajectories."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch

from list_sorting_transformer.shortcut_credit import (
    ShortcutPointerVocabulary,
    make_fitness_batches,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_forward_model,
    load_checkpoint,
    make_inner_batches,
    make_mode_batches,
    train_forward_trajectory,
    trajectory_summary,
)


def aggregate_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("at least one comparison row is required")
    deltas = [row["comparison/b_minus_a_min_accuracy"] for row in rows]
    loss_deltas = [
        row["comparison/a_minus_b_clean_loss"] for row in rows
    ]
    return {
        "replicates": float(len(rows)),
        "mean/checkpoint_a/min_mode_accuracy": statistics.mean(
            row["checkpoint_a/min_mode_accuracy"] for row in rows
        ),
        "mean/checkpoint_a/clean_loss": statistics.mean(
            row["checkpoint_a/clean_loss"] for row in rows
        ),
        "mean/checkpoint_b/min_mode_accuracy": statistics.mean(
            row["checkpoint_b/min_mode_accuracy"] for row in rows
        ),
        "mean/checkpoint_b/clean_loss": statistics.mean(
            row["checkpoint_b/clean_loss"] for row in rows
        ),
        "mean/comparison/b_minus_a_min_accuracy": statistics.mean(deltas),
        "mean/comparison/a_minus_b_clean_loss": statistics.mean(loss_deltas),
        "comparison/b_accuracy_win_fraction": (
            sum(delta > 0 for delta in deltas) / len(deltas)
        ),
        "comparison/a_accuracy_win_fraction": (
            sum(delta < 0 for delta in deltas) / len(deltas)
        ),
        "comparison/accuracy_tie_fraction": (
            sum(delta == 0 for delta in deltas) / len(deltas)
        ),
        "comparison/b_loss_win_fraction": (
            sum(delta > 0 for delta in loss_deltas) / len(loss_deltas)
        ),
    }


def comparable_config(
    checkpoint: dict[str, object],
    *,
    horizon: int,
    device: str,
) -> ShortcutCreditExperimentConfig:
    saved = ShortcutCreditExperimentConfig(**checkpoint["config"])
    return replace(
        saved,
        horizon=horizon,
        max_horizon=max(horizon, saved.max_horizon),
        device=device,
        resume=None,
        resume_horizon=None,
        wandb=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--generation-offset", type=int, default=1_000)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")

    device = torch.device(args.device)
    raw_a = torch.load(args.checkpoint_a, map_location=device)
    raw_b = torch.load(args.checkpoint_b, map_location=device)
    config_a = comparable_config(
        raw_a,
        horizon=args.horizon,
        device=args.device,
    )
    config_b = comparable_config(
        raw_b,
        horizon=args.horizon,
        device=args.device,
    )
    architecture_fields = (
        "d_model",
        "forward_layers",
        "heads",
        "batch_size",
        "fitness_examples",
        "fitness_batch_size",
        "correct_eval_examples",
        "min_length",
        "max_length",
        "leak_placement",
        "forward_learning_rate",
    )
    mismatches = [
        field
        for field in architecture_fields
        if getattr(config_a, field) != getattr(config_b, field)
    ]
    if mismatches:
        raise ValueError(
            "checkpoint experiment configurations differ: "
            + ", ".join(mismatches)
        )

    rule_a, _, _, _ = load_checkpoint(args.checkpoint_a, device=device)
    rule_b, _, _, _ = load_checkpoint(args.checkpoint_b, device=device)
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    rows: list[dict[str, float]] = []

    for replicate in range(
        args.replicate_start,
        args.replicate_start + args.replicates,
    ):
        generation = args.generation_offset + replicate
        generation_seed = config_a.seed * 1_000_003 + generation * 10_007
        base_model = initialize_forward_model(
            config_a,
            vocabulary,
            initialization_seed=generation_seed + 1,
            device=device,
        )
        base_state = {
            name: tensor.detach().clone()
            for name, tensor in base_model.state_dict().items()
        }
        del base_model
        inner_batches = make_inner_batches(
            config_a,
            horizon=args.horizon,
            vocabulary=vocabulary,
            generator=torch.Generator().manual_seed(generation_seed + 2),
            device=device,
        )
        evaluation_generator = torch.Generator().manual_seed(
            generation_seed + 4
        )
        clean_batches = make_fitness_batches(
            config_a.fitness_examples,
            min_length=config_a.min_length,
            max_length=config_a.max_length,
            batch_size=config_a.fitness_batch_size,
            generator=evaluation_generator,
            vocabulary=vocabulary,
            leak_placement=config_a.leak_placement,
            device=device,
        )
        correct_batches = make_mode_batches(
            config_a.correct_eval_examples,
            leak_mode="correct",
            config=config_a,
            vocabulary=vocabulary,
            generator=evaluation_generator,
            device=device,
        )
        trajectory_a = train_forward_trajectory(
            config_a,
            base_state=base_state,
            backward_rule=rule_a,
            inner_batches=inner_batches,
            fitness_batches=clean_batches,
            correct_batches=correct_batches,
            device=device,
        )
        trajectory_b = train_forward_trajectory(
            config_a,
            base_state=base_state,
            backward_rule=rule_b,
            inner_batches=inner_batches,
            fitness_batches=clean_batches,
            correct_batches=correct_batches,
            device=device,
        )
        row: dict[str, float] = {
            "replicate": float(replicate),
            "generation_seed": float(generation_seed),
        }
        row.update(
            trajectory_summary(
                "checkpoint_a",
                None,
                trajectory_a.clean,
                trajectory_a.correct,
            )
        )
        row.update(
            trajectory_summary(
                "checkpoint_b",
                None,
                trajectory_b.clean,
                trajectory_b.correct,
            )
        )
        row.update(
            {
                "comparison/b_minus_a_min_accuracy": (
                    row["checkpoint_b/min_mode_accuracy"]
                    - row["checkpoint_a/min_mode_accuracy"]
                ),
                "comparison/a_minus_b_clean_loss": (
                    row["checkpoint_a/clean_loss"]
                    - row["checkpoint_b/clean_loss"]
                ),
            }
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = aggregate_rows(rows)
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
