"""Replicate a learned backward-rule checkpoint against matched controls."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch

from list_sorting_transformer.shortcut_learning.shortcut_credit import (
    ShortcutPointerVocabulary,
    make_fitness_batches,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit_experiment import (
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
        raise ValueError("at least one replication row is required")
    metric_keys = (
        "center_rule/min_mode_accuracy",
        "center_rule/correct_leak_accuracy",
        "center_rule/clean_loss",
        "ordinary_rule/min_mode_accuracy",
        "ordinary_rule/correct_leak_accuracy",
        "ordinary_rule/clean_loss",
        "masked_training/min_mode_accuracy",
        "masked_training/correct_leak_accuracy",
        "masked_training/clean_loss",
        "comparison/center_minus_ordinary_min_accuracy",
        "comparison/center_clean_loss_improvement_over_ordinary",
    )
    summary = {
        f"mean/{key}": statistics.mean(row[key] for row in rows)
        for key in metric_keys
    }
    accuracy_differences = [
        row["comparison/center_minus_ordinary_min_accuracy"]
        for row in rows
    ]
    loss_differences = [
        row["comparison/center_clean_loss_improvement_over_ordinary"]
        for row in rows
    ]
    summary.update(
        {
            "replicates": float(len(rows)),
            "comparison/center_accuracy_win_fraction": (
                sum(value > 0 for value in accuracy_differences) / len(rows)
            ),
            "comparison/center_loss_win_fraction": (
                sum(value > 0 for value in loss_differences) / len(rows)
            ),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--generation-offset", type=int, default=1_000)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_config = ShortcutCreditExperimentConfig(**checkpoint["config"])
    backward_rule, _, checkpoint_horizon, _ = load_checkpoint(
        args.checkpoint,
        device=device,
    )
    horizon = checkpoint_horizon if args.horizon is None else args.horizon
    config = replace(
        saved_config,
        horizon=horizon,
        max_horizon=max(horizon, saved_config.max_horizon),
        device=args.device,
        resume=None,
        resume_horizon=None,
        wandb=False,
    )
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    rows: list[dict[str, float]] = []
    for replicate in range(
        args.replicate_start,
        args.replicate_start + args.replicates,
    ):
        generation = args.generation_offset + replicate
        generation_seed = config.seed * 1_000_003 + generation * 10_007
        base_model = initialize_forward_model(
            config,
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
            config,
            horizon=horizon,
            vocabulary=vocabulary,
            generator=torch.Generator().manual_seed(generation_seed + 2),
            device=device,
        )
        masked_inner_batches = make_inner_batches(
            config,
            horizon=horizon,
            vocabulary=vocabulary,
            generator=torch.Generator().manual_seed(generation_seed + 2),
            device=device,
            leak_mode="masked",
        )
        evaluation_generator = torch.Generator().manual_seed(
            generation_seed + 4
        )
        clean_batches = make_fitness_batches(
            config.fitness_examples,
            min_length=config.min_length,
            max_length=config.max_length,
            batch_size=config.fitness_batch_size,
            generator=evaluation_generator,
            vocabulary=vocabulary,
            leak_placement=config.leak_placement,
            device=device,
        )
        correct_batches = make_mode_batches(
            config.correct_eval_examples,
            leak_mode="correct",
            config=config,
            vocabulary=vocabulary,
            generator=evaluation_generator,
            device=device,
        )
        center = train_forward_trajectory(
            config,
            base_state=base_state,
            backward_rule=backward_rule,
            inner_batches=inner_batches,
            fitness_batches=clean_batches,
            correct_batches=correct_batches,
            device=device,
        )
        ordinary = train_forward_trajectory(
            config,
            base_state=base_state,
            backward_rule=None,
            inner_batches=inner_batches,
            fitness_batches=clean_batches,
            correct_batches=correct_batches,
            device=device,
        )
        masked = train_forward_trajectory(
            config,
            base_state=base_state,
            backward_rule=None,
            inner_batches=masked_inner_batches,
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
                "center_rule",
                None,
                center.clean,
                center.correct,
            )
        )
        row.update(
            trajectory_summary(
                "ordinary_rule",
                None,
                ordinary.clean,
                ordinary.correct,
            )
        )
        row.update(
            trajectory_summary(
                "masked_training",
                None,
                masked.clean,
                masked.correct,
            )
        )
        row.update(
            {
                "comparison/center_minus_ordinary_min_accuracy": (
                    row["center_rule/min_mode_accuracy"]
                    - row["ordinary_rule/min_mode_accuracy"]
                ),
                "comparison/center_clean_loss_improvement_over_ordinary": (
                    row["ordinary_rule/clean_loss"]
                    - row["center_rule/clean_loss"]
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
