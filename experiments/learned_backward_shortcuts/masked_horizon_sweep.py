"""Measure how long clean pointer learning takes with the shortcut masked."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch

from list_sorting_transformer.shortcut_learning.shortcut_credit import (
    ShortcutPointerVocabulary,
    evaluate_shortcut_batches,
    make_fitness_batches,
    shortcut_loss,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_forward_model,
    make_inner_batches,
    make_mode_batches,
)


def parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(sorted({int(item) for item in value.split(",")}))
    if not horizons or horizons[0] < 1:
        raise argparse.ArgumentTypeError(
            "horizons must be positive comma-separated integers"
        )
    return horizons


def aggregate_horizon_rows(
    rows: list[dict[str, float]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("at least one sweep row is required")
    horizons = sorted({int(row["horizon"]) for row in rows})
    metrics = (
        "clean_loss",
        "min_mode_accuracy",
        "masked_accuracy",
        "incorrect_accuracy",
        "correct_leak_accuracy",
    )
    by_horizon = {}
    for horizon in horizons:
        horizon_rows = [
            row for row in rows if int(row["horizon"]) == horizon
        ]
        by_horizon[str(horizon)] = {
            f"mean/{metric}": statistics.mean(
                row[metric] for row in horizon_rows
            )
            for metric in metrics
        }
    return {
        "replicates": len({int(row["replicate"]) for row in rows}),
        "horizons": by_horizon,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=parse_horizons("160,320,640,1280,2000,3000"),
    )
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--generation-offset", type=int, default=60_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")

    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    saved_config = ShortcutCreditExperimentConfig(**checkpoint["config"])
    max_horizon = max(args.horizons)
    config = replace(
        saved_config,
        horizon=max_horizon,
        max_horizon=max(max_horizon, saved_config.max_horizon),
        device=args.device,
        resume=None,
        resume_horizon=None,
        wandb=False,
    )
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    rows: list[dict[str, float]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for replicate in range(
        args.replicate_start,
        args.replicate_start + args.replicates,
    ):
        generation = args.generation_offset + replicate
        generation_seed = config.seed * 1_000_003 + generation * 10_007
        model = initialize_forward_model(
            config,
            vocabulary,
            initialization_seed=generation_seed + 1,
            device=device,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.forward_learning_rate,
        )
        training_generator = torch.Generator().manual_seed(
            generation_seed + 2
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

        completed_steps = 0
        for horizon in args.horizons:
            model.train()
            for _ in range(horizon - completed_steps):
                batch = make_inner_batches(
                    config,
                    horizon=1,
                    vocabulary=vocabulary,
                    generator=training_generator,
                    device=device,
                    leak_mode="masked",
                )[0]
                optimizer.zero_grad(set_to_none=True)
                loss = shortcut_loss(model, batch)
                loss.backward()
                optimizer.step()
            completed_steps = horizon

            clean = evaluate_shortcut_batches(model, clean_batches)
            correct = evaluate_shortcut_batches(model, correct_batches)
            row = {
                "replicate": float(replicate),
                "generation_seed": float(generation_seed),
                "horizon": float(horizon),
                "clean_loss": clean.loss,
                "min_mode_accuracy": min(clean.mode_accuracy.values()),
                "masked_accuracy": clean.mode_accuracy["masked"],
                "incorrect_accuracy": clean.mode_accuracy["incorrect"],
                "correct_leak_accuracy": correct.accuracy,
            }
            rows.append(row)
            with args.output.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)

        del model, optimizer

    summary = aggregate_horizon_rows(rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
