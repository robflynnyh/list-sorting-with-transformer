"""Replay a saved collapse window with a selected backward-rule checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from list_sorting_transformer.shortcut_collapse_window import (
    load_collapse_window,
    replay_collapse_window,
)
from list_sorting_transformer.shortcut_credit import (
    ShortcutPointerVocabulary,
    make_fitness_batches,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    load_checkpoint,
    parse_fitness_checkpoints,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_steps = parse_fitness_checkpoints(args.checkpoints)
    device = torch.device(args.device)
    raw_checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    saved_config = ShortcutCreditExperimentConfig(
        **raw_checkpoint["config"]
    )
    window = load_collapse_window(args.window)
    config = replace(
        saved_config,
        horizon=len(window.batches),
        max_horizon=max(len(window.batches), saved_config.max_horizon),
        device=args.device,
        resume=None,
        resume_horizon=None,
        wandb=False,
    )
    backward_rule, _, _, _ = load_checkpoint(
        args.checkpoint,
        device=device,
    )
    backward_rule.requires_grad_(False)

    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    fitness_generator = torch.Generator().manual_seed(config.seed + 10_000)
    fitness_batches = make_fitness_batches(
        config.fitness_examples,
        min_length=config.min_length,
        max_length=config.max_length,
        batch_size=config.fitness_batch_size,
        generator=fitness_generator,
        vocabulary=vocabulary,
        leak_placement=config.leak_placement,
        device=device,
    )
    replay = replay_collapse_window(
        config,
        window=window,
        backward_rule=backward_rule,
        fitness_batches=fitness_batches,
        device=device,
        checkpoint_steps=checkpoint_steps,
        evaluation_batch_size=config.fitness_examples,
    )
    rows = []
    for relative_step, metrics in replay.checkpoint_metrics:
        rows.append(
            {
                "relative_step": relative_step,
                "absolute_step": window.start_step + relative_step,
                "clean_loss": metrics.loss,
                "min_mode_accuracy": min(
                    metrics.mode_accuracy.values()
                ),
                "masked_accuracy": metrics.mode_accuracy["masked"],
                "incorrect_accuracy": metrics.mode_accuracy["incorrect"],
                "worst_mode_loss": max(metrics.mode_loss.values()),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
