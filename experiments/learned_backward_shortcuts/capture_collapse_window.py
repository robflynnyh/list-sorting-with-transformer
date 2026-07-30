"""Capture a replayable forward-model window around a known collapse."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from list_sorting_transformer.shortcut_collapse_window import (
    capture_collapse_window,
    replay_collapse_window,
    save_collapse_window,
)
from list_sorting_transformer.shortcut_credit import (
    ShortcutPointerVocabulary,
    make_fitness_batches,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    load_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generation-seed", type=int, required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--window-steps", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    raw_checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    saved_config = ShortcutCreditExperimentConfig(
        **raw_checkpoint["config"]
    )
    config = replace(
        saved_config,
        horizon=args.window_steps,
        max_horizon=max(args.window_steps, saved_config.max_horizon),
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
    window = capture_collapse_window(
        config,
        backward_rule=backward_rule,
        generation_seed=args.generation_seed,
        start_step=args.start_step,
        window_steps=args.window_steps,
        fitness_batches=fitness_batches,
        device=device,
        evaluation_batch_size=config.fitness_examples,
    )
    replay = replay_collapse_window(
        config,
        window=window,
        backward_rule=backward_rule,
        fitness_batches=fitness_batches,
        device=device,
        checkpoint_steps=(args.window_steps,),
        evaluation_batch_size=config.fitness_examples,
    )
    if replay.end_metrics != window.center_end_metrics:
        raise RuntimeError("captured collapse window did not replay exactly")
    save_collapse_window(args.output, window)

    summary = {
        "output": str(args.output),
        "generation_seed": args.generation_seed,
        "start_step": args.start_step,
        "end_step": window.end_step,
        "start_clean_loss": window.start_metrics.loss,
        "start_min_mode_accuracy": min(
            window.start_metrics.mode_accuracy.values()
        ),
        "center_end_clean_loss": window.center_end_metrics.loss,
        "center_end_min_mode_accuracy": min(
            window.center_end_metrics.mode_accuracy.values()
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
