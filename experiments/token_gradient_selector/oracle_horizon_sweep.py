"""Compare ordinary backprop with oracle shortcut-token gradient reversal."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict
from pathlib import Path

import torch

from list_sorting_transformer.shortcut_credit import (
    ShortcutPointerVocabulary,
    evaluate_shortcut_batches,
    make_fitness_batches,
    shortcut_loss,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_forward_model,
    make_inner_batches,
    make_mode_batches,
)
from list_sorting_transformer.token_gradient_reversal import (
    oracle_reversal_shortcut_loss,
)


def parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "horizons must be comma-separated integers"
        ) from error
    if (
        not horizons
        or horizons[0] != 0
        or any(horizon < 0 for horizon in horizons)
        or tuple(sorted(set(horizons))) != horizons
    ):
        raise argparse.ArgumentTypeError(
            "horizons must be unique increasing integers beginning at zero"
        )
    return horizons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=("ordinary", "oracle_reversal"),
        required=True,
    )
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=parse_horizons("0,10,20,40,80,160,320,640,1280,2000,3000"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fitness-examples", type=int, default=512)
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--forward-learning-rate", type=float, default=3e-4)
    parser.add_argument("--reversal-scale", type=float, default=1.0)
    parser.add_argument(
        "--leak-placement",
        choices=("suffix", "random_list"),
        default="suffix",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.reversal_scale <= 0:
        raise ValueError("reversal-scale must be positive")

    device = torch.device(args.device)
    max_horizon = args.horizons[-1]
    config = ShortcutCreditExperimentConfig(
        generations=1,
        population_size=2,
        horizon=max(1, max_horizon),
        max_horizon=max(1, max_horizon),
        batch_size=args.batch_size,
        fitness_examples=args.fitness_examples,
        min_length=args.min_length,
        max_length=args.max_length,
        forward_learning_rate=args.forward_learning_rate,
        leak_placement=args.leak_placement,
        seed=args.seed,
        device=args.device,
    )
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    initialization_seed = args.seed * 1_000_003 + 1
    model = initialize_forward_model(
        config,
        vocabulary,
        initialization_seed=initialization_seed,
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.forward_learning_rate,
    )
    training_generator = torch.Generator().manual_seed(
        args.seed * 1_000_003 + 2
    )
    fitness_generator = torch.Generator().manual_seed(args.seed + 10_000)
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
    correct_batches = make_mode_batches(
        config.correct_eval_examples,
        leak_mode="correct",
        config=config,
        vocabulary=vocabulary,
        generator=fitness_generator,
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("")
    recent_losses: deque[float] = deque(maxlen=100)
    recent_gradient_norms: deque[float] = deque(maxlen=100)
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
            )[0]
            optimizer.zero_grad(set_to_none=True)
            if args.condition == "oracle_reversal":
                loss = oracle_reversal_shortcut_loss(
                    model,
                    batch,
                    vocabulary,
                    reversal_scale=args.reversal_scale,
                )
            else:
                loss = shortcut_loss(model, batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float("inf"),
            )
            optimizer.step()
            recent_losses.append(float(loss.detach()))
            recent_gradient_norms.append(float(gradient_norm))
        completed_steps = horizon

        clean = evaluate_shortcut_batches(model, fitness_batches)
        correct = evaluate_shortcut_batches(model, correct_batches)
        row = {
            "condition": args.condition,
            "seed": args.seed,
            "horizon": horizon,
            "reversal_scale": args.reversal_scale,
            "train_loss_last_100": (
                sum(recent_losses) / len(recent_losses)
                if recent_losses
                else None
            ),
            "gradient_norm_last_100": (
                sum(recent_gradient_norms) / len(recent_gradient_norms)
                if recent_gradient_norms
                else None
            ),
            "clean_loss": clean.loss,
            "clean_accuracy": clean.accuracy,
            "masked_accuracy": clean.mode_accuracy["masked"],
            "incorrect_accuracy": clean.mode_accuracy["incorrect"],
            "correct_accuracy": correct.accuracy,
            "unique_value_predictions": clean.unique_value_prediction_count,
            "prediction_mode_fraction": clean.prediction_mode_fraction,
        }
        with args.output.open("a") as output_file:
            output_file.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        model.train()

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "condition": args.condition,
                "seed": args.seed,
                "horizon": completed_steps,
                "reversal_scale": args.reversal_scale,
                "config": asdict(config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            args.checkpoint,
        )


if __name__ == "__main__":
    main()
