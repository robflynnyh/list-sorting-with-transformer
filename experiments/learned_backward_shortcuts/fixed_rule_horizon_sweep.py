"""Train one forward seed for a long time with a frozen backward rule."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

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
    load_checkpoint,
    make_inner_batches,
    make_mode_batches,
)


def parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "horizons must be comma-separated integers"
        ) from error
    if not horizons or horizons[0] < 0:
        raise argparse.ArgumentTypeError(
            "horizons must be non-negative comma-separated integers"
        )
    return horizons


def maybe_initialize_wandb(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    run_name: str,
    config: dict[str, Any],
) -> Any | None:
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking requires the project tracking dependencies"
        ) from error
    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generation-seed", type=int, required=True)
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=parse_horizons(
            "0,320,640,1280,2000,3000,5000,10000,20000,50000"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default="list-sorting-learned-backward",
    )
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()
    if (
        args.gradient_clip_norm is not None
        and args.gradient_clip_norm <= 0
    ):
        raise ValueError("gradient_clip_norm must be positive")

    device = torch.device(args.device)
    raw_checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    saved_config = ShortcutCreditExperimentConfig(
        **raw_checkpoint["config"]
    )
    max_horizon = max(args.horizons)
    config = replace(
        saved_config,
        horizon=max(1, max_horizon),
        max_horizon=max(1, max_horizon, saved_config.max_horizon),
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
    model = initialize_forward_model(
        config,
        vocabulary,
        initialization_seed=args.generation_seed + 1,
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.forward_learning_rate,
    )
    training_generator = torch.Generator().manual_seed(
        args.generation_seed + 2
    )
    audit_generator = torch.Generator().manual_seed(
        args.generation_seed + 4
    )
    audit_batches = make_fitness_batches(
        config.fitness_examples,
        min_length=config.min_length,
        max_length=config.max_length,
        batch_size=config.fitness_batch_size,
        generator=audit_generator,
        vocabulary=vocabulary,
        leak_placement=config.leak_placement,
        device=device,
    )
    audit_correct_batches = make_mode_batches(
        config.correct_eval_examples,
        leak_mode="correct",
        config=config,
        vocabulary=vocabulary,
        generator=audit_generator,
        device=device,
    )
    fixed_generator = torch.Generator().manual_seed(config.seed + 10_000)
    fixed_batches = make_fitness_batches(
        config.fitness_examples,
        min_length=config.min_length,
        max_length=config.max_length,
        batch_size=config.fitness_batch_size,
        generator=fixed_generator,
        vocabulary=vocabulary,
        leak_placement=config.leak_placement,
        device=device,
    )

    run_config = {
        "checkpoint": str(args.checkpoint),
        "generation_seed": args.generation_seed,
        "horizons": args.horizons,
        "gradient_clip_norm": args.gradient_clip_norm,
        "experiment_config": asdict(config),
    }
    wandb_run = maybe_initialize_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.run_name,
        config=run_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("")
    recent_losses: deque[float] = deque(maxlen=1_000)
    recent_gradient_norms: deque[float] = deque(maxlen=1_000)
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
            loss = shortcut_loss(model, batch, backward_rule)
            loss.backward()
            if args.gradient_clip_norm is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.gradient_clip_norm,
                )
                recent_gradient_norms.append(float(gradient_norm))
            optimizer.step()
            recent_losses.append(float(loss.detach()))
        completed_steps = horizon

        audit = evaluate_shortcut_batches(model, audit_batches)
        audit_correct = evaluate_shortcut_batches(
            model,
            audit_correct_batches,
        )
        fixed = evaluate_shortcut_batches(model, fixed_batches)
        last_10 = tuple(recent_losses)[-10:]
        last_100 = tuple(recent_losses)[-100:]
        gradient_norm_last_100 = tuple(recent_gradient_norms)[-100:]
        row = {
            "horizon": float(horizon),
            "audit/clean_loss": audit.loss,
            "audit/min_mode_accuracy": min(audit.mode_accuracy.values()),
            "audit/masked_accuracy": audit.mode_accuracy["masked"],
            "audit/incorrect_accuracy": audit.mode_accuracy["incorrect"],
            "audit/correct_leak_accuracy": audit_correct.accuracy,
            "fixed/clean_loss": fixed.loss,
            "fixed/min_mode_accuracy": min(fixed.mode_accuracy.values()),
            "fixed/masked_accuracy": fixed.mode_accuracy["masked"],
            "fixed/incorrect_accuracy": fixed.mode_accuracy["incorrect"],
            "train_loss/last_10_mean": (
                sum(last_10) / len(last_10)
                if last_10
                else None
            ),
            "train_loss/last_100_mean": (
                sum(last_100) / len(last_100)
                if last_100
                else None
            ),
            "train_loss/last_1000_mean": (
                sum(recent_losses) / len(recent_losses)
                if recent_losses
                else None
            ),
            "gradient_norm/last_100_mean": (
                sum(gradient_norm_last_100)
                / len(gradient_norm_last_100)
                if gradient_norm_last_100
                else None
            ),
            "gradient_norm/last_100_max": (
                max(gradient_norm_last_100)
                if gradient_norm_last_100
                else None
            ),
        }
        with args.output.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if wandb_run is not None:
            wandb_run.log(row, step=horizon)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
