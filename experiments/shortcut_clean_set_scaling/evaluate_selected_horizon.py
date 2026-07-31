"""Select an EGGROLL checkpoint on fixed fitness, then audit a longer horizon."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from list_sorting_transformer.shortcut_learning.shortcut_credit import (
    ShortcutPointerVocabulary,
    evaluate_shortcut_batches,
    make_fitness_batches,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    candidate_fitness,
    initialize_forward_model,
    load_checkpoint,
    make_inner_batches,
    make_mode_batches,
    train_forward_trajectory,
)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def clean_summary(prefix: str, metrics: Any) -> dict[str, float]:
    return {
        f"{prefix}/loss": metrics.loss,
        f"{prefix}/masked_accuracy": metrics.mode_accuracy["masked"],
        f"{prefix}/incorrect_accuracy": metrics.mode_accuracy["incorrect"],
        f"{prefix}/robust_accuracy": min(metrics.mode_accuracy.values()),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(set.intersection(*(set(row) for row in rows)) - {"replicate"})
    return {f"mean/{key}": mean([row[key] for row in rows]) for key in keys}


def config_from_checkpoint(
    checkpoint: Path, *, horizon: int, device: str,
) -> ShortcutCreditExperimentConfig:
    raw = torch.load(checkpoint, map_location="cpu")
    saved = ShortcutCreditExperimentConfig(**raw["config"])
    return replace(
        saved,
        horizon=horizon,
        max_horizon=horizon,
        fitness_checkpoints=None,
        resume=None,
        resume_horizon=None,
        wandb=False,
        device=device,
    )


def fixed_fitness_batches(
    config: ShortcutCreditExperimentConfig,
    vocabulary: ShortcutPointerVocabulary,
    device: torch.device,
):
    generator = torch.Generator().manual_seed(config.seed + 10_000)
    clean = make_fitness_batches(
        config.fitness_examples,
        min_length=config.min_length,
        max_length=config.max_length,
        batch_size=config.fitness_batch_size,
        generator=generator,
        vocabulary=vocabulary,
        leak_placement=config.leak_placement,
        device=device,
    )
    correct = make_mode_batches(
        1,
        leak_mode="correct",
        config=config,
        vocabulary=vocabulary,
        generator=generator,
        device=device,
    )
    return clean, correct


def select_checkpoint(
    checkpoints: list[Path],
    *,
    config: ShortcutCreditExperimentConfig,
    replicates: int,
    device: torch.device,
) -> tuple[Path, list[dict[str, float]]]:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    fitness_batches, correct_batches = fixed_fitness_batches(config, vocabulary, device)
    checkpoint_scores: dict[Path, list[float]] = {
        checkpoint: [] for checkpoint in checkpoints
    }

    for replicate in range(replicates):
        seed = config.seed * 1_000_003 + 70_000 + replicate * 10_007
        base_model = initialize_forward_model(
            config, vocabulary, initialization_seed=seed + 1, device=device,
        )
        base_state = {
            name: value.detach().clone()
            for name, value in base_model.state_dict().items()
        }
        initial = evaluate_shortcut_batches(base_model, fitness_batches)
        del base_model
        inner_batches = make_inner_batches(
            config,
            horizon=config.horizon,
            vocabulary=vocabulary,
            generator=torch.Generator().manual_seed(seed + 2),
            device=device,
        )
        for checkpoint in checkpoints:
            rule, _, _, _ = load_checkpoint(checkpoint, device=device)
            trajectory = train_forward_trajectory(
                config,
                base_state=base_state,
                backward_rule=rule,
                inner_batches=inner_batches,
                fitness_batches=fitness_batches,
                correct_batches=correct_batches,
                device=device,
            )
            checkpoint_scores[checkpoint].append(
                candidate_fitness(
                    config.fitness_objective,
                    initial,
                    trajectory.clean,
                    checkpoint_clean=trajectory.checkpoint_clean,
                )
            )
            del rule

    rows = [
        {
            "checkpoint": str(checkpoint),
            "mean_fixed_fitness": mean(scores),
            "selection_replicates": float(replicates),
        }
        for checkpoint, scores in checkpoint_scores.items()
    ]
    selected = max(
        checkpoints, key=lambda checkpoint: mean(checkpoint_scores[checkpoint]),
    )
    return selected, rows


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    config: ShortcutCreditExperimentConfig,
    replicates: int,
    evaluation_examples: int,
    device: torch.device,
) -> list[dict[str, float]]:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    rule, _, _, _ = load_checkpoint(checkpoint, device=device)
    rows = []
    for replicate in range(replicates):
        seed = config.seed * 1_000_003 + 90_000 + replicate * 10_007
        base_model = initialize_forward_model(
            config, vocabulary, initialization_seed=seed + 1, device=device,
        )
        base_state = {
            name: value.detach().clone()
            for name, value in base_model.state_dict().items()
        }
        del base_model
        inner_batches = make_inner_batches(
            config,
            horizon=config.horizon,
            vocabulary=vocabulary,
            generator=torch.Generator().manual_seed(seed + 2),
            device=device,
        )
        masked_inner_batches = make_inner_batches(
            config,
            horizon=config.horizon,
            vocabulary=vocabulary,
            generator=torch.Generator().manual_seed(seed + 2),
            device=device,
            leak_mode="masked",
        )
        evaluation_generator = torch.Generator().manual_seed(seed + 4)
        clean_batches = make_fitness_batches(
            evaluation_examples,
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
        evolved = train_forward_trajectory(
            config,
            base_state=base_state,
            backward_rule=rule,
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
        row: dict[str, float] = {"replicate": float(replicate)}
        row.update(clean_summary("evolved", evolved.clean))
        row.update(clean_summary("ordinary", ordinary.clean))
        row.update(clean_summary("masked_training", masked.clean))
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-horizon", type=int, default=160)
    parser.add_argument("--evaluation-horizon", type=int, default=320)
    parser.add_argument("--selection-replicates", type=int, default=2)
    parser.add_argument("--evaluation-replicates", type=int, default=5)
    parser.add_argument("--evaluation-examples", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.selection_replicates, args.evaluation_replicates) < 1:
        raise ValueError("replicate counts must be positive")

    checkpoints = sorted(args.run_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under {args.run_dir}")
    device = torch.device(args.device)
    selection_config = config_from_checkpoint(
        checkpoints[0], horizon=args.selection_horizon, device=args.device
    )
    selected, selection_rows = select_checkpoint(
        checkpoints,
        config=selection_config,
        replicates=args.selection_replicates,
        device=device,
    )
    print(f"Selected checkpoint: {selected}", flush=True)
    evaluation_config = replace(
        selection_config,
        horizon=args.evaluation_horizon,
        max_horizon=args.evaluation_horizon,
    )
    evaluation_rows = evaluate_checkpoint(
        selected,
        config=evaluation_config,
        replicates=args.evaluation_replicates,
        evaluation_examples=args.evaluation_examples,
        device=device,
    )
    result = {
        "selection_horizon": args.selection_horizon,
        "evaluation_horizon": args.evaluation_horizon,
        "selected_checkpoint": str(selected),
        "selection": selection_rows,
        "evaluation": evaluation_rows,
        "summary": aggregate(evaluation_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
