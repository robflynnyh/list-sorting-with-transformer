"""Run EGGROLL evolution of a shortcut-resistant backward rule."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .evaluate import resolve_device
from .shortcut_credit import (
    BackwardRuleConfig,
    EggrollDirection,
    LearnedBackwardRule,
    ShortcutBatch,
    ShortcutDecoderTransformer,
    ShortcutMetrics,
    ShortcutPointerVocabulary,
    apply_eggroll_direction,
    clone_center_parameters,
    evaluate_shortcut_batches,
    make_fitness_batches,
    make_forward_model_config,
    make_shortcut_batch,
    paper_eggroll_update,
    sample_eggroll_direction,
    shortcut_loss,
)


@dataclass(frozen=True)
class ShortcutCreditExperimentConfig:
    run_name: str = "learned-backward-shortcuts"
    output_dir: str = "artifacts/learned_backward_shortcuts"
    generations: int = 1_000
    population_size: int = 64
    horizon: int = 10
    max_horizon: int = 640
    horizon_multiplier: int = 2
    plateau_patience: int = 100
    plateau_min_delta: float = 1e-3
    plateau_ema_decay: float = 0.95
    batch_size: int = 64
    fitness_examples: int = 512
    fitness_batch_size: int = 64
    correct_eval_examples: int = 128
    min_length: int = 8
    max_length: int = 32
    forward_learning_rate: float = 3e-4
    sigma: float = 0.02
    outer_learning_rate: float = 0.1
    d_model: int = 128
    backward_d_model: int = 128
    forward_layers: int = 3
    backward_layers: int = 2
    heads: int = 4
    seed: int = 7
    checkpoint_interval: int = 10
    device: str = "auto"
    wandb: bool = False
    wandb_project: str = "list-sorting-learned-backward"
    wandb_entity: str | None = None
    resume: str | None = None

    def __post_init__(self) -> None:
        positive_integers = (
            self.generations,
            self.population_size,
            self.horizon,
            self.max_horizon,
            self.horizon_multiplier,
            self.plateau_patience,
            self.batch_size,
            self.fitness_examples,
            self.fitness_batch_size,
            self.correct_eval_examples,
            self.min_length,
            self.max_length,
            self.d_model,
            self.backward_d_model,
            self.forward_layers,
            self.backward_layers,
            self.heads,
            self.checkpoint_interval,
        )
        if any(value < 1 for value in positive_integers):
            raise ValueError("integer configuration values must be positive")
        if self.population_size % 2:
            raise ValueError("population_size must be even for antithetic pairs")
        if self.fitness_examples % 2:
            raise ValueError("fitness_examples must be even")
        if not 2 <= self.min_length <= self.max_length:
            raise ValueError("invalid task length range")
        if self.horizon > self.max_horizon:
            raise ValueError("horizon must not exceed max_horizon")
        if not 0 <= self.plateau_ema_decay < 1:
            raise ValueError("plateau_ema_decay must be in [0, 1)")
        if min(
            self.forward_learning_rate,
            self.sigma,
            self.outer_learning_rate,
        ) <= 0:
            raise ValueError("learning rates and sigma must be positive")


@dataclass
class PlateauState:
    ema_fitness: float | None = None
    best_ema_fitness: float = float("-inf")
    stale_generations: int = 0


def make_mode_batches(
    example_count: int,
    *,
    leak_mode: str,
    config: ShortcutCreditExperimentConfig,
    vocabulary: ShortcutPointerVocabulary,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[ShortcutBatch, ...]:
    lengths = torch.randint(
        config.min_length,
        config.max_length + 1,
        (example_count,),
        generator=generator,
    )
    batches = []
    for length in sorted(set(lengths.tolist())):
        count = int(lengths.eq(length).sum())
        remaining = count
        while remaining:
            current = min(config.fitness_batch_size, remaining)
            batches.append(
                make_shortcut_batch(
                    current,
                    int(length),
                    leak_mode=leak_mode,  # type: ignore[arg-type]
                    generator=generator,
                    vocabulary=vocabulary,
                    device=device,
                )
            )
            remaining -= current
    return tuple(batches)


def initialize_forward_model(
    config: ShortcutCreditExperimentConfig,
    vocabulary: ShortcutPointerVocabulary,
    *,
    initialization_seed: int,
    device: torch.device,
) -> ShortcutDecoderTransformer:
    torch.manual_seed(initialization_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(initialization_seed)
    model_config = make_forward_model_config(
        vocabulary,
        d_model=config.d_model,
        n_layers=config.forward_layers,
        n_heads=config.heads,
    )
    return ShortcutDecoderTransformer(model_config).to(device)


def make_inner_batches(
    config: ShortcutCreditExperimentConfig,
    *,
    horizon: int,
    vocabulary: ShortcutPointerVocabulary,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[ShortcutBatch, ...]:
    return tuple(
        make_shortcut_batch(
            config.batch_size,
            int(
                torch.randint(
                    config.min_length,
                    config.max_length + 1,
                    (),
                    generator=generator,
                )
            ),
            leak_mode="correct",
            generator=generator,
            vocabulary=vocabulary,
            device=device,
        )
        for _ in range(horizon)
    )


def train_candidate(
    config: ShortcutCreditExperimentConfig,
    *,
    base_state: dict[str, Tensor],
    initialization_seed: int,
    center_rule: LearnedBackwardRule,
    center_parameters: dict[str, Tensor],
    direction: EggrollDirection,
    sign: int,
    inner_batches: tuple[ShortcutBatch, ...],
    fitness_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    initial_clean_metrics: ShortcutMetrics,
    device: torch.device,
    capture_statistics: bool,
) -> tuple[float, ShortcutMetrics, ShortcutMetrics, list[dict[str, float]]]:
    model = initialize_forward_model(
        config,
        ShortcutPointerVocabulary("numbers", 10),
        initialization_seed=initialization_seed,
        device=device,
    )
    model.load_state_dict(base_state)
    backward_rule = LearnedBackwardRule(center_rule.config).to(device)
    apply_eggroll_direction(
        backward_rule,
        center_parameters,
        direction,
        sigma=config.sigma,
        sign=sign,
    )
    backward_rule.capture_statistics = capture_statistics
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.forward_learning_rate,
    )
    model.train()
    for batch in inner_batches:
        optimizer.zero_grad(set_to_none=True)
        loss = shortcut_loss(model, batch, backward_rule)
        loss.backward()
        optimizer.step()

    clean_metrics = evaluate_shortcut_batches(model, fitness_batches)
    correct_metrics = evaluate_shortcut_batches(model, correct_batches)
    fitness = initial_clean_metrics.loss - clean_metrics.loss
    return (
        fitness,
        clean_metrics,
        correct_metrics,
        list(backward_rule.statistics),
    )


def linear_outer_learning_rate(
    config: ShortcutCreditExperimentConfig,
    generation: int,
) -> float:
    if config.generations == 1:
        return config.outer_learning_rate
    fraction_remaining = 1.0 - generation / (config.generations - 1)
    return config.outer_learning_rate * fraction_remaining


def update_plateau_state(
    state: PlateauState,
    *,
    fitness: float,
    config: ShortcutCreditExperimentConfig,
) -> bool:
    if state.ema_fitness is None:
        state.ema_fitness = fitness
    else:
        state.ema_fitness = (
            config.plateau_ema_decay * state.ema_fitness
            + (1.0 - config.plateau_ema_decay) * fitness
        )
    if state.ema_fitness > state.best_ema_fitness + config.plateau_min_delta:
        state.best_ema_fitness = state.ema_fitness
        state.stale_generations = 0
    else:
        state.stale_generations += 1
    return state.stale_generations >= config.plateau_patience


def candidate_summary(
    fitnesses: Tensor,
    clean_metrics: list[ShortcutMetrics],
    correct_metrics: list[ShortcutMetrics],
) -> dict[str, float]:
    clean_losses = torch.tensor([metrics.loss for metrics in clean_metrics])
    clean_accuracies = torch.tensor(
        [metrics.accuracy for metrics in clean_metrics]
    )
    masked_accuracies = torch.tensor(
        [metrics.mode_accuracy["masked"] for metrics in clean_metrics]
    )
    incorrect_accuracies = torch.tensor(
        [metrics.mode_accuracy["incorrect"] for metrics in clean_metrics]
    )
    correct_accuracies = torch.tensor(
        [metrics.accuracy for metrics in correct_metrics]
    )
    return {
        "fitness/mean": float(fitnesses.mean()),
        "fitness/std": float(fitnesses.std(unbiased=False)),
        "fitness/max": float(fitnesses.max()),
        "clean/loss_mean": float(clean_losses.mean()),
        "clean/accuracy_mean": float(clean_accuracies.mean()),
        "clean/masked_accuracy_mean": float(masked_accuracies.mean()),
        "clean/incorrect_accuracy_mean": float(incorrect_accuracies.mean()),
        "correct_leak/accuracy_mean": float(correct_accuracies.mean()),
    }


def save_checkpoint(
    path: Path,
    *,
    backward_rule: LearnedBackwardRule,
    config: ShortcutCreditExperimentConfig,
    generation: int,
    horizon: int,
    plateau_state: PlateauState,
) -> None:
    torch.save(
        {
            "experiment": "learned_backward_shortcuts",
            "config": asdict(config),
            "backward_rule_config": asdict(backward_rule.config),
            "backward_rule_state": backward_rule.state_dict(),
            "generation": generation,
            "horizon": horizon,
            "plateau_state": asdict(plateau_state),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> tuple[LearnedBackwardRule, int, int, PlateauState]:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("experiment") != "learned_backward_shortcuts":
        raise ValueError("checkpoint belongs to a different experiment")
    backward_rule = LearnedBackwardRule(
        BackwardRuleConfig(**checkpoint["backward_rule_config"])
    ).to(device)
    backward_rule.load_state_dict(checkpoint["backward_rule_state"])
    plateau_state = PlateauState(**checkpoint["plateau_state"])
    return (
        backward_rule,
        int(checkpoint["generation"]) + 1,
        int(checkpoint["horizon"]),
        plateau_state,
    )


def maybe_initialize_wandb(
    config: ShortcutCreditExperimentConfig,
) -> Any | None:
    if not config.wandb:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking requires the project tracking dependencies"
        ) from error
    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.run_name,
        config=asdict(config),
    )


def run(config: ShortcutCreditExperimentConfig) -> Path:
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2) + "\n")

    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    backward_config = BackwardRuleConfig(
        d_model=config.backward_d_model,
        forward_d_model=config.d_model,
        n_layers=config.backward_layers,
        n_heads=config.heads,
        forward_layers=config.forward_layers,
    )
    if config.resume is None:
        center_rule = LearnedBackwardRule(backward_config).to(device)
        start_generation = 0
        horizon = config.horizon
        plateau_state = PlateauState()
    else:
        center_rule, start_generation, horizon, plateau_state = load_checkpoint(
            Path(config.resume),
            device=device,
        )
        if center_rule.config != backward_config:
            raise ValueError("resume checkpoint architecture differs from config")

    fitness_generator = torch.Generator().manual_seed(config.seed + 10_000)
    fitness_batches = make_fitness_batches(
        config.fitness_examples,
        min_length=config.min_length,
        max_length=config.max_length,
        batch_size=config.fitness_batch_size,
        generator=fitness_generator,
        vocabulary=vocabulary,
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

    wandb_run = maybe_initialize_wandb(config)
    started_at = time.monotonic()
    for generation in range(start_generation, config.generations):
        generation_started_at = time.monotonic()
        generation_seed = config.seed * 1_000_003 + generation * 10_007
        initialization_seed = generation_seed + 1
        base_model = initialize_forward_model(
            config,
            vocabulary,
            initialization_seed=initialization_seed,
            device=device,
        )
        base_state = {
            name: tensor.detach().clone()
            for name, tensor in base_model.state_dict().items()
        }
        initial_clean_metrics = evaluate_shortcut_batches(
            base_model,
            fitness_batches,
        )
        del base_model

        inner_generator = torch.Generator().manual_seed(generation_seed + 2)
        inner_batches = make_inner_batches(
            config,
            horizon=horizon,
            vocabulary=vocabulary,
            generator=inner_generator,
            device=device,
        )
        direction_generator = torch.Generator().manual_seed(generation_seed + 3)
        directions = tuple(
            sample_eggroll_direction(
                center_rule,
                generator=direction_generator,
            )
            for _ in range(config.population_size // 2)
        )
        center_parameters = clone_center_parameters(center_rule)

        fitness_values = []
        clean_results = []
        correct_results = []
        captured_statistics: list[dict[str, float]] = []
        for direction_index, direction in enumerate(directions):
            for sign in (1, -1):
                capture_statistics = (
                    direction_index == 0 and sign == 1
                )
                (
                    fitness,
                    clean_metrics,
                    correct_metrics,
                    statistics,
                ) = train_candidate(
                    config,
                    base_state=base_state,
                    initialization_seed=initialization_seed,
                    center_rule=center_rule,
                    center_parameters=center_parameters,
                    direction=direction,
                    sign=sign,
                    inner_batches=inner_batches,
                    fitness_batches=fitness_batches,
                    correct_batches=correct_batches,
                    initial_clean_metrics=initial_clean_metrics,
                    device=device,
                    capture_statistics=capture_statistics,
                )
                fitness_values.append(fitness)
                clean_results.append(clean_metrics)
                correct_results.append(correct_metrics)
                if statistics:
                    captured_statistics = statistics

        fitness_tensor = torch.tensor(fitness_values, device=device)
        outer_learning_rate = linear_outer_learning_rate(config, generation)
        standardized = paper_eggroll_update(
            center_rule,
            directions,
            fitness_tensor,
            sigma=config.sigma,
            learning_rate=outer_learning_rate,
        )
        summary = candidate_summary(
            fitness_tensor.cpu(),
            clean_results,
            correct_results,
        )
        summary.update(
            {
                "generation": generation,
                "horizon": horizon,
                "population_size": config.population_size,
                "outer_learning_rate": outer_learning_rate,
                "fitness/standardized_mean": float(standardized.mean()),
                "fitness/standardized_std": float(
                    standardized.std(unbiased=False)
                ),
                "initial_clean/loss": initial_clean_metrics.loss,
                "initial_clean/accuracy": initial_clean_metrics.accuracy,
                "timing/generation_seconds": (
                    time.monotonic() - generation_started_at
                ),
                "timing/elapsed_seconds": time.monotonic() - started_at,
            }
        )
        if captured_statistics:
            summary["backward/gradient_cosine_mean"] = sum(
                item["cosine"] for item in captured_statistics
            ) / len(captured_statistics)
            summary["backward/correction_rms_ratio_mean"] = sum(
                item["correction_rms_ratio"]
                for item in captured_statistics
            ) / len(captured_statistics)

        with metrics_path.open("a") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
        if wandb_run is not None:
            wandb_run.log(summary, step=generation)

        promote_horizon = update_plateau_state(
            plateau_state,
            fitness=summary["fitness/mean"],
            config=config,
        )
        if promote_horizon and horizon < config.max_horizon:
            horizon = min(
                config.max_horizon,
                horizon * config.horizon_multiplier,
            )
            plateau_state = PlateauState()
            print(f"Increasing evolved horizon to {horizon}", flush=True)

        if (
            (generation + 1) % config.checkpoint_interval == 0
            or generation + 1 == config.generations
        ):
            checkpoint_path = output_dir / f"checkpoint_{generation + 1:06d}.pt"
            save_checkpoint(
                checkpoint_path,
                backward_rule=center_rule,
                config=config,
                generation=generation,
                horizon=horizon,
                plateau_state=plateau_state,
            )
            save_checkpoint(
                output_dir / "latest.pt",
                backward_rule=center_rule,
                config=config,
                generation=generation,
                horizon=horizon,
                plateau_state=plateau_state,
            )

    if wandb_run is not None:
        wandb_run.finish()
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="learned-backward-shortcuts")
    parser.add_argument(
        "--output-dir",
        default="artifacts/learned_backward_shortcuts",
    )
    parser.add_argument("--generations", type=int, default=1_000)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-horizon", type=int, default=640)
    parser.add_argument("--horizon-multiplier", type=int, default=2)
    parser.add_argument("--plateau-patience", type=int, default=100)
    parser.add_argument("--plateau-min-delta", type=float, default=1e-3)
    parser.add_argument("--plateau-ema-decay", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fitness-examples", type=int, default=512)
    parser.add_argument("--fitness-batch-size", type=int, default=64)
    parser.add_argument("--correct-eval-examples", type=int, default=128)
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--forward-learning-rate", type=float, default=3e-4)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--outer-learning-rate", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--backward-d-model", type=int, default=128)
    parser.add_argument("--forward-layers", type=int, default=3)
    parser.add_argument("--backward-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default="list-sorting-learned-backward",
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument("--resume")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ShortcutCreditExperimentConfig(**vars(args))
    output_dir = run(config)
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
