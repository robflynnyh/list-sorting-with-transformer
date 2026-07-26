"""Run EGGROLL evolution of a shortcut-resistant backward rule."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union

import torch
from torch import Tensor

from .evaluate import resolve_device
from .shortcut_credit import (
    AttentionRoutingRule,
    AttentionRoutingRuleConfig,
    BackwardRule,
    BackwardRuleConfig,
    EggrollDirection,
    LeakMode,
    LeakPlacement,
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
    move_eggroll_direction,
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
    leak_placement: LeakPlacement = "suffix"
    forward_learning_rate: float = 3e-4
    sigma: float = 0.08
    outer_learning_rate: float = 0.1
    d_model: int = 128
    backward_d_model: int = 128
    backward_rule_type: str = "gradient_transformer"
    route_output_projection: bool = False
    shared_routing_map: bool = True
    fitness_objective: str = "mean_clean_ce"
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
    candidate_devices: str | None = None

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
        if self.backward_rule_type not in {
            "gradient_transformer",
            "attention_router",
        }:
            raise ValueError("unknown backward_rule_type")
        if self.fitness_objective not in {
            "mean_clean_ce",
            "worst_mode_ce",
        }:
            raise ValueError("unknown fitness_objective")
        if self.leak_placement not in {"suffix", "random_list"}:
            raise ValueError("unknown leak placement")
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


@dataclass(frozen=True)
class ForwardTrajectoryMetrics:
    clean: ShortcutMetrics
    correct: ShortcutMetrics
    heldout_clean: ShortcutMetrics | None = None
    heldout_correct: ShortcutMetrics | None = None


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
                    leak_placement=config.leak_placement,
                    device=device,
                )
            )
            remaining -= current
    return tuple(batches)


def initialize_forward_model(
    config: ShortcutCreditExperimentConfig,
    vocabulary: ShortcutPointerVocabulary,
    *,
    initialization_seed: int | None,
    device: torch.device,
) -> ShortcutDecoderTransformer:
    if initialization_seed is not None:
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


RuleConfig = Union[BackwardRuleConfig, AttentionRoutingRuleConfig]


def make_rule_config(
    config: ShortcutCreditExperimentConfig,
    vocabulary: ShortcutPointerVocabulary,
) -> RuleConfig:
    if config.backward_rule_type == "gradient_transformer":
        return BackwardRuleConfig(
            d_model=config.backward_d_model,
            forward_d_model=config.d_model,
            n_layers=config.backward_layers,
            n_heads=config.heads,
            forward_layers=config.forward_layers,
        )
    return AttentionRoutingRuleConfig(
        vocab_size=vocabulary.size,
        d_model=config.backward_d_model,
        n_heads=config.heads,
        forward_layers=config.forward_layers,
        route_output_projection=config.route_output_projection,
        shared_routing_map=config.shared_routing_map,
    )


def initialize_backward_rule(
    config: RuleConfig,
    *,
    device: torch.device,
) -> BackwardRule:
    if isinstance(config, AttentionRoutingRuleConfig):
        return AttentionRoutingRule(config).to(device)
    return LearnedBackwardRule(config).to(device)


def initialize_fresh_backward_rule(
    config: ShortcutCreditExperimentConfig,
    vocabulary: ShortcutPointerVocabulary,
    *,
    device: torch.device,
) -> BackwardRule:
    torch.manual_seed(config.seed + 20_000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + 20_000)
    return initialize_backward_rule(
        make_rule_config(config, vocabulary),
        device=device,
    )


def make_inner_batches(
    config: ShortcutCreditExperimentConfig,
    *,
    horizon: int,
    vocabulary: ShortcutPointerVocabulary,
    generator: torch.Generator,
    device: torch.device,
    leak_mode: LeakMode = "correct",
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
            leak_mode=leak_mode,
            generator=generator,
            vocabulary=vocabulary,
            leak_placement=config.leak_placement,
            device=device,
        )
        for _ in range(horizon)
    )


def train_candidate(
    config: ShortcutCreditExperimentConfig,
    *,
    base_state: dict[str, Tensor],
    center_rule: BackwardRule,
    center_parameters: dict[str, Tensor],
    direction: EggrollDirection,
    sign: int,
    inner_batches: tuple[ShortcutBatch, ...],
    fitness_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    initial_clean_metrics: ShortcutMetrics,
    device: torch.device,
    capture_statistics: bool,
    perturbation_sigma: float | None = None,
    heldout_fitness_batches: tuple[ShortcutBatch, ...] | None = None,
    heldout_correct_batches: tuple[ShortcutBatch, ...] | None = None,
) -> tuple[float, ForwardTrajectoryMetrics, list[dict[str, float]]]:
    backward_rule = initialize_backward_rule(
        center_rule.config,
        device=device,
    )
    apply_eggroll_direction(
        backward_rule,
        center_parameters,
        direction,
        sigma=(
            config.sigma
            if perturbation_sigma is None
            else perturbation_sigma
        ),
        sign=sign,
    )
    backward_rule.capture_statistics = capture_statistics
    trajectory = train_forward_trajectory(
        config,
        base_state=base_state,
        backward_rule=backward_rule,
        inner_batches=inner_batches,
        fitness_batches=fitness_batches,
        correct_batches=correct_batches,
        heldout_fitness_batches=heldout_fitness_batches,
        heldout_correct_batches=heldout_correct_batches,
        device=device,
    )
    fitness = candidate_fitness(
        config.fitness_objective,
        initial_clean_metrics,
        trajectory.clean,
    )
    return (
        fitness,
        trajectory,
        list(backward_rule.statistics),
    )


def train_forward_trajectory(
    config: ShortcutCreditExperimentConfig,
    *,
    base_state: dict[str, Tensor],
    backward_rule: BackwardRule | None,
    inner_batches: tuple[ShortcutBatch, ...],
    fitness_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    heldout_fitness_batches: tuple[ShortcutBatch, ...] | None = None,
    heldout_correct_batches: tuple[ShortcutBatch, ...] | None = None,
    device: torch.device,
) -> ForwardTrajectoryMetrics:
    """Train one forward model from the shared per-generation state."""

    if (heldout_fitness_batches is None) != (
        heldout_correct_batches is None
    ):
        raise ValueError("both held-out batch groups must be provided together")
    model = initialize_forward_model(
        config,
        ShortcutPointerVocabulary("numbers", 10),
        initialization_seed=None,
        device=device,
    )
    model.load_state_dict(base_state)
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
    heldout_clean_metrics = (
        None
        if heldout_fitness_batches is None
        else evaluate_shortcut_batches(model, heldout_fitness_batches)
    )
    heldout_correct_metrics = (
        None
        if heldout_correct_batches is None
        else evaluate_shortcut_batches(model, heldout_correct_batches)
    )
    return ForwardTrajectoryMetrics(
        clean=clean_metrics,
        correct=correct_metrics,
        heldout_clean=heldout_clean_metrics,
        heldout_correct=heldout_correct_metrics,
    )


def candidate_fitness(
    objective: str,
    initial: ShortcutMetrics,
    trained: ShortcutMetrics,
) -> float:
    if objective == "mean_clean_ce":
        return initial.loss - trained.loss
    if objective == "worst_mode_ce":
        initial_worst = max(initial.mode_loss.values())
        trained_worst = max(trained.mode_loss.values())
        return initial_worst - trained_worst
    raise ValueError(f"unknown fitness objective: {objective}")


def trajectory_summary(
    prefix: str,
    fitness: float | None,
    clean: ShortcutMetrics,
    correct: ShortcutMetrics,
) -> dict[str, float]:
    summary = {
        f"{prefix}/clean_loss": clean.loss,
        f"{prefix}/clean_accuracy": clean.accuracy,
        f"{prefix}/masked_accuracy": clean.mode_accuracy["masked"],
        f"{prefix}/incorrect_accuracy": clean.mode_accuracy["incorrect"],
        f"{prefix}/min_mode_accuracy": min(
            clean.mode_accuracy["masked"],
            clean.mode_accuracy["incorrect"],
        ),
        f"{prefix}/correct_leak_accuracy": correct.accuracy,
        f"{prefix}/unique_value_predictions": float(
            clean.unique_value_prediction_count
        ),
        f"{prefix}/prediction_mode_fraction": (
            clean.prediction_mode_fraction
        ),
    }
    if fitness is not None:
        summary[f"{prefix}/fitness"] = fitness
    return summary


def center_rule_summary(
    fitness: float,
    clean: ShortcutMetrics,
    correct: ShortcutMetrics,
) -> dict[str, float]:
    return trajectory_summary("center_rule", fitness, clean, correct)


def parse_candidate_devices(
    configured_devices: str | None,
    primary_device: torch.device,
) -> tuple[torch.device, ...]:
    if configured_devices is None:
        return (primary_device,)
    devices = tuple(
        torch.device(item.strip())
        for item in configured_devices.split(",")
        if item.strip()
    )
    if not devices:
        raise ValueError("candidate_devices must contain at least one device")
    if len(set(devices)) != len(devices):
        raise ValueError("candidate_devices must not contain duplicates")
    if any(device.type != "cuda" for device in devices):
        raise ValueError("parallel candidate devices must be CUDA devices")
    return devices


def shard_candidate_specs(
    candidate_specs: tuple[tuple[int, int, int], ...],
    worker_count: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Keep both signs of each antithetic direction on the same worker."""

    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    return tuple(
        tuple(
            spec
            for spec in candidate_specs
            if spec[1] % worker_count == worker_index
        )
        for worker_index in range(worker_count)
    )


def train_candidate_shard(
    config: ShortcutCreditExperimentConfig,
    *,
    candidate_specs: tuple[tuple[int, int, int], ...],
    device: torch.device,
    base_state: dict[str, Tensor],
    center_rule_config: RuleConfig,
    center_parameters: dict[str, Tensor],
    directions: tuple[EggrollDirection, ...],
    inner_batches: tuple[ShortcutBatch, ...],
    fitness_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    heldout_fitness_batches: tuple[ShortcutBatch, ...],
    heldout_correct_batches: tuple[ShortcutBatch, ...],
    initial_clean_metrics: ShortcutMetrics,
) -> list[
    tuple[
        int,
        float,
        ForwardTrajectoryMetrics,
        list[dict[str, float]],
    ]
]:
    """Evaluate one deterministic subset of candidates on one CUDA device."""

    torch.cuda.set_device(device)
    worker_center_rule = initialize_backward_rule(
        center_rule_config,
        device=device,
    )
    worker_center_parameters = {
        name: tensor.to(device)
        for name, tensor in center_parameters.items()
    }
    worker_base_state = {
        name: tensor.to(device)
        for name, tensor in base_state.items()
    }
    worker_inner_batches = tuple(batch.to(device) for batch in inner_batches)
    worker_fitness_batches = tuple(
        batch.to(device) for batch in fitness_batches
    )
    worker_correct_batches = tuple(
        batch.to(device) for batch in correct_batches
    )
    worker_heldout_fitness_batches = tuple(
        batch.to(device) for batch in heldout_fitness_batches
    )
    worker_heldout_correct_batches = tuple(
        batch.to(device) for batch in heldout_correct_batches
    )
    direction_indices = {spec[1] for spec in candidate_specs}
    worker_directions = {
        index: move_eggroll_direction(directions[index], device)
        for index in direction_indices
    }

    results = []
    for candidate_index, direction_index, sign in candidate_specs:
        fitness, trajectory, statistics = train_candidate(
            config,
            base_state=worker_base_state,
            center_rule=worker_center_rule,
            center_parameters=worker_center_parameters,
            direction=worker_directions[direction_index],
            sign=sign,
            inner_batches=worker_inner_batches,
            fitness_batches=worker_fitness_batches,
            correct_batches=worker_correct_batches,
            initial_clean_metrics=initial_clean_metrics,
            device=device,
            capture_statistics=(
                isinstance(worker_center_rule, AttentionRoutingRule)
                or candidate_index == 0
            ),
            heldout_fitness_batches=worker_heldout_fitness_batches,
            heldout_correct_batches=worker_heldout_correct_batches,
        )
        results.append(
            (candidate_index, fitness, trajectory, statistics)
        )
    return results


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
    objective: float,
    config: ShortcutCreditExperimentConfig,
) -> bool:
    if state.ema_fitness is None:
        state.ema_fitness = objective
    else:
        state.ema_fitness = (
            config.plateau_ema_decay * state.ema_fitness
            + (1.0 - config.plateau_ema_decay) * objective
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
    pair_deltas = fitnesses[0::2] - fitnesses[1::2]
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
    masked_losses = torch.tensor(
        [metrics.mode_loss["masked"] for metrics in clean_metrics]
    )
    incorrect_losses = torch.tensor(
        [metrics.mode_loss["incorrect"] for metrics in clean_metrics]
    )
    worst_mode_losses = torch.maximum(masked_losses, incorrect_losses)
    clean_unique_values = torch.tensor(
        [
            metrics.unique_value_prediction_count
            for metrics in clean_metrics
        ],
        dtype=torch.float32,
    )
    clean_mode_fractions = torch.tensor(
        [metrics.prediction_mode_fraction for metrics in clean_metrics]
    )
    best_index = int(fitnesses.argmax())
    best_clean = clean_metrics[best_index]
    best_correct = correct_metrics[best_index]
    robust_index = max(
        range(len(clean_metrics)),
        key=lambda index: min(
            clean_metrics[index].mode_accuracy["masked"],
            clean_metrics[index].mode_accuracy["incorrect"],
        ),
    )
    robust_clean = clean_metrics[robust_index]
    robust_correct = correct_metrics[robust_index]
    return {
        "fitness/mean": float(fitnesses.mean()),
        "fitness/std": float(fitnesses.std(unbiased=False)),
        "fitness/pair_delta_mean_abs": float(pair_deltas.abs().mean()),
        "fitness/pair_delta_rms": float(pair_deltas.square().mean().sqrt()),
        "fitness/max": float(fitnesses.max()),
        "clean/loss_mean": float(clean_losses.mean()),
        "clean/accuracy_mean": float(clean_accuracies.mean()),
        "clean/masked_accuracy_mean": float(masked_accuracies.mean()),
        "clean/incorrect_accuracy_mean": float(incorrect_accuracies.mean()),
        "clean/masked_loss_mean": float(masked_losses.mean()),
        "clean/incorrect_loss_mean": float(incorrect_losses.mean()),
        "clean/worst_mode_loss_mean": float(worst_mode_losses.mean()),
        "clean/unique_value_predictions_mean": float(
            clean_unique_values.mean()
        ),
        "clean/prediction_mode_fraction_mean": float(
            clean_mode_fractions.mean()
        ),
        "correct_leak/accuracy_mean": float(correct_accuracies.mean()),
        "best/candidate_index": best_index,
        "best/fitness": float(fitnesses[best_index]),
        "best/clean_loss": best_clean.loss,
        "best/clean_accuracy": best_clean.accuracy,
        "best/masked_accuracy": best_clean.mode_accuracy["masked"],
        "best/incorrect_accuracy": best_clean.mode_accuracy["incorrect"],
        "best/correct_leak_accuracy": best_correct.accuracy,
        "best/unique_value_predictions": float(
            best_clean.unique_value_prediction_count
        ),
        "best/prediction_mode_fraction": (
            best_clean.prediction_mode_fraction
        ),
        "robust/candidate_index": robust_index,
        "robust/min_mode_accuracy": min(
            robust_clean.mode_accuracy["masked"],
            robust_clean.mode_accuracy["incorrect"],
        ),
        "robust/fitness": float(fitnesses[robust_index]),
        "robust/clean_loss": robust_clean.loss,
        "robust/clean_accuracy": robust_clean.accuracy,
        "robust/masked_accuracy": robust_clean.mode_accuracy["masked"],
        "robust/incorrect_accuracy": (
            robust_clean.mode_accuracy["incorrect"]
        ),
        "robust/correct_leak_accuracy": robust_correct.accuracy,
        "robust/unique_value_predictions": float(
            robust_clean.unique_value_prediction_count
        ),
        "robust/prediction_mode_fraction": (
            robust_clean.prediction_mode_fraction
        ),
    }


def heldout_candidate_summary(
    fitnesses: Tensor,
    outer_clean_metrics: list[ShortcutMetrics],
    heldout_clean_metrics: list[ShortcutMetrics],
    heldout_correct_metrics: list[ShortcutMetrics],
) -> dict[str, float]:
    """Measure whether outer-selected candidates generalize to fresh data."""

    if not (
        len(outer_clean_metrics)
        == len(heldout_clean_metrics)
        == len(heldout_correct_metrics)
        == fitnesses.numel()
    ):
        raise ValueError("candidate metric groups must have matching lengths")
    heldout_masked = torch.tensor(
        [
            metrics.mode_accuracy["masked"]
            for metrics in heldout_clean_metrics
        ]
    )
    heldout_incorrect = torch.tensor(
        [
            metrics.mode_accuracy["incorrect"]
            for metrics in heldout_clean_metrics
        ]
    )
    heldout_min_accuracy = torch.minimum(
        heldout_masked,
        heldout_incorrect,
    )
    heldout_worst_loss = torch.tensor(
        [
            max(
                metrics.mode_loss["masked"],
                metrics.mode_loss["incorrect"],
            )
            for metrics in heldout_clean_metrics
        ]
    )
    heldout_correct = torch.tensor(
        [metrics.accuracy for metrics in heldout_correct_metrics]
    )
    centered_fitness = fitnesses.float().cpu() - fitnesses.float().cpu().mean()
    heldout_objective = -heldout_worst_loss
    centered_heldout = heldout_objective - heldout_objective.mean()
    denominator = (
        centered_fitness.square().sum()
        * centered_heldout.square().sum()
    ).sqrt()
    objective_correlation = (
        float(
            (centered_fitness * centered_heldout).sum()
            / denominator
        )
        if float(denominator) > 0
        else 0.0
    )
    best_index = int(fitnesses.argmax())
    robust_index = max(
        range(len(outer_clean_metrics)),
        key=lambda index: min(
            outer_clean_metrics[index].mode_accuracy["masked"],
            outer_clean_metrics[index].mode_accuracy["incorrect"],
        ),
    )
    best_clean = heldout_clean_metrics[best_index]
    robust_clean = heldout_clean_metrics[robust_index]
    return {
        "heldout_candidates/masked_accuracy_mean": float(
            heldout_masked.mean()
        ),
        "heldout_candidates/incorrect_accuracy_mean": float(
            heldout_incorrect.mean()
        ),
        "heldout_candidates/min_mode_accuracy_mean": float(
            heldout_min_accuracy.mean()
        ),
        "heldout_candidates/correct_leak_accuracy_mean": float(
            heldout_correct.mean()
        ),
        "heldout_candidates/outer_fitness_correlation": (
            objective_correlation
        ),
        "best/heldout_clean_loss": best_clean.loss,
        "best/heldout_masked_accuracy": best_clean.mode_accuracy["masked"],
        "best/heldout_incorrect_accuracy": (
            best_clean.mode_accuracy["incorrect"]
        ),
        "best/heldout_min_mode_accuracy": float(
            heldout_min_accuracy[best_index]
        ),
        "best/heldout_correct_leak_accuracy": float(
            heldout_correct[best_index]
        ),
        "robust/heldout_clean_loss": robust_clean.loss,
        "robust/heldout_masked_accuracy": (
            robust_clean.mode_accuracy["masked"]
        ),
        "robust/heldout_incorrect_accuracy": (
            robust_clean.mode_accuracy["incorrect"]
        ),
        "robust/heldout_min_mode_accuracy": float(
            heldout_min_accuracy[robust_index]
        ),
        "robust/heldout_correct_leak_accuracy": float(
            heldout_correct[robust_index]
        ),
    }


def routing_population_summary(
    fitnesses: Tensor,
    clean_metrics: list[ShortcutMetrics],
    candidate_statistics: list[list[dict[str, float]]],
) -> dict[str, float]:
    """Report whether routing selectivity exists and fitness rewards it."""

    if not candidate_statistics or any(
        not statistics for statistics in candidate_statistics
    ):
        return {}
    query_relative_gates = torch.tensor(
        [
            sum(
                item["routing_leak_relative_gate"]
                for item in statistics
            )
            / len(statistics)
            for statistics in candidate_statistics
        ],
        dtype=torch.float32,
    )
    hint_source_relative_gates = torch.tensor(
        [
            sum(
                item["routing_hint_source_relative_gate"]
                for item in statistics
            )
            / len(statistics)
            for statistics in candidate_statistics
        ],
        dtype=torch.float32,
    )
    selectivity = 1.0 - query_relative_gates
    centered_fitness = fitnesses.float().cpu() - fitnesses.float().cpu().mean()
    centered_selectivity = selectivity - selectivity.mean()
    denominator = (
        centered_fitness.square().sum()
        * centered_selectivity.square().sum()
    ).sqrt()
    correlation = (
        float(
            (centered_fitness * centered_selectivity).sum()
            / denominator
        )
        if float(denominator) > 0
        else 0.0
    )
    hint_source_selectivity = 1.0 - hint_source_relative_gates
    centered_hint_source_selectivity = (
        hint_source_selectivity - hint_source_selectivity.mean()
    )
    hint_source_denominator = (
        centered_fitness.square().sum()
        * centered_hint_source_selectivity.square().sum()
    ).sqrt()
    hint_source_correlation = (
        float(
            (
                centered_fitness
                * centered_hint_source_selectivity
            ).sum()
            / hint_source_denominator
        )
        if float(hint_source_denominator) > 0
        else 0.0
    )
    best_index = int(fitnesses.argmax())
    robust_index = max(
        range(len(clean_metrics)),
        key=lambda index: min(
            clean_metrics[index].mode_accuracy["masked"],
            clean_metrics[index].mode_accuracy["incorrect"],
        ),
    )
    return {
        "backward/population_leak_relative_gate_min": float(
            query_relative_gates.min()
        ),
        "backward/population_leak_relative_gate_mean": float(
            query_relative_gates.mean()
        ),
        "backward/population_leak_relative_gate_max": float(
            query_relative_gates.max()
        ),
        "backward/population_selective_fraction": float(
            (query_relative_gates < 0.9).float().mean()
        ),
        "backward/selectivity_fitness_correlation": correlation,
        "backward/best_fitness_leak_relative_gate": float(
            query_relative_gates[best_index]
        ),
        "backward/robust_candidate_leak_relative_gate": float(
            query_relative_gates[robust_index]
        ),
        "backward/population_hint_source_relative_gate_min": float(
            hint_source_relative_gates.min()
        ),
        "backward/population_hint_source_relative_gate_mean": float(
            hint_source_relative_gates.mean()
        ),
        "backward/population_hint_source_relative_gate_max": float(
            hint_source_relative_gates.max()
        ),
        "backward/population_hint_source_selective_fraction": float(
            (hint_source_relative_gates < 0.9).float().mean()
        ),
        "backward/hint_source_selectivity_fitness_correlation": (
            hint_source_correlation
        ),
        "backward/best_fitness_hint_source_relative_gate": float(
            hint_source_relative_gates[best_index]
        ),
        "backward/robust_candidate_hint_source_relative_gate": float(
            hint_source_relative_gates[robust_index]
        ),
    }


def center_update_summary(
    backward_rule: BackwardRule,
    previous_parameters: dict[str, Tensor],
) -> dict[str, float]:
    """Summarize the actual EGGROLL center displacement."""

    squared_update = 0.0
    squared_center = 0.0
    parameter_count = 0
    for name, parameter in backward_rule.named_parameters():
        previous = previous_parameters[name]
        squared_update += float((parameter - previous).square().sum())
        squared_center += float(previous.square().sum())
        parameter_count += parameter.numel()

    update_rms = (squared_update / parameter_count) ** 0.5
    center_rms = (squared_center / parameter_count) ** 0.5
    gates = backward_rule.gates.detach()
    return {
        "outer/update_rms": update_rms,
        "outer/update_to_center_rms": update_rms / max(center_rms, 1e-12),
        "backward/center_gate_abs_mean": float(gates.abs().mean()),
        "backward/center_gate_abs_max": float(gates.abs().max()),
    }


@torch.inference_mode()
def center_routing_summary(
    backward_rule: BackwardRule,
    token_ids: Tensor,
) -> dict[str, float]:
    if not isinstance(backward_rule, AttentionRoutingRule):
        return {}
    capture_statistics = backward_rule.capture_statistics
    backward_rule.capture_statistics = True
    backward_rule.clear_statistics()
    backward_rule.attention_gates(token_ids)
    statistics = backward_rule.statistics.pop()
    backward_rule.capture_statistics = capture_statistics
    return {
        f"backward/center_{key}": value
        for key, value in statistics.items()
        if key != "layer"
    }


def save_checkpoint(
    path: Path,
    *,
    backward_rule: BackwardRule,
    config: ShortcutCreditExperimentConfig,
    generation: int,
    horizon: int,
    plateau_state: PlateauState,
) -> None:
    torch.save(
        {
            "experiment": "learned_backward_shortcuts",
            "backward_rule_type": (
                "attention_router"
                if isinstance(backward_rule, AttentionRoutingRule)
                else "gradient_transformer"
            ),
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
) -> tuple[BackwardRule, int, int, PlateauState]:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("experiment") != "learned_backward_shortcuts":
        raise ValueError("checkpoint belongs to a different experiment")
    rule_type = checkpoint.get(
        "backward_rule_type",
        "gradient_transformer",
    )
    if rule_type == "attention_router":
        rule_config: RuleConfig = AttentionRoutingRuleConfig(
            **checkpoint["backward_rule_config"]
        )
    elif rule_type == "gradient_transformer":
        rule_config = BackwardRuleConfig(
            **checkpoint["backward_rule_config"]
        )
    else:
        raise ValueError(f"unknown checkpoint backward rule: {rule_type}")
    backward_rule = initialize_backward_rule(rule_config, device=device)
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
    candidate_devices = parse_candidate_devices(
        config.candidate_devices,
        device,
    )
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2) + "\n")

    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    backward_config = make_rule_config(config, vocabulary)
    if config.resume is None:
        center_rule = initialize_fresh_backward_rule(
            config,
            vocabulary,
            device=device,
        )
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

    wandb_run = maybe_initialize_wandb(config)
    started_at = time.monotonic()
    for generation in range(start_generation, config.generations):
        generation_started_at = time.monotonic()
        generation_seed = config.seed * 1_000_003 + generation * 10_007
        initialization_seed = generation_seed + 1
        heldout_generator = torch.Generator().manual_seed(
            generation_seed + 4
        )
        heldout_fitness_batches = make_fitness_batches(
            config.fitness_examples,
            min_length=config.min_length,
            max_length=config.max_length,
            batch_size=config.fitness_batch_size,
            generator=heldout_generator,
            vocabulary=vocabulary,
            leak_placement=config.leak_placement,
            device=device,
        )
        heldout_correct_batches = make_mode_batches(
            config.correct_eval_examples,
            leak_mode="correct",
            config=config,
            vocabulary=vocabulary,
            generator=heldout_generator,
            device=device,
        )
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
        masked_inner_generator = torch.Generator().manual_seed(
            generation_seed + 2
        )
        masked_inner_batches = make_inner_batches(
            config,
            horizon=horizon,
            vocabulary=vocabulary,
            generator=masked_inner_generator,
            device=device,
            leak_mode="masked",
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

        candidate_specs = tuple(
            (
                2 * direction_index + sign_index,
                direction_index,
                sign,
            )
            for direction_index in range(len(directions))
            for sign_index, sign in enumerate((1, -1))
        )
        candidate_outputs: list[
            tuple[
                int,
                float,
                ForwardTrajectoryMetrics,
                list[dict[str, float]],
            ]
        ] = []
        if len(candidate_devices) == 1:
            for candidate_index, direction_index, sign in candidate_specs:
                fitness, trajectory, statistics = train_candidate(
                    config,
                    base_state=base_state,
                    center_rule=center_rule,
                    center_parameters=center_parameters,
                    direction=directions[direction_index],
                    sign=sign,
                    inner_batches=inner_batches,
                    fitness_batches=fitness_batches,
                    correct_batches=correct_batches,
                    initial_clean_metrics=initial_clean_metrics,
                    device=device,
                    capture_statistics=(
                        isinstance(center_rule, AttentionRoutingRule)
                        or candidate_index == 0
                    ),
                    heldout_fitness_batches=heldout_fitness_batches,
                    heldout_correct_batches=heldout_correct_batches,
                )
                candidate_outputs.append(
                    (
                        candidate_index,
                        fitness,
                        trajectory,
                        statistics,
                    )
                )
        else:
            shards = shard_candidate_specs(
                candidate_specs,
                len(candidate_devices),
            )
            with ThreadPoolExecutor(
                max_workers=len(candidate_devices)
            ) as executor:
                futures = [
                    executor.submit(
                        train_candidate_shard,
                        config,
                        candidate_specs=shard,
                        device=worker_device,
                        base_state=base_state,
                        center_rule_config=center_rule.config,
                        center_parameters=center_parameters,
                        directions=directions,
                        inner_batches=inner_batches,
                        fitness_batches=fitness_batches,
                        correct_batches=correct_batches,
                        heldout_fitness_batches=heldout_fitness_batches,
                        heldout_correct_batches=heldout_correct_batches,
                        initial_clean_metrics=initial_clean_metrics,
                    )
                    for shard, worker_device in zip(
                        shards,
                        candidate_devices,
                    )
                    if shard
                ]
                for future in futures:
                    candidate_outputs.extend(future.result())

        candidate_outputs.sort(key=lambda result: result[0])
        fitness_values = []
        clean_results = []
        correct_results = []
        heldout_clean_results = []
        heldout_correct_results = []
        candidate_statistics: list[list[dict[str, float]]] = []
        captured_statistics: list[dict[str, float]] = []
        for candidate_index, fitness, trajectory, statistics in (
            candidate_outputs
        ):
            fitness_values.append(fitness)
            clean_results.append(trajectory.clean)
            correct_results.append(trajectory.correct)
            if (
                trajectory.heldout_clean is None
                or trajectory.heldout_correct is None
            ):
                raise RuntimeError(
                    "candidate held-out metrics were not produced"
                )
            heldout_clean_results.append(trajectory.heldout_clean)
            heldout_correct_results.append(trajectory.heldout_correct)
            candidate_statistics.append(statistics)
            if candidate_index == 0 and statistics:
                captured_statistics = statistics

        center_fitness, center_trajectory, _ = train_candidate(
            config,
            base_state=base_state,
            center_rule=center_rule,
            center_parameters=center_parameters,
            direction=directions[0],
            sign=1,
            inner_batches=inner_batches,
            fitness_batches=fitness_batches,
            correct_batches=correct_batches,
            initial_clean_metrics=initial_clean_metrics,
            device=device,
            capture_statistics=False,
            perturbation_sigma=0.0,
            heldout_fitness_batches=heldout_fitness_batches,
            heldout_correct_batches=heldout_correct_batches,
        )
        ordinary_trajectory = train_forward_trajectory(
            config,
            base_state=base_state,
            backward_rule=None,
            inner_batches=inner_batches,
            fitness_batches=fitness_batches,
            correct_batches=correct_batches,
            heldout_fitness_batches=heldout_fitness_batches,
            heldout_correct_batches=heldout_correct_batches,
            device=device,
        )
        masked_training_trajectory = train_forward_trajectory(
            config,
            base_state=base_state,
            backward_rule=None,
            inner_batches=masked_inner_batches,
            fitness_batches=fitness_batches,
            correct_batches=correct_batches,
            heldout_fitness_batches=heldout_fitness_batches,
            heldout_correct_batches=heldout_correct_batches,
            device=device,
        )
        ordinary_fitness = candidate_fitness(
            config.fitness_objective,
            initial_clean_metrics,
            ordinary_trajectory.clean,
        )
        masked_training_fitness = candidate_fitness(
            config.fitness_objective,
            initial_clean_metrics,
            masked_training_trajectory.clean,
        )
        center_clean = center_trajectory.clean
        center_correct = center_trajectory.correct
        ordinary_clean = ordinary_trajectory.clean
        ordinary_correct = ordinary_trajectory.correct
        masked_training_clean = masked_training_trajectory.clean
        masked_training_correct = masked_training_trajectory.correct
        if any(
            metrics is None
            for metrics in (
                center_trajectory.heldout_clean,
                center_trajectory.heldout_correct,
                ordinary_trajectory.heldout_clean,
                ordinary_trajectory.heldout_correct,
                masked_training_trajectory.heldout_clean,
                masked_training_trajectory.heldout_correct,
            )
        ):
            raise RuntimeError("held-out trajectory metrics were not produced")
        center_heldout_clean = center_trajectory.heldout_clean
        center_heldout_correct = center_trajectory.heldout_correct
        ordinary_heldout_clean = ordinary_trajectory.heldout_clean
        ordinary_heldout_correct = ordinary_trajectory.heldout_correct
        masked_training_heldout_clean = (
            masked_training_trajectory.heldout_clean
        )
        masked_training_heldout_correct = (
            masked_training_trajectory.heldout_correct
        )
        assert center_heldout_clean is not None
        assert center_heldout_correct is not None
        assert ordinary_heldout_clean is not None
        assert ordinary_heldout_correct is not None
        assert masked_training_heldout_clean is not None
        assert masked_training_heldout_correct is not None
        fitness_tensor = torch.tensor(fitness_values, device=device)
        outer_learning_rate = linear_outer_learning_rate(config, generation)
        standardized = paper_eggroll_update(
            center_rule,
            directions,
            fitness_tensor,
            sigma=config.sigma,
            learning_rate=outer_learning_rate,
        )
        if isinstance(center_rule, AttentionRoutingRule):
            center_rule.project_parameters_()
        summary = candidate_summary(
            fitness_tensor.cpu(),
            clean_results,
            correct_results,
        )
        summary.update(
            heldout_candidate_summary(
                fitness_tensor.cpu(),
                clean_results,
                heldout_clean_results,
                heldout_correct_results,
            )
        )
        summary.update(
            center_rule_summary(
                center_fitness,
                center_clean,
                center_correct,
            )
        )
        summary.update(
            trajectory_summary(
                "ordinary_rule",
                ordinary_fitness,
                ordinary_clean,
                ordinary_correct,
            )
        )
        summary.update(
            trajectory_summary(
                "masked_training",
                masked_training_fitness,
                masked_training_clean,
                masked_training_correct,
            )
        )
        summary.update(
            trajectory_summary(
                "heldout_center_rule",
                None,
                center_heldout_clean,
                center_heldout_correct,
            )
        )
        summary.update(
            trajectory_summary(
                "heldout_ordinary_rule",
                None,
                ordinary_heldout_clean,
                ordinary_heldout_correct,
            )
        )
        summary.update(
            trajectory_summary(
                "heldout_masked_training",
                None,
                masked_training_heldout_clean,
                masked_training_heldout_correct,
            )
        )
        center_min_accuracy = min(
            center_clean.mode_accuracy["masked"],
            center_clean.mode_accuracy["incorrect"],
        )
        ordinary_min_accuracy = min(
            ordinary_clean.mode_accuracy["masked"],
            ordinary_clean.mode_accuracy["incorrect"],
        )
        masked_training_min_accuracy = min(
            masked_training_clean.mode_accuracy["masked"],
            masked_training_clean.mode_accuracy["incorrect"],
        )
        center_heldout_min_accuracy = min(
            center_heldout_clean.mode_accuracy["masked"],
            center_heldout_clean.mode_accuracy["incorrect"],
        )
        ordinary_heldout_min_accuracy = min(
            ordinary_heldout_clean.mode_accuracy["masked"],
            ordinary_heldout_clean.mode_accuracy["incorrect"],
        )
        summary.update(
            {
                "comparison/center_minus_ordinary_min_accuracy": (
                    center_min_accuracy - ordinary_min_accuracy
                ),
                "comparison/masked_training_minus_ordinary_min_accuracy": (
                    masked_training_min_accuracy - ordinary_min_accuracy
                ),
                "comparison/center_clean_loss_improvement_over_ordinary": (
                    ordinary_clean.loss - center_clean.loss
                ),
                "comparison/masked_training_clean_loss_improvement_over_ordinary": (
                    ordinary_clean.loss - masked_training_clean.loss
                ),
                "heldout_comparison/center_minus_ordinary_min_accuracy": (
                    center_heldout_min_accuracy
                    - ordinary_heldout_min_accuracy
                ),
                "heldout_comparison/center_clean_loss_improvement_over_ordinary": (
                    ordinary_heldout_clean.loss - center_heldout_clean.loss
                ),
            }
        )
        if isinstance(center_rule, AttentionRoutingRule):
            summary.update(
                routing_population_summary(
                    fitness_tensor,
                    clean_results,
                    candidate_statistics,
                )
            )
        summary.update(center_update_summary(center_rule, center_parameters))
        summary.update(
            center_routing_summary(
                center_rule,
                inner_batches[-1].input_ids,
            )
        )
        summary.update(
            {
                "generation": generation,
                "horizon": horizon,
                "population_size": config.population_size,
                "search/sigma": config.sigma,
                "candidate_device_count": len(candidate_devices),
                "outer_learning_rate": outer_learning_rate,
                "fitness/standardized_mean": float(standardized.mean()),
                "fitness/standardized_std": float(
                    standardized.std(unbiased=False)
                ),
                "initial_clean/loss": initial_clean_metrics.loss,
                "initial_clean/accuracy": initial_clean_metrics.accuracy,
                "initial_clean/unique_value_predictions": (
                    initial_clean_metrics.unique_value_prediction_count
                ),
                "initial_clean/prediction_mode_fraction": (
                    initial_clean_metrics.prediction_mode_fraction
                ),
                "timing/generation_seconds": (
                    time.monotonic() - generation_started_at
                ),
                "timing/elapsed_seconds": time.monotonic() - started_at,
            }
        )
        if captured_statistics:
            for key in captured_statistics[0]:
                if key == "layer":
                    continue
                metric_key = (
                    "gradient_cosine" if key == "cosine" else key
                )
                summary[f"backward/{metric_key}_mean"] = sum(
                    item[key] for item in captured_statistics
                ) / len(captured_statistics)

        # Mean fitness cannot be compared across generations because each
        # generation starts from a different model and initial clean loss.
        # Post-training clean CE is the stable cross-generation objective.
        plateau_objective = -summary[
            (
                "clean/worst_mode_loss_mean"
                if config.fitness_objective == "worst_mode_ce"
                else "clean/loss_mean"
            )
        ]
        promote_horizon = update_plateau_state(
            plateau_state,
            objective=plateau_objective,
            config=config,
        )
        summary.update(
            {
                "curriculum/objective_negative_clean_loss": plateau_objective,
                "curriculum/ema_objective": plateau_state.ema_fitness,
                "curriculum/stale_generations": plateau_state.stale_generations,
                "curriculum/promoted": float(
                    promote_horizon and horizon < config.max_horizon
                ),
            }
        )

        with metrics_path.open("a") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
        if wandb_run is not None:
            wandb_run.log(summary, step=generation)

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
    parser.add_argument(
        "--leak-placement",
        choices=("suffix", "random_list"),
        default="suffix",
        help="place the leak at the suffix or after a random list value",
    )
    parser.add_argument("--forward-learning-rate", type=float, default=3e-4)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--outer-learning-rate", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--backward-d-model", type=int, default=128)
    parser.add_argument(
        "--backward-rule-type",
        choices=("gradient_transformer", "attention_router"),
        default="gradient_transformer",
    )
    parser.add_argument(
        "--route-output-projection",
        action="store_true",
        help=(
            "apply attention routing to output-projection parameter gradients"
        ),
    )
    parser.add_argument(
        "--shared-routing-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse one backward suppression map across all layers and heads",
    )
    parser.add_argument(
        "--fitness-objective",
        choices=("mean_clean_ce", "worst_mode_ce"),
        default="mean_clean_ce",
        help="candidate objective used by the EGGROLL update",
    )
    parser.add_argument("--forward-layers", type=int, default=3)
    parser.add_argument("--backward-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--candidate-devices",
        help="comma-separated CUDA devices for parallel candidate shards",
    )
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
