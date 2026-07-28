"""Run EGGROLL evolution of a shortcut-resistant backward rule."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union

import torch
from torch import Tensor

from .direction_sampling import sample_function_diverse_directions
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
    make_clean_pointer_batch,
    make_shortcut_batch,
    move_eggroll_direction,
    paper_eggroll_update,
    sample_eggroll_direction,
    shortcut_loss,
)
from .tokens import PointerNextVocabulary


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
    acceptance_fitness_examples: int = 0
    fitness_batch_size: int = 64
    correct_eval_examples: int = 128
    heldout_examples: int = 128
    report_interval: int = 1
    task_variant: str = "shortcut"
    min_length: int = 8
    max_length: int = 32
    fitness_length: int | None = None
    heldout_length: int | None = None
    leak_placement: LeakPlacement = "suffix"
    forward_learning_rate: float = 3e-4
    forward_training_precision: str = "fp32"
    sigma: float = 0.08
    outer_learning_rate: float = 0.1
    outer_update_rule: str = "paper_standardized"
    elite_count: int = 8
    elite_interpolation: float = 0.5
    elite_backtracking: bool = False
    elite_rejection_sigma_decay: float = 0.5
    elite_min_sigma: float = 1e-4
    elite_acceptance_patience: int = 3
    elite_acceptance_sigma_growth: float = 2.0
    elite_acceptance_trajectories: int = 1
    candidate_ranking_trajectories: int = 1
    adaptive_elite_counts: str | None = None
    deduplicate_antithetic_elites: bool = True
    adaptive_commit_scale: float | None = None
    adaptive_commit_scale_multiplier: float = 2.0
    horizon_promotion_mode: str = "plateau"
    horizon_rejection_patience: int = 5
    horizon_probe_min_improvement: float = 0.0
    horizon_score_window: int = 10
    horizon_min_generations: int = 20
    horizon_max_generations: int = 30
    horizon_failed_extension_limit: int = 2
    d_model: int = 128
    backward_d_model: int = 128
    backward_rule_type: str = "gradient_transformer"
    routing_credit_mode: str = "suppress_renorm"
    route_output_projection: bool = False
    shared_routing_map: bool = True
    condition_on_forward_state: bool = False
    fitness_objective: str = "mean_clean_ce"
    fitness_checkpoints: str | None = None
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
    resume_horizon: int | None = None
    candidate_devices: str | None = None
    vectorized_population: bool = False
    vectorized_chunk_size: int = 16
    successive_halving_rungs: str | None = None
    direction_sampler: str = "random"
    direction_candidate_multiplier: int = 4
    direction_probe_examples: int = 8
    direction_signature_size: int = 1024

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
            self.heldout_examples,
            self.report_interval,
            self.min_length,
            self.max_length,
            self.d_model,
            self.backward_d_model,
            self.forward_layers,
            self.backward_layers,
            self.heads,
            self.checkpoint_interval,
            self.elite_count,
            self.elite_acceptance_patience,
            self.elite_acceptance_trajectories,
            self.candidate_ranking_trajectories,
            self.horizon_rejection_patience,
            self.horizon_score_window,
            self.horizon_min_generations,
            self.horizon_max_generations,
            self.horizon_failed_extension_limit,
            self.vectorized_chunk_size,
            self.direction_candidate_multiplier,
            self.direction_probe_examples,
            self.direction_signature_size,
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
        if self.routing_credit_mode not in {
            "suppress_renorm",
            "signed",
        }:
            raise ValueError("unknown routing credit mode")
        if self.forward_training_precision not in {"fp32", "bf16"}:
            raise ValueError("unknown forward training precision")
        if self.direction_sampler not in {"random", "function_diverse"}:
            raise ValueError("unknown direction sampler")
        if (
            self.direction_sampler == "function_diverse"
            and self.backward_rule_type != "attention_router"
        ):
            raise ValueError(
                "function-diverse directions require an attention router"
            )
        if (
            self.routing_credit_mode != "suppress_renorm"
            and self.backward_rule_type != "attention_router"
        ):
            raise ValueError(
                "nonstandard routing credit requires an attention router"
            )
        if self.vectorized_population and (
            self.backward_rule_type != "attention_router"
            or self.route_output_projection
            or self.condition_on_forward_state
        ):
            raise ValueError(
                "vectorized populations require an unconditioned attention "
                "router without output-projection routing"
            )
        if self.fitness_objective not in {
            "mean_clean_ce",
            "worst_mode_ce",
            "worst_checkpoint_mode_ce",
        }:
            raise ValueError("unknown fitness_objective")
        if self.task_variant not in {"shortcut", "pointer_next_length"}:
            raise ValueError("unknown task variant")
        checkpoints = parse_fitness_checkpoints(self.fitness_checkpoints)
        if self.fitness_objective == "worst_checkpoint_mode_ce":
            if not checkpoints:
                raise ValueError(
                    "worst_checkpoint_mode_ce requires fitness_checkpoints"
                )
            if checkpoints[-1] > self.max_horizon:
                raise ValueError(
                    "fitness checkpoints must not exceed max_horizon"
                )
        elif checkpoints:
            raise ValueError(
                "fitness_checkpoints require worst_checkpoint_mode_ce"
            )
        if self.outer_update_rule not in {
            "paper_standardized",
            "elite_centroid",
        }:
            raise ValueError("unknown outer_update_rule")
        adaptive_counts = parse_adaptive_elite_counts(
            self.adaptive_elite_counts
        )
        halving_rungs = parse_successive_halving_rungs(
            self.successive_halving_rungs
        )
        if adaptive_counts and (
            self.outer_update_rule != "elite_centroid"
            or not self.elite_backtracking
            or not self.vectorized_population
        ):
            raise ValueError(
                "adaptive elite selection requires vectorized elite-centroid "
                "backtracking"
            )
        maximum_elite_count = (
            self.population_size // 2
            if self.deduplicate_antithetic_elites
            else self.population_size
        )
        if adaptive_counts and adaptive_counts[-1] > maximum_elite_count:
            raise ValueError(
                "adaptive elite counts exceed the selectable population"
            )
        if self.adaptive_commit_scale is not None:
            if not adaptive_counts:
                raise ValueError(
                    "adaptive commit-scale search requires adaptive elites"
                )
            if self.adaptive_commit_scale <= 0:
                raise ValueError(
                    "adaptive commit scale must be positive"
                )
            if self.adaptive_commit_scale_multiplier <= 1:
                raise ValueError(
                    "adaptive commit-scale multiplier must exceed 1"
                )
        if halving_rungs:
            if not self.vectorized_population:
                raise ValueError(
                    "successive halving requires a vectorized population"
                )
            if self.horizon_promotion_mode != "fixed":
                raise ValueError(
                    "successive halving requires fixed horizon mode"
                )
            if halving_rungs[-1][0] != self.horizon:
                raise ValueError(
                    "final halving rung must equal the fixed horizon"
                )
            if halving_rungs[0][1] > self.population_size:
                raise ValueError(
                    "halving survivors must not exceed population size"
                )
            if adaptive_counts and halving_rungs[-1][1] < adaptive_counts[-1]:
                raise ValueError(
                    "final halving survivors must cover adaptive elite counts"
                )
            if self.candidate_ranking_trajectories != 1:
                raise ValueError(
                    "successive halving currently supports one ranking trajectory"
                )
            if self.outer_update_rule != "elite_centroid":
                raise ValueError(
                    "successive halving requires elite-centroid updates"
                )
            if self.fitness_checkpoints is not None:
                raise ValueError(
                    "successive halving does not support fitness checkpoints"
                )
        if self.horizon_promotion_mode not in {
            "fixed",
            "plateau",
            "rejection_probe",
            "performance_plateau",
        }:
            raise ValueError("unknown horizon promotion mode")
        if (
            self.horizon_promotion_mode == "fixed"
            and self.horizon != self.max_horizon
        ):
            raise ValueError(
                "fixed horizon mode requires horizon to equal max_horizon"
            )
        if (
            self.horizon_promotion_mode != "fixed"
            and self.report_interval != 1
        ):
            raise ValueError(
                "sparse reporting requires fixed horizon mode"
            )
        if (
            self.horizon_promotion_mode == "rejection_probe"
            and not adaptive_counts
        ):
            raise ValueError(
                "rejection-probe horizon promotion requires adaptive elites"
            )
        if (
            self.outer_update_rule == "elite_centroid"
            and not adaptive_counts
            and self.elite_count > maximum_elite_count
        ):
            raise ValueError(
                "elite_count exceeds the selectable population"
            )
        if not 0 < self.elite_interpolation <= 1:
            raise ValueError("elite_interpolation must be in (0, 1]")
        if not 0 < self.elite_rejection_sigma_decay < 1:
            raise ValueError(
                "elite_rejection_sigma_decay must be in (0, 1)"
            )
        if not 0 < self.elite_min_sigma <= self.sigma:
            raise ValueError("elite_min_sigma must be in (0, sigma]")
        if self.elite_acceptance_sigma_growth <= 1:
            raise ValueError(
                "elite_acceptance_sigma_growth must be greater than 1"
            )
        if self.leak_placement not in {"suffix", "random_list"}:
            raise ValueError("unknown leak placement")
        if self.fitness_examples % 2:
            raise ValueError("fitness_examples must be even")
        if self.acceptance_fitness_examples < 0:
            raise ValueError(
                "acceptance_fitness_examples must be nonnegative"
            )
        if not 2 <= self.min_length <= self.max_length:
            raise ValueError("invalid task length range")
        if self.task_variant == "pointer_next_length":
            if self.fitness_length is None or self.heldout_length is None:
                raise ValueError(
                    "pointer-next length task requires fitness and held-out "
                    "lengths"
                )
            if self.fitness_objective != "mean_clean_ce":
                raise ValueError(
                    "pointer-next length task uses mean clean CE fitness"
                )
            if self.fitness_length <= self.max_length:
                raise ValueError(
                    "fitness length must exceed the training length range"
                )
            if self.heldout_length <= self.fitness_length:
                raise ValueError(
                    "held-out length must exceed the fitness length"
                )
        elif self.fitness_length is not None or self.heldout_length is not None:
            raise ValueError(
                "fixed evaluation lengths require pointer-next length task"
            )
        if self.horizon > self.max_horizon:
            raise ValueError("horizon must not exceed max_horizon")
        if self.resume_horizon is not None:
            if self.resume is None:
                raise ValueError("resume_horizon requires a resume checkpoint")
            if not 1 <= self.resume_horizon <= self.max_horizon:
                raise ValueError(
                    "resume_horizon must be in [1, max_horizon]"
                )
        if not 0 <= self.plateau_ema_decay < 1:
            raise ValueError("plateau_ema_decay must be in [0, 1)")
        if min(
            self.forward_learning_rate,
            self.sigma,
            self.outer_learning_rate,
        ) <= 0:
            raise ValueError("learning rates and sigma must be positive")
        if self.horizon_probe_min_improvement < 0:
            raise ValueError(
                "horizon probe minimum improvement must be nonnegative"
            )
        if self.horizon_min_generations < self.horizon_score_window:
            raise ValueError(
                "horizon minimum generations must cover the score window"
            )
        if self.horizon_max_generations < self.horizon_min_generations:
            raise ValueError(
                "horizon maximum generations must be at least the minimum"
            )


@dataclass
class PlateauState:
    ema_fitness: float | None = None
    best_ema_fitness: float = float("-inf")
    stale_generations: int = 0
    search_sigma: float | None = None
    commit_scale: float | None = None
    consecutive_accepted_updates: int = 0
    consecutive_rejected_updates: int = 0
    horizon_scores: list[float] = field(default_factory=list)
    horizon_generations: int = 0
    horizon_best_average: float = float("-inf")
    horizon_stale_generations: int = 0
    horizon_reference_average: float | None = None
    failed_horizon_extensions: int = 0


@dataclass(frozen=True)
class ForwardTrajectoryMetrics:
    clean: ShortcutMetrics
    correct: ShortcutMetrics
    heldout_clean: ShortcutMetrics | None = None
    heldout_correct: ShortcutMetrics | None = None
    checkpoint_clean: tuple[tuple[int, ShortcutMetrics], ...] = ()


@dataclass(frozen=True)
class AdaptiveEliteResult:
    accepted: bool
    selected_count: int
    selected_indices: Tensor
    selected_parameters: dict[str, Tensor]
    center_fitnesses: tuple[float, ...]
    selected_fitnesses: tuple[float, ...]
    mean_fitness_by_count: dict[int, float]
    selected_commit_scale: float | None = None
    selection_center_fitness: float | None = None
    selection_selected_fitness: float | None = None
    selection_fitness_by_proposal: tuple[
        tuple[int, float, float], ...
    ] = ()


@dataclass(frozen=True)
class HorizonProbeResult:
    promoted: bool
    current_fitness: float
    longer_fitness: float
    next_horizon: int


@dataclass(frozen=True)
class PerformanceHorizonDecision:
    promote: bool = False
    stop: bool = False
    stop_reason: str | None = None
    rolling_average: float | None = None
    plateau_detected: bool = False
    maximum_dwell_reached: bool = False
    extension_improved: bool | None = None
    completed_horizon_average: float | None = None


CandidateRankingInput = tuple[
    dict[str, Tensor],
    tuple[ShortcutBatch, ...],
    ShortcutMetrics,
]


def parse_adaptive_elite_counts(
    value: str | None,
) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        counts = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError(
            "adaptive elite counts must be integers"
        ) from error
    if (
        not counts
        or any(count < 1 for count in counts)
        or tuple(sorted(set(counts))) != counts
    ):
        raise ValueError(
            "adaptive elite counts must be unique increasing positives"
        )
    return counts


def adaptive_commit_scale_grid(
    center: float,
    multiplier: float,
) -> tuple[float, float, float]:
    if center <= 0:
        raise ValueError("commit-scale center must be positive")
    if multiplier <= 1:
        raise ValueError("commit-scale multiplier must exceed 1")
    return (center / multiplier, center, center * multiplier)


def parse_successive_halving_rungs(
    value: str | None,
) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    try:
        rungs = tuple(
            tuple(int(part.strip()) for part in item.split(":"))
            for item in value.split(",")
        )
    except ValueError as error:
        raise ValueError(
            "successive halving rungs must be horizon:survivors pairs"
        ) from error
    if (
        not rungs
        or any(len(rung) != 2 for rung in rungs)
        or any(horizon < 1 or survivors < 1 for horizon, survivors in rungs)
        or tuple(sorted(horizon for horizon, _ in rungs))
        != tuple(horizon for horizon, _ in rungs)
        or len({horizon for horizon, _ in rungs}) != len(rungs)
        or any(
            next_survivors > survivors
            for (_, survivors), (_, next_survivors) in zip(rungs, rungs[1:])
        )
    ):
        raise ValueError(
            "halving horizons must increase and survivor counts must decrease"
        )
    return rungs


def parse_fitness_checkpoints(value: str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        checkpoints = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("fitness checkpoints must be integers") from error
    if (
        not checkpoints
        or any(checkpoint < 1 for checkpoint in checkpoints)
        or tuple(sorted(set(checkpoints))) != checkpoints
    ):
        raise ValueError(
            "fitness checkpoints must be unique increasing positive integers"
        )
    return checkpoints


def make_mode_batches(
    example_count: int,
    *,
    leak_mode: str,
    config: ShortcutCreditExperimentConfig,
    vocabulary: PointerNextVocabulary,
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
                make_task_batch(
                    config,
                    current,
                    int(length),
                    leak_mode=leak_mode,
                    generator=generator,
                    vocabulary=vocabulary,
                    device=device,
                )
            )
            remaining -= current
    return tuple(batches)


def make_task_batch(
    config: ShortcutCreditExperimentConfig,
    batch_size: int,
    length: int,
    *,
    leak_mode: str,
    generator: torch.Generator,
    vocabulary: PointerNextVocabulary,
    device: torch.device,
) -> ShortcutBatch:
    if config.task_variant == "pointer_next_length":
        return make_clean_pointer_batch(
            batch_size,
            length,
            generator=generator,
            vocabulary=vocabulary,
            device=device,
        )
    if not isinstance(vocabulary, ShortcutPointerVocabulary):
        raise TypeError("shortcut task requires shortcut vocabulary")
    return make_shortcut_batch(
        batch_size,
        length,
        leak_mode=leak_mode,  # type: ignore[arg-type]
        generator=generator,
        vocabulary=vocabulary,
        leak_placement=config.leak_placement,
        device=device,
    )


def make_fixed_length_batches(
    example_count: int,
    *,
    length: int,
    config: ShortcutCreditExperimentConfig,
    vocabulary: PointerNextVocabulary,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[ShortcutBatch, ...]:
    batches = []
    remaining = example_count
    while remaining:
        current = min(config.fitness_batch_size, remaining)
        batches.append(
            make_task_batch(
                config,
                current,
                length,
                leak_mode="clean",
                generator=generator,
                vocabulary=vocabulary,
                device=device,
            )
        )
        remaining -= current
    return tuple(batches)


def make_fixed_fitness_batch_sets(
    config: ShortcutCreditExperimentConfig,
    *,
    vocabulary: PointerNextVocabulary,
    device: torch.device,
) -> tuple[tuple[ShortcutBatch, ...], tuple[ShortcutBatch, ...]]:
    """Build fixed, sequential ranking and acceptance slices."""

    if config.task_variant != "pointer_next_length":
        raise ValueError("fixed fitness slices require pointer-next length task")
    if config.fitness_length is None:
        raise ValueError("fixed fitness slices require a fitness length")

    generator = torch.Generator().manual_seed(config.seed + 10_000)
    ranking_batches = make_fixed_length_batches(
        config.fitness_examples,
        length=config.fitness_length,
        config=config,
        vocabulary=vocabulary,
        generator=generator,
        device=device,
    )
    if config.acceptance_fitness_examples == 0:
        return ranking_batches, ranking_batches
    acceptance_batches = make_fixed_length_batches(
        config.acceptance_fitness_examples,
        length=config.fitness_length,
        config=config,
        vocabulary=vocabulary,
        generator=generator,
        device=device,
    )
    return ranking_batches, acceptance_batches


def make_experiment_vocabulary(
    config: ShortcutCreditExperimentConfig,
) -> PointerNextVocabulary:
    if config.task_variant == "pointer_next_length":
        return PointerNextVocabulary("numbers", 10)
    return ShortcutPointerVocabulary("numbers", 10)


def initialize_forward_model(
    config: ShortcutCreditExperimentConfig,
    vocabulary: PointerNextVocabulary,
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
    vocabulary: PointerNextVocabulary,
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
        forward_d_model=config.d_model,
        n_heads=config.heads,
        forward_layers=config.forward_layers,
        routing_credit_mode=config.routing_credit_mode,
        route_output_projection=config.route_output_projection,
        shared_routing_map=config.shared_routing_map,
        condition_on_forward_state=config.condition_on_forward_state,
        leak_token=(
            vocabulary.leak_token
            if isinstance(vocabulary, ShortcutPointerVocabulary)
            else None
        ),
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
    vocabulary: PointerNextVocabulary,
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
    vocabulary: PointerNextVocabulary,
    generator: torch.Generator,
    device: torch.device,
    leak_mode: LeakMode = "correct",
) -> tuple[ShortcutBatch, ...]:
    return tuple(
        make_task_batch(
            config,
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
    additional_ranking_inputs: tuple[CandidateRankingInput, ...] = (),
) -> tuple[
    float,
    ForwardTrajectoryMetrics,
    list[dict[str, float]],
    tuple[float, ...],
]:
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
    primary_fitness = candidate_fitness(
        config.fitness_objective,
        initial_clean_metrics,
        trajectory.clean,
        checkpoint_clean=trajectory.checkpoint_clean,
    )
    statistics = list(backward_rule.statistics)
    ranking_fitnesses = [primary_fitness]
    backward_rule.capture_statistics = False
    for (
        ranking_base_state,
        ranking_inner_batches,
        ranking_initial_clean,
    ) in additional_ranking_inputs:
        ranking_trajectory = train_forward_trajectory(
            config,
            base_state=ranking_base_state,
            backward_rule=backward_rule,
            inner_batches=ranking_inner_batches,
            fitness_batches=fitness_batches,
            correct_batches=correct_batches,
            device=device,
        )
        ranking_fitnesses.append(
            candidate_fitness(
                config.fitness_objective,
                ranking_initial_clean,
                ranking_trajectory.clean,
                checkpoint_clean=ranking_trajectory.checkpoint_clean,
            )
        )
    return (
        sum(ranking_fitnesses) / len(ranking_fitnesses),
        trajectory,
        statistics,
        tuple(ranking_fitnesses),
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
        make_experiment_vocabulary(config),
        initialization_seed=None,
        device=device,
    )
    model.load_state_dict(base_state)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.forward_learning_rate,
    )
    checkpoint_steps = parse_fitness_checkpoints(config.fitness_checkpoints)
    if checkpoint_steps and checkpoint_steps[-1] > len(inner_batches):
        raise ValueError(
            "fitness checkpoints must not exceed the trajectory horizon"
        )
    checkpoint_step_set = set(checkpoint_steps)
    checkpoint_clean = []
    model.train()
    for step, batch in enumerate(inner_batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        with forward_training_autocast_context(config, device):
            loss = shortcut_loss(model, batch, backward_rule)
        loss.backward()
        optimizer.step()
        if step in checkpoint_step_set:
            checkpoint_clean.append(
                (step, evaluate_shortcut_batches(model, fitness_batches))
            )
            model.train()

    clean_metrics = (
        checkpoint_clean[-1][1]
        if checkpoint_clean and checkpoint_clean[-1][0] == len(inner_batches)
        else evaluate_shortcut_batches(model, fitness_batches)
    )
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
        checkpoint_clean=tuple(checkpoint_clean),
    )


def forward_training_autocast_context(
    config: ShortcutCreditExperimentConfig,
    device: torch.device,
    *,
    vmap_compatible: bool = False,
):
    if config.forward_training_precision == "fp32" or device.type != "cuda":
        return nullcontext()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 training is not supported on this CUDA device")
    contexts = ExitStack()
    if vmap_compatible:
        # SDPA backend flags are process-global in this PyTorch version.
        # Context managers race when candidate shards run in worker threads.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    contexts.enter_context(
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    )
    return contexts


def worst_checkpoint_mode_loss(
    checkpoint_clean: tuple[tuple[int, ShortcutMetrics], ...],
) -> float:
    if not checkpoint_clean:
        raise ValueError("checkpoint fitness requires checkpoint metrics")
    return max(
        max(metrics.mode_loss.values()) for _, metrics in checkpoint_clean
    )


def candidate_fitness(
    objective: str,
    initial: ShortcutMetrics,
    trained: ShortcutMetrics,
    *,
    checkpoint_clean: tuple[tuple[int, ShortcutMetrics], ...] = (),
) -> float:
    if objective == "mean_clean_ce":
        return initial.loss - trained.loss
    if objective == "worst_mode_ce":
        initial_worst = max(initial.mode_loss.values())
        trained_worst = max(trained.mode_loss.values())
        return initial_worst - trained_worst
    if objective == "worst_checkpoint_mode_ce":
        initial_worst = max(initial.mode_loss.values())
        return initial_worst - worst_checkpoint_mode_loss(checkpoint_clean)
    raise ValueError(f"unknown fitness objective: {objective}")


def trajectory_summary(
    prefix: str,
    fitness: float | None,
    clean: ShortcutMetrics,
    correct: ShortcutMetrics,
) -> dict[str, float]:
    minimum_accuracy = min(clean.mode_accuracy.values())
    summary = {
        f"{prefix}/clean_loss": clean.loss,
        f"{prefix}/clean_accuracy": clean.accuracy,
        f"{prefix}/min_mode_accuracy": minimum_accuracy,
        f"{prefix}/correct_leak_accuracy": correct.accuracy,
        f"{prefix}/unique_value_predictions": float(
            clean.unique_value_prediction_count
        ),
        f"{prefix}/prediction_mode_fraction": (
            clean.prediction_mode_fraction
        ),
    }
    for mode, accuracy in clean.mode_accuracy.items():
        summary[f"{prefix}/{mode}_accuracy"] = accuracy
    if fitness is not None:
        summary[f"{prefix}/fitness"] = fitness
    return summary


def strip_shortcut_only_metrics(
    summary: dict[str, float | str | None],
) -> dict[str, float | str | None]:
    """Remove legacy shortcut-condition aliases from clean-task reporting."""

    shortcut_terms = ("masked", "incorrect", "correct_leak")
    return {
        key: value
        for key, value in summary.items()
        if not any(term in key for term in shortcut_terms)
    }


def checkpoint_trajectory_summary(
    prefix: str,
    trajectory: ForwardTrajectoryMetrics,
) -> dict[str, float]:
    if not trajectory.checkpoint_clean:
        return {}
    summary = {
        f"{prefix}/worst_checkpoint_mode_loss": (
            worst_checkpoint_mode_loss(trajectory.checkpoint_clean)
        )
    }
    for step, metrics in trajectory.checkpoint_clean:
        summary.update(
            {
                f"{prefix}/checkpoint_{step}_clean_loss": metrics.loss,
                f"{prefix}/checkpoint_{step}_min_mode_accuracy": min(
                    metrics.mode_accuracy.values()
                ),
                f"{prefix}/checkpoint_{step}_worst_mode_loss": max(
                    metrics.mode_loss.values()
                ),
            }
        )
    return summary


def checkpoint_population_summary(
    trajectories: list[ForwardTrajectoryMetrics],
) -> dict[str, float]:
    checkpoint_groups = [
        trajectory.checkpoint_clean for trajectory in trajectories
    ]
    if not checkpoint_groups or not checkpoint_groups[0]:
        return {}
    expected_steps = tuple(step for step, _ in checkpoint_groups[0])
    if any(
        tuple(step for step, _ in checkpoint_group) != expected_steps
        for checkpoint_group in checkpoint_groups
    ):
        raise ValueError("candidate checkpoint steps must match")
    summary = {
        "transition/worst_checkpoint_mode_loss_mean": (
            sum(
                worst_checkpoint_mode_loss(checkpoint_group)
                for checkpoint_group in checkpoint_groups
            )
            / len(checkpoint_groups)
        )
    }
    for checkpoint_index, step in enumerate(expected_steps):
        metrics = [
            checkpoint_group[checkpoint_index][1]
            for checkpoint_group in checkpoint_groups
        ]
        summary.update(
            {
                f"transition/checkpoint_{step}_clean_loss_mean": (
                    sum(item.loss for item in metrics) / len(metrics)
                ),
                f"transition/checkpoint_{step}_min_mode_accuracy_mean": (
                    sum(min(item.mode_accuracy.values()) for item in metrics)
                    / len(metrics)
                ),
                f"transition/checkpoint_{step}_worst_mode_loss_mean": (
                    sum(max(item.mode_loss.values()) for item in metrics)
                    / len(metrics)
                ),
            }
        )
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


def train_successive_halving_population(
    config: ShortcutCreditExperimentConfig,
    *,
    candidate_specs: tuple[tuple[int, int, int], ...],
    candidate_devices: tuple[torch.device, ...],
    base_state: dict[str, Tensor],
    center_rule: AttentionRoutingRule,
    center_parameters: dict[str, Tensor],
    directions: tuple[EggrollDirection, ...],
    inner_batches: tuple[ShortcutBatch, ...],
    fitness_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    heldout_fitness_batches: tuple[ShortcutBatch, ...] | None,
    heldout_correct_batches: tuple[ShortcutBatch, ...] | None,
    initial_clean_metrics: ShortcutMetrics,
    perturbation_sigma: float,
) -> tuple[list[Any], tuple[int, ...], dict[str, float]]:
    """Train and prune vectorized candidates without restarting survivors."""

    from .vectorized_routing_population import (
        VectorizedRoutingCandidateState,
        train_vectorized_routing_halving_stage,
    )

    rungs = parse_successive_halving_rungs(
        config.successive_halving_rungs
    )
    if not rungs:
        raise ValueError("successive halving requires configured rungs")
    active_specs = candidate_specs
    candidate_states: dict[int, VectorizedRoutingCandidateState] | None = None
    latest_outputs: dict[int, Any] = {}
    previous_horizon = 0
    summary: dict[str, float] = {}

    for rung_index, (rung_horizon, survivor_count) in enumerate(rungs):
        segment = inner_batches[previous_horizon:rung_horizon]
        if not segment:
            raise ValueError("successive halving rung has no training steps")
        final_rung = rung_index + 1 == len(rungs)
        shards = shard_candidate_specs(active_specs, len(candidate_devices))

        def run_worker(
            shard: tuple[tuple[int, int, int], ...],
            worker_device: torch.device,
        ) -> tuple[list[Any], dict[int, VectorizedRoutingCandidateState]]:
            return train_vectorized_routing_halving_stage(
                config=config,
                candidate_specs=shard,
                chunk_size=config.vectorized_chunk_size,
                device=worker_device,
                base_state=base_state,
                center_rule_config=center_rule.config,
                center_parameters=center_parameters,
                directions=directions,
                inner_batches=segment,
                fitness_batches=fitness_batches,
                correct_batches=correct_batches,
                heldout_fitness_batches=(
                    heldout_fitness_batches if final_rung else None
                ),
                heldout_correct_batches=(
                    heldout_correct_batches if final_rung else None
                ),
                initial_clean_metrics=initial_clean_metrics,
                perturbation_sigma=perturbation_sigma,
                candidate_states=candidate_states,
            )

        active_workers = [
            (shard, worker_device)
            for shard, worker_device in zip(shards, candidate_devices)
            if shard
        ]
        if len(active_workers) == 1:
            worker_outputs = [run_worker(*active_workers[0])]
        else:
            with ThreadPoolExecutor(
                max_workers=len(active_workers)
            ) as executor:
                worker_outputs = [
                    future.result()
                    for future in (
                        executor.submit(run_worker, shard, worker_device)
                        for shard, worker_device in active_workers
                    )
                ]
        stage_outputs = []
        stage_states = {}
        for outputs, states in worker_outputs:
            stage_outputs.extend(outputs)
            stage_states.update(states)
        stage_outputs.sort(key=lambda output: output[1], reverse=True)
        if survivor_count > len(stage_outputs):
            raise ValueError("halving rung retains too many candidates")
        surviving_indices = {
            output[0] for output in stage_outputs[:survivor_count]
        }
        latest_outputs.update(
            {output[0]: output for output in stage_outputs}
        )
        candidate_states = {
            index: stage_states[index]
            for index in surviving_indices
        }
        active_specs = tuple(
            spec for spec in active_specs if spec[0] in surviving_indices
        )
        prefix = f"halving/rung_{rung_index}"
        summary[f"{prefix}/horizon"] = float(rung_horizon)
        summary[f"{prefix}/candidates"] = float(len(stage_outputs))
        summary[f"{prefix}/survivors"] = float(len(active_specs))
        summary[f"{prefix}/best_fitness"] = float(stage_outputs[0][1])
        previous_horizon = rung_horizon

    return (
        [latest_outputs[index] for index in sorted(latest_outputs)],
        tuple(spec[0] for spec in active_specs),
        summary,
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
    perturbation_sigma: float,
    additional_ranking_inputs: tuple[CandidateRankingInput, ...] = (),
) -> list[
    tuple[
        int,
        float,
        ForwardTrajectoryMetrics,
        list[dict[str, float]],
        tuple[float, ...],
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
    worker_additional_ranking_inputs = tuple(
        (
            {
                name: tensor.to(device)
                for name, tensor in ranking_base_state.items()
            },
            tuple(batch.to(device) for batch in ranking_inner_batches),
            ranking_initial_clean,
        )
        for (
            ranking_base_state,
            ranking_inner_batches,
            ranking_initial_clean,
        ) in additional_ranking_inputs
    )
    direction_indices = {spec[1] for spec in candidate_specs}
    worker_directions = {
        index: move_eggroll_direction(directions[index], device)
        for index in direction_indices
    }

    results = []
    for candidate_index, direction_index, sign in candidate_specs:
        fitness, trajectory, statistics, ranking_fitnesses = train_candidate(
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
            perturbation_sigma=perturbation_sigma,
            heldout_fitness_batches=worker_heldout_fitness_batches,
            heldout_correct_batches=worker_heldout_correct_batches,
            additional_ranking_inputs=worker_additional_ranking_inputs,
        )
        results.append(
            (
                candidate_index,
                fitness,
                trajectory,
                statistics,
                ranking_fitnesses,
            )
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


def outer_update_hyperparameter_summary(
    config: ShortcutCreditExperimentConfig,
    *,
    sigma: float,
    paper_learning_rate: float,
    commit_scale: float | None = None,
) -> dict[str, float]:
    if config.outer_update_rule == "paper_standardized":
        return {
            "outer/paper_learning_rate": paper_learning_rate,
        }
    return {
        "outer/elite_count": float(config.elite_count),
        "outer/elite_interpolation": config.elite_interpolation,
        "outer/elite_step_scale": (
            sigma * config.elite_interpolation
            if commit_scale is None
            else commit_scale
        ),
    }


def top_unique_antithetic_indices(
    fitnesses: Tensor,
    elite_count: int,
) -> Tensor:
    """Keep only the fitter sign from each antithetic direction."""

    if fitnesses.ndim != 1 or fitnesses.numel() % 2:
        raise ValueError(
            "fitnesses must contain positive/negative antithetic pairs"
        )
    direction_count = fitnesses.numel() // 2
    if not 1 <= elite_count <= direction_count:
        raise ValueError(
            "elite_count must select unique antithetic directions"
        )
    pair_fitnesses = fitnesses.view(direction_count, 2)
    preferred_fitnesses, preferred_sign_indices = pair_fitnesses.max(dim=1)
    elite_directions = preferred_fitnesses.topk(elite_count).indices
    return 2 * elite_directions + preferred_sign_indices[elite_directions]


def top_elite_indices(
    fitnesses: Tensor,
    elite_count: int,
    *,
    deduplicate_antithetic: bool,
) -> Tensor:
    if deduplicate_antithetic:
        return top_unique_antithetic_indices(fitnesses, elite_count)
    if fitnesses.ndim != 1 or not 1 <= elite_count <= fitnesses.numel():
        raise ValueError("elite_count must select from the population")
    return fitnesses.topk(elite_count).indices


@torch.no_grad()
def elite_centroid_update(
    module: BackwardRule,
    directions: tuple[EggrollDirection, ...],
    fitnesses: Tensor,
    *,
    sigma: float,
    elite_count: int,
    interpolation: float,
    commit_scale: float | None = None,
    deduplicate_antithetic: bool = False,
) -> Tensor:
    """Move the centre toward the mean of the highest-fitness candidates."""

    if fitnesses.ndim != 1 or fitnesses.numel() != 2 * len(directions):
        raise ValueError("fitnesses must contain one positive/negative pair")
    if not 0 < interpolation <= 1:
        raise ValueError("interpolation must be in (0, 1]")
    step_scale = sigma * interpolation
    if commit_scale is not None:
        if commit_scale <= 0:
            raise ValueError("commit scale must be positive")
        step_scale = commit_scale
    elite_indices = top_elite_indices(
        fitnesses,
        elite_count,
        deduplicate_antithetic=deduplicate_antithetic,
    )
    for name, parameter in module.named_parameters():
        centroid_delta = torch.zeros_like(parameter)
        for candidate_index in elite_indices.tolist():
            direction = directions[candidate_index // 2]
            sign = 1 if candidate_index % 2 == 0 else -1
            centroid_delta.add_(
                sign * direction.tensors[name]
            )
        parameter.add_(
            step_scale * centroid_delta / elite_count
        )
    return elite_indices


@torch.no_grad()
def elite_centroid_parameters(
    module: BackwardRule,
    center_parameters: dict[str, Tensor],
    directions: tuple[EggrollDirection, ...],
    fitnesses: Tensor,
    *,
    sigma: float,
    elite_count: int,
    interpolation: float,
    commit_scale: float | None = None,
    deduplicate_antithetic: bool = False,
) -> tuple[dict[str, Tensor], Tensor]:
    """Build one elite proposal without changing the supplied centre."""

    restore_center_parameters(module, center_parameters)
    elite_indices = elite_centroid_update(
        module,
        directions,
        fitnesses,
        sigma=sigma,
        elite_count=elite_count,
        interpolation=interpolation,
        commit_scale=commit_scale,
        deduplicate_antithetic=deduplicate_antithetic,
    )
    if isinstance(module, AttentionRoutingRule):
        module.project_parameters_()
    proposal = clone_center_parameters(module)
    restore_center_parameters(module, center_parameters)
    return proposal, elite_indices


@torch.no_grad()
def restore_center_parameters(
    module: BackwardRule,
    center_parameters: dict[str, Tensor],
) -> None:
    for name, parameter in module.named_parameters():
        parameter.copy_(center_parameters[name])


def update_elite_search_state(
    state: PlateauState,
    *,
    accepted: bool,
    config: ShortcutCreditExperimentConfig,
) -> None:
    if state.search_sigma is None:
        raise ValueError("search sigma must be initialized")
    if not accepted:
        state.consecutive_accepted_updates = 0
        state.consecutive_rejected_updates += 1
        state.search_sigma = max(
            config.elite_min_sigma,
            state.search_sigma * config.elite_rejection_sigma_decay,
        )
        return

    state.consecutive_rejected_updates = 0
    state.consecutive_accepted_updates += 1
    if (
        state.consecutive_accepted_updates
        < config.elite_acceptance_patience
    ):
        return
    state.search_sigma = min(
        config.sigma,
        state.search_sigma * config.elite_acceptance_sigma_growth,
    )
    state.consecutive_accepted_updates = 0


def elite_proposal_mean_improvement(
    center_fitnesses: list[float],
    proposal_fitnesses: list[float],
) -> float:
    if not center_fitnesses:
        raise ValueError("at least one acceptance trajectory is required")
    if len(center_fitnesses) != len(proposal_fitnesses):
        raise ValueError("center and proposal fitness counts must match")
    return (
        sum(proposal_fitnesses) - sum(center_fitnesses)
    ) / len(center_fitnesses)


def elite_proposal_improves_every_trajectory(
    center_fitnesses: list[float] | tuple[float, ...],
    proposal_fitnesses: list[float] | tuple[float, ...],
) -> bool:
    if not center_fitnesses:
        raise ValueError("at least one acceptance trajectory is required")
    if len(center_fitnesses) != len(proposal_fitnesses):
        raise ValueError("center and proposal fitness counts must match")
    return all(
        proposal > center
        for center, proposal in zip(center_fitnesses, proposal_fitnesses)
    )


def elite_acceptance_seed(
    generation_seed: int,
    trajectory_index: int,
) -> int:
    if trajectory_index < 1:
        raise ValueError("extra acceptance trajectory index must be positive")
    return generation_seed + trajectory_index * 1_000_000_007


def candidate_ranking_seeds(
    generation_seed: int,
    trajectory_count: int,
) -> tuple[int, ...]:
    if trajectory_count < 1:
        raise ValueError("ranking trajectory count must be positive")
    return (generation_seed,) + tuple(
        elite_acceptance_seed(generation_seed, trajectory_index)
        for trajectory_index in range(1, trajectory_count)
    )


def independent_elite_acceptance_seeds(
    generation_seed: int,
    trajectory_count: int,
    *,
    start_index: int = 1,
) -> tuple[int, ...]:
    if trajectory_count < 1:
        raise ValueError("acceptance trajectory count must be positive")
    if start_index < 1:
        raise ValueError("acceptance trajectory start index must be positive")
    return tuple(
        elite_acceptance_seed(generation_seed, trajectory_index)
        for trajectory_index in range(
            start_index,
            start_index + trajectory_count,
        )
    )


def select_adaptive_elite_proposal(
    config: ShortcutCreditExperimentConfig,
    *,
    center_rule: AttentionRoutingRule,
    center_parameters: dict[str, Tensor],
    directions: tuple[EggrollDirection, ...],
    fitnesses: Tensor,
    sigma: float,
    generation_seed: int,
    horizon: int,
    vocabulary: PointerNextVocabulary,
    acceptance_fitness_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    acceptance_devices: tuple[torch.device, ...],
    commit_scale: float | None = None,
) -> AdaptiveEliteResult:
    """Choose a nested elite proposal on matched independent trajectories."""

    if commit_scale is not None:
        return select_adaptive_commit_scale_proposal(
            config,
            center_rule=center_rule,
            center_parameters=center_parameters,
            directions=directions,
            fitnesses=fitnesses,
            sigma=sigma,
            commit_scale=commit_scale,
            generation_seed=generation_seed,
            horizon=horizon,
            vocabulary=vocabulary,
            acceptance_fitness_batches=acceptance_fitness_batches,
            correct_batches=correct_batches,
            acceptance_devices=acceptance_devices,
        )

    from .vectorized_routing_population import (
        stack_rule_parameter_sets,
        train_vectorized_routing_population,
    )

    elite_counts = parse_adaptive_elite_counts(
        config.adaptive_elite_counts
    )
    if not elite_counts:
        raise ValueError("adaptive elite selection requires elite counts")
    proposals = []
    indices_by_count = {}
    for elite_count in elite_counts:
        parameters, indices = elite_centroid_parameters(
            center_rule,
            center_parameters,
            directions,
            fitnesses,
            sigma=sigma,
            elite_count=elite_count,
            interpolation=config.elite_interpolation,
            deduplicate_antithetic=(
                config.deduplicate_antithetic_elites
            ),
        )
        proposals.append(parameters)
        indices_by_count[elite_count] = indices
    parameter_sets = (center_parameters, *proposals)
    fitness_groups = [
        [] for _ in parameter_sets
    ]
    acceptance_seeds = independent_elite_acceptance_seeds(
        generation_seed,
        config.elite_acceptance_trajectories,
        start_index=config.candidate_ranking_trajectories,
    )
    cpu = torch.device("cpu")
    prepared_trajectories = []
    for acceptance_seed in acceptance_seeds:
        acceptance_model = initialize_forward_model(
            config,
            vocabulary,
            initialization_seed=acceptance_seed + 1,
            device=cpu,
        )
        acceptance_base_state = {
            name: tensor.detach().clone()
            for name, tensor in acceptance_model.state_dict().items()
        }
        del acceptance_model
        acceptance_inner_batches = make_inner_batches(
            config,
            horizon=horizon,
            vocabulary=vocabulary,
            generator=torch.Generator().manual_seed(
                acceptance_seed + 2
            ),
            device=cpu,
        )
        prepared_trajectories.append(
            (acceptance_base_state, acceptance_inner_batches)
        )

    cpu_parameter_sets = tuple(
        {
            name: tensor.detach().to(cpu)
            for name, tensor in parameters.items()
        }
        for parameters in parameter_sets
    )
    cpu_fitness_batches = tuple(
        batch.to(cpu) for batch in acceptance_fitness_batches
    )
    cpu_correct_batches = tuple(batch.to(cpu) for batch in correct_batches)
    worker_jobs = [[] for _ in acceptance_devices]
    for index, prepared in enumerate(prepared_trajectories):
        worker_jobs[index % len(acceptance_devices)].append(
            (index, prepared)
        )

    def evaluate_worker(
        worker_device: torch.device,
        jobs: list[
            tuple[
                int,
                tuple[
                    dict[str, Tensor],
                    tuple[ShortcutBatch, ...],
                ],
            ]
        ],
    ) -> list[
        tuple[
            int,
            ShortcutMetrics,
            tuple[ForwardTrajectoryMetrics, ...],
        ]
    ]:
        if worker_device.type == "cuda":
            torch.cuda.set_device(worker_device)
        worker_rule = AttentionRoutingRule(center_rule.config).to(
            worker_device
        )
        stacked_parameters = stack_rule_parameter_sets(
            cpu_parameter_sets,
            device=worker_device,
        )
        worker_fitness_batches = tuple(
            batch.to(worker_device) for batch in cpu_fitness_batches
        )
        worker_correct_batches = tuple(
            batch.to(worker_device) for batch in cpu_correct_batches
        )
        results = []
        for trajectory_index, (
            acceptance_base_state,
            acceptance_inner_batches,
        ) in jobs:
            acceptance_model = initialize_forward_model(
                config,
                vocabulary,
                initialization_seed=None,
                device=worker_device,
            )
            acceptance_model.load_state_dict(acceptance_base_state)
            acceptance_initial = evaluate_shortcut_batches(
                acceptance_model,
                worker_fitness_batches,
            )
            del acceptance_model
            population = train_vectorized_routing_population(
                config=config,
                base_state=acceptance_base_state,
                center_rule=worker_rule,
                rule_parameters=stacked_parameters,
                inner_batches=acceptance_inner_batches,
                fitness_batches=worker_fitness_batches,
                correct_batches=worker_correct_batches,
                heldout_fitness_batches=None,
                heldout_correct_batches=None,
                device=worker_device,
            )
            results.append(
                (
                    trajectory_index,
                    acceptance_initial,
                    population.trajectories,
                )
            )
        return results

    active_workers = [
        (worker_device, jobs)
        for worker_device, jobs in zip(acceptance_devices, worker_jobs)
        if jobs
    ]
    if len(active_workers) == 1:
        acceptance_results = evaluate_worker(*active_workers[0])
    else:
        with ThreadPoolExecutor(
            max_workers=len(active_workers)
        ) as executor:
            futures = [
                executor.submit(evaluate_worker, worker_device, jobs)
                for worker_device, jobs in active_workers
            ]
            acceptance_results = [
                result
                for future in futures
                for result in future.result()
            ]
    acceptance_results.sort(key=lambda result: result[0])
    for _, acceptance_initial, trajectories in acceptance_results:
        for index, trajectory in enumerate(trajectories):
            fitness_groups[index].append(
                candidate_fitness(
                    config.fitness_objective,
                    acceptance_initial,
                    trajectory.clean,
                    checkpoint_clean=trajectory.checkpoint_clean,
                )
            )
    means = tuple(
        sum(group) / len(group)
        for group in fitness_groups
    )
    selected_offset = max(
        range(len(elite_counts)),
        key=lambda index: (means[index + 1], -elite_counts[index]),
    )
    selected_count = elite_counts[selected_offset]
    accepted = elite_proposal_improves_every_trajectory(
        fitness_groups[0],
        fitness_groups[selected_offset + 1],
    )
    return AdaptiveEliteResult(
        accepted=accepted,
        selected_count=selected_count,
        selected_indices=indices_by_count[selected_count],
        selected_parameters=proposals[selected_offset],
        center_fitnesses=tuple(fitness_groups[0]),
        selected_fitnesses=tuple(
            fitness_groups[selected_offset + 1]
        ),
        mean_fitness_by_count={
            elite_count: means[index + 1]
            for index, elite_count in enumerate(elite_counts)
        },
    )


def select_adaptive_commit_scale_proposal(
    config: ShortcutCreditExperimentConfig,
    *,
    center_rule: AttentionRoutingRule,
    center_parameters: dict[str, Tensor],
    directions: tuple[EggrollDirection, ...],
    fitnesses: Tensor,
    sigma: float,
    commit_scale: float,
    generation_seed: int,
    horizon: int,
    vocabulary: PointerNextVocabulary,
    acceptance_fitness_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    acceptance_devices: tuple[torch.device, ...],
) -> AdaptiveEliteResult:
    """Select a centroid step size, then confirm it on fresh trajectories."""

    from .vectorized_routing_population import (
        stack_rule_parameter_sets,
        train_vectorized_routing_population,
    )

    elite_counts = parse_adaptive_elite_counts(
        config.adaptive_elite_counts
    )
    if not elite_counts:
        raise ValueError("commit-scale search requires adaptive elite counts")
    commit_scales = adaptive_commit_scale_grid(
        commit_scale,
        config.adaptive_commit_scale_multiplier,
    )
    proposal_specs = tuple(
        (elite_count, proposal_scale)
        for elite_count in elite_counts
        for proposal_scale in commit_scales
    )
    proposals = []
    indices_by_count = {}
    for elite_count, proposal_scale in proposal_specs:
        parameters, indices = elite_centroid_parameters(
            center_rule,
            center_parameters,
            directions,
            fitnesses,
            sigma=sigma,
            elite_count=elite_count,
            interpolation=config.elite_interpolation,
            commit_scale=proposal_scale,
            deduplicate_antithetic=(
                config.deduplicate_antithetic_elites
            ),
        )
        proposals.append(parameters)
        indices_by_count[elite_count] = indices
    parameter_sets = (center_parameters, *proposals)

    cpu = torch.device("cpu")
    cpu_parameter_sets = tuple(
        {
            name: tensor.detach().to(cpu)
            for name, tensor in parameters.items()
        }
        for parameters in parameter_sets
    )
    cpu_fitness_batches = tuple(
        batch.to(cpu) for batch in acceptance_fitness_batches
    )
    cpu_correct_batches = tuple(batch.to(cpu) for batch in correct_batches)

    def prepare_trajectory(seed: int) -> tuple[
        dict[str, Tensor],
        tuple[ShortcutBatch, ...],
    ]:
        model = initialize_forward_model(
            config,
            vocabulary,
            initialization_seed=seed + 1,
            device=cpu,
        )
        base_state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        del model
        inner_batches = make_inner_batches(
            config,
            horizon=horizon,
            vocabulary=vocabulary,
            generator=torch.Generator().manual_seed(seed + 2),
            device=cpu,
        )
        return base_state, inner_batches

    Job = tuple[
        int,
        tuple[int, ...],
        dict[str, Tensor],
        tuple[ShortcutBatch, ...],
    ]

    def evaluate_worker(
        worker_device: torch.device,
        jobs: list[Job],
    ) -> list[tuple[int, int, float]]:
        if worker_device.type == "cuda":
            torch.cuda.set_device(worker_device)
        worker_rule = AttentionRoutingRule(center_rule.config).to(
            worker_device
        )
        worker_fitness_batches = tuple(
            batch.to(worker_device) for batch in cpu_fitness_batches
        )
        worker_correct_batches = tuple(
            batch.to(worker_device) for batch in cpu_correct_batches
        )
        results = []
        for trajectory_index, parameter_indices, base_state, inner_batches in jobs:
            selected_parameters = tuple(
                cpu_parameter_sets[index] for index in parameter_indices
            )
            stacked_parameters = stack_rule_parameter_sets(
                selected_parameters,
                device=worker_device,
            )
            model = initialize_forward_model(
                config,
                vocabulary,
                initialization_seed=None,
                device=worker_device,
            )
            model.load_state_dict(base_state)
            initial = evaluate_shortcut_batches(
                model,
                worker_fitness_batches,
            )
            del model
            population = train_vectorized_routing_population(
                config=config,
                base_state=base_state,
                center_rule=worker_rule,
                rule_parameters=stacked_parameters,
                inner_batches=inner_batches,
                fitness_batches=worker_fitness_batches,
                correct_batches=worker_correct_batches,
                heldout_fitness_batches=None,
                heldout_correct_batches=None,
                device=worker_device,
            )
            for parameter_index, trajectory in zip(
                parameter_indices,
                population.trajectories,
            ):
                results.append(
                    (
                        trajectory_index,
                        parameter_index,
                        candidate_fitness(
                            config.fitness_objective,
                            initial,
                            trajectory.clean,
                            checkpoint_clean=trajectory.checkpoint_clean,
                        ),
                    )
                )
        return results

    def run_jobs(
        jobs_by_device: list[list[Job]],
    ) -> list[tuple[int, int, float]]:
        active_workers = [
            (worker_device, jobs)
            for worker_device, jobs in zip(
                acceptance_devices,
                jobs_by_device,
            )
            if jobs
        ]
        if len(active_workers) == 1:
            results = evaluate_worker(*active_workers[0])
        else:
            with ThreadPoolExecutor(
                max_workers=len(active_workers)
            ) as executor:
                futures = [
                    executor.submit(evaluate_worker, worker_device, jobs)
                    for worker_device, jobs in active_workers
                ]
                results = [
                    result
                    for future in futures
                    for result in future.result()
                ]
        results.sort(key=lambda item: (item[0], item[1]))
        return results

    selection_seed = elite_acceptance_seed(
        generation_seed,
        config.candidate_ranking_trajectories,
    )
    selection_base_state, selection_inner_batches = prepare_trajectory(
        selection_seed
    )
    selection_indices_by_device = [
        [] for _ in acceptance_devices
    ]
    for parameter_index in range(len(parameter_sets)):
        selection_indices_by_device[
            parameter_index % len(acceptance_devices)
        ].append(parameter_index)
    selection_jobs = [
        [
            (
                0,
                tuple(parameter_indices),
                selection_base_state,
                selection_inner_batches,
            )
        ]
        if parameter_indices
        else []
        for parameter_indices in selection_indices_by_device
    ]
    selection_results = run_jobs(selection_jobs)
    selection_fitnesses = {
        parameter_index: fitness
        for _, parameter_index, fitness in selection_results
    }
    if len(selection_fitnesses) != len(parameter_sets):
        raise RuntimeError("commit-scale selection did not evaluate all proposals")

    selected_offset = max(
        range(len(proposal_specs)),
        key=lambda index: (
            selection_fitnesses[index + 1],
            -abs(proposal_specs[index][1] - commit_scale),
            -proposal_specs[index][0],
        ),
    )
    selected_count, selected_scale = proposal_specs[selected_offset]
    selected_parameter_index = selected_offset + 1

    confirmation_seeds = independent_elite_acceptance_seeds(
        generation_seed,
        config.elite_acceptance_trajectories,
        start_index=config.candidate_ranking_trajectories + 1,
    )
    confirmation_jobs = [[] for _ in acceptance_devices]
    for trajectory_index, confirmation_seed in enumerate(confirmation_seeds):
        base_state, inner_batches = prepare_trajectory(confirmation_seed)
        confirmation_jobs[
            trajectory_index % len(acceptance_devices)
        ].append(
            (
                trajectory_index,
                (0, selected_parameter_index),
                base_state,
                inner_batches,
            )
        )
    confirmation_results = run_jobs(confirmation_jobs)
    center_fitnesses = []
    selected_fitnesses = []
    by_trajectory = {}
    for trajectory_index, parameter_index, fitness in confirmation_results:
        by_trajectory.setdefault(trajectory_index, {})[parameter_index] = fitness
    for trajectory_index in range(len(confirmation_seeds)):
        trajectory_fitnesses = by_trajectory[trajectory_index]
        center_fitnesses.append(trajectory_fitnesses[0])
        selected_fitnesses.append(
            trajectory_fitnesses[selected_parameter_index]
        )

    return AdaptiveEliteResult(
        accepted=elite_proposal_improves_every_trajectory(
            center_fitnesses,
            selected_fitnesses,
        ),
        selected_count=selected_count,
        selected_indices=indices_by_count[selected_count],
        selected_parameters=proposals[selected_offset],
        center_fitnesses=tuple(center_fitnesses),
        selected_fitnesses=tuple(selected_fitnesses),
        mean_fitness_by_count={
            elite_count: max(
                selection_fitnesses[index + 1]
                for index, (count, _) in enumerate(proposal_specs)
                if count == elite_count
            )
            for elite_count in elite_counts
        },
        selected_commit_scale=selected_scale,
        selection_center_fitness=selection_fitnesses[0],
        selection_selected_fitness=selection_fitnesses[
            selected_parameter_index
        ],
        selection_fitness_by_proposal=tuple(
            (
                elite_count,
                proposal_scale,
                selection_fitnesses[index + 1],
            )
            for index, (elite_count, proposal_scale) in enumerate(
                proposal_specs
            )
        ),
    )


def probe_longer_horizon(
    config: ShortcutCreditExperimentConfig,
    *,
    center_rule: BackwardRule,
    generation_seed: int,
    horizon: int,
    vocabulary: PointerNextVocabulary,
    fitness_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    device: torch.device,
) -> HorizonProbeResult:
    """Compare current and doubled horizons on one matched fresh trajectory."""

    next_horizon = min(
        config.max_horizon,
        horizon * config.horizon_multiplier,
    )
    if next_horizon <= horizon:
        raise ValueError("horizon probe requires a larger available horizon")
    probe_seed = generation_seed + 9_000_000_063
    probe_model = initialize_forward_model(
        config,
        vocabulary,
        initialization_seed=probe_seed + 1,
        device=device,
    )
    base_state = {
        name: tensor.detach().clone()
        for name, tensor in probe_model.state_dict().items()
    }
    initial = evaluate_shortcut_batches(
        probe_model,
        fitness_batches,
    )
    del probe_model
    longer_batches = make_inner_batches(
        config,
        horizon=next_horizon,
        vocabulary=vocabulary,
        generator=torch.Generator().manual_seed(probe_seed + 2),
        device=device,
    )
    current = train_forward_trajectory(
        config,
        base_state=base_state,
        backward_rule=center_rule,
        inner_batches=longer_batches[:horizon],
        fitness_batches=fitness_batches,
        correct_batches=correct_batches,
        device=device,
    )
    longer = train_forward_trajectory(
        config,
        base_state=base_state,
        backward_rule=center_rule,
        inner_batches=longer_batches,
        fitness_batches=fitness_batches,
        correct_batches=correct_batches,
        device=device,
    )
    current_fitness = candidate_fitness(
        config.fitness_objective,
        initial,
        current.clean,
        checkpoint_clean=current.checkpoint_clean,
    )
    longer_fitness = candidate_fitness(
        config.fitness_objective,
        initial,
        longer.clean,
        checkpoint_clean=longer.checkpoint_clean,
    )
    return HorizonProbeResult(
        promoted=(
            longer_fitness
            > current_fitness + config.horizon_probe_min_improvement
        ),
        current_fitness=current_fitness,
        longer_fitness=longer_fitness,
        next_horizon=next_horizon,
    )


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


def reset_horizon_tracking(state: PlateauState) -> None:
    state.horizon_scores = []
    state.horizon_generations = 0
    state.horizon_best_average = float("-inf")
    state.horizon_stale_generations = 0


def update_performance_horizon_state(
    state: PlateauState,
    *,
    objective: float,
    horizon: int,
    config: ShortcutCreditExperimentConfig,
) -> PerformanceHorizonDecision:
    """Advance or stop after persistent centre performance plateaus."""

    state.horizon_generations += 1
    state.horizon_scores.append(objective)
    if len(state.horizon_scores) > config.horizon_score_window:
        del state.horizon_scores[0]
    if len(state.horizon_scores) < config.horizon_score_window:
        return PerformanceHorizonDecision()

    rolling_average = sum(state.horizon_scores) / len(state.horizon_scores)
    if (
        rolling_average
        > state.horizon_best_average + config.plateau_min_delta
    ):
        state.horizon_best_average = rolling_average
        state.horizon_stale_generations = 0
    else:
        state.horizon_stale_generations += 1

    plateau_detected = (
        state.horizon_generations >= config.horizon_min_generations
        and state.horizon_stale_generations >= config.plateau_patience
    )
    maximum_dwell_reached = (
        state.horizon_generations >= config.horizon_max_generations
    )
    if not plateau_detected and not maximum_dwell_reached:
        return PerformanceHorizonDecision(
            rolling_average=rolling_average,
        )

    extension_improved = None
    if state.horizon_reference_average is None:
        state.horizon_reference_average = rolling_average
        state.failed_horizon_extensions = 0
    else:
        extension_improved = (
            rolling_average
            > state.horizon_reference_average + config.plateau_min_delta
        )
        if extension_improved:
            state.horizon_reference_average = rolling_average
            state.failed_horizon_extensions = 0
        else:
            state.failed_horizon_extensions += 1

    common = {
        "rolling_average": rolling_average,
        "plateau_detected": plateau_detected,
        "maximum_dwell_reached": maximum_dwell_reached,
        "extension_improved": extension_improved,
        "completed_horizon_average": rolling_average,
    }
    if (
        state.failed_horizon_extensions
        >= config.horizon_failed_extension_limit
    ):
        return PerformanceHorizonDecision(
            stop=True,
            stop_reason="failed_horizon_extensions",
            **common,
        )
    if horizon >= config.max_horizon:
        return PerformanceHorizonDecision(
            stop=True,
            stop_reason="max_horizon_plateau",
            **common,
        )

    return PerformanceHorizonDecision(
        promote=True,
        **common,
    )


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
        [
            metrics.mode_accuracy.get("masked", metrics.accuracy)
            for metrics in clean_metrics
        ]
    )
    incorrect_accuracies = torch.tensor(
        [
            metrics.mode_accuracy.get("incorrect", metrics.accuracy)
            for metrics in clean_metrics
        ]
    )
    correct_accuracies = torch.tensor(
        [metrics.accuracy for metrics in correct_metrics]
    )
    masked_losses = torch.tensor(
        [metrics.mode_loss.get("masked", metrics.loss) for metrics in clean_metrics]
    )
    incorrect_losses = torch.tensor(
        [
            metrics.mode_loss.get("incorrect", metrics.loss)
            for metrics in clean_metrics
        ]
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
        key=lambda index: min(clean_metrics[index].mode_accuracy.values()),
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
        "best/masked_accuracy": best_clean.mode_accuracy.get(
            "masked", best_clean.accuracy
        ),
        "best/incorrect_accuracy": best_clean.mode_accuracy.get(
            "incorrect", best_clean.accuracy
        ),
        "best/correct_leak_accuracy": best_correct.accuracy,
        "best/unique_value_predictions": float(
            best_clean.unique_value_prediction_count
        ),
        "best/prediction_mode_fraction": (
            best_clean.prediction_mode_fraction
        ),
        "robust/candidate_index": robust_index,
        "robust/min_mode_accuracy": min(
            robust_clean.mode_accuracy.values()
        ),
        "robust/fitness": float(fitnesses[robust_index]),
        "robust/clean_loss": robust_clean.loss,
        "robust/clean_accuracy": robust_clean.accuracy,
        "robust/masked_accuracy": robust_clean.mode_accuracy.get(
            "masked", robust_clean.accuracy
        ),
        "robust/incorrect_accuracy": (
            robust_clean.mode_accuracy.get("incorrect", robust_clean.accuracy)
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
            metrics.mode_accuracy.get("masked", metrics.accuracy)
            for metrics in heldout_clean_metrics
        ]
    )
    heldout_incorrect = torch.tensor(
        [
            metrics.mode_accuracy.get("incorrect", metrics.accuracy)
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
                metrics.mode_loss.values(),
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
            outer_clean_metrics[index].mode_accuracy.values()
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
        "best/heldout_masked_accuracy": best_clean.mode_accuracy.get(
            "masked", best_clean.accuracy
        ),
        "best/heldout_incorrect_accuracy": (
            best_clean.mode_accuracy.get("incorrect", best_clean.accuracy)
        ),
        "best/heldout_min_mode_accuracy": float(
            heldout_min_accuracy[best_index]
        ),
        "best/heldout_correct_leak_accuracy": float(
            heldout_correct[best_index]
        ),
        "robust/heldout_clean_loss": robust_clean.loss,
        "robust/heldout_masked_accuracy": (
            robust_clean.mode_accuracy.get("masked", robust_clean.accuracy)
        ),
        "robust/heldout_incorrect_accuracy": (
            robust_clean.mode_accuracy.get("incorrect", robust_clean.accuracy)
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
    if not all(
        "routing_leak_relative_gate" in item
        for statistics in candidate_statistics
        for item in statistics
    ):
        mean_gates = torch.tensor(
            [
                sum(item["routing_gate"] for item in statistics)
                / len(statistics)
                for statistics in candidate_statistics
            ]
        )
        suppressed_fractions = torch.tensor(
            [
                sum(
                    item["routing_suppressed_fraction"]
                    for item in statistics
                )
                / len(statistics)
                for statistics in candidate_statistics
            ]
        )
        return {
            "backward/population_gate_mean": float(mean_gates.mean()),
            "backward/population_gate_min": float(mean_gates.min()),
            "backward/population_gate_max": float(mean_gates.max()),
            "backward/population_suppressed_fraction_mean": float(
                suppressed_fractions.mean()
            ),
        }
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
        key=lambda index: min(clean_metrics[index].mode_accuracy.values()),
    )
    summary = {
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
    if all(
        "routing_input_conditioned_rms" in item
        and "routing_position_profile_std" in item
        for statistics in candidate_statistics
        for item in statistics
    ):
        input_conditioned_rms = torch.tensor(
            [
                sum(
                    item["routing_input_conditioned_rms"]
                    for item in statistics
                )
                / len(statistics)
                for statistics in candidate_statistics
            ],
            dtype=torch.float32,
        )
        position_profile_std = torch.tensor(
            [
                sum(
                    item["routing_position_profile_std"]
                    for item in statistics
                )
                / len(statistics)
                for statistics in candidate_statistics
            ],
            dtype=torch.float32,
        )
        centered_conditioning = (
            input_conditioned_rms - input_conditioned_rms.mean()
        )
        conditioning_denominator = (
            centered_fitness.square().sum()
            * centered_conditioning.square().sum()
        ).sqrt()
        conditioning_correlation = (
            float(
                (centered_fitness * centered_conditioning).sum()
                / conditioning_denominator
            )
            if float(conditioning_denominator) > 0
            else 0.0
        )
        centered_position_profile = (
            position_profile_std - position_profile_std.mean()
        )
        position_profile_denominator = (
            centered_fitness.square().sum()
            * centered_position_profile.square().sum()
        ).sqrt()
        position_profile_correlation = (
            float(
                (centered_fitness * centered_position_profile).sum()
                / position_profile_denominator
            )
            if float(position_profile_denominator) > 0
            else 0.0
        )
        summary.update(
            {
                "backward/population_input_conditioned_rms_mean": float(
                    input_conditioned_rms.mean()
                ),
                "backward/population_input_conditioned_rms_min": float(
                    input_conditioned_rms.min()
                ),
                "backward/population_input_conditioned_rms_max": float(
                    input_conditioned_rms.max()
                ),
                "backward/input_conditioning_fitness_correlation": (
                    conditioning_correlation
                ),
                "backward/best_fitness_input_conditioned_rms": float(
                    input_conditioned_rms[best_index]
                ),
                "backward/robust_candidate_input_conditioned_rms": float(
                    input_conditioned_rms[robust_index]
                ),
                "backward/population_position_profile_std_mean": float(
                    position_profile_std.mean()
                ),
                "backward/population_position_profile_std_min": float(
                    position_profile_std.min()
                ),
                "backward/population_position_profile_std_max": float(
                    position_profile_std.max()
                ),
                "backward/position_profile_fitness_correlation": (
                    position_profile_correlation
                ),
                "backward/best_fitness_position_profile_std": float(
                    position_profile_std[best_index]
                ),
                "backward/robust_candidate_position_profile_std": float(
                    position_profile_std[robust_index]
                ),
            }
        )
    return summary


def function_delta_alignment_summary(
    candidate_deltas: Tensor,
    fitnesses: Tensor,
    *,
    robust_index: int,
    top_k: int = 8,
) -> dict[str, float]:
    """Summarize whether high-fitness function changes point together."""

    if candidate_deltas.ndim != 2:
        raise ValueError("candidate deltas must have shape [P, D]")
    population_size = candidate_deltas.shape[0]
    if fitnesses.shape != (population_size,):
        raise ValueError("fitness count must match candidate deltas")
    if not 0 <= robust_index < population_size:
        raise ValueError("robust index must select a candidate")
    if top_k < 2:
        raise ValueError("top_k must be at least two")
    top_k = min(top_k, population_size)

    deltas = candidate_deltas.float()
    norms = deltas.norm(dim=1)
    normalized = deltas / norms.clamp_min(1e-12).unsqueeze(1)
    top_indices = fitnesses.float().topk(top_k).indices
    top_normalized = normalized[top_indices]
    similarities = top_normalized @ top_normalized.T
    off_diagonal = ~torch.eye(
        top_k,
        dtype=torch.bool,
        device=similarities.device,
    )
    top_pairwise = similarities[off_diagonal]

    standardized = (
        fitnesses.float() - fitnesses.float().mean()
    ) / torch.sqrt(fitnesses.float().var(unbiased=False) + 1e-5)
    weighted_delta = (
        standardized.unsqueeze(1) * deltas
    ).mean(dim=0) * population_size**0.5
    unweighted_delta = deltas.mean(dim=0)
    top_centroid = deltas[top_indices].mean(dim=0)
    best_index = int(fitnesses.argmax())

    def cosine(left: Tensor, right: Tensor) -> float:
        denominator = left.norm() * right.norm()
        if float(denominator) <= 1e-12:
            return 0.0
        return float((left @ right) / denominator)

    def rms(value: Tensor) -> float:
        return float(value.square().mean().sqrt())

    summary = {
        "backward/function_delta_rms_mean": float(
            deltas.square().mean(dim=1).sqrt().mean()
        ),
        "backward/best_fitness_function_delta_rms": rms(
            deltas[best_index]
        ),
        "backward/robust_candidate_function_delta_rms": rms(
            deltas[robust_index]
        ),
        "backward/top_function_pairwise_cosine_mean": float(
            top_pairwise.mean()
        ),
        "backward/top_function_pairwise_cosine_abs_mean": float(
            top_pairwise.abs().mean()
        ),
        "backward/top_function_centroid_rms": rms(top_centroid),
        "backward/fitness_weighted_function_delta_rms": rms(
            weighted_delta
        ),
        "backward/unweighted_function_delta_rms": rms(unweighted_delta),
        "backward/fitness_weighted_cosine_with_best": cosine(
            weighted_delta,
            deltas[best_index],
        ),
        "backward/fitness_weighted_cosine_with_robust": cosine(
            weighted_delta,
            deltas[robust_index],
        ),
    }
    summary.update(
        {
            f"backward/top_function_candidate_{rank}_index": float(index)
            for rank, index in enumerate(top_indices.tolist())
        }
    )
    return summary


@torch.inference_mode()
def routing_population_function_summary(
    center_rule: AttentionRoutingRule,
    center_parameters: dict[str, Tensor],
    directions: tuple[EggrollDirection, ...],
    fitnesses: Tensor,
    clean_metrics: list[ShortcutMetrics],
    *,
    token_ids: Tensor,
    sigma: float,
) -> dict[str, float]:
    """Measure candidate agreement in backward-routing function space."""

    center_gates = center_rule.attention_gates(token_ids)[0]
    sequence_length = token_ids.shape[1]
    causal = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=center_gates.device,
    ).tril()
    center_vector = center_gates[..., causal].flatten().float()
    candidate_deltas = []
    work_rule = initialize_backward_rule(
        center_rule.config,
        device=token_ids.device,
    )
    for direction in directions:
        for sign in (1, -1):
            apply_eggroll_direction(
                work_rule,
                center_parameters,
                direction,
                sigma=sigma,
                sign=sign,
            )
            candidate_vector = (
                work_rule.attention_gates(token_ids)[0][..., causal]
                .flatten()
                .float()
            )
            candidate_deltas.append(
                (candidate_vector - center_vector).cpu()
            )
    robust_index = max(
        range(len(clean_metrics)),
        key=lambda index: min(clean_metrics[index].mode_accuracy.values()),
    )
    return function_delta_alignment_summary(
        torch.stack(candidate_deltas),
        fitnesses.detach().cpu(),
        robust_index=robust_index,
    )


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
        routing_config = dict(checkpoint["backward_rule_config"])
        shortcut_vocabulary = ShortcutPointerVocabulary("numbers", 10)
        if (
            "leak_token" not in routing_config
            and routing_config["vocab_size"] == shortcut_vocabulary.size
        ):
            routing_config["leak_token"] = shortcut_vocabulary.leak_token
        rule_config: RuleConfig = AttentionRoutingRuleConfig(
            **routing_config
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


def resolve_resume_horizon(
    config: ShortcutCreditExperimentConfig,
    checkpoint_horizon: int,
) -> int:
    if config.resume_horizon is None:
        return checkpoint_horizon
    return config.resume_horizon


def apply_resume_horizon(
    config: ShortcutCreditExperimentConfig,
    checkpoint_horizon: int,
    state: PlateauState,
) -> int:
    horizon = resolve_resume_horizon(config, checkpoint_horizon)
    if horizon != checkpoint_horizon:
        state.consecutive_accepted_updates = 0
        state.consecutive_rejected_updates = 0
        reset_horizon_tracking(state)
    return horizon


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

    vocabulary = make_experiment_vocabulary(config)
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
        horizon = apply_resume_horizon(config, horizon, plateau_state)
        if center_rule.config != backward_config:
            raise ValueError("resume checkpoint architecture differs from config")
    if plateau_state.search_sigma is None:
        plateau_state.search_sigma = config.sigma
    if (
        config.adaptive_commit_scale is not None
        and plateau_state.commit_scale is None
    ):
        plateau_state.commit_scale = config.adaptive_commit_scale

    if config.task_variant == "pointer_next_length":
        assert config.fitness_length is not None
        assert config.heldout_length is not None
        fitness_batches, acceptance_fitness_batches = (
            make_fixed_fitness_batch_sets(
                config,
                vocabulary=vocabulary,
                device=device,
            )
        )
        fitness_generator = torch.Generator().manual_seed(
            config.seed + 12_500
        )
    else:
        fitness_generator = torch.Generator().manual_seed(
            config.seed + 10_000
        )
        if not isinstance(vocabulary, ShortcutPointerVocabulary):
            raise TypeError("shortcut task requires shortcut vocabulary")
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
        acceptance_fitness_batches = fitness_batches
    correct_batches = make_mode_batches(
        config.correct_eval_examples,
        leak_mode=(
            "clean"
            if config.task_variant == "pointer_next_length"
            else "correct"
        ),
        config=config,
        vocabulary=vocabulary,
        generator=fitness_generator,
        device=device,
    )
    fixed_heldout_fitness_batches = None
    fixed_heldout_correct_batches = None
    if config.task_variant == "pointer_next_length":
        fixed_heldout_generator = torch.Generator().manual_seed(
            config.seed + 20_000
        )
        fixed_heldout_fitness_batches = make_fixed_length_batches(
            config.heldout_examples,
            length=config.heldout_length,
            config=config,
            vocabulary=vocabulary,
            generator=fixed_heldout_generator,
            device=device,
        )
        fixed_heldout_correct_batches = make_mode_batches(
            config.heldout_examples,
            leak_mode="clean",
            config=config,
            vocabulary=vocabulary,
            generator=fixed_heldout_generator,
            device=device,
        )

    wandb_run = maybe_initialize_wandb(config)
    started_at = time.monotonic()
    for generation in range(start_generation, config.generations):
        generation_started_at = time.monotonic()
        report_generation = (
            generation % config.report_interval == 0
            or generation + 1 == config.generations
        )
        for worker_device in candidate_devices:
            if worker_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(worker_device)
        generation_sigma = plateau_state.search_sigma
        if generation_sigma is None:
            raise RuntimeError("search sigma was not initialized")
        generation_seed = config.seed * 1_000_003 + generation * 10_007
        initialization_seed = generation_seed + 1
        if not report_generation:
            heldout_fitness_batches = None
            heldout_correct_batches = None
        elif fixed_heldout_fitness_batches is not None:
            assert fixed_heldout_correct_batches is not None
            heldout_fitness_batches = fixed_heldout_fitness_batches
            heldout_correct_batches = fixed_heldout_correct_batches
        else:
            heldout_generator = torch.Generator().manual_seed(
                generation_seed + 4
            )
            if not isinstance(vocabulary, ShortcutPointerVocabulary):
                raise TypeError("shortcut task requires shortcut vocabulary")
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
        additional_ranking_inputs_list = []
        for ranking_seed in candidate_ranking_seeds(
            generation_seed,
            config.candidate_ranking_trajectories,
        )[1:]:
            ranking_model = initialize_forward_model(
                config,
                vocabulary,
                initialization_seed=ranking_seed + 1,
                device=device,
            )
            ranking_base_state = {
                name: tensor.detach().clone()
                for name, tensor in ranking_model.state_dict().items()
            }
            ranking_initial_clean = evaluate_shortcut_batches(
                ranking_model,
                fitness_batches,
            )
            del ranking_model
            ranking_inner_batches = make_inner_batches(
                config,
                horizon=horizon,
                vocabulary=vocabulary,
                generator=torch.Generator().manual_seed(ranking_seed + 2),
                device=device,
            )
            additional_ranking_inputs_list.append(
                (
                    ranking_base_state,
                    ranking_inner_batches,
                    ranking_initial_clean,
                )
            )
        additional_ranking_inputs = tuple(additional_ranking_inputs_list)
        masked_inner_batches = (
            make_inner_batches(
                config,
                horizon=horizon,
                vocabulary=vocabulary,
                generator=torch.Generator().manual_seed(
                    generation_seed + 2
                ),
                device=device,
                leak_mode="masked",
            )
            if report_generation
            else None
        )
        direction_generator = torch.Generator().manual_seed(generation_seed + 3)
        direction_sampling_summary = {
            "direction_sampling/method_function_diverse": 0.0,
            "direction_sampling/pool_size": float(
                config.population_size // 2
            ),
        }
        if config.direction_sampler == "function_diverse":
            if not isinstance(center_rule, AttentionRoutingRule):
                raise TypeError(
                    "function-diverse directions require an attention router"
                )
            sampling_result = sample_function_diverse_directions(
                center_rule,
                generator=direction_generator,
                count=config.population_size // 2,
                candidate_multiplier=config.direction_candidate_multiplier,
                sigma=generation_sigma,
                probe_token_ids=inner_batches[0].input_ids[
                    : config.direction_probe_examples
                ],
                signature_size=config.direction_signature_size,
            )
            directions = sampling_result.directions
            direction_sampling_summary.update(sampling_result.metrics)
            direction_sampling_summary[
                "direction_sampling/method_function_diverse"
            ] = 1.0
        else:
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
                tuple[float, ...],
            ]
        ] = []
        halving_eligible_indices: tuple[int, ...] | None = None
        halving_summary: dict[str, float] = {}
        population_started_at = time.monotonic()
        if config.successive_halving_rungs is not None:
            if not isinstance(center_rule, AttentionRoutingRule):
                raise TypeError(
                    "successive halving requires an attention router"
                )
            (
                candidate_outputs,
                halving_eligible_indices,
                halving_summary,
            ) = train_successive_halving_population(
                config,
                candidate_specs=candidate_specs,
                candidate_devices=candidate_devices,
                base_state=base_state,
                center_rule=center_rule,
                center_parameters=center_parameters,
                directions=directions,
                inner_batches=inner_batches,
                fitness_batches=fitness_batches,
                correct_batches=correct_batches,
                heldout_fitness_batches=heldout_fitness_batches,
                heldout_correct_batches=heldout_correct_batches,
                initial_clean_metrics=initial_clean_metrics,
                perturbation_sigma=generation_sigma,
            )
        elif config.vectorized_population:
            from .vectorized_routing_population import (
                train_vectorized_routing_candidate_chunks,
            )

            if not isinstance(center_rule, AttentionRoutingRule):
                raise TypeError(
                    "vectorized populations require an attention router"
                )
            shards = shard_candidate_specs(
                candidate_specs,
                len(candidate_devices),
            )
            if len(candidate_devices) == 1:
                candidate_outputs.extend(
                    train_vectorized_routing_candidate_chunks(
                        config=config,
                        candidate_specs=shards[0],
                        chunk_size=config.vectorized_chunk_size,
                        device=candidate_devices[0],
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
                        perturbation_sigma=generation_sigma,
                        additional_ranking_inputs=(
                            additional_ranking_inputs
                        ),
                    )
                )
            else:
                with ThreadPoolExecutor(
                    max_workers=len(candidate_devices)
                ) as executor:
                    futures = [
                        executor.submit(
                            train_vectorized_routing_candidate_chunks,
                            config=config,
                            candidate_specs=shard,
                            chunk_size=config.vectorized_chunk_size,
                            device=worker_device,
                            base_state=base_state,
                            center_rule_config=center_rule.config,
                            center_parameters=center_parameters,
                            directions=directions,
                            inner_batches=inner_batches,
                            fitness_batches=fitness_batches,
                            correct_batches=correct_batches,
                            heldout_fitness_batches=(
                                heldout_fitness_batches
                            ),
                            heldout_correct_batches=(
                                heldout_correct_batches
                            ),
                            initial_clean_metrics=initial_clean_metrics,
                            perturbation_sigma=generation_sigma,
                            additional_ranking_inputs=(
                                additional_ranking_inputs
                            ),
                        )
                        for shard, worker_device in zip(
                            shards,
                            candidate_devices,
                        )
                        if shard
                    ]
                    for future in futures:
                        candidate_outputs.extend(future.result())
        elif len(candidate_devices) == 1:
            for candidate_index, direction_index, sign in candidate_specs:
                (
                    fitness,
                    trajectory,
                    statistics,
                    ranking_fitnesses,
                ) = train_candidate(
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
                    perturbation_sigma=generation_sigma,
                    heldout_fitness_batches=heldout_fitness_batches,
                    heldout_correct_batches=heldout_correct_batches,
                    additional_ranking_inputs=additional_ranking_inputs,
                )
                candidate_outputs.append(
                    (
                        candidate_index,
                        fitness,
                        trajectory,
                        statistics,
                        ranking_fitnesses,
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
                        perturbation_sigma=generation_sigma,
                        additional_ranking_inputs=(
                            additional_ranking_inputs
                        ),
                    )
                    for shard, worker_device in zip(
                        shards,
                        candidate_devices,
                    )
                    if shard
                ]
                for future in futures:
                    candidate_outputs.extend(future.result())
        population_seconds = time.monotonic() - population_started_at

        candidate_outputs.sort(key=lambda result: result[0])
        fitness_values = []
        clean_results = []
        correct_results = []
        heldout_clean_results = []
        heldout_correct_results = []
        candidate_trajectories = []
        candidate_statistics: list[list[dict[str, float]]] = []
        ranking_fitness_groups = []
        captured_statistics: list[dict[str, float]] = []
        for (
            candidate_index,
            fitness,
            trajectory,
            statistics,
            ranking_fitnesses,
        ) in candidate_outputs:
            fitness_values.append(fitness)
            ranking_fitness_groups.append(ranking_fitnesses)
            candidate_trajectories.append(trajectory)
            clean_results.append(trajectory.clean)
            correct_results.append(trajectory.correct)
            if report_generation:
                if (
                    trajectory.heldout_clean is None
                    or trajectory.heldout_correct is None
                ):
                    if halving_eligible_indices is None:
                        raise RuntimeError(
                            "candidate held-out metrics were not produced"
                        )
                else:
                    heldout_clean_results.append(trajectory.heldout_clean)
                    heldout_correct_results.append(
                        trajectory.heldout_correct
                    )
            candidate_statistics.append(statistics)
            if candidate_index == 0 and statistics:
                captured_statistics = statistics

        center_fitness = None
        center_trajectory = None
        ordinary_trajectory = None
        masked_training_trajectory = None
        if report_generation:
            center_fitness, center_trajectory, _, _ = train_candidate(
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
            if config.task_variant == "pointer_next_length":
                masked_training_trajectory = ordinary_trajectory
            else:
                if masked_inner_batches is None:
                    raise RuntimeError("masked reporting batches are missing")
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
        fitness_tensor = torch.tensor(fitness_values, device=device)
        selection_fitness_tensor = fitness_tensor
        if halving_eligible_indices is not None:
            eligible_mask = torch.zeros(
                len(fitness_values),
                dtype=torch.bool,
                device=device,
            )
            eligible_mask[list(halving_eligible_indices)] = True
            selection_fitness_tensor = fitness_tensor.masked_fill(
                ~eligible_mask,
                float("-inf"),
            )
        ranking_fitness_tensor = torch.tensor(
            ranking_fitness_groups,
            device=device,
        )
        torch.testing.assert_close(
            ranking_fitness_tensor.mean(dim=1),
            fitness_tensor,
        )
        function_space_summary = (
            routing_population_function_summary(
                center_rule,
                center_parameters,
                directions,
                fitness_tensor,
                clean_results,
                token_ids=inner_batches[-1].input_ids,
                sigma=generation_sigma,
            )
            if (
                isinstance(center_rule, AttentionRoutingRule)
                and halving_eligible_indices is None
            )
            else {}
        )
        outer_learning_rate = linear_outer_learning_rate(config, generation)
        standardized = (
            fitness_tensor - fitness_tensor.mean()
        ) / torch.sqrt(fitness_tensor.var(unbiased=False) + 1e-5)
        adaptive_elite_counts = parse_adaptive_elite_counts(
            config.adaptive_elite_counts
        )
        adaptive_elite_fitnesses: dict[int, float] = {}
        adaptive_commit_fitnesses: tuple[
            tuple[int, float, float], ...
        ] = ()
        adaptive_selection_center_fitness: float | None = None
        adaptive_selection_selected_fitness: float | None = None
        acceptance_seconds: float | None = None
        selected_elite_count: int | None = None
        selected_commit_scale: float | None = None
        if config.outer_update_rule == "paper_standardized":
            standardized = paper_eggroll_update(
                center_rule,
                directions,
                fitness_tensor,
                sigma=generation_sigma,
                learning_rate=outer_learning_rate,
            )
            elite_indices = None
            elite_proposal_fitness = None
            elite_proposal_trajectory = None
            elite_update_accepted = None
            elite_acceptance_center_fitnesses = None
            elite_acceptance_proposal_fitnesses = None
        elif adaptive_elite_counts:
            if not isinstance(center_rule, AttentionRoutingRule):
                raise TypeError(
                    "adaptive elites require an attention router"
                )
            acceptance_started_at = time.monotonic()
            adaptive_result = select_adaptive_elite_proposal(
                config,
                center_rule=center_rule,
                center_parameters=center_parameters,
                directions=directions,
                fitnesses=selection_fitness_tensor,
                sigma=generation_sigma,
                generation_seed=generation_seed,
                horizon=horizon,
                vocabulary=vocabulary,
                acceptance_fitness_batches=acceptance_fitness_batches,
                correct_batches=correct_batches,
                acceptance_devices=candidate_devices,
                commit_scale=plateau_state.commit_scale,
            )
            acceptance_seconds = time.monotonic() - acceptance_started_at
            selected_elite_count = adaptive_result.selected_count
            selected_commit_scale = adaptive_result.selected_commit_scale
            elite_indices = adaptive_result.selected_indices
            adaptive_elite_fitnesses = (
                adaptive_result.mean_fitness_by_count
            )
            adaptive_commit_fitnesses = (
                adaptive_result.selection_fitness_by_proposal
            )
            adaptive_selection_center_fitness = (
                adaptive_result.selection_center_fitness
            )
            adaptive_selection_selected_fitness = (
                adaptive_result.selection_selected_fitness
            )
            restore_center_parameters(
                center_rule,
                adaptive_result.selected_parameters,
            )
            if report_generation:
                elite_proposal_trajectory = train_forward_trajectory(
                    config,
                    base_state=base_state,
                    backward_rule=center_rule,
                    inner_batches=inner_batches,
                    fitness_batches=fitness_batches,
                    correct_batches=correct_batches,
                    heldout_fitness_batches=heldout_fitness_batches,
                    heldout_correct_batches=heldout_correct_batches,
                    device=device,
                )
                elite_proposal_fitness = candidate_fitness(
                    config.fitness_objective,
                    initial_clean_metrics,
                    elite_proposal_trajectory.clean,
                    checkpoint_clean=(
                        elite_proposal_trajectory.checkpoint_clean
                    ),
                )
            else:
                elite_proposal_trajectory = None
                elite_proposal_fitness = None
            elite_update_accepted = adaptive_result.accepted
            elite_acceptance_center_fitnesses = list(
                adaptive_result.center_fitnesses
            )
            elite_acceptance_proposal_fitnesses = list(
                adaptive_result.selected_fitnesses
            )
            if not elite_update_accepted:
                restore_center_parameters(
                    center_rule,
                    center_parameters,
                )
            elif selected_commit_scale is not None:
                plateau_state.commit_scale = selected_commit_scale
            center_rule.project_parameters_()
            update_elite_search_state(
                plateau_state,
                accepted=elite_update_accepted,
                config=config,
            )
        else:
            selected_elite_count = config.elite_count
            elite_indices = elite_centroid_update(
                center_rule,
                directions,
                selection_fitness_tensor,
                sigma=generation_sigma,
                elite_count=config.elite_count,
                interpolation=config.elite_interpolation,
                deduplicate_antithetic=(
                    config.deduplicate_antithetic_elites
                ),
            )
        if isinstance(center_rule, AttentionRoutingRule):
            center_rule.project_parameters_()
        if (
            config.outer_update_rule == "elite_centroid"
            and config.elite_backtracking
            and not adaptive_elite_counts
        ):
            elite_proposal_trajectory = train_forward_trajectory(
                config,
                base_state=base_state,
                backward_rule=center_rule,
                inner_batches=inner_batches,
                fitness_batches=fitness_batches,
                correct_batches=correct_batches,
                heldout_fitness_batches=heldout_fitness_batches,
                heldout_correct_batches=heldout_correct_batches,
                device=device,
            )
            elite_proposal_fitness = candidate_fitness(
                config.fitness_objective,
                initial_clean_metrics,
                elite_proposal_trajectory.clean,
                checkpoint_clean=(
                    elite_proposal_trajectory.checkpoint_clean
                ),
            )
            # The trajectory used to rank the population is optimistically
            # biased toward the selected proposal. Keep its result separate
            # and accept only on independently seeded trajectories.
            elite_acceptance_center_fitnesses = []
            elite_acceptance_proposal_fitnesses = []
            proposal_parameters = clone_center_parameters(center_rule)
            for acceptance_seed in independent_elite_acceptance_seeds(
                generation_seed,
                config.elite_acceptance_trajectories,
                start_index=config.candidate_ranking_trajectories,
            ):
                acceptance_model = initialize_forward_model(
                    config,
                    vocabulary,
                    initialization_seed=acceptance_seed + 1,
                    device=device,
                )
                acceptance_base_state = {
                    name: tensor.detach().clone()
                    for name, tensor in acceptance_model.state_dict().items()
                }
                acceptance_initial = evaluate_shortcut_batches(
                    acceptance_model,
                    acceptance_fitness_batches,
                )
                del acceptance_model
                acceptance_inner_batches = make_inner_batches(
                    config,
                    horizon=horizon,
                    vocabulary=vocabulary,
                    generator=torch.Generator().manual_seed(
                        acceptance_seed + 2
                    ),
                    device=device,
                )

                restore_center_parameters(
                    center_rule,
                    center_parameters,
                )
                acceptance_center = train_forward_trajectory(
                    config,
                    base_state=acceptance_base_state,
                    backward_rule=center_rule,
                    inner_batches=acceptance_inner_batches,
                    fitness_batches=acceptance_fitness_batches,
                    correct_batches=correct_batches,
                    device=device,
                )
                restore_center_parameters(
                    center_rule,
                    proposal_parameters,
                )
                acceptance_proposal = train_forward_trajectory(
                    config,
                    base_state=acceptance_base_state,
                    backward_rule=center_rule,
                    inner_batches=acceptance_inner_batches,
                    fitness_batches=acceptance_fitness_batches,
                    correct_batches=correct_batches,
                    device=device,
                )
                elite_acceptance_center_fitnesses.append(
                    candidate_fitness(
                        config.fitness_objective,
                        acceptance_initial,
                        acceptance_center.clean,
                        checkpoint_clean=(
                            acceptance_center.checkpoint_clean
                        ),
                    )
                )
                elite_acceptance_proposal_fitnesses.append(
                    candidate_fitness(
                        config.fitness_objective,
                        acceptance_initial,
                        acceptance_proposal.clean,
                        checkpoint_clean=(
                            acceptance_proposal.checkpoint_clean
                        ),
                    )
                )
            restore_center_parameters(
                center_rule,
                proposal_parameters,
            )
            elite_update_accepted = (
                elite_proposal_improves_every_trajectory(
                    elite_acceptance_center_fitnesses,
                    elite_acceptance_proposal_fitnesses,
                )
            )
            if not elite_update_accepted:
                restore_center_parameters(center_rule, center_parameters)
                if isinstance(center_rule, AttentionRoutingRule):
                    center_rule.project_parameters_()
            update_elite_search_state(
                plateau_state,
                accepted=elite_update_accepted,
                config=config,
            )
        elif (
            config.outer_update_rule == "elite_centroid"
            and not adaptive_elite_counts
        ):
            elite_proposal_fitness = None
            elite_proposal_trajectory = None
            elite_update_accepted = True
            elite_acceptance_center_fitnesses = None
            elite_acceptance_proposal_fitnesses = None
        summary = candidate_summary(
            fitness_tensor.cpu(),
            clean_results,
            correct_results,
        )
        ranking_best_index = int(selection_fitness_tensor.argmax())
        summary.update(
            {
                "fitness/ranking_trajectory_count": float(
                    ranking_fitness_tensor.shape[1]
                ),
                "fitness/ranking_within_candidate_std_mean": float(
                    ranking_fitness_tensor.std(
                        dim=1,
                        unbiased=False,
                    ).mean()
                ),
                "best/ranking_fitness_std": float(
                    ranking_fitness_tensor[ranking_best_index].std(
                        unbiased=False
                    )
                ),
            }
        )
        if acceptance_seconds is not None:
            summary["timing/acceptance_seconds"] = acceptance_seconds
        for ranking_index in range(ranking_fitness_tensor.shape[1]):
            summary[
                f"fitness/ranking_trajectory_{ranking_index}_mean"
            ] = float(ranking_fitness_tensor[:, ranking_index].mean())
        summary.update(
            checkpoint_population_summary(candidate_trajectories)
        )
        if report_generation:
            if (
                center_fitness is None
                or center_trajectory is None
                or ordinary_trajectory is None
                or masked_training_trajectory is None
            ):
                raise RuntimeError("reporting trajectories were not produced")
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
                raise RuntimeError(
                    "held-out reporting metrics were not produced"
                )
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
            ordinary_fitness = candidate_fitness(
                config.fitness_objective,
                initial_clean_metrics,
                ordinary_clean,
                checkpoint_clean=ordinary_trajectory.checkpoint_clean,
            )
            masked_training_fitness = candidate_fitness(
                config.fitness_objective,
                initial_clean_metrics,
                masked_training_clean,
                checkpoint_clean=(
                    masked_training_trajectory.checkpoint_clean
                ),
            )
            if halving_eligible_indices is None:
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
                checkpoint_trajectory_summary(
                    "center_rule",
                    center_trajectory,
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
                checkpoint_trajectory_summary(
                    "ordinary_rule",
                    ordinary_trajectory,
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
                checkpoint_trajectory_summary(
                    "masked_training",
                    masked_training_trajectory,
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
            summary.update(
                {
                    "comparison/center_minus_ordinary_min_accuracy": (
                        min(center_clean.mode_accuracy.values())
                        - min(ordinary_clean.mode_accuracy.values())
                    ),
                    "comparison/masked_training_minus_ordinary_min_accuracy": (
                        min(masked_training_clean.mode_accuracy.values())
                        - min(ordinary_clean.mode_accuracy.values())
                    ),
                    "comparison/center_clean_loss_improvement_over_ordinary": (
                        ordinary_clean.loss - center_clean.loss
                    ),
                    "comparison/masked_training_clean_loss_improvement_over_ordinary": (
                        ordinary_clean.loss - masked_training_clean.loss
                    ),
                    "heldout_comparison/center_minus_ordinary_min_accuracy": (
                        min(center_heldout_clean.mode_accuracy.values())
                        - min(ordinary_heldout_clean.mode_accuracy.values())
                    ),
                    "heldout_comparison/center_clean_loss_improvement_over_ordinary": (
                        ordinary_heldout_clean.loss
                        - center_heldout_clean.loss
                    ),
                }
            )
            if config.task_variant == "pointer_next_length":
                assert config.fitness_length is not None
                assert config.heldout_length is not None
                fitness_prefix = f"length_{config.fitness_length}"
                heldout_prefix = f"length_{config.heldout_length}"
                summary.update(
                    {
                        "task/train_max_length": float(config.max_length),
                        "task/fitness_length": float(config.fitness_length),
                        "task/heldout_length": float(config.heldout_length),
                        f"{fitness_prefix}/center_accuracy": (
                            center_clean.accuracy
                        ),
                        f"{fitness_prefix}/center_loss": center_clean.loss,
                        f"{fitness_prefix}/ordinary_accuracy": (
                            ordinary_clean.accuracy
                        ),
                        f"{fitness_prefix}/ordinary_loss": ordinary_clean.loss,
                        f"{fitness_prefix}/center_minus_ordinary_accuracy": (
                            center_clean.accuracy - ordinary_clean.accuracy
                        ),
                        f"{heldout_prefix}/center_accuracy": (
                            center_heldout_clean.accuracy
                        ),
                        f"{heldout_prefix}/center_loss": (
                            center_heldout_clean.loss
                        ),
                        f"{heldout_prefix}/ordinary_accuracy": (
                            ordinary_heldout_clean.accuracy
                        ),
                        f"{heldout_prefix}/ordinary_loss": (
                            ordinary_heldout_clean.loss
                        ),
                        f"{heldout_prefix}/center_minus_ordinary_accuracy": (
                            center_heldout_clean.accuracy
                            - ordinary_heldout_clean.accuracy
                        ),
                        "train_domain/center_accuracy": (
                            center_correct.accuracy
                        ),
                        "train_domain/ordinary_accuracy": (
                            ordinary_correct.accuracy
                        ),
                        "train_domain/center_minus_ordinary_accuracy": (
                            center_correct.accuracy
                            - ordinary_correct.accuracy
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
        summary.update(function_space_summary)
        summary.update(halving_summary)
        summary.update(direction_sampling_summary)
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
                "report/full_generation": float(report_generation),
                "report/interval": float(config.report_interval),
                "search/sigma": generation_sigma,
                "search/next_sigma": plateau_state.search_sigma,
                "outer/commit_scale": float(
                    selected_commit_scale
                    if selected_commit_scale is not None
                    else (
                        plateau_state.commit_scale
                        if plateau_state.commit_scale is not None
                        else generation_sigma * config.elite_interpolation
                    )
                ),
                "outer/next_commit_scale": float(
                    plateau_state.commit_scale
                    if plateau_state.commit_scale is not None
                    else generation_sigma * config.elite_interpolation
                ),
                "search/consecutive_accepted_updates": float(
                    plateau_state.consecutive_accepted_updates
                ),
                "search/consecutive_rejected_updates": float(
                    plateau_state.consecutive_rejected_updates
                ),
                "candidate_device_count": len(candidate_devices),
                "vectorized_population": float(
                    config.vectorized_population
                ),
                "vectorized_chunk_size": float(
                    config.vectorized_chunk_size
                ),
                "forward/training_bf16": float(
                    config.forward_training_precision == "bf16"
                ),
                "outer/update_rule_elite_centroid": float(
                    config.outer_update_rule == "elite_centroid"
                ),
                "outer/selected_elite_count": float(
                    selected_elite_count or 0
                ),
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
                "timing/population_seconds": population_seconds,
                "timing/elapsed_seconds": time.monotonic() - started_at,
            }
        )
        summary.update(
            outer_update_hyperparameter_summary(
                config,
                sigma=generation_sigma,
                paper_learning_rate=outer_learning_rate,
                commit_scale=selected_commit_scale,
            )
        )
        cuda_peak_allocated = [
            torch.cuda.max_memory_allocated(worker_device) / 2**30
            for worker_device in candidate_devices
            if worker_device.type == "cuda"
        ]
        if cuda_peak_allocated:
            summary["memory/peak_allocated_gib_max"] = max(
                cuda_peak_allocated
            )
        if elite_update_accepted is not None:
            summary["outer/update_accepted"] = float(
                elite_update_accepted
            )
        for elite_count, mean_fitness in adaptive_elite_fitnesses.items():
            summary[
                f"outer/adaptive_elite_{elite_count}_acceptance_fitness"
            ] = mean_fitness
        for index, (
            elite_count,
            commit_scale,
            selection_fitness,
        ) in enumerate(adaptive_commit_fitnesses):
            prefix = f"outer/commit_search_{index}"
            summary[f"{prefix}_elite_count"] = float(elite_count)
            summary[f"{prefix}_scale"] = commit_scale
            summary[f"{prefix}_selection_fitness"] = selection_fitness
        if (
            adaptive_selection_center_fitness is not None
            and adaptive_selection_selected_fitness is not None
        ):
            summary["outer/commit_selection_center_fitness"] = (
                adaptive_selection_center_fitness
            )
            summary["outer/commit_selection_winner_fitness"] = (
                adaptive_selection_selected_fitness
            )
            summary["outer/commit_selection_winner_delta"] = (
                adaptive_selection_selected_fitness
                - adaptive_selection_center_fitness
            )
        if (
            elite_proposal_fitness is not None
            and elite_proposal_trajectory is not None
        ):
            summary.update(
                trajectory_summary(
                    "elite_proposal",
                    elite_proposal_fitness,
                    elite_proposal_trajectory.clean,
                    elite_proposal_trajectory.correct,
                )
            )
            summary.update(
                checkpoint_trajectory_summary(
                    "elite_proposal",
                    elite_proposal_trajectory,
                )
            )
            summary[
                "outer/proposal_fitness_minus_center"
            ] = elite_proposal_fitness - center_fitness
        if (
            elite_acceptance_center_fitnesses is not None
            and elite_acceptance_proposal_fitnesses is not None
        ):
            summary["outer/acceptance_trajectory_count"] = float(
                len(elite_acceptance_center_fitnesses)
            )
            summary[
                "outer/proposal_mean_fitness_minus_center"
            ] = elite_proposal_mean_improvement(
                elite_acceptance_center_fitnesses,
                elite_acceptance_proposal_fitnesses,
            )
            summary[
                "outer/proposal_min_fitness_minus_center"
            ] = min(
                proposal - center
                for center, proposal in zip(
                    elite_acceptance_center_fitnesses,
                    elite_acceptance_proposal_fitnesses,
                )
            )
            for acceptance_index, (
                acceptance_center_fitness,
                acceptance_proposal_fitness,
            ) in enumerate(
                zip(
                    elite_acceptance_center_fitnesses,
                    elite_acceptance_proposal_fitnesses,
                )
            ):
                summary[
                    f"outer/acceptance_trajectory_{acceptance_index}_delta"
                ] = (
                    acceptance_proposal_fitness
                    - acceptance_center_fitness
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
        if elite_indices is not None:
            for rank, index in enumerate(elite_indices.tolist()):
                summary[f"outer/elite_candidate_{rank}_index"] = float(index)

        # Mean fitness cannot be compared across generations because each
        # generation starts from a different model and initial clean loss.
        # Post-training clean CE is the stable cross-generation objective.
        if config.fitness_objective == "worst_checkpoint_mode_ce":
            plateau_objective = -summary[
                "transition/worst_checkpoint_mode_loss_mean"
            ]
        else:
            plateau_objective = -summary[
                (
                    "clean/worst_mode_loss_mean"
                    if config.fitness_objective == "worst_mode_ce"
                    else "clean/loss_mean"
                )
            ]
        center_plateau_objective = None
        if center_trajectory is not None:
            if config.fitness_objective == "worst_checkpoint_mode_ce":
                center_plateau_objective = -worst_checkpoint_mode_loss(
                    center_trajectory.checkpoint_clean
                )
            elif config.fitness_objective == "worst_mode_ce":
                center_plateau_objective = -max(
                    center_trajectory.clean.mode_loss.values()
                )
            else:
                center_plateau_objective = -center_trajectory.clean.loss

        plateau_promote_horizon = False
        horizon_probe_result = None
        performance_horizon_decision = None
        stop_training = False
        if config.horizon_promotion_mode == "fixed":
            promote_horizon = False
        elif config.horizon_promotion_mode == "plateau":
            plateau_promote_horizon = update_plateau_state(
                plateau_state,
                objective=plateau_objective,
                config=config,
            )
            promote_horizon = plateau_promote_horizon
        elif config.horizon_promotion_mode == "rejection_probe":
            promote_horizon = False
            if (
                horizon < config.max_horizon
                and plateau_state.consecutive_rejected_updates
                >= config.horizon_rejection_patience
            ):
                horizon_probe_result = probe_longer_horizon(
                    config,
                    center_rule=center_rule,
                    generation_seed=generation_seed,
                    horizon=horizon,
                    vocabulary=vocabulary,
                    fitness_batches=fitness_batches,
                    correct_batches=correct_batches,
                    device=device,
                )
                promote_horizon = horizon_probe_result.promoted
                plateau_state.consecutive_rejected_updates = 0
        else:
            if center_plateau_objective is None:
                raise RuntimeError(
                    "performance horizon mode requires centre reporting"
                )
            performance_horizon_decision = (
                update_performance_horizon_state(
                    plateau_state,
                    objective=center_plateau_objective,
                    horizon=horizon,
                    config=config,
                )
            )
            promote_horizon = performance_horizon_decision.promote
            stop_training = performance_horizon_decision.stop
        summary.update(
            {
                "curriculum/objective_negative_clean_loss": plateau_objective,
                "curriculum/ema_objective": plateau_state.ema_fitness,
                "curriculum/stale_generations": plateau_state.stale_generations,
                "curriculum/fixed_horizon_mode": float(
                    config.horizon_promotion_mode == "fixed"
                ),
                "curriculum/rejection_probe_mode": float(
                    config.horizon_promotion_mode == "rejection_probe"
                ),
                "curriculum/performance_plateau_mode": float(
                    config.horizon_promotion_mode == "performance_plateau"
                ),
                "curriculum/promoted": float(
                    promote_horizon and horizon < config.max_horizon
                ),
                "curriculum/stop_triggered": float(stop_training),
            }
        )
        if center_plateau_objective is not None:
            summary["curriculum/center_objective"] = (
                center_plateau_objective
            )
        if horizon_probe_result is not None:
            summary.update(
                {
                    "curriculum/probe_current_fitness": (
                        horizon_probe_result.current_fitness
                    ),
                    "curriculum/probe_longer_fitness": (
                        horizon_probe_result.longer_fitness
                    ),
                    "curriculum/probe_improvement": (
                        horizon_probe_result.longer_fitness
                        - horizon_probe_result.current_fitness
                    ),
                    "curriculum/probe_next_horizon": float(
                        horizon_probe_result.next_horizon
                    ),
                }
            )
        if performance_horizon_decision is not None:
            summary.update(
                {
                    "curriculum/horizon_generations": float(
                        plateau_state.horizon_generations
                    ),
                    "curriculum/horizon_stale_generations": float(
                        plateau_state.horizon_stale_generations
                    ),
                    "curriculum/failed_horizon_extensions": float(
                        plateau_state.failed_horizon_extensions
                    ),
                    "curriculum/plateau_detected": float(
                        performance_horizon_decision.plateau_detected
                    ),
                    "curriculum/maximum_dwell_reached": float(
                        performance_horizon_decision.maximum_dwell_reached
                    ),
                }
            )
            if plateau_state.horizon_best_average != float("-inf"):
                summary["curriculum/horizon_best_average"] = (
                    plateau_state.horizon_best_average
                )
            if performance_horizon_decision.rolling_average is not None:
                summary["curriculum/horizon_rolling_average"] = (
                    performance_horizon_decision.rolling_average
                )
            if (
                performance_horizon_decision.completed_horizon_average
                is not None
            ):
                summary["curriculum/completed_horizon_average"] = (
                    performance_horizon_decision.completed_horizon_average
                )
            if performance_horizon_decision.promote:
                summary["curriculum/next_horizon"] = float(
                    min(
                        config.max_horizon,
                        horizon * config.horizon_multiplier,
                    )
                )
            if plateau_state.horizon_reference_average is not None:
                summary["curriculum/reference_average"] = (
                    plateau_state.horizon_reference_average
                )
            if performance_horizon_decision.extension_improved is not None:
                summary["curriculum/extension_improved"] = float(
                    performance_horizon_decision.extension_improved
                )
            if performance_horizon_decision.stop_reason is not None:
                summary["curriculum/stop_reason"] = (
                    performance_horizon_decision.stop_reason
                )
        summary["search/consecutive_rejected_updates"] = float(
            plateau_state.consecutive_rejected_updates
        )
        if config.task_variant == "pointer_next_length":
            summary = strip_shortcut_only_metrics(summary)

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
            if config.horizon_promotion_mode != "performance_plateau":
                search_sigma = plateau_state.search_sigma
                consecutive_accepted_updates = (
                    plateau_state.consecutive_accepted_updates
                )
                plateau_state = PlateauState(
                    search_sigma=search_sigma,
                    commit_scale=plateau_state.commit_scale,
                    consecutive_accepted_updates=(
                        consecutive_accepted_updates
                    ),
                )
            else:
                reset_horizon_tracking(plateau_state)
            print(f"Increasing evolved horizon to {horizon}", flush=True)

        if (
            (generation + 1) % config.checkpoint_interval == 0
            or generation + 1 == config.generations
            or stop_training
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

        if stop_training:
            print(
                "Stopping curriculum: "
                f"{performance_horizon_decision.stop_reason}",
                flush=True,
            )
            break

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
    parser.add_argument(
        "--acceptance-fitness-examples",
        type=int,
        default=0,
        help=(
            "fixed examples reserved for proposal acceptance; zero reuses "
            "the candidate-ranking fitness set"
        ),
    )
    parser.add_argument("--fitness-batch-size", type=int, default=64)
    parser.add_argument("--correct-eval-examples", type=int, default=128)
    parser.add_argument("--heldout-examples", type=int, default=128)
    parser.add_argument(
        "--report-interval",
        type=int,
        default=1,
        help=(
            "run reporting-only held-out and control trajectories every N "
            "generations; sparse reporting requires fixed horizon mode"
        ),
    )
    parser.add_argument(
        "--task-variant",
        choices=("shortcut", "pointer_next_length"),
        default="shortcut",
        help="shortcut resistance or clean pointer-next length generalization",
    )
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--fitness-length", type=int)
    parser.add_argument("--heldout-length", type=int)
    parser.add_argument(
        "--leak-placement",
        choices=("suffix", "random_list"),
        default="suffix",
        help="place the leak at the suffix or after a random list value",
    )
    parser.add_argument("--forward-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--forward-training-precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help=(
            "autocast forward-model training to BF16 while retaining FP32 "
            "parameters, optimizer state, evaluation, and outer evolution"
        ),
    )
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--outer-learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--outer-update-rule",
        choices=("paper_standardized", "elite_centroid"),
        default="paper_standardized",
    )
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--elite-interpolation", type=float, default=0.5)
    parser.add_argument(
        "--elite-backtracking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="accept an elite proposal only when it improves centre fitness",
    )
    parser.add_argument(
        "--elite-rejection-sigma-decay",
        type=float,
        default=0.5,
        help="multiply sigma by this factor after rejecting a proposal",
    )
    parser.add_argument("--elite-min-sigma", type=float, default=1e-4)
    parser.add_argument("--elite-acceptance-patience", type=int, default=3)
    parser.add_argument(
        "--elite-acceptance-sigma-growth",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--elite-acceptance-trajectories",
        type=int,
        default=1,
        help=(
            "number of independently seeded shortcut-training trajectories "
            "used to accept or reject an elite centroid; the population-"
            "ranking trajectory is excluded"
        ),
    )
    parser.add_argument(
        "--candidate-ranking-trajectories",
        type=int,
        default=1,
        help=(
            "number of shared model/data trajectories used to rank every "
            "population candidate"
        ),
    )
    parser.add_argument(
        "--adaptive-elite-counts",
        help=(
            "comma-separated nested elite counts selected automatically on "
            "matched independent acceptance trajectories"
        ),
    )
    parser.add_argument(
        "--deduplicate-antithetic-elites",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "allow at most one sign from each antithetic direction in an "
            "elite centroid; use --no-deduplicate-antithetic-elites to "
            "reproduce the historical population-top-k controller"
        ),
    )
    parser.add_argument(
        "--adaptive-commit-scale",
        type=float,
        help=(
            "initial absolute elite-centroid step scale; when set, search "
            "half/current/double scales independently of candidate sigma"
        ),
    )
    parser.add_argument(
        "--adaptive-commit-scale-multiplier",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--horizon-promotion-mode",
        choices=(
            "fixed",
            "plateau",
            "rejection_probe",
            "performance_plateau",
        ),
        default="plateau",
    )
    parser.add_argument(
        "--horizon-rejection-patience",
        type=int,
        default=5,
        help="rejected adaptive updates before a matched longer-horizon probe",
    )
    parser.add_argument(
        "--horizon-probe-min-improvement",
        type=float,
        default=0.0,
        help="fitness gain required to accept a longer-horizon probe",
    )
    parser.add_argument(
        "--horizon-score-window",
        type=int,
        default=10,
        help="recent centre-score generations averaged at each horizon",
    )
    parser.add_argument(
        "--horizon-min-generations",
        type=int,
        default=20,
        help="minimum generations spent at a horizon before promotion",
    )
    parser.add_argument(
        "--horizon-max-generations",
        type=int,
        default=30,
        help="maximum generations spent at a horizon before promotion",
    )
    parser.add_argument(
        "--horizon-failed-extension-limit",
        type=int,
        default=2,
        help="non-improving horizon extensions allowed before stopping",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--backward-d-model", type=int, default=128)
    parser.add_argument(
        "--backward-rule-type",
        choices=("gradient_transformer", "attention_router"),
        default="gradient_transformer",
    )
    parser.add_argument(
        "--routing-credit-mode",
        choices=("suppress_renorm", "signed"),
        default="suppress_renorm",
        help="positive suppression or signed attention-edge credit",
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
        "--condition-on-forward-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "condition backward attention suppression on detached forward "
            "input embeddings"
        ),
    )
    parser.add_argument(
        "--fitness-objective",
        choices=(
            "mean_clean_ce",
            "worst_mode_ce",
            "worst_checkpoint_mode_ce",
        ),
        default="mean_clean_ce",
        help="candidate objective used by the EGGROLL update",
    )
    parser.add_argument(
        "--fitness-checkpoints",
        help=(
            "comma-separated forward-update checkpoints used by a "
            "checkpoint-aware fitness objective"
        ),
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
    parser.add_argument(
        "--vectorized-population",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="batch independent attention-router candidates with torch.func",
    )
    parser.add_argument(
        "--vectorized-chunk-size",
        type=int,
        default=16,
        help="number of candidate trajectories batched on each GPU",
    )
    parser.add_argument(
        "--successive-halving-rungs",
        help=(
            "opt-in horizon:survivors schedule, for example "
            "80:16,160:8,320:8"
        ),
    )
    parser.add_argument(
        "--direction-sampler",
        choices=("random", "function_diverse"),
        default="random",
    )
    parser.add_argument(
        "--direction-candidate-multiplier",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--direction-probe-examples",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--direction-signature-size",
        type=int,
        default=1024,
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default="list-sorting-learned-backward",
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument("--resume")
    parser.add_argument(
        "--resume-horizon",
        type=int,
        help="override the evolved horizon stored in a resume checkpoint",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ShortcutCreditExperimentConfig(**vars(args))
    output_dir = run(config)
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
