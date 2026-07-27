"""Train a token gradient-reversal selector with group-relative policy updates."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from list_sorting_transformer.shortcut_credit import (
    ShortcutBatch,
    ShortcutMetrics,
    ShortcutPointerVocabulary,
    evaluate_shortcut_batches,
    make_fitness_batches,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_forward_model,
    make_inner_batches,
    make_mode_batches,
)
from list_sorting_transformer.token_gradient_reversal import (
    forward_with_source_gradient_reversal,
)
from list_sorting_transformer.token_gradient_selector import (
    SelectorTrajectory,
    TokenGradientSelector,
    TokenGradientSelectorConfig,
    grouped_trajectory_policy_terms,
    sample_selector_trajectories,
    selector_probability_statistics,
    standardize_group_rewards,
)
from list_sorting_transformer.vectorized_reversal_population import (
    train_vectorized_candidate_chunks,
)


@dataclass(frozen=True)
class CandidateResult:
    candidate_index: int
    reward: float
    clean: ShortcutMetrics
    heldout_clean: ShortcutMetrics
    correct: ShortcutMetrics


@dataclass
class PlateauState:
    ema_reward: float | None = None
    best_ema_reward: float = float("-inf")
    stale_generations: int = 0


def parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices:
        raise argparse.ArgumentTypeError("at least one device is required")
    return devices


def metrics_summary(
    prefix: str,
    metrics: ShortcutMetrics,
) -> dict[str, float]:
    return {
        f"{prefix}/loss": metrics.loss,
        f"{prefix}/accuracy": metrics.accuracy,
        f"{prefix}/masked_accuracy": metrics.mode_accuracy["masked"],
        f"{prefix}/incorrect_accuracy": metrics.mode_accuracy["incorrect"],
        f"{prefix}/unique_value_predictions": float(
            metrics.unique_value_prediction_count
        ),
        f"{prefix}/prediction_mode_fraction": (
            metrics.prediction_mode_fraction
        ),
    }


def pearson_correlation(left: Tensor, right: Tensor) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("correlation inputs must be matching vectors")
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator) < 1e-12:
        return 0.0
    return float((left * right).sum() / denominator)


def train_candidate(
    *,
    candidate_index: int,
    config: ShortcutCreditExperimentConfig,
    base_state: dict[str, Tensor],
    inner_batches: tuple[ShortcutBatch, ...],
    actions: tuple[Tensor, ...],
    fitness_batches: tuple[ShortcutBatch, ...],
    heldout_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    initial_clean_loss: float,
    reversal_scale: float,
    vocabulary: ShortcutPointerVocabulary,
    device: torch.device,
) -> CandidateResult:
    model = initialize_forward_model(
        config,
        vocabulary,
        initialization_seed=None,
        device=device,
    )
    model.load_state_dict(base_state)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.forward_learning_rate,
    )
    model.train()
    for batch, selection in zip(inner_batches, actions):
        device_batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_with_source_gradient_reversal(
            model,
            device_batch.input_ids,
            selection.to(device),
            reversal_scale=reversal_scale,
            reversal_scope="attention_scores",
        )
        loss = F.cross_entropy(
            logits[:, -1],
            device_batch.targets,
        )
        loss.backward()
        optimizer.step()

    clean = evaluate_shortcut_batches(model, fitness_batches)
    heldout_clean = evaluate_shortcut_batches(model, heldout_batches)
    correct = evaluate_shortcut_batches(model, correct_batches)
    return CandidateResult(
        candidate_index=candidate_index,
        reward=initial_clean_loss - clean.loss,
        clean=clean,
        heldout_clean=heldout_clean,
        correct=correct,
    )


def train_candidate_shard(
    *,
    candidate_indices: tuple[int, ...],
    device_name: str,
    config: ShortcutCreditExperimentConfig,
    base_state: dict[str, Tensor],
    inner_batches: tuple[ShortcutBatch, ...],
    trajectories: tuple[SelectorTrajectory, ...],
    fitness_batches_cpu: tuple[ShortcutBatch, ...],
    heldout_batches_cpu: tuple[ShortcutBatch, ...],
    correct_batches_cpu: tuple[ShortcutBatch, ...],
    initial_clean_loss: float,
    reversal_scale: float,
    vocabulary: ShortcutPointerVocabulary,
) -> list[CandidateResult]:
    device = torch.device(device_name)
    inner_batches = tuple(batch.to(device) for batch in inner_batches)
    fitness_batches = tuple(
        batch.to(device) for batch in fitness_batches_cpu
    )
    heldout_batches = tuple(
        batch.to(device) for batch in heldout_batches_cpu
    )
    correct_batches = tuple(
        batch.to(device) for batch in correct_batches_cpu
    )
    results = []
    for candidate_index in candidate_indices:
        results.append(
            train_candidate(
                candidate_index=candidate_index,
                config=config,
                base_state=base_state,
                inner_batches=inner_batches,
                actions=trajectories[candidate_index].actions,
                fitness_batches=fitness_batches,
                heldout_batches=heldout_batches,
                correct_batches=correct_batches,
                initial_clean_loss=initial_clean_loss,
                reversal_scale=reversal_scale,
                vocabulary=vocabulary,
                device=device,
            )
        )
    return results


def evaluate_population(
    *,
    devices: tuple[str, ...],
    config: ShortcutCreditExperimentConfig,
    base_state: dict[str, Tensor],
    inner_batches: tuple[ShortcutBatch, ...],
    trajectories: tuple[SelectorTrajectory, ...],
    fitness_batches: tuple[ShortcutBatch, ...],
    heldout_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    initial_clean_loss: float,
    reversal_scale: float,
    vocabulary: ShortcutPointerVocabulary,
) -> tuple[CandidateResult, ...]:
    shards = tuple(
        tuple(
            range(
                device_index,
                len(trajectories),
                len(devices),
            )
        )
        for device_index in range(len(devices))
    )
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(
                train_candidate_shard,
                candidate_indices=indices,
                device_name=device,
                config=config,
                base_state=base_state,
                inner_batches=inner_batches,
                trajectories=trajectories,
                fitness_batches_cpu=fitness_batches,
                heldout_batches_cpu=heldout_batches,
                correct_batches_cpu=correct_batches,
                initial_clean_loss=initial_clean_loss,
                reversal_scale=reversal_scale,
                vocabulary=vocabulary,
            )
            for device, indices in zip(devices, shards)
            if indices
        ]
        results = [
            result
            for future in futures
            for result in future.result()
        ]
    return tuple(sorted(results, key=lambda result: result.candidate_index))


def evaluate_population_vectorized(
    *,
    devices: tuple[str, ...],
    config: ShortcutCreditExperimentConfig,
    base_state: dict[str, Tensor],
    inner_batches: tuple[ShortcutBatch, ...],
    trajectories: tuple[SelectorTrajectory, ...],
    fitness_batches: tuple[ShortcutBatch, ...],
    heldout_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    initial_clean_loss: float,
    reversal_scale: float,
    vocabulary: ShortcutPointerVocabulary,
    chunk_size: int,
) -> tuple[CandidateResult, ...]:
    shards = tuple(
        tuple(range(device_index, len(trajectories), len(devices)))
        for device_index in range(len(devices))
    )
    actions = tuple(
        trajectory.actions for trajectory in trajectories
    )
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(
                train_vectorized_candidate_chunks,
                candidate_indices=indices,
                chunk_size=chunk_size,
                device_name=device,
                config=config,
                base_state=base_state,
                inner_batches=inner_batches,
                actions=actions,
                fitness_batches=fitness_batches,
                heldout_batches=heldout_batches,
                correct_batches=correct_batches,
                initial_clean_loss=initial_clean_loss,
                reversal_scale=reversal_scale,
                vocabulary=vocabulary,
            )
            for device, indices in zip(devices, shards)
            if indices
        ]
        results = [
            CandidateResult(
                candidate_index=result.candidate_index,
                reward=result.reward,
                clean=result.clean,
                heldout_clean=result.heldout_clean,
                correct=result.correct,
            )
            for future in futures
            for result in future.result()
        ]
    return tuple(sorted(results, key=lambda result: result.candidate_index))


def update_plateau(
    state: PlateauState,
    reward: float,
    *,
    decay: float,
    patience: int,
    minimum_delta: float,
) -> bool:
    state.ema_reward = (
        reward
        if state.ema_reward is None
        else decay * state.ema_reward + (1 - decay) * reward
    )
    if state.ema_reward > state.best_ema_reward + minimum_delta:
        state.best_ema_reward = state.ema_reward
        state.stale_generations = 0
    else:
        state.stale_generations += 1
    return state.stale_generations >= patience


def maybe_initialize_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str,
    config: dict[str, Any],
) -> Any | None:
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B is not installed") from error
    return wandb.init(project=project, name=run_name, config=config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-horizon", type=int, default=3_000)
    parser.add_argument("--horizon-multiplier", type=int, default=2)
    parser.add_argument("--plateau-patience", type=int, default=25)
    parser.add_argument("--plateau-min-delta", type=float, default=0.005)
    parser.add_argument("--plateau-ema-decay", type=float, default=0.9)
    parser.add_argument("--minimum-reward-std", type=float, default=1e-3)
    parser.add_argument("--tied-group-patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fitness-examples", type=int, default=512)
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--forward-learning-rate", type=float, default=1e-4)
    parser.add_argument("--reversal-scale", type=float, default=4.0)
    parser.add_argument("--selector-learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--selector-d-model", type=int, default=64)
    parser.add_argument("--selector-heads", type=int, default=4)
    parser.add_argument(
        "--initial-reverse-probability",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--leak-placement",
        choices=("suffix", "random_list"),
        default="random_list",
    )
    parser.add_argument(
        "--candidate-devices",
        type=parse_devices,
        default=parse_devices("cuda:0,cuda:1,cuda:2"),
    )
    parser.add_argument("--vectorized-population", action="store_true")
    parser.add_argument("--vectorized-chunk-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-horizon", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default="list-sorting-token-gradient-selector",
    )
    parser.add_argument("--run-name", default="token-selector-grpo")
    args = parser.parse_args()
    positive_integers = (
        args.generations,
        args.group_size,
        args.horizon,
        args.max_horizon,
        args.horizon_multiplier,
        args.plateau_patience,
        args.tied_group_patience,
        args.batch_size,
        args.fitness_examples,
        args.selector_d_model,
        args.selector_heads,
        args.checkpoint_interval,
        args.vectorized_chunk_size,
    )
    if any(value < 1 for value in positive_integers):
        raise ValueError("integer settings must be positive")
    if args.group_size < 2:
        raise ValueError("group-size must be at least two")
    if args.horizon > args.max_horizon:
        raise ValueError("horizon must not exceed max-horizon")
    if (
        args.resume_horizon is not None
        and (
            args.resume is None
            or not 1 <= args.resume_horizon <= args.max_horizon
        )
    ):
        raise ValueError(
            "resume-horizon requires resume and must not exceed max-horizon"
        )
    if not 0 <= args.plateau_ema_decay < 1:
        raise ValueError("plateau-ema-decay must be in [0, 1)")
    if min(
        args.forward_learning_rate,
        args.reversal_scale,
        args.selector_learning_rate,
        args.gradient_clip_norm,
    ) <= 0:
        raise ValueError("learning rates and gradient clip must be positive")
    if args.entropy_coefficient < 0:
        raise ValueError("entropy-coefficient must be nonnegative")
    if args.minimum_reward_std < 0:
        raise ValueError("minimum-reward-std must be nonnegative")
    if not 0 < args.initial_reverse_probability < 1:
        raise ValueError(
            "initial-reverse-probability must be in (0, 1)"
        )

    torch.set_num_threads(1)
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    forward_config = ShortcutCreditExperimentConfig(
        generations=1,
        population_size=2,
        horizon=args.horizon,
        max_horizon=args.max_horizon,
        batch_size=args.batch_size,
        fitness_examples=args.fitness_examples,
        min_length=args.min_length,
        max_length=args.max_length,
        forward_learning_rate=args.forward_learning_rate,
        leak_placement=args.leak_placement,
        seed=args.seed,
        device=args.candidate_devices[0],
    )
    selector_config = TokenGradientSelectorConfig(
        vocab_size=vocabulary.size,
        d_model=args.selector_d_model,
        n_layers=2,
        n_heads=args.selector_heads,
        dropout=0.0,
        initial_reverse_probability=args.initial_reverse_probability,
    )
    torch.manual_seed(args.seed + 20_000)
    selector_device = torch.device(args.candidate_devices[0])
    selector = TokenGradientSelector(selector_config).to(selector_device)
    selector_optimizer = torch.optim.Adam(
        selector.parameters(),
        lr=args.selector_learning_rate,
    )

    if args.resume is None:
        fixed_generator = torch.Generator().manual_seed(args.seed + 10_000)
        fitness_batches = make_fitness_batches(
            args.fitness_examples,
            min_length=args.min_length,
            max_length=args.max_length,
            batch_size=forward_config.fitness_batch_size,
            generator=fixed_generator,
            vocabulary=vocabulary,
            leak_placement=args.leak_placement,
            device="cpu",
        )
        correct_batches = make_mode_batches(
            forward_config.correct_eval_examples,
            leak_mode="correct",
            config=forward_config,
            vocabulary=vocabulary,
            generator=fixed_generator,
            device=torch.device("cpu"),
        )
        heldout_generator = torch.Generator().manual_seed(
            args.seed + 30_000
        )
        heldout_batches = make_fitness_batches(
            args.fitness_examples,
            min_length=args.min_length,
            max_length=args.max_length,
            batch_size=forward_config.fitness_batch_size,
            generator=heldout_generator,
            vocabulary=vocabulary,
            leak_placement=args.leak_placement,
            device="cpu",
        )
        start_generation = 0
        horizon = args.horizon
        plateau = PlateauState()
        consecutive_tied_groups = 0
    else:
        checkpoint = torch.load(args.resume, map_location="cpu")
        checkpoint_selector_config = checkpoint["selector_config"]
        current_selector_config = selector_config.as_dict()
        shared_current_config = {
            key: current_selector_config[key]
            for key in checkpoint_selector_config
        }
        if checkpoint_selector_config != shared_current_config:
            raise ValueError(
                "resume selector configuration does not match"
            )
        selector.load_state_dict(checkpoint["selector"])
        selector_optimizer.load_state_dict(
            checkpoint["selector_optimizer"]
        )
        fitness_batches = checkpoint["fitness_batches"]
        heldout_batches = checkpoint["heldout_batches"]
        correct_batches = checkpoint["correct_batches"]
        start_generation = int(checkpoint["generation"]) + 1
        horizon = (
            args.resume_horizon
            if args.resume_horizon is not None
            else int(checkpoint["horizon"])
        )
        plateau = PlateauState(**checkpoint["plateau"])
        consecutive_tied_groups = int(
            checkpoint["consecutive_tied_groups"]
        )

    run_config = {
        **vars(args),
        "output_dir": str(args.output_dir),
        "forward_config": asdict(forward_config),
        "selector_config": selector_config.as_dict(),
        "fitness_set_seed": args.seed + 10_000,
        "heldout_set_seed": args.seed + 30_000,
    }
    wandb_run = maybe_initialize_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.run_name,
        config=run_config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    if args.resume is None:
        metrics_path.write_text("")
        (args.output_dir / "config.json").write_text(
            json.dumps(
                run_config,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
    elif not metrics_path.exists():
        metrics_path.write_text("")
        (args.output_dir / "config.json").write_text(
            json.dumps(
                run_config,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )

    for generation in range(start_generation, args.generations):
        generation_seed = args.seed * 1_000_003 + generation * 10_007
        base_model = initialize_forward_model(
            forward_config,
            vocabulary,
            initialization_seed=generation_seed + 1,
            device=torch.device("cpu"),
        )
        base_state = {
            name: tensor.detach().clone()
            for name, tensor in base_model.state_dict().items()
        }
        initial_clean = evaluate_shortcut_batches(
            base_model,
            fitness_batches,
        )
        del base_model
        inner_batches = make_inner_batches(
            forward_config,
            horizon=horizon,
            vocabulary=vocabulary,
            generator=torch.Generator().manual_seed(generation_seed + 2),
            device=torch.device("cpu"),
        )
        trajectories = sample_selector_trajectories(
            selector,
            inner_batches,
            group_size=args.group_size,
            vocabulary=vocabulary,
            generators=tuple(
                torch.Generator(device=selector_device).manual_seed(
                    generation_seed + 100 + candidate_index
                )
                for candidate_index in range(args.group_size)
            ),
        )
        population_started_at = time.perf_counter()
        population_arguments = {
            "devices": args.candidate_devices,
            "config": forward_config,
            "base_state": base_state,
            "inner_batches": inner_batches,
            "trajectories": trajectories,
            "fitness_batches": fitness_batches,
            "heldout_batches": heldout_batches,
            "correct_batches": correct_batches,
            "initial_clean_loss": initial_clean.loss,
            "reversal_scale": args.reversal_scale,
            "vocabulary": vocabulary,
        }
        if args.vectorized_population:
            results = evaluate_population_vectorized(
                **population_arguments,
                chunk_size=args.vectorized_chunk_size,
            )
        else:
            results = evaluate_population(**population_arguments)
        population_seconds = time.perf_counter() - population_started_at
        rewards = torch.tensor(
            [result.reward for result in results],
            dtype=torch.float32,
        )
        reward_standard_deviation = float(
            rewards.std(unbiased=False)
        )
        tied_group = reward_standard_deviation < args.minimum_reward_std
        advantages = standardize_group_rewards(
            rewards,
            minimum_standard_deviation=args.minimum_reward_std,
        )

        selector.train()
        selector_optimizer.zero_grad(set_to_none=True)
        log_probabilities, member_entropy = (
            grouped_trajectory_policy_terms(
                selector,
                inner_batches,
                trajectories,
            )
        )
        policy_objective = (
            -(advantages.to(log_probabilities.device) * log_probabilities)
            .mean()
            - args.entropy_coefficient * member_entropy
        )
        policy_loss = float(policy_objective.detach())
        entropy = float(member_entropy.detach())
        if tied_group:
            gradient_norm = torch.zeros(())
            consecutive_tied_groups += 1
        else:
            policy_objective.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                selector.parameters(),
                args.gradient_clip_norm,
            )
            selector_optimizer.step()
            consecutive_tied_groups = 0

        best = max(results, key=lambda result: result.reward)
        oracle_selection_rates = torch.tensor(
            [
                trajectory.oracle_selected_fraction
                for trajectory in trajectories
            ]
        )
        other_selection_rates = torch.tensor(
            [
                trajectory.other_selected_fraction
                for trajectory in trajectories
            ]
        )
        selection_rates = torch.tensor(
            [
                trajectory.selected_fraction
                for trajectory in trajectories
            ]
        )
        probabilities = selector_probability_statistics(
            selector,
            inner_batches,
            vocabulary=vocabulary,
        )
        row: dict[str, float | int | str] = {
            "generation": generation,
            "horizon": horizon,
            "reward/mean": float(rewards.mean()),
            "reward/std": reward_standard_deviation,
            "reward/min": float(rewards.min()),
            "reward/max": float(rewards.max()),
            "policy/loss": policy_loss,
            "policy/entropy": entropy,
            "policy/gradient_norm": float(gradient_norm),
            "policy/update_applied": int(not tied_group),
            "reward/tied_group": int(tied_group),
            "runtime/population_seconds": population_seconds,
            "runtime/candidate_steps_per_second": (
                args.group_size * horizon / population_seconds
            ),
            "reward/oracle_selection_correlation": pearson_correlation(
                rewards,
                oracle_selection_rates,
            ),
            "reward/other_selection_correlation": pearson_correlation(
                rewards,
                other_selection_rates,
            ),
            "reward/selection_fraction_correlation": pearson_correlation(
                rewards,
                selection_rates,
            ),
            "sample/selected_fraction": sum(
                trajectory.selected_fraction for trajectory in trajectories
            )
            / args.group_size,
            "sample/oracle_selected_fraction": sum(
                trajectory.oracle_selected_fraction
                for trajectory in trajectories
            )
            / args.group_size,
            "sample/other_selected_fraction": sum(
                trajectory.other_selected_fraction
                for trajectory in trajectories
            )
            / args.group_size,
            **probabilities,
            **metrics_summary("best/fitness", best.clean),
            **metrics_summary("best/heldout", best.heldout_clean),
            "best/correct_accuracy": best.correct.accuracy,
        }
        promoted = (
            consecutive_tied_groups >= args.tied_group_patience
            or update_plateau(
                plateau,
                float(rewards.mean()),
                decay=args.plateau_ema_decay,
                patience=args.plateau_patience,
                minimum_delta=args.plateau_min_delta,
            )
        )
        if promoted and horizon < args.max_horizon:
            horizon = min(
                args.max_horizon,
                horizon * args.horizon_multiplier,
            )
            plateau = PlateauState()
            consecutive_tied_groups = 0
            row["next_horizon"] = horizon

        with metrics_path.open("a") as metrics_file:
            metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if wandb_run is not None:
            wandb_run.log(row, step=generation)

        if (
            (generation + 1) % args.checkpoint_interval == 0
            or generation + 1 == args.generations
        ):
            torch.save(
                {
                    "generation": generation,
                    "horizon": horizon,
                    "plateau": asdict(plateau),
                    "consecutive_tied_groups": consecutive_tied_groups,
                    "selector_config": selector_config.as_dict(),
                    "selector": selector.state_dict(),
                    "selector_optimizer": selector_optimizer.state_dict(),
                    "run_config": run_config,
                    "fitness_batches": fitness_batches,
                    "heldout_batches": heldout_batches,
                    "correct_batches": correct_batches,
                },
                args.output_dir / "latest.pt",
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
