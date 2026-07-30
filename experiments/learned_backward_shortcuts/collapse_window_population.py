"""Screen EGGROLL backward-rule perturbations on saved collapse windows."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import torch

from list_sorting_transformer.shortcut_learning.shortcut_collapse_window import (
    CollapseWindow,
    load_collapse_window,
    replay_collapse_window,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit import (
    BackwardRule,
    EggrollDirection,
    ShortcutMetrics,
    ShortcutPointerVocabulary,
    apply_eggroll_direction,
    clone_center_parameters,
    make_fitness_batches,
    move_eggroll_direction,
    sample_eggroll_direction,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_backward_rule,
    load_checkpoint,
    parse_candidate_devices,
)


def worst_mode_loss(metrics: ShortcutMetrics) -> float:
    return max(metrics.mode_loss.values())


def worst_checkpoint(
    checkpoint_metrics: tuple[tuple[int, ShortcutMetrics], ...],
) -> tuple[int, ShortcutMetrics]:
    if not checkpoint_metrics:
        raise ValueError("collapse fitness requires checkpoint metrics")
    return max(
        checkpoint_metrics,
        key=lambda item: worst_mode_loss(item[1]),
    )


def minimum_mode_accuracy(
    checkpoint_metrics: tuple[tuple[int, ShortcutMetrics], ...],
) -> float:
    if not checkpoint_metrics:
        raise ValueError("collapse fitness requires checkpoint metrics")
    return min(
        metrics.mode_accuracy[mode]
        for _, metrics in checkpoint_metrics
        for mode in metrics.mode_accuracy
    )


def collapse_fitness(
    metric: str,
    *,
    center_worst_mode_loss: float,
    candidate_worst_mode_loss: float,
    center_minimum_mode_accuracy: float,
    candidate_minimum_mode_accuracy: float,
) -> float:
    if metric == "worst_mode_ce":
        return center_worst_mode_loss - candidate_worst_mode_loss
    if metric == "minimum_mode_accuracy":
        return (
            candidate_minimum_mode_accuracy
            - center_minimum_mode_accuracy
        )
    raise ValueError(f"unknown collapse fitness metric: {metric}")


def aggregate_window_fitness(
    fitnesses: list[float],
    *,
    aggregation: str,
) -> float:
    if not fitnesses:
        raise ValueError("at least one window fitness is required")
    if aggregation == "mean":
        return sum(fitnesses) / len(fitnesses)
    if aggregation == "minimum":
        return min(fitnesses)
    raise ValueError(
        f"unknown window fitness aggregation: {aggregation}"
    )


def parse_parameter_prefixes(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    prefixes = tuple(
        prefix.strip() for prefix in value.split(",") if prefix.strip()
    )
    if not prefixes or len(prefixes) != len(set(prefixes)):
        raise argparse.ArgumentTypeError(
            "parameter prefixes must be unique nonempty comma-separated names"
        )
    return prefixes


def restrict_direction(
    direction: EggrollDirection,
    *,
    parameter_prefixes: tuple[str, ...],
) -> EggrollDirection:
    if not parameter_prefixes:
        return direction
    matched = {
        name
        for name in direction.tensors
        if any(name.startswith(prefix) for prefix in parameter_prefixes)
    }
    if not matched:
        raise ValueError("parameter prefixes did not match any rule parameters")
    return EggrollDirection(
        {
            name: (
                tensor
                if name in matched
                else torch.zeros_like(tensor)
            )
            for name, tensor in direction.tensors.items()
        }
    )


def evaluate_rule_windows(
    *,
    config: ShortcutCreditExperimentConfig,
    rule: BackwardRule,
    windows: tuple[CollapseWindow, ...],
    device: torch.device,
) -> list[dict[str, object]]:
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
    rows = []
    for window in windows:
        checkpoint_steps = tuple(range(1, len(window.batches) + 1))
        replay = replay_collapse_window(
            config,
            window=window,
            backward_rule=rule,
            fitness_batches=fitness_batches,
            device=device,
            checkpoint_steps=checkpoint_steps,
            evaluation_batch_size=config.fitness_examples,
        )
        worst_step, worst_metrics = worst_checkpoint(
            replay.checkpoint_metrics
        )
        rows.append(
            {
                "generation_seed": window.generation_seed,
                "start_step": window.start_step,
                "end_step": window.end_step,
                "worst_relative_step": worst_step,
                "worst_mode_loss": worst_mode_loss(worst_metrics),
                "worst_min_mode_accuracy": min(
                    worst_metrics.mode_accuracy.values()
                ),
                "minimum_mode_accuracy": min(
                    metrics.mode_accuracy[mode]
                    for _, metrics in replay.checkpoint_metrics
                    for mode in metrics.mode_accuracy
                ),
                "end_clean_loss": replay.end_metrics.loss,
                "end_min_mode_accuracy": min(
                    replay.end_metrics.mode_accuracy.values()
                ),
            }
        )
    return rows


def evaluate_candidate_shard(
    *,
    config: ShortcutCreditExperimentConfig,
    center_rule_config: object,
    center_parameters: dict[str, torch.Tensor],
    directions: tuple[EggrollDirection, ...],
    candidate_indices: tuple[int, ...],
    sigma: float,
    windows: tuple[CollapseWindow, ...],
    device: torch.device,
    fitness_metric: str,
    window_aggregation: str,
) -> list[dict[str, object]]:
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
    worker_center = {
        name: tensor.to(device)
        for name, tensor in center_parameters.items()
    }
    direction_indices = {index // 2 for index in candidate_indices}
    worker_directions = {
        index: move_eggroll_direction(directions[index], device)
        for index in direction_indices
    }
    checkpoint_steps_by_window = tuple(
        tuple(range(1, len(window.batches) + 1))
        for window in windows
    )
    center_rule = initialize_backward_rule(
        center_rule_config,  # type: ignore[arg-type]
        device=device,
    )
    center_rule.load_state_dict(worker_center)
    center_results_by_window = []
    for window, checkpoint_steps in zip(
        windows,
        checkpoint_steps_by_window,
    ):
        center_replay = replay_collapse_window(
            config,
            window=window,
            backward_rule=center_rule,
            fitness_batches=fitness_batches,
            device=device,
            checkpoint_steps=checkpoint_steps,
            evaluation_batch_size=config.fitness_examples,
        )
        center_results_by_window.append(
            (
                worst_checkpoint(center_replay.checkpoint_metrics),
                minimum_mode_accuracy(
                    center_replay.checkpoint_metrics
                ),
            )
        )
    results = []
    for candidate_index in candidate_indices:
        direction_index = candidate_index // 2
        sign = 1 if candidate_index % 2 == 0 else -1
        rule = initialize_backward_rule(
            center_rule_config,  # type: ignore[arg-type]
            device=device,
        )
        apply_eggroll_direction(
            rule,
            worker_center,
            worker_directions[direction_index],
            sigma=sigma,
            sign=sign,
        )
        window_results = []
        for window, checkpoint_steps, center_result in zip(
            windows,
            checkpoint_steps_by_window,
            center_results_by_window,
        ):
            replay = replay_collapse_window(
                config,
                window=window,
                backward_rule=rule,
                fitness_batches=fitness_batches,
                device=device,
                checkpoint_steps=checkpoint_steps,
                evaluation_batch_size=config.fitness_examples,
            )
            candidate_worst = worst_checkpoint(
                replay.checkpoint_metrics
            )
            center_worst, center_minimum_accuracy = center_result
            center_worst_step, center_worst_metrics = center_worst
            candidate_worst_step, candidate_worst_metrics = candidate_worst
            center_worst_loss = worst_mode_loss(center_worst_metrics)
            candidate_worst_loss = worst_mode_loss(
                candidate_worst_metrics
            )
            candidate_minimum_accuracy = minimum_mode_accuracy(
                replay.checkpoint_metrics
            )
            window_results.append(
                {
                    "generation_seed": window.generation_seed,
                    "start_step": window.start_step,
                    "end_step": window.end_step,
                    "fitness": collapse_fitness(
                        fitness_metric,
                        center_worst_mode_loss=center_worst_loss,
                        candidate_worst_mode_loss=candidate_worst_loss,
                        center_minimum_mode_accuracy=(
                            center_minimum_accuracy
                        ),
                        candidate_minimum_mode_accuracy=(
                            candidate_minimum_accuracy
                        ),
                    ),
                    "clean_loss": replay.end_metrics.loss,
                    "end_min_mode_accuracy": min(
                        replay.end_metrics.mode_accuracy.values()
                    ),
                    "center_worst_relative_step": center_worst_step,
                    "center_worst_mode_loss": center_worst_loss,
                    "center_worst_min_mode_accuracy": min(
                        center_worst_metrics.mode_accuracy.values()
                    ),
                    "center_minimum_mode_accuracy": (
                        center_minimum_accuracy
                    ),
                    "candidate_worst_relative_step": (
                        candidate_worst_step
                    ),
                    "candidate_worst_mode_loss": candidate_worst_loss,
                    "candidate_worst_min_mode_accuracy": min(
                        candidate_worst_metrics.mode_accuracy.values()
                    ),
                    "min_mode_accuracy": candidate_minimum_accuracy,
                }
            )
        window_fitnesses = [
            float(result["fitness"]) for result in window_results
        ]
        results.append(
            {
                "candidate_index": candidate_index,
                "direction_index": direction_index,
                "sign": sign,
                "fitness": aggregate_window_fitness(
                    window_fitnesses,
                    aggregation=window_aggregation,
                ),
                "mean_window_fitness": (
                    sum(window_fitnesses) / len(window_fitnesses)
                ),
                "windows": window_results,
            }
        )
    return results


def save_proposal_checkpoint(
    path: Path,
    *,
    source_checkpoint: dict[str, object],
    proposal_state: dict[str, torch.Tensor],
    metadata: dict[str, object],
) -> None:
    checkpoint = dict(source_checkpoint)
    checkpoint["backward_rule_state"] = {
        name: tensor.detach().cpu()
        for name, tensor in proposal_state.items()
    }
    checkpoint["collapse_window_update"] = metadata
    torch.save(checkpoint, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--windows",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--acceptance-windows",
        type=Path,
        nargs="*",
        default=(),
        help="seed-disjoint windows that only decide proposal acceptance",
    )
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument(
        "--fitness-metric",
        choices=("worst_mode_ce", "minimum_mode_accuracy"),
        default="worst_mode_ce",
    )
    parser.add_argument(
        "--window-aggregation",
        choices=("mean", "minimum"),
        default="mean",
        help="combine each candidate's per-window fitness values",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--perturb-parameter-prefixes",
        type=parse_parameter_prefixes,
        help=(
            "comma-separated rule parameter prefixes to perturb; all other "
            "direction tensors are zeroed"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-devices")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.population_size < 2 or args.population_size % 2:
        raise ValueError("population size must be positive and even")
    if args.sigma <= 0:
        raise ValueError("sigma must be positive")

    started_at = time.monotonic()
    primary_device = torch.device(args.device)
    candidate_devices = parse_candidate_devices(
        args.candidate_devices,
        primary_device,
    )
    raw_checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    saved_config = ShortcutCreditExperimentConfig(
        **raw_checkpoint["config"]
    )
    windows = tuple(load_collapse_window(path) for path in args.windows)
    acceptance_windows = tuple(
        load_collapse_window(path) for path in args.acceptance_windows
    )
    config = replace(
        saved_config,
        horizon=max(len(window.batches) for window in windows),
        max_horizon=max(
            saved_config.max_horizon,
            *(len(window.batches) for window in windows),
        ),
        device=args.device,
        resume=None,
        resume_horizon=None,
        wandb=False,
    )
    center_rule, _, _, _ = load_checkpoint(
        args.checkpoint,
        device=primary_device,
    )
    center_parameters = {
        name: tensor.cpu()
        for name, tensor in clone_center_parameters(center_rule).items()
    }
    direction_generator = torch.Generator().manual_seed(args.seed)
    directions = tuple(
        move_eggroll_direction(
            restrict_direction(
                sample_eggroll_direction(
                    center_rule,
                    generator=direction_generator,
                ),
                parameter_prefixes=args.perturb_parameter_prefixes,
            ),
            "cpu",
        )
        for _ in range(args.population_size // 2)
    )

    candidate_indices = tuple(range(args.population_size))
    shards = tuple(
        candidate_indices[index :: len(candidate_devices)]
        for index in range(len(candidate_devices))
    )
    if len(candidate_devices) == 1:
        candidate_rows = evaluate_candidate_shard(
            config=config,
            center_rule_config=center_rule.config,
            center_parameters=center_parameters,
            directions=directions,
            candidate_indices=shards[0],
            sigma=args.sigma,
            windows=windows,
            device=candidate_devices[0],
            fitness_metric=args.fitness_metric,
            window_aggregation=args.window_aggregation,
        )
    else:
        with ThreadPoolExecutor(max_workers=len(candidate_devices)) as pool:
            futures = [
                pool.submit(
                    evaluate_candidate_shard,
                    config=config,
                    center_rule_config=center_rule.config,
                    center_parameters=center_parameters,
                    directions=directions,
                    candidate_indices=shard,
                    sigma=args.sigma,
                    windows=windows,
                    device=worker_device,
                    fitness_metric=args.fitness_metric,
                    window_aggregation=args.window_aggregation,
                )
                for shard, worker_device in zip(
                    shards,
                    candidate_devices,
                )
            ]
            candidate_rows = [
                row
                for future in futures
                for row in future.result()
            ]
    candidate_rows.sort(key=lambda row: int(row["candidate_index"]))
    fitnesses = torch.tensor(
        [float(row["fitness"]) for row in candidate_rows]
    )
    best_index = int(fitnesses.argmax())
    best_row = candidate_rows[best_index]
    best_direction_index = best_index // 2
    best_sign = 1 if best_index % 2 == 0 else -1

    proposal_rule = initialize_backward_rule(
        center_rule.config,
        device=primary_device,
    )
    apply_eggroll_direction(
        proposal_rule,
        {
            name: tensor.to(primary_device)
            for name, tensor in center_parameters.items()
        },
        move_eggroll_direction(
            directions[best_direction_index],
            primary_device,
        ),
        sigma=args.sigma,
        sign=best_sign,
    )
    acceptance_rows = []
    accepted = None
    if acceptance_windows:
        center_acceptance = evaluate_rule_windows(
            config=config,
            rule=center_rule,
            windows=acceptance_windows,
            device=primary_device,
        )
        proposal_acceptance = evaluate_rule_windows(
            config=config,
            rule=proposal_rule,
            windows=acceptance_windows,
            device=primary_device,
        )
        for center_result, proposal_result in zip(
            center_acceptance,
            proposal_acceptance,
        ):
            center_worst_loss = float(
                center_result["worst_mode_loss"]
            )
            proposal_worst_loss = float(
                proposal_result["worst_mode_loss"]
            )
            center_minimum_accuracy = float(
                center_result["minimum_mode_accuracy"]
            )
            proposal_minimum_accuracy = float(
                proposal_result["minimum_mode_accuracy"]
            )
            acceptance_rows.append(
                {
                    "generation_seed": center_result["generation_seed"],
                    "start_step": center_result["start_step"],
                    "end_step": center_result["end_step"],
                    "center_worst_mode_loss": center_result[
                        "worst_mode_loss"
                    ],
                    "proposal_worst_mode_loss": proposal_result[
                        "worst_mode_loss"
                    ],
                    "fitness": collapse_fitness(
                        args.fitness_metric,
                        center_worst_mode_loss=center_worst_loss,
                        candidate_worst_mode_loss=proposal_worst_loss,
                        center_minimum_mode_accuracy=(
                            center_minimum_accuracy
                        ),
                        candidate_minimum_mode_accuracy=(
                            proposal_minimum_accuracy
                        ),
                    ),
                    "center_minimum_mode_accuracy": center_result[
                        "minimum_mode_accuracy"
                    ],
                    "proposal_minimum_mode_accuracy": proposal_result[
                        "minimum_mode_accuracy"
                    ],
                }
            )
        accepted = (
            aggregate_window_fitness(
                [
                    float(row["fitness"])
                    for row in acceptance_rows
                ],
                aggregation=args.window_aggregation,
            )
            > 0
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidates.jsonl"
    candidate_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in candidate_rows
        )
    )
    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "windows": [str(path) for path in args.windows],
        "acceptance_windows": [
            str(path) for path in args.acceptance_windows
        ],
        "population_size": args.population_size,
        "sigma": args.sigma,
        "fitness_metric": args.fitness_metric,
        "window_aggregation": args.window_aggregation,
        "perturb_parameter_prefixes": list(
            args.perturb_parameter_prefixes
        ),
        "seed": args.seed,
        "selected_candidate_index": best_index,
        "selected_fitness": float(fitnesses[best_index]),
        "accepted": accepted,
    }
    proposal_path = args.output_dir / "proposal_checkpoint.pt"
    save_proposal_checkpoint(
        proposal_path,
        source_checkpoint=raw_checkpoint,
        proposal_state=proposal_rule.state_dict(),
        metadata=metadata,
    )
    next_rule = proposal_rule if accepted is not False else center_rule
    next_path = args.output_dir / "next_checkpoint.pt"
    save_proposal_checkpoint(
        next_path,
        source_checkpoint=raw_checkpoint,
        proposal_state=next_rule.state_dict(),
        metadata=metadata,
    )
    summary = {
        **metadata,
        "candidate_devices": [str(device) for device in candidate_devices],
        "fitness_mean": float(fitnesses.mean()),
        "fitness_std": float(fitnesses.std(unbiased=False)),
        "fitness_max": float(fitnesses.max()),
        "fitness_min": float(fitnesses.min()),
        "best": best_row,
        "acceptance": acceptance_rows,
        "elapsed_seconds": time.monotonic() - started_at,
        "proposal_checkpoint": str(proposal_path),
        "next_checkpoint": str(next_path),
        "config": asdict(config),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
