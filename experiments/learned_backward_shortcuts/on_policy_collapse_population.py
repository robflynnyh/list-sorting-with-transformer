"""Evaluate backward-rule perturbations from forward-model initialization."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import NamedTuple

import torch

from experiments.learned_backward_shortcuts.collapse_window_population import (
    aggregate_window_fitness,
    parse_parameter_prefixes,
    restrict_direction,
)
from list_sorting_transformer.shortcut_learning.shortcut_collapse_window import (
    optimizer_for_model,
    train_forward_step,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit import (
    BackwardRule,
    EggrollDirection,
    ShortcutBatch,
    ShortcutPointerVocabulary,
    apply_eggroll_direction,
    clone_center_parameters,
    evaluate_shortcut_batches,
    make_fitness_batches,
    move_eggroll_direction,
    sample_eggroll_direction,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_backward_rule,
    initialize_forward_model,
    load_checkpoint,
    make_inner_batches,
    parse_candidate_devices,
)


class CollapseEvent(NamedTuple):
    generation_seed: int
    step: int


def parse_events(value: str) -> tuple[CollapseEvent, ...]:
    try:
        events = tuple(
            CollapseEvent(*(int(part) for part in item.split(":")))
            for item in value.split(",")
        )
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "events must be comma-separated generation_seed:step pairs"
        ) from error
    if (
        not events
        or any(seed < 0 or step < 1 for seed, step in events)
        or len({event.generation_seed for event in events}) != len(events)
    ):
        raise argparse.ArgumentTypeError(
            "events require unique nonnegative seeds and positive steps"
        )
    return events


def dense_evaluation_steps(
    event_step: int,
    *,
    radius: int,
) -> tuple[int, ...]:
    if event_step < 1:
        raise ValueError("event step must be positive")
    if radius < 0:
        raise ValueError("evaluation radius must be nonnegative")
    return tuple(
        range(max(1, event_step - radius), event_step + radius + 1)
    )


def evaluate_trajectory(
    *,
    config: ShortcutCreditExperimentConfig,
    rule: BackwardRule,
    base_state: dict[str, torch.Tensor],
    event: CollapseEvent,
    evaluation_radius: int,
    fitness_batches: tuple[ShortcutBatch, ...],
    device: torch.device,
) -> dict[str, object]:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    model = initialize_forward_model(
        config,
        vocabulary,
        initialization_seed=None,
        device=device,
    )
    model.load_state_dict(base_state)
    optimizer = optimizer_for_model(
        model,
        learning_rate=config.forward_learning_rate,
        device=device,
    )
    generator = torch.Generator().manual_seed(event.generation_seed + 2)
    checkpoint_steps = dense_evaluation_steps(
        event.step,
        radius=evaluation_radius,
    )
    checkpoint_set = set(checkpoint_steps)
    horizon = checkpoint_steps[-1]
    batches = make_inner_batches(
        config,
        horizon=horizon,
        vocabulary=vocabulary,
        generator=generator,
        device=device,
    )
    minimum_accuracy = float("inf")
    minimum_accuracy_step = -1
    maximum_loss = float("-inf")
    maximum_loss_step = -1
    for step, batch in enumerate(batches, start=1):
        train_forward_step(model, optimizer, batch, rule)
        if step not in checkpoint_set:
            continue
        metrics = evaluate_shortcut_batches(
            model,
            fitness_batches,
            evaluation_batch_size=config.fitness_examples,
        )
        step_accuracy = min(metrics.mode_accuracy.values())
        step_loss = max(metrics.mode_loss.values())
        if step_accuracy < minimum_accuracy:
            minimum_accuracy = step_accuracy
            minimum_accuracy_step = step
        if step_loss > maximum_loss:
            maximum_loss = step_loss
            maximum_loss_step = step
    return {
        "generation_seed": event.generation_seed,
        "event_step": event.step,
        "evaluation_start_step": checkpoint_steps[0],
        "evaluation_end_step": checkpoint_steps[-1],
        "minimum_mode_accuracy": minimum_accuracy,
        "minimum_accuracy_step": minimum_accuracy_step,
        "worst_mode_loss": maximum_loss,
        "worst_loss_step": maximum_loss_step,
    }


def evaluate_center_shard(
    *,
    config: ShortcutCreditExperimentConfig,
    checkpoint: Path,
    base_states: dict[int, dict[str, torch.Tensor]],
    events: tuple[CollapseEvent, ...],
    evaluation_radius: int,
    fitness_batches_cpu: tuple[ShortcutBatch, ...],
    device: torch.device,
) -> list[dict[str, object]]:
    rule, _, _, _ = load_checkpoint(checkpoint, device=device)
    rule.requires_grad_(False)
    fitness_batches = tuple(
        batch.to(device) for batch in fitness_batches_cpu
    )
    return [
        evaluate_trajectory(
            config=config,
            rule=rule,
            base_state={
                name: tensor.to(device)
                for name, tensor in base_states[event.generation_seed].items()
            },
            event=event,
            evaluation_radius=evaluation_radius,
            fitness_batches=fitness_batches,
            device=device,
        )
        for event in events
    ]


def evaluate_candidate_shard(
    *,
    config: ShortcutCreditExperimentConfig,
    rule_config: object,
    center_parameters: dict[str, torch.Tensor],
    base_states: dict[int, dict[str, torch.Tensor]],
    center_results: dict[int, dict[str, object]],
    directions: tuple[EggrollDirection, ...],
    candidate_indices: tuple[int, ...],
    sigma: float,
    events: tuple[CollapseEvent, ...],
    evaluation_radius: int,
    fitness_batches_cpu: tuple[ShortcutBatch, ...],
    device: torch.device,
    window_aggregation: str,
) -> list[dict[str, object]]:
    worker_center = {
        name: tensor.to(device)
        for name, tensor in center_parameters.items()
    }
    worker_states = {
        seed: {
            name: tensor.to(device)
            for name, tensor in base_states[seed].items()
        }
        for seed, _ in events
    }
    worker_directions = {
        index: move_eggroll_direction(directions[index], device)
        for index in {candidate // 2 for candidate in candidate_indices}
    }
    fitness_batches = tuple(
        batch.to(device) for batch in fitness_batches_cpu
    )
    rows = []
    for candidate_index in candidate_indices:
        direction_index = candidate_index // 2
        sign = 1 if candidate_index % 2 == 0 else -1
        rule = initialize_backward_rule(
            rule_config,  # type: ignore[arg-type]
            device=device,
        )
        rule.requires_grad_(False)
        apply_eggroll_direction(
            rule,
            worker_center,
            worker_directions[direction_index],
            sigma=sigma,
            sign=sign,
        )
        event_results = []
        for event in events:
            result = evaluate_trajectory(
                config=config,
                rule=rule,
                base_state=worker_states[event.generation_seed],
                event=event,
                evaluation_radius=evaluation_radius,
                fitness_batches=fitness_batches,
                device=device,
            )
            center_accuracy = float(
                center_results[event.generation_seed][
                    "minimum_mode_accuracy"
                ]
            )
            result["center_minimum_mode_accuracy"] = center_accuracy
            result["fitness"] = (
                float(result["minimum_mode_accuracy"]) - center_accuracy
            )
            event_results.append(result)
        fitnesses = [
            float(result["fitness"]) for result in event_results
        ]
        rows.append(
            {
                "candidate_index": candidate_index,
                "direction_index": direction_index,
                "sign": sign,
                "fitness": aggregate_window_fitness(
                    fitnesses,
                    aggregation=window_aggregation,
                ),
                "mean_window_fitness": sum(fitnesses) / len(fitnesses),
                "events": event_results,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--events", type=parse_events, required=True)
    parser.add_argument("--evaluation-radius", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument(
        "--window-aggregation",
        choices=("mean", "minimum"),
        default="minimum",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--perturb-parameter-prefixes",
        type=parse_parameter_prefixes,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-devices")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.population_size < 2 or args.population_size % 2:
        raise ValueError("population size must be positive and even")
    if args.sigma <= 0:
        raise ValueError("sigma must be positive")
    if args.evaluation_radius < 0:
        raise ValueError("evaluation radius must be nonnegative")

    started_at = time.monotonic()
    primary_device = torch.device(args.device)
    devices = parse_candidate_devices(
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
    horizon = max(
        event.step + args.evaluation_radius for event in args.events
    )
    config = replace(
        saved_config,
        horizon=horizon,
        max_horizon=max(horizon, saved_config.max_horizon),
        device=args.device,
        resume=None,
        resume_horizon=None,
        wandb=False,
    )
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    base_states = {}
    for event in args.events:
        model = initialize_forward_model(
            config,
            vocabulary,
            initialization_seed=event.generation_seed + 1,
            device=torch.device("cpu"),
        )
        base_states[event.generation_seed] = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        del model
    fitness_generator = torch.Generator().manual_seed(config.seed + 10_000)
    fitness_batches_cpu = make_fitness_batches(
        config.fitness_examples,
        min_length=config.min_length,
        max_length=config.max_length,
        batch_size=config.fitness_batch_size,
        generator=fitness_generator,
        vocabulary=vocabulary,
        leak_placement=config.leak_placement,
        device=torch.device("cpu"),
    )

    event_shards = tuple(
        args.events[index :: len(devices)]
        for index in range(len(devices))
    )
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        center_futures = [
            pool.submit(
                evaluate_center_shard,
                config=config,
                checkpoint=args.checkpoint,
                base_states=base_states,
                events=shard,
                evaluation_radius=args.evaluation_radius,
                fitness_batches_cpu=fitness_batches_cpu,
                device=device,
            )
            for shard, device in zip(event_shards, devices)
        ]
        center_rows = [
            row for future in center_futures for row in future.result()
        ]
    center_results = {
        int(row["generation_seed"]): row for row in center_rows
    }

    center_rule, _, _, _ = load_checkpoint(
        args.checkpoint,
        device=primary_device,
    )
    center_parameters = {
        name: tensor.cpu()
        for name, tensor in clone_center_parameters(center_rule).items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    directions = tuple(
        move_eggroll_direction(
            restrict_direction(
                sample_eggroll_direction(
                    center_rule,
                    generator=generator,
                ),
                parameter_prefixes=args.perturb_parameter_prefixes,
            ),
            "cpu",
        )
        for _ in range(args.population_size // 2)
    )
    candidate_indices = tuple(range(args.population_size))
    candidate_shards = tuple(
        candidate_indices[index :: len(devices)]
        for index in range(len(devices))
    )
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        candidate_futures = [
            pool.submit(
                evaluate_candidate_shard,
                config=config,
                rule_config=center_rule.config,
                center_parameters=center_parameters,
                base_states=base_states,
                center_results=center_results,
                directions=directions,
                candidate_indices=shard,
                sigma=args.sigma,
                events=args.events,
                evaluation_radius=args.evaluation_radius,
                fitness_batches_cpu=fitness_batches_cpu,
                device=device,
                window_aggregation=args.window_aggregation,
            )
            for shard, device in zip(candidate_shards, devices)
        ]
        candidate_rows = [
            row for future in candidate_futures for row in future.result()
        ]
    candidate_rows.sort(key=lambda row: int(row["candidate_index"]))
    best = max(
        candidate_rows,
        key=lambda row: (
            float(row["fitness"]),
            float(row["mean_window_fitness"]),
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "center.json").write_text(
        json.dumps(
            sorted(
                center_rows,
                key=lambda row: int(row["generation_seed"]),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.output_dir / "candidates.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in candidate_rows
        )
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "events": [event._asdict() for event in args.events],
        "evaluation_radius": args.evaluation_radius,
        "population_size": args.population_size,
        "sigma": args.sigma,
        "window_aggregation": args.window_aggregation,
        "seed": args.seed,
        "parameter_prefixes": list(args.perturb_parameter_prefixes),
        "devices": [str(device) for device in devices],
        "best": best,
        "elapsed_seconds": time.monotonic() - started_at,
        "config": asdict(config),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
