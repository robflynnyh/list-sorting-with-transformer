"""Scan forward seeds for late optimizer events that may cause collapse."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import torch

from list_sorting_transformer.shortcut_collapse_window import (
    train_forward_step,
)
from list_sorting_transformer.shortcut_credit import (
    ShortcutBatch,
    ShortcutPointerVocabulary,
    evaluate_shortcut_batches,
    make_fitness_batches,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_forward_model,
    load_checkpoint,
    make_inner_batches,
    parse_candidate_devices,
)


def parse_generation_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "generation seeds must be comma-separated integers"
        ) from error
    if not seeds or min(seeds) < 0 or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError(
            "generation seeds must be unique nonnegative integers"
        )
    return seeds


def add_peak_event(
    events: list[dict[str, float]],
    *,
    event: dict[str, float],
    event_count: int,
) -> None:
    events.append(event)
    events.sort(key=lambda item: item["training_loss"], reverse=True)
    del events[event_count:]


def should_evaluate_clean_accuracy(
    step: int,
    *,
    minimum_event_step: int,
    horizon: int,
    evaluation_interval: int,
) -> bool:
    if evaluation_interval == 0 or step < minimum_event_step:
        return False
    return step == horizon or step % evaluation_interval == 0


def scan_seed_shard(
    *,
    config: ShortcutCreditExperimentConfig,
    checkpoint: Path,
    base_states: dict[int, dict[str, torch.Tensor]],
    generation_seeds: tuple[int, ...],
    horizon: int,
    minimum_event_step: int,
    event_count: int,
    evaluation_interval: int,
    fitness_batches_cpu: tuple[ShortcutBatch, ...],
    device: torch.device,
) -> list[dict[str, object]]:
    backward_rule, _, _, _ = load_checkpoint(
        checkpoint,
        device=device,
    )
    backward_rule.requires_grad_(False)
    fitness_batches = tuple(
        batch.to(device)
        for batch in fitness_batches_cpu
    )
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    rows = []
    for generation_seed in generation_seeds:
        model = initialize_forward_model(
            config,
            vocabulary,
            initialization_seed=None,
            device=device,
        )
        model.load_state_dict(base_states[generation_seed])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.forward_learning_rate,
        )
        generator = torch.Generator().manual_seed(generation_seed + 2)
        inner_batches = make_inner_batches(
            config,
            horizon=horizon,
            vocabulary=vocabulary,
            generator=generator,
            device=device,
        )
        recent_losses: deque[float] = deque(maxlen=100)
        peak_events: list[dict[str, float]] = []
        minimum_clean_event: dict[str, float] | None = None
        for step, batch in enumerate(inner_batches, start=1):
            trailing_mean = (
                sum(recent_losses) / len(recent_losses)
                if recent_losses
                else 0.0
            )
            training_loss = train_forward_step(
                model,
                optimizer,
                batch,
                backward_rule,
            )
            if step >= minimum_event_step:
                add_peak_event(
                    peak_events,
                    event={
                        "step": float(step),
                        "training_loss": training_loss,
                        "trailing_100_mean": trailing_mean,
                        "loss_excess": training_loss - trailing_mean,
                        "loss_ratio": training_loss
                        / max(trailing_mean, 1e-12),
                    },
                    event_count=event_count,
                )
            recent_losses.append(training_loss)
            if should_evaluate_clean_accuracy(
                step,
                minimum_event_step=minimum_event_step,
                horizon=horizon,
                evaluation_interval=evaluation_interval,
            ):
                metrics = evaluate_shortcut_batches(
                    model,
                    fitness_batches,
                    evaluation_batch_size=config.fitness_examples,
                )
                event = {
                    "step": float(step),
                    "clean_loss": metrics.loss,
                    "minimum_mode_accuracy": min(
                        metrics.mode_accuracy.values()
                    ),
                    "worst_mode_loss": max(metrics.mode_loss.values()),
                }
                if (
                    minimum_clean_event is None
                    or event["minimum_mode_accuracy"]
                    < minimum_clean_event["minimum_mode_accuracy"]
                    or (
                        event["minimum_mode_accuracy"]
                        == minimum_clean_event["minimum_mode_accuracy"]
                        and event["worst_mode_loss"]
                        > minimum_clean_event["worst_mode_loss"]
                    )
                ):
                    minimum_clean_event = event
        fixed_metrics = evaluate_shortcut_batches(
            model,
            fitness_batches,
            evaluation_batch_size=config.fitness_examples,
        )
        rows.append(
            {
                "generation_seed": generation_seed,
                "horizon": horizon,
                "maximum_late_training_loss": peak_events[0][
                    "training_loss"
                ],
                "maximum_late_loss_step": int(
                    peak_events[0]["step"]
                ),
                "maximum_late_loss_excess": peak_events[0][
                    "loss_excess"
                ],
                "fixed_end_clean_loss": fixed_metrics.loss,
                "fixed_end_min_mode_accuracy": min(
                    fixed_metrics.mode_accuracy.values()
                ),
                "peak_events": peak_events,
                "minimum_clean_event": minimum_clean_event,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--generation-seeds",
        type=parse_generation_seeds,
        required=True,
    )
    parser.add_argument("--horizon", type=int, default=3600)
    parser.add_argument("--minimum-event-step", type=int, default=1000)
    parser.add_argument("--event-count", type=int, default=10)
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=0,
        help=(
            "measure fixed-set clean accuracy every N updates; zero disables "
            "periodic evaluation"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-devices")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    if not 1 <= args.minimum_event_step <= args.horizon:
        raise ValueError("minimum event step must index the horizon")
    if args.event_count < 1:
        raise ValueError("event count must be positive")
    if args.evaluation_interval < 0:
        raise ValueError("evaluation interval must be nonnegative")

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
    config = replace(
        saved_config,
        horizon=args.horizon,
        max_horizon=max(args.horizon, saved_config.max_horizon),
        device=args.device,
        resume=None,
        resume_horizon=None,
        wandb=False,
    )
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    base_states = {}
    for generation_seed in args.generation_seeds:
        # Build states serially before worker threads mutate process-wide RNG.
        # The model constructor initializes on CPU before moving to its device.
        model = initialize_forward_model(
            config,
            vocabulary,
            initialization_seed=generation_seed + 1,
            device=torch.device("cpu"),
        )
        base_states[generation_seed] = {
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
    shards = tuple(
        args.generation_seeds[index :: len(devices)]
        for index in range(len(devices))
    )
    if len(devices) == 1:
        rows = scan_seed_shard(
            config=config,
            checkpoint=args.checkpoint,
            base_states=base_states,
            generation_seeds=shards[0],
            horizon=args.horizon,
            minimum_event_step=args.minimum_event_step,
            event_count=args.event_count,
            evaluation_interval=args.evaluation_interval,
            fitness_batches_cpu=fitness_batches_cpu,
            device=devices[0],
        )
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [
                pool.submit(
                    scan_seed_shard,
                    config=config,
                    checkpoint=args.checkpoint,
                    base_states=base_states,
                    generation_seeds=shard,
                    horizon=args.horizon,
                    minimum_event_step=args.minimum_event_step,
                    event_count=args.event_count,
                    evaluation_interval=args.evaluation_interval,
                    fitness_batches_cpu=fitness_batches_cpu,
                    device=device,
                )
                for shard, device in zip(shards, devices)
            ]
            rows = [
                row
                for future in futures
                for row in future.result()
            ]
    rows.sort(
        key=lambda row: float(row["maximum_late_training_loss"]),
        reverse=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "events.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "generation_seeds": args.generation_seeds,
        "horizon": args.horizon,
        "minimum_event_step": args.minimum_event_step,
        "event_count": args.event_count,
        "evaluation_interval": args.evaluation_interval,
        "devices": [str(device) for device in devices],
        "elapsed_seconds": time.monotonic() - started_at,
        "largest_events": [
            {
                "generation_seed": row["generation_seed"],
                "step": row["maximum_late_loss_step"],
                "training_loss": row["maximum_late_training_loss"],
                "loss_excess": row["maximum_late_loss_excess"],
            }
            for row in rows
        ],
        "deepest_clean_events": sorted(
            (
                {
                    "generation_seed": row["generation_seed"],
                    **row["minimum_clean_event"],
                }
                for row in rows
                if row["minimum_clean_event"] is not None
            ),
            key=lambda event: (
                event["minimum_mode_accuracy"],
                -event["worst_mode_loss"],
            ),
        ),
        "config": asdict(config),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
