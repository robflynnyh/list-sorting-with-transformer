"""Train an autoregressive Transformer or LSTM to sort symbol lists."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .data import (
    make_adjacent_sort_batch,
    make_auto_advance_sort_batch,
    make_local_window_sort_batch,
    make_pointer_next_batch,
    make_pointer_quicksort_batch,
    make_quicksort_trace_batch,
    make_sorting_batch,
    sample_length,
)
from .evaluate import resolve_device
from .evaluation import (
    aggregate_length_ranges,
    autocast_context,
    evaluate_lengths,
    output_cross_entropy,
)
from .metrics import masked_token_accuracy
from .model import DecoderTransformer, ModelConfig
from .local_window_sort import WINDOW_TOOL_EVENTS
from .plots import plot_length_generalization, plot_training_history
from .quicksort import SNAPSHOT_MODES, SnapshotMode
from .recurrent import LSTMConfig, LSTMSorter
from .tokens import (
    LOCAL_WINDOW_PAIR_ENCODINGS,
    AdjacentSortVocabulary,
    AutoAdvanceSortVocabulary,
    LocalWindowSortVocabulary,
    PointerNextVocabulary,
    PointerQuicksortVocabulary,
    QuicksortTraceVocabulary,
    SymbolVocabulary,
    make_vocabulary,
)


@dataclass(frozen=True)
class TrainConfig:
    task: str = "direct"
    representation: str = "numbers"
    symbol_count: int = 10
    train_min_length: int = 2
    train_max_length: int = 20
    eval_max_length: int = 40
    steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    log_interval: int = 50
    eval_interval: int = 1_000
    eval_examples: int = 128
    eval_batch_size: int = 128
    seed: int = 7
    trace_snapshot_mode: SnapshotMode = "partition"
    gradient_accumulation_steps: int = 1
    checkpoint_interval: int = 1_000
    window_tool_events: tuple[str, ...] = WINDOW_TOOL_EVENTS
    window_pair_encoding: str = "separate"

    def __post_init__(self) -> None:
        if self.task not in {
            "direct",
            "pointer_next",
            "quicksort_trace",
            "pointer_quicksort",
            "pointer_quicksort_no_tool",
            "adjacent_sort",
            "adjacent_sort_no_tool",
            "adjacent_sort_auto_advance",
            "adjacent_sort_auto_advance_no_tool",
            "adjacent_sort_local_window",
        }:
            raise ValueError("invalid sorting task")
        if self.representation not in {"alphabet", "numbers"}:
            raise ValueError("invalid representation")
        if self.trace_snapshot_mode not in SNAPSHOT_MODES:
            raise ValueError("invalid trace snapshot mode")
        if not 1 <= self.train_min_length <= self.train_max_length:
            raise ValueError("invalid training length range")
        if self.eval_max_length < self.train_max_length:
            raise ValueError("eval_max_length must include the training range")
        if self.task == "pointer_next" and self.train_min_length < 2:
            raise ValueError("pointer_next requires train_min_length >= 2")
        integer_fields = (
            self.steps,
            self.batch_size,
            self.log_interval,
            self.eval_interval,
            self.eval_examples,
            self.eval_batch_size,
            self.gradient_accumulation_steps,
            self.checkpoint_interval,
        )
        if any(value < 1 for value in integer_fields):
            raise ValueError("step, batch, logging, and evaluation sizes must be positive")
        if not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be between zero and steps")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if not set(self.window_tool_events) <= set(WINDOW_TOOL_EVENTS):
            allowed = ", ".join(WINDOW_TOOL_EVENTS)
            raise ValueError(
                f"window_tool_events may only contain {allowed}"
            )
        if self.window_pair_encoding not in LOCAL_WINDOW_PAIR_ENCODINGS:
            allowed = ", ".join(LOCAL_WINDOW_PAIR_ENCODINGS)
            raise ValueError(
                f"window_pair_encoding must be one of: {allowed}"
            )


def learning_rate_at_step(config: TrainConfig, step: int) -> float:
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    decay_steps = max(config.steps - config.warmup_steps, 1)
    progress = min(max((step - config.warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.learning_rate * (0.1 + 0.9 * cosine)


def selected_evaluation_lengths(config: TrainConfig) -> list[int]:
    midpoint = (config.train_min_length + config.train_max_length) // 2
    candidates = {
        config.train_min_length,
        midpoint,
        config.train_max_length,
        min(config.eval_max_length, config.train_max_length + 5),
        config.eval_max_length,
    }
    return sorted(candidates)


def parse_window_tool_events(specification: str) -> tuple[str, ...]:
    """Parse the transition events whose windows use the executor."""

    if specification.strip().lower() in {"", "none"}:
        return ()
    events = tuple(
        component.strip().upper()
        for component in specification.split(",")
        if component.strip()
    )
    if not set(events) <= set(WINDOW_TOOL_EVENTS):
        allowed = ", ".join(WINDOW_TOOL_EVENTS)
        raise argparse.ArgumentTypeError(
            f"window tool events must use {allowed}, or none"
        )
    return events


def save_checkpoint(
    path: Path,
    *,
    model: DecoderTransformer | LSTMSorter,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    step: int,
    generator: torch.Generator,
) -> None:
    torch.save(
        {
            "architecture": model.architecture,
            "model_config": model.config.as_dict(),
            "train_config": asdict(train_config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "step": step,
        },
        path,
    )


def train(
    model: DecoderTransformer | LSTMSorter,
    config: TrainConfig,
    *,
    vocabulary: SymbolVocabulary,
    output_directory: Path,
    device: torch.device,
    tracker: Any | None = None,
    resume_checkpoint: Path | None = None,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")

    generator = torch.Generator().manual_seed(config.seed + 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    start_step = 0
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for optimizer_state in optimizer.state.values():
            for name, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[name] = value.to(device)
        start_step = int(checkpoint["step"])
        if "generator_state" in checkpoint:
            generator.set_state(checkpoint["generator_state"])
        if start_step >= config.steps:
            raise ValueError(
                "resume checkpoint is already at or beyond the requested steps"
            )
        print(
            json.dumps(
                {
                    "resumed_from": str(resume_checkpoint),
                    "resume_step": start_step,
                }
            ),
            flush=True,
        )
    history = []
    evaluations = []
    started_at = time.monotonic()
    model.train()
    for step in range(start_step + 1, config.steps + 1):
        current_learning_rate = learning_rate_at_step(config, step)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_accuracy = 0.0
        microbatch_lengths = []
        for _ in range(config.gradient_accumulation_steps):
            length = sample_length(
                config.train_min_length,
                config.train_max_length,
                generator=generator,
            )
            microbatch_lengths.append(length)
            if config.task == "direct":
                batch = make_sorting_batch(
                    config.batch_size,
                    length,
                    generator=generator,
                    symbol_count=config.symbol_count,
                    device=device,
                )
            elif config.task == "pointer_next":
                if not isinstance(vocabulary, PointerNextVocabulary):
                    raise TypeError("pointer_next requires PointerNextVocabulary")
                batch = make_pointer_next_batch(
                    config.batch_size,
                    length,
                    generator=generator,
                    vocabulary=vocabulary,
                    device=device,
                )
            elif config.task == "quicksort_trace":
                if not isinstance(vocabulary, QuicksortTraceVocabulary):
                    raise TypeError(
                        "quicksort_trace requires QuicksortTraceVocabulary"
                    )
                batch = make_quicksort_trace_batch(
                    config.batch_size,
                    length,
                    generator=generator,
                    vocabulary=vocabulary,
                    snapshot_mode=config.trace_snapshot_mode,
                    device=device,
                )
            elif config.task in {
                "pointer_quicksort",
                "pointer_quicksort_no_tool",
            }:
                if not isinstance(vocabulary, PointerQuicksortVocabulary):
                    raise TypeError(
                        f"{config.task} requires PointerQuicksortVocabulary"
                    )
                batch = make_pointer_quicksort_batch(
                    config.batch_size,
                    length,
                    generator=generator,
                    vocabulary=vocabulary,
                    supervise_observations=(
                        config.task == "pointer_quicksort_no_tool"
                    ),
                    device=device,
                )
            elif config.task in {"adjacent_sort", "adjacent_sort_no_tool"}:
                if not isinstance(vocabulary, AdjacentSortVocabulary):
                    raise TypeError(
                        f"{config.task} requires AdjacentSortVocabulary"
                    )
                batch = make_adjacent_sort_batch(
                    config.batch_size,
                    length,
                    generator=generator,
                    vocabulary=vocabulary,
                    supervise_observations=(
                        config.task == "adjacent_sort_no_tool"
                    ),
                    device=device,
                )
            elif config.task in {
                "adjacent_sort_auto_advance",
                "adjacent_sort_auto_advance_no_tool",
            }:
                if not isinstance(vocabulary, AutoAdvanceSortVocabulary):
                    raise TypeError(
                        f"{config.task} requires AutoAdvanceSortVocabulary"
                    )
                batch = make_auto_advance_sort_batch(
                    config.batch_size,
                    length,
                    generator=generator,
                    vocabulary=vocabulary,
                    supervise_observations=(
                        config.task == "adjacent_sort_auto_advance_no_tool"
                    ),
                    device=device,
                )
            elif config.task == "adjacent_sort_local_window":
                if not isinstance(vocabulary, LocalWindowSortVocabulary):
                    raise TypeError(
                        f"{config.task} requires LocalWindowSortVocabulary"
                    )
                batch = make_local_window_sort_batch(
                    config.batch_size,
                    length,
                    generator=generator,
                    vocabulary=vocabulary,
                    tool_events=config.window_tool_events,
                    device=device,
                )
            else:
                raise ValueError(f"unsupported sorting task: {config.task}")
            with autocast_context(device):
                logits = model(batch.model_inputs)
                loss = output_cross_entropy(logits, batch.labels)
            (loss / config.gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.item())
            accumulated_accuracy += masked_token_accuracy(logits, batch.labels)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )
        optimizer.step()

        if step % config.checkpoint_interval == 0:
            save_checkpoint(
                output_directory / "checkpoint.pt",
                model=model,
                optimizer=optimizer,
                train_config=config,
                step=step,
                generator=generator,
            )

        if step == 1 or step % config.log_interval == 0:
            row = {
                "step": float(step),
                "length": sum(microbatch_lengths) / len(microbatch_lengths),
                "minimum_length": float(min(microbatch_lengths)),
                "maximum_length": float(max(microbatch_lengths)),
                "loss": accumulated_loss / config.gradient_accumulation_steps,
                "token_accuracy": (
                    accumulated_accuracy / config.gradient_accumulation_steps
                ),
                "learning_rate": current_learning_rate,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.monotonic() - started_at,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if tracker is not None:
                tracker.log(
                    {
                        "step": step,
                        **{
                            f"train/{name}": value
                            for name, value in row.items()
                            if name != "step"
                        },
                    }
                )

        if step % config.eval_interval == 0 or step == config.steps:
            if step % config.checkpoint_interval:
                save_checkpoint(
                    output_directory / "checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    train_config=config,
                    step=step,
                    generator=generator,
                )
            per_length = evaluate_lengths(
                model,
                vocabulary,
                selected_evaluation_lengths(config),
                examples_per_length=config.eval_examples,
                batch_size=config.eval_batch_size,
                seed=config.seed + 20_000,
                device=device,
                task=config.task,
                trace_snapshot_mode=config.trace_snapshot_mode,
                window_tool_events=config.window_tool_events,
                train_max_length=config.train_max_length,
            )
            evaluation_row = {
                "step": step,
                "per_length": {
                    str(eval_length): metrics
                    for eval_length, metrics in per_length.items()
                },
            }
            evaluations.append(evaluation_row)
            print(
                json.dumps(
                    {
                        "step": step,
                        "evaluation_exact_match": {
                            str(eval_length): metrics["exact_match"]
                            for eval_length, metrics in per_length.items()
                        },
                    }
                ),
                flush=True,
            )
            if tracker is not None:
                tracker.log(
                    {
                        "step": step,
                        **{
                            f"eval/length_{eval_length}/{name}": value
                            for eval_length, metrics in per_length.items()
                            for name, value in metrics.items()
                        },
                    }
                )
            model.train()

    checkpoint_path = output_directory / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        train_config=config,
        step=config.steps,
        generator=generator,
    )
    final_lengths = range(config.train_min_length, config.eval_max_length + 1)
    final_per_length = evaluate_lengths(
        model,
        vocabulary,
        final_lengths,
        examples_per_length=config.eval_examples,
        batch_size=config.eval_batch_size,
        seed=config.seed + 30_000,
        device=device,
        task=config.task,
        trace_snapshot_mode=config.trace_snapshot_mode,
        window_tool_events=config.window_tool_events,
        train_max_length=config.train_max_length,
    )
    results = {
        "architecture": model.architecture,
        "model_config": model.config.as_dict(),
        "train_config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "wall_time_seconds": time.monotonic() - started_at,
        "history": history,
        "intermediate_evaluations": evaluations,
        "final_per_length": {
            str(length): metrics
            for length, metrics in final_per_length.items()
        },
        "final_aggregate": aggregate_length_ranges(
            final_per_length,
            train_min_length=config.train_min_length,
            train_max_length=config.train_max_length,
        ),
    }
    (output_directory / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    plot_training_history(history, output_directory / "training.png")
    plot_length_generalization(
        final_per_length,
        output_directory / "length_generalization.png",
        train_max_length=config.train_max_length,
    )
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("transformer", "lstm"),
        default="transformer",
    )
    parser.add_argument(
        "--task",
        choices=(
            "direct",
            "pointer_next",
            "quicksort_trace",
            "pointer_quicksort",
            "pointer_quicksort_no_tool",
            "adjacent_sort",
            "adjacent_sort_no_tool",
            "adjacent_sort_auto_advance",
            "adjacent_sort_auto_advance_no_tool",
            "adjacent_sort_local_window",
        ),
        default="direct",
    )
    parser.add_argument(
        "--representation",
        choices=("alphabet", "numbers"),
        default="numbers",
    )
    parser.add_argument("--symbol-count", type=int, default=10)
    parser.add_argument("--train-min-length", type=int, default=2)
    parser.add_argument("--train-max-length", type=int, default=20)
    parser.add_argument("--eval-max-length", type=int, default=40)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument(
        "--window-tool-events",
        type=parse_window_tool_events,
        default=WINDOW_TOOL_EVENTS,
        metavar="EVENTS",
        help=(
            "comma-separated KEEP, SWAP, RESET, and FINISH transition "
            "windows supplied by the executor; use 'none' for no tools"
        ),
    )
    parser.add_argument(
        "--window-pair-encoding",
        choices=LOCAL_WINDOW_PAIR_ENCODINGS,
        default="separate",
        help=(
            "encode the active values as two separate tokens or one atomic "
            "ordered-pair token plus PAIR_END"
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument(
        "--position-pattern",
        choices=("alternating", "rotary", "none"),
        default="alternating",
    )
    parser.add_argument("--rotary-base", type=float, default=10_000.0)
    parser.add_argument(
        "--trace-snapshot-mode",
        choices=SNAPSHOT_MODES,
        default="partition",
        help="when quicksort traces include complete array snapshots",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    train_config = TrainConfig(
        task=args.task,
        representation=args.representation,
        symbol_count=args.symbol_count,
        train_min_length=args.train_min_length,
        train_max_length=args.train_max_length,
        eval_max_length=args.eval_max_length,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_examples=args.eval_examples,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        trace_snapshot_mode=args.trace_snapshot_mode,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_interval=args.checkpoint_interval,
        window_tool_events=args.window_tool_events,
        window_pair_encoding=args.window_pair_encoding,
    )
    vocabulary = make_vocabulary(
        train_config.task,
        representation=train_config.representation,
        symbol_count=train_config.symbol_count,
        window_pair_encoding=train_config.window_pair_encoding,
    )
    if args.architecture == "transformer":
        model_config = ModelConfig(
            vocab_size=vocabulary.size,
            symbol_count=train_config.symbol_count,
            representation=train_config.representation,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            ffn_multiplier=args.ffn_multiplier,
            dropout=args.dropout,
            position_pattern=args.position_pattern,
            rotary_base=args.rotary_base,
        )
    else:
        model_config = LSTMConfig(
            vocab_size=vocabulary.size,
            symbol_count=train_config.symbol_count,
            representation=train_config.representation,
            d_model=args.d_model,
            hidden_size=args.lstm_hidden_size,
            n_layers=args.lstm_layers,
            dropout=args.dropout,
        )
    torch.manual_seed(train_config.seed)
    if args.architecture == "transformer":
        model = DecoderTransformer(model_config)
    else:
        model = LSTMSorter(model_config)
    device = resolve_device(args.device)
    model.to(device)
    output_directory = args.output_directory
    if output_directory is None:
        if args.architecture == "transformer":
            run_name = (
                f"{train_config.task}_{train_config.representation}_"
                f"{model_config.position_pattern}_seed{train_config.seed}"
            )
        else:
            run_name = (
                f"lstm_{train_config.task}_"
                f"{train_config.representation}_seed{train_config.seed}"
            )
        output_directory = Path("artifacts") / run_name
    metadata = {
        "architecture": model.architecture,
        "task": train_config.task,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "output_directory": str(output_directory),
    }
    if isinstance(model, DecoderTransformer):
        metadata["layer_position_modes"] = model.layer_position_modes
    else:
        metadata["recurrent_layers"] = model.config.n_layers
    print(json.dumps(metadata), flush=True)
    tracker = None
    if args.wandb_project is not None:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "install the 'tracking' extra to use W&B logging"
            ) from error
        tracker = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "architecture": model.architecture,
                "model": model.config.as_dict(),
                "training": asdict(train_config),
                "parameter_count": metadata["parameter_count"],
            },
        )
        print(json.dumps({"wandb_url": tracker.url}), flush=True)
    try:
        results = train(
            model,
            train_config,
            vocabulary=vocabulary,
            output_directory=output_directory,
            device=device,
            tracker=tracker,
            resume_checkpoint=args.resume_checkpoint,
        )
    finally:
        if tracker is not None:
            tracker.finish()
    print(
        json.dumps(
            {
                "completed": True,
                "output_directory": str(output_directory),
                "aggregate": results["final_aggregate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
