"""Evaluate a saved sorting Transformer over specified list lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .evaluation import (
    aggregate_length_ranges,
    evaluate_lengths,
    load_model_checkpoint,
)
from .plots import plot_length_generalization
from .tokens import make_vocabulary


def parse_lengths(specification: str) -> list[int]:
    lengths = set()
    for component in specification.split(","):
        component = component.strip()
        if not component:
            continue
        if "-" in component:
            start_text, end_text = component.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if start < 1 or end < start:
                raise ValueError(f"invalid length range: {component}")
            lengths.update(range(start, end + 1))
        else:
            value = int(component)
            if value < 1:
                raise ValueError("lengths must be positive")
            lengths.add(value)
    if not lengths:
        raise ValueError("at least one evaluation length is required")
    return sorted(lengths)


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(specification)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--lengths", default="2-40")
    parser.add_argument("--examples-per-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17_003)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_model_checkpoint(str(args.checkpoint), device=device)
    config = model.config
    train_config = checkpoint.get("train_config", {})
    task = str(train_config.get("task", "direct"))
    trace_snapshot_mode = str(
        train_config.get("trace_snapshot_mode", "partition")
    )
    window_tool_events = tuple(
        str(event)
        for event in train_config.get(
            "window_tool_events",
            ("KEEP", "SWAP", "RESET", "FINISH"),
        )
    )
    vocabulary = make_vocabulary(
        task,
        representation=config.representation,
        symbol_count=config.symbol_count,
    )
    lengths = parse_lengths(args.lengths)
    per_length = evaluate_lengths(
        model,
        vocabulary,
        lengths,
        examples_per_length=args.examples_per_length,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        task=task,
        trace_snapshot_mode=trace_snapshot_mode,
        window_tool_events=window_tool_events,
    )
    train_min_length = int(train_config.get("train_min_length", min(lengths)))
    train_max_length = int(train_config.get("train_max_length", max(lengths)))
    output = {
        "checkpoint": str(args.checkpoint),
        "step": checkpoint.get("step"),
        "task": task,
        "trace_snapshot_mode": trace_snapshot_mode,
        "window_tool_events": list(window_tool_events),
        "model_config": config.as_dict(),
        "train_length_range": [train_min_length, train_max_length],
        "per_length": {str(length): metrics for length, metrics in per_length.items()},
        "aggregate": aggregate_length_ranges(
            per_length,
            train_min_length=train_min_length,
            train_max_length=train_max_length,
        ),
    }
    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_length_generalization(
            per_length,
            args.plot,
            train_max_length=train_max_length,
        )


if __name__ == "__main__":
    main()
