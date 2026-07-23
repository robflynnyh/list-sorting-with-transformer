"""Plot per-length metrics from two or more completed sorting runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .plots import plot_representation_comparison


def load_run(path: Path) -> tuple[str, dict[int, dict[str, float]], int]:
    payload = json.loads(path.read_text())
    representation = payload["train_config"]["representation"]
    architecture = payload.get("architecture", "transformer")
    if architecture == "transformer":
        position_pattern = payload["model_config"]["position_pattern"]
        label = f"{representation} Transformer ({position_pattern})"
    else:
        label = f"{representation} {architecture.upper()}"
    per_length = {
        int(length): metrics
        for length, metrics in payload["final_per_length"].items()
    }
    train_max_length = int(payload["train_config"]["train_max_length"])
    return label, per_length, train_max_length


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    loaded = [load_run(path) for path in args.metrics]
    train_boundaries = {run[2] for run in loaded}
    if len(train_boundaries) != 1:
        raise ValueError("all compared runs must have the same training boundary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_representation_comparison(
        [(label, per_length) for label, per_length, _ in loaded],
        args.output,
        train_max_length=train_boundaries.pop(),
    )


if __name__ == "__main__":
    main()
