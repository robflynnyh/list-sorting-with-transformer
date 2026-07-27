"""Reconstruct one deterministic collapse-window population candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from list_sorting_transformer.shortcut_credit import (
    apply_eggroll_direction,
    clone_center_parameters,
    sample_eggroll_direction,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    load_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--population-size", type=int, required=True)
    parser.add_argument("--population-seed", type=int, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.population_size < 2 or args.population_size % 2:
        raise ValueError("population size must be positive and even")
    if not 0 <= args.candidate_index < args.population_size:
        raise ValueError("candidate index is outside the population")
    if args.sigma <= 0:
        raise ValueError("sigma must be positive")

    raw_checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    rule, _, _, _ = load_checkpoint(
        args.checkpoint,
        device=torch.device("cpu"),
    )
    center_parameters = clone_center_parameters(rule)
    generator = torch.Generator().manual_seed(args.population_seed)
    directions = tuple(
        sample_eggroll_direction(rule, generator=generator)
        for _ in range(args.population_size // 2)
    )
    direction_index = args.candidate_index // 2
    sign = 1 if args.candidate_index % 2 == 0 else -1
    apply_eggroll_direction(
        rule,
        center_parameters,
        directions[direction_index],
        sigma=args.sigma,
        sign=sign,
    )

    checkpoint = dict(raw_checkpoint)
    checkpoint["backward_rule_state"] = {
        name: tensor.detach().cpu()
        for name, tensor in rule.state_dict().items()
    }
    checkpoint["collapse_window_update"] = {
        "source_checkpoint": str(args.checkpoint),
        "population_size": args.population_size,
        "population_seed": args.population_seed,
        "sigma": args.sigma,
        "candidate_index": args.candidate_index,
        "direction_index": direction_index,
        "sign": sign,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
