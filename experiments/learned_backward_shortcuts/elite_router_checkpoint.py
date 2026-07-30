"""Create router checkpoints interpolated toward an elite candidate centroid."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from list_sorting_transformer.shortcut_learning.shortcut_credit import (
    AttentionRoutingRule,
    EggrollDirection,
    clone_center_parameters,
    sample_eggroll_direction,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit_experiment import (
    PlateauState,
    ShortcutCreditExperimentConfig,
    update_elite_search_state,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit_routing_plot import (
    load_attention_router,
)


def parse_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(item) for item in value.split(","))
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError(
            "elite indices must be unique nonnegative integers"
        )
    return indices


def checkpoint_search_sigma(
    checkpoint: dict[str, object],
    config: ShortcutCreditExperimentConfig,
) -> float:
    plateau_state = checkpoint.get("plateau_state", {})
    if not isinstance(plateau_state, dict):
        raise ValueError("checkpoint plateau state must be a mapping")
    sigma = plateau_state.get("search_sigma")
    if sigma is None:
        return config.sigma
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("checkpoint search sigma must be positive")
    return sigma


def elite_parameter_delta(
    directions: Sequence[EggrollDirection],
    candidate_indices: Sequence[int],
    *,
    parameter_name: str,
    sigma: float,
) -> Tensor:
    """Average signed parameter perturbations for selected candidates."""

    perturbations = []
    for candidate_index in candidate_indices:
        direction_index = candidate_index // 2
        if direction_index >= len(directions):
            raise ValueError("candidate index exceeds sampled directions")
        sign = 1 if candidate_index % 2 == 0 else -1
        perturbations.append(
            sign * sigma * directions[direction_index].tensors[parameter_name]
        )
    return torch.stack(perturbations).mean(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument(
        "--horizon",
        type=int,
        help="training horizon used to evaluate the selected candidates",
    )
    parser.add_argument("--elite-indices", type=parse_indices, required=True)
    parser.add_argument(
        "--alpha",
        action="append",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--mark-accepted",
        action="store_true",
        help="advance the saved adaptive search state as an accepted update",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.generation < 0:
        raise ValueError("generation must be nonnegative")
    if args.horizon is not None and args.horizon < 1:
        raise ValueError("horizon must be positive")
    if any(not 0 <= alpha <= 1 for alpha in args.alpha):
        raise ValueError("alpha must be in [0, 1]")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = ShortcutCreditExperimentConfig(**checkpoint["config"])
    search_sigma = checkpoint_search_sigma(checkpoint, config)
    if max(args.elite_indices) >= config.population_size:
        raise ValueError("elite index exceeds saved population size")
    center_rule = load_attention_router(args.checkpoint)
    center_parameters = clone_center_parameters(center_rule)
    generation_seed = config.seed * 1_000_003 + args.generation * 10_007
    generator = torch.Generator().manual_seed(generation_seed + 3)
    directions = tuple(
        sample_eggroll_direction(center_rule, generator=generator)
        for _ in range(config.population_size // 2)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for alpha in args.alpha:
        rule = load_attention_router(args.checkpoint)
        with torch.no_grad():
            for name, parameter in rule.named_parameters():
                delta = elite_parameter_delta(
                    directions,
                    args.elite_indices,
                    parameter_name=name,
                    sigma=search_sigma,
                )
                parameter.copy_(center_parameters[name] + alpha * delta)
            rule.project_parameters_()
        output = args.output_dir / (
            f"elite_alpha_{alpha:.4f}".replace(".", "p") + ".pt"
        )
        derived = dict(checkpoint)
        derived["backward_rule_state"] = rule.state_dict()
        # This checkpoint contains the update produced by args.generation, so
        # resuming it should begin at the following generation.
        derived["generation"] = args.generation
        if args.horizon is not None:
            derived["horizon"] = args.horizon
        derived["elite_interpolation"] = {
            "source_checkpoint": str(args.checkpoint),
            "generation": args.generation,
            "horizon": args.horizon,
            "candidate_indices": list(args.elite_indices),
            "alpha": alpha,
            "search_sigma": search_sigma,
            "marked_accepted": args.mark_accepted,
        }
        if args.mark_accepted:
            plateau_state = PlateauState(**checkpoint["plateau_state"])
            update_elite_search_state(
                plateau_state,
                accepted=True,
                config=config,
            )
            derived["plateau_state"] = asdict(plateau_state)
        torch.save(derived, output)
        print(output)


if __name__ == "__main__":
    main()
