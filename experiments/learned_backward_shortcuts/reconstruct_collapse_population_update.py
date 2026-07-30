"""Reconstruct a fitness-weighted EGGROLL update from collapse candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.learned_backward_shortcuts.collapse_window_population import (
    parse_parameter_prefixes,
    restrict_direction,
)
from list_sorting_transformer.shortcut_credit import (
    clone_center_parameters,
    paper_eggroll_update,
    sample_eggroll_direction,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    load_checkpoint,
)


def load_fitnesses(
    path: Path,
    *,
    population_size: int,
    fitness_field: str = "fitness",
) -> torch.Tensor:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: int(row["candidate_index"]))
    candidate_indices = [int(row["candidate_index"]) for row in rows]
    if candidate_indices != list(range(population_size)):
        raise ValueError(
            "candidate file must contain every population index exactly once"
        )
    return torch.tensor(
        [float(row[fitness_field]) for row in rows],
        dtype=torch.float32,
    )


def reconstruct_update(
    *,
    checkpoint: Path,
    candidates: Path,
    population_size: int,
    population_seed: int,
    sigma: float,
    outer_learning_rate: float,
    fitness_field: str,
    parameter_prefixes: tuple[str, ...],
    output: Path,
) -> dict[str, object]:
    if population_size < 2 or population_size % 2:
        raise ValueError("population size must be positive and even")
    if sigma <= 0 or outer_learning_rate <= 0:
        raise ValueError("sigma and outer learning rate must be positive")

    raw_checkpoint = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    rule, _, _, _ = load_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    center = clone_center_parameters(rule)
    generator = torch.Generator().manual_seed(population_seed)
    directions = tuple(
        restrict_direction(
            sample_eggroll_direction(rule, generator=generator),
            parameter_prefixes=parameter_prefixes,
        )
        for _ in range(population_size // 2)
    )
    fitnesses = load_fitnesses(
        candidates,
        population_size=population_size,
        fitness_field=fitness_field,
    )
    standardized = paper_eggroll_update(
        rule,
        directions,
        fitnesses,
        sigma=sigma,
        learning_rate=outer_learning_rate,
    )
    changed = {}
    for name, parameter in rule.named_parameters():
        delta = parameter.detach() - center[name]
        if bool(delta.count_nonzero()):
            changed[name] = {
                "rms": float(delta.square().mean().sqrt()),
                "norm": float(delta.norm()),
                "rank": (
                    int(torch.linalg.matrix_rank(delta))
                    if delta.ndim == 2
                    else None
                ),
            }

    reconstructed = dict(raw_checkpoint)
    reconstructed["backward_rule_state"] = {
        name: tensor.detach().cpu()
        for name, tensor in rule.state_dict().items()
    }
    reconstructed["collapse_population_update"] = {
        "source_checkpoint": str(checkpoint),
        "candidates": str(candidates),
        "population_size": population_size,
        "population_seed": population_seed,
        "sigma": sigma,
        "outer_learning_rate": outer_learning_rate,
        "fitness_field": fitness_field,
        "parameter_prefixes": list(parameter_prefixes),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reconstructed, output)
    return {
        "output_checkpoint": str(output),
        "fitness_mean": float(fitnesses.mean()),
        "fitness_std": float(fitnesses.std(unbiased=False)),
        "standardized_mean": float(standardized.mean()),
        "standardized_std": float(standardized.std(unbiased=False)),
        "changed_parameters": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--population-size", type=int, required=True)
    parser.add_argument("--population-seed", type=int, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--outer-learning-rate", type=float, required=True)
    parser.add_argument(
        "--fitness-field",
        choices=("fitness", "mean_window_fitness"),
        default="fitness",
        help="candidate JSONL field standardized to form the EGGROLL update",
    )
    parser.add_argument(
        "--perturb-parameter-prefixes",
        type=parse_parameter_prefixes,
        default=(),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            reconstruct_update(
                checkpoint=args.checkpoint,
                candidates=args.candidates,
                population_size=args.population_size,
                population_seed=args.population_seed,
                sigma=args.sigma,
                outer_learning_rate=args.outer_learning_rate,
                fitness_field=args.fitness_field,
                parameter_prefixes=args.perturb_parameter_prefixes,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
