"""Inspect deterministic EGGROLL candidates around saved router centres."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from list_sorting_transformer.shortcut_credit import (
    AttentionRoutingRule,
    ShortcutPointerVocabulary,
    apply_eggroll_direction,
    clone_center_parameters,
    make_shortcut_batch,
    sample_eggroll_direction,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
)
from list_sorting_transformer.shortcut_credit_routing_plot import (
    load_attention_router,
    plot_routing_roles,
    position_matched_role_gates,
    query_role_gates,
    routing_structure_summary,
)


def parse_candidate(value: str) -> tuple[str, Path, int, int]:
    """Parse ``LABEL=CHECKPOINT@GENERATION@CANDIDATE_INDEX``."""

    try:
        label, specification = value.split("=", maxsplit=1)
        raw_path, raw_generation, raw_candidate = specification.rsplit(
            "@",
            maxsplit=2,
        )
        generation = int(raw_generation)
        candidate_index = int(raw_candidate)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "candidate must use "
            "LABEL=CHECKPOINT@GENERATION@CANDIDATE_INDEX"
        ) from error
    if not label or not raw_path or generation < 0 or candidate_index < 0:
        raise argparse.ArgumentTypeError(
            "candidate label/path must be nonempty and indices nonnegative"
        )
    return label, Path(raw_path), generation, candidate_index


def restore_candidate(
    checkpoint_path: Path,
    *,
    generation: int,
    candidate_index: int,
) -> AttentionRoutingRule:
    """Recreate one deterministic antithetic candidate from its centre."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = ShortcutCreditExperimentConfig(**checkpoint["config"])
    if candidate_index >= config.population_size:
        raise ValueError("candidate index exceeds saved population size")
    rule = load_attention_router(checkpoint_path)
    generation_seed = config.seed * 1_000_003 + generation * 10_007
    generator = torch.Generator().manual_seed(generation_seed + 3)
    direction_index = candidate_index // 2
    direction = None
    for _ in range(direction_index + 1):
        direction = sample_eggroll_direction(rule, generator=generator)
    assert direction is not None
    apply_eggroll_direction(
        rule,
        clone_center_parameters(rule),
        direction,
        sigma=config.sigma,
        sign=1 if candidate_index % 2 == 0 else -1,
    )
    return rule


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=parse_candidate,
        metavar="LABEL=CHECKPOINT@GENERATION@CANDIDATE_INDEX",
    )
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument(
        "--leak-mode",
        choices=("correct", "masked", "incorrect"),
        default="correct",
    )
    parser.add_argument(
        "--leak-placement",
        choices=("suffix", "random_list"),
        default="random_list",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args(argv)

    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    batch = make_shortcut_batch(
        args.examples,
        args.length,
        leak_mode=args.leak_mode,
        generator=torch.Generator().manual_seed(args.seed),
        vocabulary=vocabulary,
        leak_placement=args.leak_placement,
    )
    rows = []
    role_summaries = []
    for label, checkpoint, generation, candidate_index in args.candidate:
        rule = restore_candidate(
            checkpoint,
            generation=generation,
            candidate_index=candidate_index,
        )
        gates = rule.attention_gates(batch.input_ids)[0]
        roles = query_role_gates(gates, batch, vocabulary)
        matched = position_matched_role_gates(gates, batch, vocabulary)
        structure = routing_structure_summary(gates)
        row = {
            "label": label,
            "checkpoint": str(checkpoint),
            "generation": generation,
            "candidate_index": candidate_index,
            "roles": roles,
            "position_matched_roles": matched,
            "structure": structure,
        }
        rows.append(row)
        role_summaries.append((label, roles))
        print(json.dumps(row, sort_keys=True), flush=True)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2) + "\n")
    plot_routing_roles(role_summaries, args.output_plot)


if __name__ == "__main__":
    main()
