"""Restore an old centre after retrospectively rejecting an elite proposal."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from list_sorting_transformer.shortcut_credit_experiment import (
    PlateauState,
    ShortcutCreditExperimentConfig,
    update_elite_search_state,
)


def corrected_rejection_state(
    *,
    center_checkpoint: dict[str, object],
    run_checkpoint: dict[str, object],
    config: ShortcutCreditExperimentConfig,
) -> PlateauState:
    center_state = PlateauState(**center_checkpoint["plateau_state"])
    if center_checkpoint["horizon"] != run_checkpoint["horizon"]:
        center_state.consecutive_accepted_updates = 0
    update_elite_search_state(
        center_state,
        accepted=False,
        config=config,
    )

    run_state = PlateauState(**run_checkpoint["plateau_state"])
    run_state.search_sigma = center_state.search_sigma
    run_state.consecutive_accepted_updates = (
        center_state.consecutive_accepted_updates
    )
    return run_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-checkpoint", type=Path, required=True)
    parser.add_argument("--run-checkpoint", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    center = torch.load(
        args.center_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    run = torch.load(
        args.run_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if center.get("experiment") != "learned_backward_shortcuts":
        raise ValueError("centre checkpoint belongs to another experiment")
    if run.get("experiment") != "learned_backward_shortcuts":
        raise ValueError("run checkpoint belongs to another experiment")
    if int(run["generation"]) != int(center["generation"]) + 1:
        raise ValueError("run checkpoint must be one generation after centre")
    for key in ("backward_rule_type", "backward_rule_config"):
        if center[key] != run[key]:
            raise ValueError("centre and run rule architectures differ")

    config = ShortcutCreditExperimentConfig(**run["config"])
    corrected_state = corrected_rejection_state(
        center_checkpoint=center,
        run_checkpoint=run,
        config=config,
    )
    corrected = dict(run)
    corrected["backward_rule_state"] = center["backward_rule_state"]
    corrected["plateau_state"] = asdict(corrected_state)
    corrected["elite_decision_override"] = {
        "decision": "rejected",
        "reason": args.reason,
        "restored_center_checkpoint": str(args.center_checkpoint),
        "original_run_checkpoint": str(args.run_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(corrected, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
