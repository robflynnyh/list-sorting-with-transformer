"""Add zero-initialized forward-state conditioning to a router checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from list_sorting_transformer.shortcut_learning.shortcut_credit import (
    AttentionRoutingRule,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit_experiment import (
    initialize_backward_rule,
    load_checkpoint,
)


def upgrade_checkpoint(source: Path, output: Path) -> dict[str, object]:
    raw_checkpoint = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
    )
    old_rule, _, _, _ = load_checkpoint(
        source,
        device=torch.device("cpu"),
    )
    if not isinstance(old_rule, AttentionRoutingRule):
        raise ValueError("only attention-router checkpoints can be upgraded")
    if old_rule.config.condition_on_forward_state:
        raise ValueError("checkpoint is already state-conditioned")

    forward_d_model = int(raw_checkpoint["config"]["d_model"])
    new_config = replace(
        old_rule.config,
        forward_d_model=forward_d_model,
        condition_on_forward_state=True,
    )
    new_rule = initialize_backward_rule(
        new_config,
        device=torch.device("cpu"),
    )
    incompatible = new_rule.load_state_dict(
        old_rule.state_dict(),
        strict=False,
    )
    expected_missing = ["forward_state_projection.weight"]
    if (
        incompatible.missing_keys != expected_missing
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "unexpected checkpoint conversion mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    projection = new_rule.forward_state_projection
    if projection is None or bool(projection.weight.count_nonzero()):
        raise RuntimeError("new state projection must start at exact zero")

    upgraded = dict(raw_checkpoint)
    upgraded_config = dict(raw_checkpoint["config"])
    upgraded_config["condition_on_forward_state"] = True
    upgraded["config"] = upgraded_config
    upgraded["backward_rule_config"] = asdict(new_config)
    upgraded["backward_rule_state"] = {
        name: tensor.detach().cpu()
        for name, tensor in new_rule.state_dict().items()
    }
    upgraded["state_conditioned_upgrade"] = {
        "source_checkpoint": str(source),
        "forward_d_model": forward_d_model,
        "zero_initialized": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(upgraded, output)
    return {
        "source_checkpoint": str(source),
        "output_checkpoint": str(output),
        "forward_d_model": forward_d_model,
        "new_parameter_count": projection.weight.numel(),
        "new_parameter_nonzero_count": int(
            projection.weight.count_nonzero()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            upgrade_checkpoint(args.checkpoint, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
