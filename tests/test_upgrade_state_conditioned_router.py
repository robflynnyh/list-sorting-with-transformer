import importlib.util
from dataclasses import asdict
from pathlib import Path

import torch

from list_sorting_transformer.shortcut_learning.shortcut_credit import (
    AttentionRoutingRule,
    AttentionRoutingRuleConfig,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit_experiment import (
    PlateauState,
    load_checkpoint,
)


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "learned_backward_shortcuts"
    / "upgrade_state_conditioned_router.py"
)
SPEC = importlib.util.spec_from_file_location(
    "upgrade_state_conditioned_router",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_upgrade_preserves_rule_and_adds_zero_projection(
    tmp_path: Path,
) -> None:
    torch.manual_seed(5)
    rule = AttentionRoutingRule(
        AttentionRoutingRuleConfig(
            vocab_size=20,
            d_model=16,
            forward_d_model=16,
            n_heads=4,
            forward_layers=2,
            shared_routing_map=True,
        )
    )
    source = tmp_path / "source.pt"
    output = tmp_path / "conditioned.pt"
    torch.save(
        {
            "experiment": "learned_backward_shortcuts",
            "backward_rule_type": "attention_router",
            "config": {
                "d_model": 16,
                "condition_on_forward_state": False,
            },
            "backward_rule_config": asdict(rule.config),
            "backward_rule_state": rule.state_dict(),
            "generation": 3,
            "horizon": 10,
            "plateau_state": asdict(PlateauState()),
        },
        source,
    )

    summary = MODULE.upgrade_checkpoint(source, output)
    upgraded, generation, horizon, _ = load_checkpoint(
        output,
        device=torch.device("cpu"),
    )

    assert summary["new_parameter_nonzero_count"] == 0
    assert generation == 4
    assert horizon == 10
    assert isinstance(upgraded, AttentionRoutingRule)
    assert upgraded.config.condition_on_forward_state
    assert upgraded.forward_state_projection is not None
    assert not bool(upgraded.forward_state_projection.weight.count_nonzero())
    for name, tensor in rule.state_dict().items():
        torch.testing.assert_close(
            upgraded.state_dict()[name],
            tensor,
            rtol=0,
            atol=0,
        )
