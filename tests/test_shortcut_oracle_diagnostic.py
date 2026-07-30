from __future__ import annotations

import torch

from experiments.learned_backward_shortcuts.oracle_route_diagnostic import (
    UniformAttentionRouter,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit import (
    AttentionRoutingRuleConfig,
)


def test_uniform_router_uses_the_requested_gate_on_every_edge() -> None:
    rule = UniformAttentionRouter(
        AttentionRoutingRuleConfig(
            vocab_size=20,
            d_model=16,
            n_heads=4,
            forward_layers=2,
        ),
        gate_value=0.72,
    )

    gates = rule.attention_gates(torch.tensor([[1, 2, 3], [4, 5, 6]]))

    assert len(gates) == 2
    assert all(gate.shape == (2, 4, 3, 3) for gate in gates)
    assert all(torch.equal(gate, torch.full_like(gate, 0.72)) for gate in gates)
