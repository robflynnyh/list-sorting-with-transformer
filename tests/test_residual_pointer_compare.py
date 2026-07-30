from __future__ import annotations

import torch
import pytest

from list_sorting_transformer.core.data import make_pointer_pair_batch
from list_sorting_transformer.core.model import ModelConfig
from list_sorting_transformer.length_generalisation.residual_pointer_compare import (
    ResidualPointerCompareConfig,
    ResidualPointerCompareModel,
    learning_rate_at_step,
    residual_pointer_compare_loss,
)
from list_sorting_transformer.core.tokens import PointerCompareVocabulary


def test_residual_pipeline_loss_trains_all_internal_operations() -> None:
    torch.manual_seed(4)
    vocabulary = PointerCompareVocabulary("numbers", 10)
    model = ResidualPointerCompareModel(
        ModelConfig(
            vocab_size=vocabulary.size,
            d_model=32,
            n_layers=3,
            n_heads=2,
            ffn_multiplier=2,
            dropout=0,
            position_pattern="none",
        ),
        (3, 5, 7, 11),
    )
    config = ResidualPointerCompareConfig(
        steps=10,
        batch_size=4,
        warmup_steps=2,
        position_moduli=(3, 5, 7, 11),
    )
    batch = make_pointer_pair_batch(
        4,
        5,
        generator=torch.Generator().manual_seed(5),
        vocabulary=vocabulary,
    )
    offsets = torch.tensor([-100, -3, 17, 91])

    loss, metrics = residual_pointer_compare_loss(
        model,
        batch,
        offsets,
        config,
    )
    loss.backward()

    assert loss.isfinite()
    assert model.encoder.blocks[0].attention.qkv.weight.grad is not None
    assert model.encoder.blocks[1].attention.qkv.weight.grad is not None
    assert model.action_head.weight.grad is not None
    assert set(metrics) >= {
        "action_accuracy",
        "pointer_address_accuracy",
        "marked_address_accuracy",
        "following_address_accuracy",
        "marked_value_accuracy",
        "following_value_accuracy",
        "pointer_route_accuracy",
        "marked_route_accuracy",
        "following_route_accuracy",
    }


def test_learning_rate_warms_up_and_decays() -> None:
    config = ResidualPointerCompareConfig(
        steps=100,
        warmup_steps=10,
        learning_rate=3e-4,
        minimum_learning_rate=1e-5,
    )

    assert learning_rate_at_step(config, 1) == pytest.approx(3e-5)
    assert learning_rate_at_step(config, 10) == pytest.approx(3e-4)
    assert learning_rate_at_step(config, 100) == pytest.approx(1e-5)


def test_routing_top_k_uses_hard_forward_and_soft_backward() -> None:
    torch.manual_seed(8)
    vocabulary = PointerCompareVocabulary("numbers", 10)
    model_config = ModelConfig(
        vocab_size=vocabulary.size,
        d_model=32,
        n_layers=3,
        n_heads=2,
        ffn_multiplier=2,
        dropout=0,
        position_pattern="none",
    )
    dense = ResidualPointerCompareModel(
        model_config,
        (3, 5, 7, 11),
    )
    straight_through = ResidualPointerCompareModel(
        model_config,
        (3, 5, 7, 11),
        routing_top_k=1,
        routing_top_k_straight_through=True,
    )
    straight_through.load_state_dict(dense.state_dict())
    batch = make_pointer_pair_batch(
        4,
        5,
        generator=torch.Generator().manual_seed(9),
        vocabulary=vocabulary,
    )
    offsets = torch.tensor([-10, -2, 7, 19])

    straight_through.train()
    training_outputs = straight_through(batch.prompt_ids, offsets=offsets)
    straight_through.eval()
    evaluation_outputs = straight_through(batch.prompt_ids, offsets=offsets)
    assert torch.equal(
        training_outputs["action_logits"],
        evaluation_outputs["action_logits"],
    )

    straight_through.train()
    straight_through.zero_grad(set_to_none=True)
    straight_through(batch.prompt_ids, offsets=offsets)[
        "action_logits"
    ].sum().backward()
    straight_through_gradient = (
        straight_through.encoder.blocks[1].attention.qkv.weight.grad.clone()
    )

    hard = ResidualPointerCompareModel(
        model_config,
        (3, 5, 7, 11),
        routing_top_k=1,
    )
    hard.load_state_dict(dense.state_dict())
    hard.train()
    hard(batch.prompt_ids, offsets=offsets)["action_logits"].sum().backward()
    hard_gradient = hard.encoder.blocks[1].attention.qkv.weight.grad

    assert straight_through_gradient.isfinite().all()
    assert straight_through_gradient.abs().sum() > 0
    assert not torch.allclose(straight_through_gradient, hard_gradient)
