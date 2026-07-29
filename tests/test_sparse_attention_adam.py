from __future__ import annotations

import torch
import torch.nn.functional as F

from list_sorting_transformer.data import make_pointer_next_batch
from list_sorting_transformer.positions import sample_position_offsets
from list_sorting_transformer.sparse_attention_adam import (
    AdaptiveEntmaxSelfAttention,
    SparseAttentionAdamConfig,
    SparseAttentionPointerTransformer,
    entmax15,
    evaluation_batch_size,
    pointer_targets,
)


def small_config(**overrides: object) -> SparseAttentionAdamConfig:
    values = {
        "run_name": "test",
        "steps": 10,
        "batch_size": 4,
        "train_max_length": 4,
        "eval_lengths": (2, 4),
        "final_eval_lengths": (8,),
        "eval_examples": 4,
        "final_eval_examples": 2,
        "d_model": 64,
        "layers": 2,
        "heads": 4,
        "alibi_heads": 2,
        "position_moduli": (3, 5, 7, 11),
        "position_offset_min": -100,
        "position_offset_max": 100,
        "warmup_steps": 1,
        "log_interval": 1,
        "eval_interval": 2,
        "checkpoint_interval": 2,
        "device": "cpu",
        "precision": "float32",
    }
    values.update(overrides)
    return SparseAttentionAdamConfig(**values)


def test_entmax15_is_normalized_sparse_and_differentiable() -> None:
    scores = torch.tensor(
        [[3.0, 1.0, -2.0, -10.0], [0.1, 0.2, 0.3, 0.4]],
        requires_grad=True,
    )
    probabilities = entmax15(scores)

    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2),
    )
    assert bool(probabilities[0, -1].eq(0))
    assert bool((probabilities >= 0).all())
    probabilities.square().sum().backward()
    assert scores.grad is not None
    assert bool(torch.isfinite(scores.grad).all())


def test_nape_split_and_adaptive_scalers_receive_gradients() -> None:
    config = small_config()
    model = SparseAttentionPointerTransformer(config)
    attention = model.encoder.blocks[0].attention
    assert isinstance(attention, AdaptiveEntmaxSelfAttention)
    torch.testing.assert_close(
        attention.slopes,
        torch.tensor([1.0, 0.5, 0.0, 0.0]),
    )

    generator = torch.Generator().manual_seed(17)
    batch = make_pointer_next_batch(
        config.batch_size,
        4,
        generator=generator,
        vocabulary=model.vocabulary,
    )
    offsets = sample_position_offsets(
        config.batch_size,
        minimum=config.position_offset_min,
        maximum=config.position_offset_max,
        generator=generator,
        device=torch.device("cpu"),
    )
    logits = model(batch.prompt_ids, offsets=offsets)
    loss = F.cross_entropy(logits, pointer_targets(batch))
    loss.backward()

    assert attention.beta_projection.weight.grad is not None
    assert attention.gamma_projection.weight.grad is not None
    assert bool(torch.isfinite(attention.beta_projection.weight.grad).all())
    metrics = model.attention_metrics()
    assert 0 < metrics["attention/support_fraction"] < 1
    assert metrics["attention/support_size"] >= 1


def test_evaluation_batch_size_respects_attention_budget() -> None:
    config = small_config(
        eval_batch_size=32,
        eval_attention_element_budget=4 * 100 * 100,
    )
    assert evaluation_batch_size(config, prompt_length=10) == 32
    assert evaluation_batch_size(config, prompt_length=100) == 1
