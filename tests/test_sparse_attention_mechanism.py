from __future__ import annotations

import torch

from experiments.sparse_attention_adam.analyze_alibi_nope_mechanism import (
    ROLE_INDEX,
    ablate_heads,
    attention_components,
    source_roles,
)
from list_sorting_transformer.core.data import make_pointer_next_batch
from list_sorting_transformer.length_generalisation.sparse_attention_adam import (
    AdaptiveEntmaxSelfAttention,
    SparseAttentionAdamConfig,
    SparseAttentionPointerTransformer,
)


def mechanism_config() -> SparseAttentionAdamConfig:
    return SparseAttentionAdamConfig(
        run_name="mechanism-test",
        batch_size=4,
        train_max_length=5,
        eval_lengths=(5,),
        final_eval_lengths=(),
        eval_examples=4,
        d_model=64,
        layers=2,
        heads=4,
        alibi_heads=2,
        attention_normalizer="softmax",
        scaling_mode="none",
        input_position_mode="nape_only",
        value_input_mode="embedding",
        log_interval=1,
        eval_interval=1,
        checkpoint_interval=1,
        device="cpu",
        precision="float32",
    )


def test_source_roles_identify_pointer_chain_positions() -> None:
    config = mechanism_config()
    model = SparseAttentionPointerTransformer(config)
    batch = make_pointer_next_batch(
        4,
        5,
        generator=torch.Generator().manual_seed(41),
        vocabulary=model.vocabulary,
    )

    roles = source_roles(batch)
    rows = torch.arange(batch.values.shape[0])
    ptr_positions = 1 + 2 * batch.pointers

    assert torch.equal(roles[rows, ptr_positions], torch.full((4,), ROLE_INDEX["ptr"]))
    assert torch.equal(
        roles[rows, ptr_positions + 1],
        torch.full((4,), ROLE_INDEX["marked_value"]),
    )
    assert torch.equal(
        roles[rows, ptr_positions + 3],
        torch.full((4,), ROLE_INDEX["target_value"]),
    )
    assert torch.equal(
        roles[:, -1],
        torch.full((4,), ROLE_INDEX["separator"]),
    )


def test_attention_reconstruction_matches_forward() -> None:
    config = mechanism_config()
    attention = AdaptiveEntmaxSelfAttention(config)
    hidden = torch.randn(3, 9, config.d_model)

    expected = attention(hidden)
    reconstructed = attention_components(attention, hidden).output

    torch.testing.assert_close(reconstructed, expected)


def test_all_one_head_masks_leave_model_output_unchanged() -> None:
    config = mechanism_config()
    model = SparseAttentionPointerTransformer(config)
    batch = make_pointer_next_batch(
        4,
        5,
        generator=torch.Generator().manual_seed(43),
        vocabulary=model.vocabulary,
    )
    offsets = torch.zeros(4, dtype=torch.long)
    expected = model(batch.prompt_ids, offsets=offsets)

    with ablate_heads(
        model,
        {
            layer: torch.ones(config.heads)
            for layer in range(config.layers)
        },
    ):
        reconstructed = model(batch.prompt_ids, offsets=offsets)

    torch.testing.assert_close(reconstructed, expected)
