from __future__ import annotations

import torch

from list_sorting_transformer.data import make_pointer_next_batch
from list_sorting_transformer.hard_attention_eggroll import (
    HardAttentionEggrollConfig,
    make_model,
)
from list_sorting_transformer.hard_attention_sweep import (
    checkpoint_axis_labels,
    exact_top1_logits,
    resolved_batch_size,
)
from list_sorting_transformer.positions import sample_position_offsets


def small_config() -> HardAttentionEggrollConfig:
    return HardAttentionEggrollConfig(
        run_name="test",
        generations=1,
        population_size=4,
        population_chunk_size=4,
        batch_size=3,
        train_min_length=2,
        train_max_length=4,
        eval_lengths=(2, 4),
        eval_examples=4,
        d_model=64,
        layers=2,
        heads=4,
        position_moduli=(3, 5, 7, 11),
        position_offset_min=-100,
        position_offset_max=100,
        log_interval=1,
        eval_interval=1,
        checkpoint_interval=1,
        device="cpu",
    )


def test_chunked_top1_matches_model_with_pruned_heads() -> None:
    config = small_config()
    model = make_model(config, device=torch.device("cpu"))
    model.set_active_head_indices(((1, 3), (0, 2)))
    model.eval()
    generator = torch.Generator().manual_seed(17)
    batch = make_pointer_next_batch(
        3,
        6,
        generator=generator,
        vocabulary=model.vocabulary,
    )
    offsets = sample_position_offsets(
        3,
        minimum=config.position_offset_min,
        maximum=config.position_offset_max,
        generator=generator,
        device=torch.device("cpu"),
    )

    expected = model(batch.prompt_ids, offsets=offsets)
    actual = exact_top1_logits(
        model,
        batch.prompt_ids,
        offsets,
        query_chunk_size=2,
    )

    torch.testing.assert_close(actual, expected)


def test_resolved_batch_size_obeys_score_budget() -> None:
    assert (
        resolved_batch_size(
            examples=64,
            sequence_length=5_000,
            active_heads=1,
            query_chunk_size=128,
            score_element_budget=32_000_000,
            maximum_batch_size=64,
        )
        == 50
    )


def test_checkpoint_labels_mark_head_pruning_transitions() -> None:
    rows = [
        {"generation": "8000", "attention_top_k": "1", "active_heads": "4"},
        {"generation": "8361", "attention_top_k": "1", "active_heads": "3"},
        {"generation": "9000", "attention_top_k": "1", "active_heads": "3"},
        {"generation": "10000", "attention_top_k": "1", "active_heads": "2"},
    ]

    generations, labels = checkpoint_axis_labels(rows)

    assert generations == [8000, 8361, 9000, 10000]
    assert labels == [
        "8,000  |  top-1, 4 heads",
        "8,361  |  pruned to 3 heads",
        "9,000",
        "10,000  |  pruned to 2 heads",
    ]
