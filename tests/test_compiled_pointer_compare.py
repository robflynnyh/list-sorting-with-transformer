from __future__ import annotations

import torch

from list_sorting_transformer.length_generalisation.compiled_pointer_compare import (
    CompiledPointerCompareConfig,
    CompiledPointerCompareTransformer,
    evaluate_compiled_model,
)
from list_sorting_transformer.core.model import SplitInputDecoderTransformer
from list_sorting_transformer.core.tokens import PointerCompareVocabulary


def compiled_model() -> CompiledPointerCompareTransformer:
    return CompiledPointerCompareTransformer(
        CompiledPointerCompareConfig()
    )


def test_compiler_targets_existing_pipeline_architecture() -> None:
    model = compiled_model()

    assert isinstance(model.encoder, SplitInputDecoderTransformer)
    assert model.encoder.config.d_model == 128
    assert model.encoder.config.n_layers == 4
    assert model.encoder.config.n_heads == 4
    assert model.encoder.content_dim == 64
    assert model.position_embedding.moduli == (
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
    )
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_compiled_routing_and_actions_are_exact_with_random_offsets() -> None:
    model = compiled_model()
    results = evaluate_compiled_model(
        model,
        lengths=(2, 11, 40),
        examples=32,
        batch_size=8,
        seed=11,
        offset_min=-1_000_000,
        offset_max=1_000_000,
        device=torch.device("cpu"),
    )

    for metrics in results.values():
        assert metrics["pointer_route_accuracy"] == 1.0
        assert metrics["marked_route_accuracy"] == 1.0
        assert metrics["following_route_accuracy"] == 1.0
        assert metrics["action_accuracy"] == 1.0
        assert metrics["minimum_pointer_route_logit_margin"] > 0
        assert metrics["minimum_marked_route_logit_margin"] > 0
        assert metrics["minimum_following_route_logit_margin"] > 0
        assert metrics["minimum_action_margin"] > 0


def test_compiled_actions_cover_all_digit_pairs() -> None:
    model = compiled_model()
    vocabulary = PointerCompareVocabulary("numbers", 10)
    pairs = [(left, right) for left in range(10) for right in range(10)]
    prompt_ids = torch.tensor(
        [
            vocabulary.encode_prompt_with_pointer(pair, 0)
            for pair in pairs
        ]
    )
    offsets = torch.linspace(
        -1_000_000,
        1_000_000,
        len(pairs),
        dtype=torch.long,
    )
    predictions = model(prompt_ids, offsets=offsets).argmax(-1)
    expected = torch.tensor(
        [int(left > right) for left, right in pairs]
    )

    torch.testing.assert_close(predictions, expected)
