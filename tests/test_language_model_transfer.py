from __future__ import annotations

import json
from pathlib import Path

import torch

from list_sorting_transformer.language_model_transfer import (
    LanguageModelTransferConfig,
    build_language_model,
    evaluation_batch_size,
    learning_rate_at_step,
    load_byte_corpus,
    sample_byte_batch,
)


def _config(initialization: str, *, seed: int = 7) -> LanguageModelTransferConfig:
    return LanguageModelTransferConfig(
        initialization=initialization,
        seed=seed,
        steps=10,
        batch_size=2,
        sequence_length=8,
        train_bytes=32,
        validation_bytes=16,
        evaluation_batches=1,
        warmup_steps=2,
    )


def test_byte_corpus_uses_disjoint_deterministic_documents(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pile.jsonl"
    records = [
        {"text": "validation zero"},
        {"text": "training one"},
        {"text": "training two"},
        {"text": "validation three"},
        {"text": "training four"},
        {"text": "training five"},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    corpus = load_byte_corpus(
        path,
        train_bytes=40,
        validation_bytes=25,
        validation_document_stride=3,
    )

    assert corpus.train.numel() == 40
    assert corpus.validation.numel() == 25
    assert corpus.metadata["split_rule"] == (
        "document_index % 3 == 0 is validation"
    )
    assert corpus.metadata["train_sha256"] != corpus.metadata["validation_sha256"]
    assert path.read_text().startswith('{"text": "validation zero"}')


def test_compiled_middle_changes_only_middle_blocks() -> None:
    random = build_language_model(_config("random"))
    compiled = build_language_model(_config("compiled_middle"))

    random_state = random.state_dict()
    compiled_state = compiled.state_dict()
    changed = {
        name
        for name in random_state
        if not torch.equal(random_state[name], compiled_state[name])
    }

    assert changed
    assert all(
        name.startswith(("encoder.blocks.2.", "encoder.blocks.3."))
        for name in changed
    )
    assert any(name.startswith("encoder.blocks.2.") for name in changed)
    assert any(name.startswith("encoder.blocks.3.") for name in changed)


def test_frozen_compiled_middle_has_identical_weights_and_is_not_trainable() -> None:
    trainable = build_language_model(_config("compiled_middle"))
    frozen = build_language_model(_config("compiled_middle_frozen"))

    for name, parameter in frozen.named_parameters():
        assert torch.equal(
            parameter,
            dict(trainable.named_parameters())[name],
        )
        expected_trainable = not name.startswith(
            (
                "encoder.blocks.2.",
                "encoder.blocks.3.",
                "position_embedding.",
            )
        )
        assert parameter.requires_grad == expected_trainable


def test_language_model_is_causal_and_has_byte_logits() -> None:
    model = build_language_model(_config("compiled_middle"))
    model.eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    second = first.clone()
    second[0, -1] = 99

    with torch.inference_mode():
        first_logits = model(first)
        second_logits = model(second)

    assert first_logits.shape == (1, 8, 256)
    assert torch.allclose(first_logits[:, :-1], second_logits[:, :-1])


def test_sampled_byte_batches_are_next_token_shifted() -> None:
    tokens = torch.arange(64, dtype=torch.uint8)
    inputs, targets = sample_byte_batch(
        tokens,
        batch_size=4,
        sequence_length=8,
        generator=torch.Generator().manual_seed(11),
        device=torch.device("cpu"),
    )

    assert inputs.shape == targets.shape == (4, 8)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_learning_rate_warms_up_and_decays() -> None:
    config = _config("random")

    assert learning_rate_at_step(1, config) == config.learning_rate / 2
    assert learning_rate_at_step(2, config) == config.learning_rate
    assert learning_rate_at_step(config.steps, config) == (
        config.minimum_learning_rate
    )


def test_length_evaluation_keeps_target_byte_count_constant() -> None:
    config = LanguageModelTransferConfig(
        initialization="random",
        seed=7,
    )

    assert [
        evaluation_batch_size(config, length)
        for length in (256, 512, 1_024, 2_048)
    ] == [64, 32, 16, 8]
