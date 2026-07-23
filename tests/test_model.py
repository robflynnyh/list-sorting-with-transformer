from __future__ import annotations

import torch

from list_sorting_transformer.data import make_sorting_batch
from list_sorting_transformer.evaluation import output_cross_entropy
from list_sorting_transformer.model import DecoderTransformer, ModelConfig
from list_sorting_transformer.tokens import SymbolVocabulary


def small_config(representation: str = "numbers") -> ModelConfig:
    vocabulary = SymbolVocabulary(representation, 10)
    return ModelConfig(
        vocab_size=vocabulary.size,
        representation=representation,
        symbol_count=10,
        d_model=32,
        n_layers=4,
        n_heads=4,
        ffn_multiplier=2.0,
    )


def test_default_layers_interleave_rotary_and_nope() -> None:
    model = DecoderTransformer(small_config())
    assert model.layer_position_modes == ("rotary", "none", "rotary", "none")


def test_causal_mask_prevents_future_tokens_changing_prefix_logits() -> None:
    torch.manual_seed(4)
    model = DecoderTransformer(small_config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 12))
    changed = tokens.clone()
    changed[:, 7:] = torch.randint(0, model.config.vocab_size, (2, 5))

    original_logits = model(tokens)
    changed_logits = model(changed)

    torch.testing.assert_close(original_logits[:, :7], changed_logits[:, :7])


def test_number_mode_adds_ordered_scalar_feature_but_alphabet_does_not() -> None:
    number_model = DecoderTransformer(small_config("numbers"))
    alphabet_model = DecoderTransformer(small_config("alphabet"))
    token_ids = torch.tensor([[5, 9, 14]])

    assert number_model.number_projection is not None
    assert alphabet_model.number_projection is None
    assert not torch.equal(
        number_model.embed(token_ids),
        number_model.token_embedding(token_ids),
    )
    torch.testing.assert_close(
        alphabet_model.embed(token_ids),
        alphabet_model.token_embedding(token_ids),
    )


def test_model_backpropagates_output_loss_and_accepts_longer_sequences() -> None:
    model = DecoderTransformer(small_config())
    batch = make_sorting_batch(
        4,
        6,
        generator=torch.Generator().manual_seed(8),
    )
    logits = model(batch.model_inputs)
    loss = output_cross_entropy(logits, batch.labels)
    loss.backward()

    assert logits.shape == (4, 24, model.config.vocab_size)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
    long_logits = model(torch.randint(0, model.config.vocab_size, (2, 96)))
    assert long_logits.shape == (2, 96, model.config.vocab_size)
