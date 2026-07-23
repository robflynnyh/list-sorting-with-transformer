from __future__ import annotations

import torch

from list_sorting_transformer.data import make_sorting_batch
from list_sorting_transformer.evaluation import output_cross_entropy
from list_sorting_transformer.model import DecoderTransformer, ModelConfig
from list_sorting_transformer.recurrent import LSTMConfig, LSTMSorter
from list_sorting_transformer.tokens import SymbolVocabulary


def small_config(
    representation: str = "numbers",
    *,
    rotate_values_with_rope: bool = False,
) -> ModelConfig:
    vocabulary = SymbolVocabulary(representation, 10)
    return ModelConfig(
        vocab_size=vocabulary.size,
        representation=representation,
        symbol_count=10,
        d_model=32,
        n_layers=4,
        n_heads=4,
        ffn_multiplier=2.0,
        rotate_values_with_rope=rotate_values_with_rope,
    )


def test_default_layers_interleave_rotary_and_nope() -> None:
    model = DecoderTransformer(small_config())
    assert model.layer_position_modes == ("rotary", "none", "rotary", "none")


def test_value_rotary_mode_is_reported_for_rotary_layers() -> None:
    model = DecoderTransformer(small_config(rotate_values_with_rope=True))
    assert model.layer_position_modes == (
        "rotary+value",
        "none",
        "rotary+value",
        "none",
    )


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


def test_transformer_cache_matches_full_prefix_logits() -> None:
    torch.manual_seed(13)
    model = DecoderTransformer(small_config()).eval()
    prompt = torch.randint(0, model.config.vocab_size, (2, 9))
    next_token = torch.randint(0, model.config.vocab_size, (2, 1))

    prompt_logits, caches = model.forward_with_cache(prompt)
    full_prompt_logits = model(prompt)
    torch.testing.assert_close(prompt_logits, full_prompt_logits)

    cached_logits, caches = model.forward_with_cache(
        next_token,
        caches=caches,
    )
    full_logits = model(torch.cat((prompt, next_token), dim=1))
    torch.testing.assert_close(
        cached_logits[:, -1],
        full_logits[:, -1],
        atol=1e-5,
        rtol=1e-4,
    )
    assert all(cache[0].shape[-2] == 10 for cache in caches)


def test_transformer_cache_matches_full_prefix_logits_with_value_rope() -> None:
    torch.manual_seed(14)
    model = DecoderTransformer(
        small_config(rotate_values_with_rope=True)
    ).eval()
    prompt = torch.randint(0, model.config.vocab_size, (2, 9))
    next_token = torch.randint(0, model.config.vocab_size, (2, 1))

    prompt_logits, caches = model.forward_with_cache(prompt)
    full_prompt_logits = model(prompt)
    torch.testing.assert_close(prompt_logits, full_prompt_logits)

    cached_logits, caches = model.forward_with_cache(
        next_token,
        caches=caches,
    )
    full_logits = model(torch.cat((prompt, next_token), dim=1))
    torch.testing.assert_close(
        cached_logits[:, -1],
        full_logits[:, -1],
        atol=1e-5,
        rtol=1e-4,
    )
    assert all(cache[1].shape[-2] == 10 for cache in caches)


def test_transformer_cached_generation_matches_full_prefix_generation() -> None:
    torch.manual_seed(17)
    model = DecoderTransformer(small_config()).eval()
    prompt = torch.randint(0, model.config.vocab_size, (2, 7))

    generated = model.generate(prompt, max_new_tokens=8)
    full_sequence = prompt
    reference_tokens = []
    for _ in range(8):
        next_token = model(full_sequence)[:, -1].argmax(dim=-1)
        reference_tokens.append(next_token)
        full_sequence = torch.cat((full_sequence, next_token[:, None]), dim=1)
    reference = torch.stack(reference_tokens, dim=1)

    torch.testing.assert_close(generated, reference)


def test_transformer_generation_accepts_a_task_specific_stop_token() -> None:
    torch.manual_seed(19)
    model = DecoderTransformer(small_config()).eval()
    prompt = torch.randint(0, model.config.vocab_size, (1, 7))
    first_token = int(model(prompt)[:, -1].argmax(dim=-1).item())

    generated = model.generate(
        prompt,
        max_new_tokens=8,
        stop_token=first_token,
    )

    assert generated.shape == (1, 1)
    assert int(generated[0, 0]) == first_token


def test_lstm_baseline_backpropagates_and_generates_with_cached_state() -> None:
    vocabulary = SymbolVocabulary("alphabet", 10)
    model = LSTMSorter(
        LSTMConfig(
            vocab_size=vocabulary.size,
            representation="alphabet",
            d_model=24,
            hidden_size=32,
            n_layers=2,
        )
    )
    batch = make_sorting_batch(
        4,
        5,
        generator=torch.Generator().manual_seed(11),
    )
    logits = model(batch.model_inputs)
    loss = output_cross_entropy(logits, batch.labels)
    loss.backward()
    generated = model.generate(batch.prompt_ids, max_new_tokens=12)

    assert logits.shape == (4, 20, vocabulary.size)
    assert 1 <= generated.shape[1] <= 12
    assert any(parameter.grad is not None for parameter in model.parameters())
