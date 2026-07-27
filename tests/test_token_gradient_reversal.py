from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from list_sorting_transformer.model import (
    ModelConfig,
    source_reversed_scaled_dot_product_attention,
)
from list_sorting_transformer.shortcut_credit import (
    ShortcutDecoderTransformer,
    ShortcutPointerVocabulary,
    make_shortcut_batch,
    shortcut_loss,
)
from list_sorting_transformer.token_gradient_reversal import (
    forward_with_source_gradient_reversal,
    oracle_reversal_shortcut_loss,
    oracle_shortcut_selection,
    source_gradient_multipliers,
)


def test_source_reversal_preserves_forward_and_unit_gradients() -> None:
    generator = torch.Generator().manual_seed(12)
    ordinary_tensors = [
        torch.randn(
            2,
            2,
            4,
            6,
            generator=generator,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    reversed_tensors = [
        tensor.detach().clone().requires_grad_()
        for tensor in ordinary_tensors
    ]
    output_gradient = torch.randn(
        2,
        2,
        4,
        6,
        generator=generator,
    )

    ordinary = F.scaled_dot_product_attention(
        *ordinary_tensors,
        dropout_p=0.0,
        is_causal=True,
    )
    routed, _ = source_reversed_scaled_dot_product_attention(
        *reversed_tensors,
        source_multipliers=torch.ones(2, 4),
        is_causal=True,
    )
    torch.testing.assert_close(routed, ordinary, rtol=0, atol=0)

    ordinary.backward(output_gradient)
    routed.backward(output_gradient)
    for ordinary_tensor, reversed_tensor in zip(
        ordinary_tensors,
        reversed_tensors,
    ):
        torch.testing.assert_close(
            reversed_tensor.grad,
            ordinary_tensor.grad,
            rtol=1e-5,
            atol=1e-6,
        )


def test_source_reversal_negates_selected_value_credit_only() -> None:
    generator = torch.Generator().manual_seed(19)
    query = torch.randn(
        1, 1, 3, 4, generator=generator, requires_grad=True
    )
    key = torch.randn(
        1, 1, 3, 4, generator=generator, requires_grad=True
    )
    ordinary_value = torch.randn(
        1, 1, 3, 4, generator=generator, requires_grad=True
    )
    reversed_value = ordinary_value.detach().clone().requires_grad_()
    output_gradient = torch.randn(
        1, 1, 3, 4, generator=generator
    )

    ordinary = F.scaled_dot_product_attention(
        query,
        key,
        ordinary_value,
        dropout_p=0.0,
        is_causal=True,
    )
    reversed_output, _ = source_reversed_scaled_dot_product_attention(
        query.detach().clone().requires_grad_(),
        key.detach().clone().requires_grad_(),
        reversed_value,
        source_multipliers=torch.tensor([[1.0, -1.0, 1.0]]),
        is_causal=True,
    )
    ordinary.backward(output_gradient)
    reversed_output.backward(output_gradient)

    torch.testing.assert_close(
        reversed_value.grad[:, :, 1],
        -ordinary_value.grad[:, :, 1],
    )
    torch.testing.assert_close(
        reversed_value.grad[:, :, (0, 2), :],
        ordinary_value.grad[:, :, (0, 2), :],
    )


def test_score_only_reversal_keeps_ordinary_value_credit() -> None:
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(
        1, 1, 3, 4, generator=generator, requires_grad=True
    )
    key = torch.randn(
        1, 1, 3, 4, generator=generator, requires_grad=True
    )
    ordinary_value = torch.randn(
        1, 1, 3, 4, generator=generator, requires_grad=True
    )
    reversed_value = ordinary_value.detach().clone().requires_grad_()
    output_gradient = torch.randn(1, 1, 3, 4, generator=generator)

    ordinary = F.scaled_dot_product_attention(
        query,
        key,
        ordinary_value,
        dropout_p=0.0,
        is_causal=True,
    )
    reversed_output, _ = source_reversed_scaled_dot_product_attention(
        query.detach().clone().requires_grad_(),
        key.detach().clone().requires_grad_(),
        reversed_value,
        source_multipliers=torch.tensor([[1.0, -1.0, 1.0]]),
        reverse_value_credit=False,
        is_causal=True,
    )
    ordinary.backward(output_gradient)
    reversed_output.backward(output_gradient)

    torch.testing.assert_close(reversed_value.grad, ordinary_value.grad)


def test_attention_penalty_matches_explicit_selected_mass_loss() -> None:
    generator = torch.Generator().manual_seed(29)
    explicit_query = torch.randn(
        1, 2, 3, 4, generator=generator, requires_grad=True
    )
    explicit_key = torch.randn(
        1, 2, 3, 4, generator=generator, requires_grad=True
    )
    value = torch.randn(1, 2, 3, 4, generator=generator)
    routed_query = explicit_query.detach().clone().requires_grad_()
    routed_key = explicit_key.detach().clone().requires_grad_()
    scale = explicit_query.shape[-1] ** -0.5
    causal = torch.ones(3, 3, dtype=torch.bool).tril()
    scores = (
        explicit_query @ explicit_key.transpose(-2, -1) * scale
    ).masked_fill(~causal, float("-inf"))
    weights = scores.softmax(dim=-1)
    explicit_loss = weights[..., 1].mean()

    routed, _ = source_reversed_scaled_dot_product_attention(
        routed_query,
        routed_key,
        value,
        source_multipliers=torch.tensor([[1.0, -1.0, 1.0]]),
        reverse_score_credit=False,
        reverse_value_credit=False,
        attention_penalty_strength=1.0,
        is_causal=True,
    )
    explicit_loss.backward()
    (routed * 0).sum().backward()

    torch.testing.assert_close(routed_query.grad, explicit_query.grad)
    torch.testing.assert_close(routed_key.grad, explicit_key.grad)


@pytest.mark.parametrize("leak_placement", ("suffix", "random_list"))
def test_oracle_selects_answer_after_leak(leak_placement: str) -> None:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    batch = make_shortcut_batch(
        8,
        6,
        leak_mode="correct",
        leak_placement=leak_placement,
        generator=torch.Generator().manual_seed(3),
        vocabulary=vocabulary,
    )

    selection = oracle_shortcut_selection(batch.input_ids, vocabulary)

    assert selection.sum(dim=1).eq(1).all()
    for row in range(batch.batch_size):
        selected = int(selection[row].to(torch.long).argmax())
        assert batch.input_ids[row, selected - 1] == vocabulary.leak_token
        assert batch.input_ids[row, selected] == batch.targets[row]


def test_source_gradient_multiplier_validation() -> None:
    with pytest.raises(ValueError):
        source_gradient_multipliers(torch.ones(2, 3))
    with pytest.raises(ValueError):
        source_gradient_multipliers(
            torch.ones(2, 3, dtype=torch.bool),
            reversal_scale=0,
        )


def test_oracle_reversal_preserves_model_forward_but_changes_gradient() -> None:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    config = ModelConfig(
        vocab_size=vocabulary.size,
        symbol_count=10,
        d_model=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
    )
    torch.manual_seed(4)
    ordinary = ShortcutDecoderTransformer(config)
    reversed_model = deepcopy(ordinary)
    batch = make_shortcut_batch(
        8,
        6,
        leak_mode="correct",
        generator=torch.Generator().manual_seed(5),
        vocabulary=vocabulary,
    )

    ordinary_logits = ordinary(batch.input_ids)
    selection = oracle_shortcut_selection(batch.input_ids, vocabulary)
    reversed_logits = forward_with_source_gradient_reversal(
        reversed_model,
        batch.input_ids,
        selection,
    )
    torch.testing.assert_close(
        reversed_logits,
        ordinary_logits,
        rtol=0,
        atol=0,
    )

    shortcut_loss(ordinary, batch).backward()
    oracle_reversal_shortcut_loss(
        reversed_model,
        batch,
        vocabulary,
    ).backward()
    assert not torch.equal(
        ordinary.blocks[0].attention.qkv.weight.grad,
        reversed_model.blocks[0].attention.qkv.weight.grad,
    )


def test_reversal_scope_validation() -> None:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    model = ShortcutDecoderTransformer(
        ModelConfig(
            vocab_size=vocabulary.size,
            symbol_count=10,
            d_model=16,
            n_layers=1,
            n_heads=2,
        )
    )
    token_ids = torch.tensor(
        [[vocabulary.leak_token, vocabulary.value_token(2)]]
    )
    with pytest.raises(ValueError):
        forward_with_source_gradient_reversal(
            model,
            token_ids,
            torch.tensor([[False, True]]),
            reversal_scope="unknown",
        )
    with pytest.raises(ValueError):
        forward_with_source_gradient_reversal(
            model,
            token_ids,
            torch.tensor([[False, True]]),
            reversal_scope="attention_penalty",
        )
