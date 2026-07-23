from __future__ import annotations

import torch

from list_sorting_transformer.data import IGNORE_INDEX, make_sorting_batch
from list_sorting_transformer.tokens import (
    BOS,
    COMMA,
    EOS,
    SEP,
    SymbolVocabulary,
)


def test_number_and_alphabet_vocabularies_share_structure() -> None:
    numbers = SymbolVocabulary("numbers", 10)
    alphabet = SymbolVocabulary("alphabet", 10)
    values = [8, 2, 5, 2]

    assert numbers.encode_example(values) == alphabet.encode_example(values)
    assert numbers.render_tokens(numbers.encode_example(values)) == (
        "<bos>8,2,5,2=2,2,5,8<eos>"
    )
    assert alphabet.render_tokens(alphabet.encode_example(values)) == (
        "<bos>i,c,f,c=c,c,f,i<eos>"
    )


def test_decode_requires_exact_comma_grammar_and_eos() -> None:
    vocabulary = SymbolVocabulary()
    assert vocabulary.decode_list(vocabulary.encode_target([1, 1, 7])) == [1, 1, 7]
    assert vocabulary.decode_list(
        [vocabulary.value_token(1), vocabulary.value_token(7), EOS]
    ) is None
    assert vocabulary.decode_list(
        [vocabulary.value_token(1), COMMA, EOS]
    ) is None
    assert vocabulary.decode_list([vocabulary.value_token(1)]) is None


def test_online_batch_contains_prompt_sorted_target_and_output_only_labels() -> None:
    vocabulary = SymbolVocabulary()
    batch = make_sorting_batch(
        8,
        5,
        generator=torch.Generator().manual_seed(3),
    )
    separator_index = 10

    assert batch.token_ids.shape == (8, 21)
    assert batch.model_inputs.shape == (8, 20)
    assert batch.prompt_ids.shape == (8, 11)
    assert torch.all(batch.token_ids[:, 0].eq(BOS))
    assert torch.all(batch.token_ids[:, separator_index].eq(SEP))
    assert torch.all(batch.token_ids[:, -1].eq(EOS))
    assert torch.all(batch.labels[:, :separator_index].eq(IGNORE_INDEX))
    assert torch.all(batch.labels[:, separator_index].ne(IGNORE_INDEX))

    for values, tokens in zip(batch.values.tolist(), batch.token_ids.tolist()):
        assert tokens == vocabulary.encode_example(values)
