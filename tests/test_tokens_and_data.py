from __future__ import annotations

import torch

from list_sorting_transformer.data import (
    IGNORE_INDEX,
    make_pointer_next_batch,
    make_pointer_value_batch,
    make_sorting_batch,
)
from list_sorting_transformer.metrics import (
    generated_pointer_next_metrics,
    generated_pointer_value_metrics,
)
from list_sorting_transformer.tokens import (
    BOS,
    COMMA,
    EOS,
    PointerNextVocabulary,
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


def test_pointer_next_prompt_marks_pointer_and_targets_next_value() -> None:
    vocabulary = PointerNextVocabulary()
    example = vocabulary.encode_example_with_pointer([7, 4, 2], 1)

    assert vocabulary.render_tokens(example) == "<bos>7,<PTR>4,2=2<eos>"
    assert example[-2:] == [vocabulary.value_token(2), EOS]


def test_pointer_value_prompt_marks_pointer_and_targets_marked_value() -> None:
    vocabulary = PointerNextVocabulary()
    example = vocabulary.encode_value_example_with_pointer([7, 4, 2], 1)

    assert vocabulary.render_tokens(example) == "<bos>7,<PTR>4,2=4<eos>"
    assert example[-2:] == [vocabulary.value_token(4), EOS]


def test_pointer_next_batch_masks_prompt_and_samples_valid_pointers() -> None:
    vocabulary = PointerNextVocabulary()
    batch = make_pointer_next_batch(
        16,
        5,
        generator=torch.Generator().manual_seed(13),
        vocabulary=vocabulary,
    )

    assert batch.token_ids.shape == (16, 14)
    assert batch.model_inputs.shape == (16, 13)
    assert batch.prompt_ids.shape == (16, 12)
    assert torch.all(batch.token_ids[:, 0].eq(BOS))
    assert torch.all(batch.token_ids[:, batch.prompt_length - 1].eq(SEP))
    assert torch.all(batch.token_ids[:, -1].eq(EOS))
    assert torch.all(batch.pointers.ge(0))
    assert torch.all(batch.pointers.lt(4))
    assert torch.all(batch.labels[:, : batch.prompt_length - 1].eq(IGNORE_INDEX))
    assert torch.all(batch.labels[:, batch.prompt_length - 1].ne(IGNORE_INDEX))

    for row, pointer, tokens in zip(
        batch.values.tolist(),
        batch.pointers.tolist(),
        batch.token_ids.tolist(),
    ):
        assert tokens == vocabulary.encode_example_with_pointer(row, pointer)


def test_pointer_value_batch_can_mark_final_value() -> None:
    vocabulary = PointerNextVocabulary()
    batch = make_pointer_value_batch(
        64,
        5,
        generator=torch.Generator().manual_seed(17),
        vocabulary=vocabulary,
    )

    assert torch.all(batch.pointers.ge(0))
    assert torch.all(batch.pointers.lt(5))
    assert int(batch.pointers.max()) == 4

    for row, pointer, tokens in zip(
        batch.values.tolist(),
        batch.pointers.tolist(),
        batch.token_ids.tolist(),
    ):
        assert tokens == vocabulary.encode_value_example_with_pointer(row, pointer)


def test_pointer_next_metrics_split_unseen_pointer_positions() -> None:
    vocabulary = PointerNextVocabulary()
    values = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    pointers = torch.tensor([0, 2])
    generated = torch.tensor(
        [
            [vocabulary.value_token(2), EOS],
            [vocabulary.value_token(0), EOS],
        ]
    )

    metrics = generated_pointer_next_metrics(
        values,
        pointers,
        generated,
        vocabulary,
        train_max_pointer_index=1,
    )

    assert metrics["exact_match"] == 0.5
    assert metrics["seen_pointer_exact_match"] == 1.0
    assert metrics["unseen_pointer_exact_match"] == 0.0
    assert metrics["unseen_pointer_fraction"] == 0.5


def test_pointer_value_metrics_score_marked_value_not_next_value() -> None:
    vocabulary = PointerNextVocabulary()
    values = torch.tensor([[7, 4, 2], [1, 3, 9]])
    pointers = torch.tensor([1, 2])
    generated = torch.tensor(
        [
            [vocabulary.value_token(4), EOS],
            [vocabulary.value_token(9), EOS],
        ]
    )

    metrics = generated_pointer_value_metrics(
        values,
        pointers,
        generated,
        vocabulary,
        train_max_pointer_index=1,
    )

    assert metrics["exact_match"] == 1.0
    assert metrics["value_accuracy"] == 1.0
    assert metrics["seen_pointer_exact_match"] == 1.0
    assert metrics["unseen_pointer_exact_match"] == 1.0
