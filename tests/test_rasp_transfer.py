from __future__ import annotations

import torch

from list_sorting_transformer.data import make_pointer_pair_batch
from list_sorting_transformer.rasp_transfer import (
    ROUND_CLOSE,
    ROUND_OPEN,
    SQUARE_CLOSE,
    SQUARE_OPEN,
    RaspTransferConfig,
    build_transfer_model,
    make_associative_recall_batch,
    make_dyck_2_completion_batch,
    task_targets,
)
from list_sorting_transformer.tokens import VALUE_OFFSET, PointerCompareVocabulary


def test_transfer_targets_use_the_marked_pair() -> None:
    vocabulary = PointerCompareVocabulary("numbers", 10)
    batch = make_pointer_pair_batch(
        32,
        8,
        generator=torch.Generator().manual_seed(3),
        vocabulary=vocabulary,
    )
    rows = torch.arange(batch.values.shape[0])
    marked = batch.values[rows, batch.pointers]
    following = batch.values[rows, batch.pointers + 1]

    assert torch.equal(
        task_targets("following_value", batch),
        following,
    )
    expected_relation = torch.where(
        marked < following,
        torch.zeros_like(marked),
        torch.where(
            marked == following,
            torch.ones_like(marked),
            torch.full_like(marked, 2),
        ),
    )
    assert torch.equal(
        task_targets("three_way_relation", batch),
        expected_relation,
    )


def test_dyck_2_completion_has_no_pointer_and_closes_the_prefix() -> None:
    vocabulary = PointerCompareVocabulary("numbers", 10)
    batch = make_dyck_2_completion_batch(
        32,
        20,
        generator=torch.Generator().manual_seed(13),
        device=torch.device("cpu"),
    )
    pointer_token = vocabulary.marker_token("PTR")

    assert batch.prompt_ids.shape == (32, 46)
    assert not batch.prompt_ids.eq(pointer_token).any()
    for prompt, target in zip(batch.prompt_ids, batch.targets):
        stack: list[int] = []
        for token in prompt[1:-1].tolist():
            if token in (ROUND_OPEN, SQUARE_OPEN):
                stack.append(token)
            elif token == ROUND_CLOSE:
                assert stack.pop() == ROUND_OPEN
            elif token == SQUARE_CLOSE:
                assert stack.pop() == SQUARE_OPEN
        expected_open = ROUND_OPEN if int(target) == 0 else SQUARE_OPEN
        assert len(stack) == 4
        assert stack[-1] == expected_open


def test_associative_recall_retrieves_the_queried_mapping() -> None:
    vocabulary = PointerCompareVocabulary("numbers", 10)
    batch = make_associative_recall_batch(
        32,
        20,
        generator=torch.Generator().manual_seed(14),
        device=torch.device("cpu"),
    )
    pointer_token = vocabulary.marker_token("PTR")

    assert batch.prompt_ids.shape == (32, 43)
    assert not batch.prompt_ids.eq(pointer_token).any()
    keys = batch.prompt_ids[:, 1:-2:2] - VALUE_OFFSET
    values = batch.prompt_ids[:, 2:-2:2] - VALUE_OFFSET
    queries = batch.prompt_ids[:, -2] - VALUE_OFFSET
    for row in range(batch.prompt_ids.shape[0]):
        matching = keys[row].eq(queries[row])
        assert matching.any()
        assert values[row, matching].eq(batch.targets[row]).all()


def test_compiled_initializations_preserve_exact_routing() -> None:
    vocabulary = PointerCompareVocabulary("numbers", 10)
    batch = make_pointer_pair_batch(
        64,
        40,
        generator=torch.Generator().manual_seed(5),
        vocabulary=vocabulary,
    )
    offsets = torch.randint(
        -1_000_000,
        1_000_001,
        (64,),
        generator=torch.Generator().manual_seed(6),
    )
    pointer_offsets = 1 + 2 * batch.pointers

    for initialization in ("compiled_prefix", "compiled_full"):
        model = build_transfer_model(
            RaspTransferConfig(
                task="following_value",
                initialization=initialization,
                seed=7,
                steps=1,
                batch_size=4,
                eval_examples=4,
            )
        )
        model.eval()
        with torch.inference_mode():
            outputs = model(
                batch.prompt_ids,
                offsets=offsets,
                diagnostics=True,
            )
        assert outputs["pointer_route_logits"].argmax(-1).eq(
            pointer_offsets
        ).all()
        assert outputs["marked_route_logits"].argmax(-1).eq(
            pointer_offsets + 1
        ).all()
        assert outputs["following_route_logits"].argmax(-1).eq(
            pointer_offsets + 3
        ).all()


def test_transfer_model_has_a_fresh_trainable_head() -> None:
    config = RaspTransferConfig(
        task="three_way_relation",
        initialization="compiled_full",
        seed=11,
        steps=1,
        batch_size=8,
        eval_examples=8,
    )
    model = build_transfer_model(config)
    vocabulary = PointerCompareVocabulary("numbers", 10)
    batch = make_pointer_pair_batch(
        8,
        5,
        generator=torch.Generator().manual_seed(12),
        vocabulary=vocabulary,
    )
    offsets = torch.arange(8)
    logits = model(batch.prompt_ids, offsets=offsets)
    loss = torch.nn.functional.cross_entropy(
        logits,
        task_targets(config.task, batch),
    )
    loss.backward()

    assert logits.shape == (8, 3)
    assert model.output_head.weight.grad is not None
    assert model.output_head.weight.grad.abs().sum() > 0
