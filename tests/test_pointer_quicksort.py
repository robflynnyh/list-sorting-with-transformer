from __future__ import annotations

import torch

from list_sorting_transformer.data import (
    IGNORE_INDEX,
    make_pointer_quicksort_batch,
)
from list_sorting_transformer.evaluation import (
    generate_pointer_quicksort_rollouts,
)
from list_sorting_transformer.metrics import (
    generated_pointer_no_tool_metrics,
    generated_pointer_quicksort_metrics,
)
from list_sorting_transformer.pointer_quicksort import (
    PointerQuicksortMachine,
    execute_pointer_quicksort_actions,
    generate_pointer_quicksort_trace,
    replay_pointer_quicksort_transcript,
)
from list_sorting_transformer.tokens import (
    VALUE_OFFSET,
    PointerQuicksortVocabulary,
)


class ScriptedPointerPolicy:
    def __init__(
        self,
        action_rows: tuple[tuple[int, ...], ...],
        vocab_size: int,
    ) -> None:
        self.action_rows = action_rows
        self.vocab_size = vocab_size

    def forward_with_state(
        self,
        token_ids: torch.Tensor,
        state: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        action_index = 0 if state is None else (state + 1) // 2
        logits = torch.full(
            (*token_ids.shape, self.vocab_size),
            -1_000.0,
            device=token_ids.device,
        )
        for row_index, actions in enumerate(self.action_rows):
            selected_index = min(action_index, len(actions) - 1)
            logits[row_index, -1, actions[selected_index]] = 1_000.0
        return logits, 0 if state is None else state + 1


def test_pointer_trace_sorts_without_position_tokens() -> None:
    vocabulary = PointerQuicksortVocabulary()
    trace = generate_pointer_quicksort_trace([3, 1, 2], vocabulary)

    assert trace.final_values == (1, 2, 3)
    assert len(trace.target_tokens) == 2 * len(trace.action_tokens) - 1
    assert trace.target_prediction_mask[-1]
    assert all(
        token in vocabulary.action_tokens
        for token, predicted in zip(
            trace.target_tokens,
            trace.target_prediction_mask,
        )
        if predicted
    )
    assert all(
        token not in vocabulary.action_tokens
        for token, predicted in zip(
            trace.target_tokens,
            trace.target_prediction_mask,
        )
        if not predicted
    )
    assert [vocabulary.action_name(token) for token in trace.action_tokens[:4]] == [
        "INIT_RANGE",
        "CHECK_RANGE",
        "LOAD_PIVOT_LO",
        "SET_LT_LO",
    ]


def test_pointer_machine_sorts_duplicate_heavy_random_lists() -> None:
    vocabulary = PointerQuicksortVocabulary(symbol_count=5)
    generator = torch.Generator().manual_seed(29)

    for length in range(1, 41):
        values = torch.randint(0, 5, (length,), generator=generator).tolist()
        trace = generate_pointer_quicksort_trace(values, vocabulary)
        assert list(trace.final_values) == sorted(values)


def test_pointer_machine_rejects_an_action_from_the_wrong_phase() -> None:
    vocabulary = PointerQuicksortVocabulary()
    machine = PointerQuicksortMachine([2, 0, 1], vocabulary)

    observation = machine.step(vocabulary.action_token("GET_SCAN"))

    assert observation == vocabulary.observation_token("INVALID")
    assert machine.finished
    assert not machine.valid
    assert machine.last_error == "expected INIT_RANGE, received GET_SCAN"


def test_pointer_batch_masks_prompts_observations_and_padding() -> None:
    vocabulary = PointerQuicksortVocabulary()
    batch = make_pointer_quicksort_batch(
        8,
        7,
        generator=torch.Generator().manual_seed(3),
        vocabulary=vocabulary,
    )

    assert batch.prompt_length == 15
    assert torch.all(batch.labels[:, : batch.prompt_length - 1].eq(IGNORE_INDEX))
    for row_index, trace in enumerate(batch.traces):
        for target_index, (token, predicted) in enumerate(
            zip(trace.target_tokens, trace.target_prediction_mask)
        ):
            label_index = batch.prompt_length - 1 + target_index
            expected = token if predicted else IGNORE_INDEX
            assert int(batch.labels[row_index, label_index]) == expected
    included = batch.labels.ne(IGNORE_INDEX)
    assert torch.all(
        batch.labels[included].ge(VALUE_OFFSET + vocabulary.symbol_count)
    )


def test_no_tool_batch_supervises_actions_and_observations() -> None:
    vocabulary = PointerQuicksortVocabulary()
    batch = make_pointer_quicksort_batch(
        4,
        6,
        generator=torch.Generator().manual_seed(5),
        vocabulary=vocabulary,
        supervise_observations=True,
    )

    for row_index, trace in enumerate(batch.traces):
        target_start = batch.prompt_length - 1
        expected = torch.tensor(trace.target_tokens)
        labels = batch.labels[
            row_index,
            target_start : target_start + len(trace.target_tokens),
        ]
        torch.testing.assert_close(labels, expected)


def test_scripted_policy_completes_interactive_cached_execution() -> None:
    vocabulary = PointerQuicksortVocabulary()
    batch = make_pointer_quicksort_batch(
        4,
        6,
        generator=torch.Generator().manual_seed(11),
        vocabulary=vocabulary,
    )
    model = ScriptedPointerPolicy(
        tuple(trace.action_tokens for trace in batch.traces),
        vocabulary.size,
    )

    rollouts = generate_pointer_quicksort_rollouts(
        model,  # type: ignore[arg-type]
        batch,
        vocabulary,
    )

    assert all(rollout.completed for rollout in rollouts)
    assert all(rollout.valid_execution for rollout in rollouts)
    assert all(
        list(rollout.final_values) == sorted(values)
        for rollout, values in zip(rollouts, batch.values.tolist())
    )


def test_pointer_metrics_require_valid_completion() -> None:
    vocabulary = PointerQuicksortVocabulary()
    values = torch.tensor([[3, 1, 2]])
    trace = generate_pointer_quicksort_trace(values[0].tolist(), vocabulary)
    perfect = execute_pointer_quicksort_actions(
        values[0].tolist(),
        trace.action_tokens,
        vocabulary,
    )

    metrics = generated_pointer_quicksort_metrics(
        values,
        [perfect],
        vocabulary,
        [trace],
    )
    assert metrics["exact_match"] == 1.0
    assert metrics["trace_exact_match"] == 1.0
    assert metrics["operation_prefix_fraction"] == 1.0

    invalid = execute_pointer_quicksort_actions(
        values[0].tolist(),
        [vocabulary.action_token("GET_SCAN")],
        vocabulary,
    )
    metrics = generated_pointer_quicksort_metrics(
        values,
        [invalid],
        vocabulary,
        [trace],
    )
    assert metrics["exact_match"] == 0.0
    assert metrics["execution_completed"] == 0.0
    assert metrics["operation_prefix_fraction"] == 0.0


def test_no_tool_replay_separates_execution_from_observation_accuracy() -> None:
    vocabulary = PointerQuicksortVocabulary()
    values = torch.tensor([[3, 1, 2]])
    trace = generate_pointer_quicksort_trace(values[0].tolist(), vocabulary)

    perfect = replay_pointer_quicksort_transcript(
        values[0].tolist(),
        trace.target_tokens,
        vocabulary,
    )
    assert perfect.completed
    assert perfect.observations_valid
    assert perfect.final_values == (1, 2, 3)
    perfect_metrics = generated_pointer_no_tool_metrics(
        values,
        torch.tensor([trace.target_tokens]),
        vocabulary,
        [trace],
    )
    assert perfect_metrics["exact_match"] == 1.0
    assert perfect_metrics["observation_token_accuracy"] == 1.0
    assert perfect_metrics["full_target_token_accuracy"] == 1.0

    corrupted_tokens = list(trace.target_tokens)
    corrupted_tokens[1] = vocabulary.observation_token("ACTIVE")
    corrupted = replay_pointer_quicksort_transcript(
        values[0].tolist(),
        corrupted_tokens,
        vocabulary,
    )
    assert corrupted.completed
    assert corrupted.valid_execution
    assert not corrupted.observations_valid
    assert corrupted.final_values == (1, 2, 3)

    metrics = generated_pointer_no_tool_metrics(
        values,
        torch.tensor([corrupted_tokens]),
        vocabulary,
        [trace],
    )
    assert metrics["execution_completed"] == 1.0
    assert metrics["observation_exact_match"] == 0.0
    assert metrics["exact_match"] == 0.0
    assert metrics["trace_exact_match"] == 0.0
    assert metrics["observation_token_accuracy"] < 1.0
