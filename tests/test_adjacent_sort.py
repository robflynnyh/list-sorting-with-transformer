from __future__ import annotations

import torch

from list_sorting_transformer.adjacent_sort import (
    AdjacentSortMachine,
    AutoAdvanceSortMachine,
    execute_adjacent_sort_actions,
    execute_auto_advance_sort_actions,
    generate_adjacent_sort_trace,
    generate_auto_advance_sort_trace,
    replay_adjacent_sort_transcript,
    replay_auto_advance_sort_transcript,
)
from list_sorting_transformer.data import (
    IGNORE_INDEX,
    make_adjacent_sort_batch,
    make_auto_advance_sort_batch,
)
from list_sorting_transformer.evaluation import (
    generate_adjacent_sort_rollouts,
    generate_auto_advance_sort_rollouts,
)
from list_sorting_transformer.metrics import (
    generated_adjacent_no_tool_metrics,
    generated_adjacent_sort_metrics,
    generated_auto_advance_no_tool_metrics,
    generated_auto_advance_sort_metrics,
)
from list_sorting_transformer.tokens import (
    VALUE_OFFSET,
    AdjacentSortVocabulary,
    AutoAdvanceSortVocabulary,
)


class ScriptedAdjacentPolicy:
    def __init__(
        self,
        action_rows: tuple[tuple[int, ...], ...],
        vocabulary: AdjacentSortVocabulary | AutoAdvanceSortVocabulary,
    ) -> None:
        self.action_rows = action_rows
        self.action_tokens = frozenset(vocabulary.action_tokens)
        self.vocab_size = vocabulary.size

    def forward_with_state(
        self,
        token_ids: torch.Tensor,
        state: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        counters = [0] * len(self.action_rows) if state is None else list(state)
        if state is not None:
            for row_index, token in enumerate(token_ids[:, -1].tolist()):
                if token in self.action_tokens:
                    counters[row_index] += 1
        logits = torch.full(
            (*token_ids.shape, self.vocab_size),
            -1_000.0,
            device=token_ids.device,
        )
        for row_index, actions in enumerate(self.action_rows):
            selected_index = min(counters[row_index], len(actions) - 1)
            logits[row_index, -1, actions[selected_index]] = 1_000.0
        return logits, tuple(counters)


def test_adjacent_trace_exposes_each_local_pair() -> None:
    vocabulary = AdjacentSortVocabulary()
    trace = generate_adjacent_sort_trace([3, 1, 2], vocabulary)

    assert trace.final_values == (1, 2, 3)
    assert vocabulary.render_tokens(trace.target_tokens) == (
        "<READ_PAIR> 3 1 "
        "<SWAP> 1 3 "
        "<RIGHT> 3 2 "
        "<SWAP> 2 3 "
        "<END_PASS> <CHANGED> "
        "<RESET> 1 2 "
        "<KEEP> 1 2 "
        "<END_PASS> <UNCHANGED> "
        "<DONE>"
    )
    assert sum(trace.target_prediction_mask) == len(trace.action_tokens)
    assert all(
        token in vocabulary.action_tokens
        for token, predicted in zip(
            trace.target_tokens,
            trace.target_prediction_mask,
        )
        if predicted
    )


def test_adjacent_machine_sorts_duplicate_heavy_random_lists() -> None:
    vocabulary = AdjacentSortVocabulary(symbol_count=5)
    generator = torch.Generator().manual_seed(41)

    for length in range(1, 41):
        for _ in range(4):
            values = torch.randint(
                0,
                5,
                (length,),
                generator=generator,
            ).tolist()
            trace = generate_adjacent_sort_trace(values, vocabulary)
            assert list(trace.final_values) == sorted(values)


def test_adjacent_machine_rejects_an_action_from_the_wrong_phase() -> None:
    vocabulary = AdjacentSortVocabulary()
    machine = AdjacentSortMachine([2, 0, 1], vocabulary)

    observations = machine.step(vocabulary.action_token("SWAP"))

    assert observations == (vocabulary.observation_token("INVALID"),)
    assert machine.finished
    assert not machine.valid
    assert machine.last_error == "expected READ_PAIR, received SWAP"


def test_adjacent_batch_masks_tool_observations_but_no_tool_supervises_them() -> None:
    vocabulary = AdjacentSortVocabulary()
    tool_batch = make_adjacent_sort_batch(
        8,
        7,
        generator=torch.Generator().manual_seed(3),
        vocabulary=vocabulary,
    )

    assert tool_batch.prompt_length == 15
    assert torch.all(
        tool_batch.labels[:, : tool_batch.prompt_length - 1].eq(IGNORE_INDEX)
    )
    for row_index, trace in enumerate(tool_batch.traces):
        for target_index, (token, predicted) in enumerate(
            zip(trace.target_tokens, trace.target_prediction_mask)
        ):
            label_index = tool_batch.prompt_length - 1 + target_index
            expected = token if predicted else IGNORE_INDEX
            assert int(tool_batch.labels[row_index, label_index]) == expected
    included = tool_batch.labels.ne(IGNORE_INDEX)
    assert torch.all(
        tool_batch.labels[included].ge(VALUE_OFFSET + vocabulary.symbol_count)
    )

    no_tool_batch = make_adjacent_sort_batch(
        4,
        6,
        generator=torch.Generator().manual_seed(5),
        vocabulary=vocabulary,
        supervise_observations=True,
    )
    for row_index, trace in enumerate(no_tool_batch.traces):
        target_start = no_tool_batch.prompt_length - 1
        expected = torch.tensor(trace.target_tokens)
        labels = no_tool_batch.labels[
            row_index,
            target_start : target_start + len(trace.target_tokens),
        ]
        torch.testing.assert_close(labels, expected)


def test_scripted_policy_completes_interactive_adjacent_execution() -> None:
    vocabulary = AdjacentSortVocabulary()
    batch = make_adjacent_sort_batch(
        4,
        6,
        generator=torch.Generator().manual_seed(11),
        vocabulary=vocabulary,
    )
    model = ScriptedAdjacentPolicy(
        tuple(trace.action_tokens for trace in batch.traces),
        vocabulary,
    )

    rollouts = generate_adjacent_sort_rollouts(
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


def test_adjacent_metrics_require_valid_completion() -> None:
    vocabulary = AdjacentSortVocabulary()
    values = torch.tensor([[3, 1, 2]])
    trace = generate_adjacent_sort_trace(values[0].tolist(), vocabulary)
    perfect = execute_adjacent_sort_actions(
        values[0].tolist(),
        trace.action_tokens,
        vocabulary,
    )

    metrics = generated_adjacent_sort_metrics(
        values,
        [perfect],
        vocabulary,
        [trace],
    )
    assert metrics["exact_match"] == 1.0
    assert metrics["trace_exact_match"] == 1.0
    assert metrics["operation_prefix_fraction"] == 1.0

    invalid = execute_adjacent_sort_actions(
        values[0].tolist(),
        [vocabulary.action_token("SWAP")],
        vocabulary,
    )
    metrics = generated_adjacent_sort_metrics(
        values,
        [invalid],
        vocabulary,
        [trace],
    )
    assert metrics["exact_match"] == 0.0
    assert metrics["execution_completed"] == 0.0
    assert metrics["operation_prefix_fraction"] == 0.0


def test_adjacent_no_tool_replay_checks_generated_pair_values() -> None:
    vocabulary = AdjacentSortVocabulary()
    values = torch.tensor([[3, 1, 2]])
    trace = generate_adjacent_sort_trace(values[0].tolist(), vocabulary)

    perfect = replay_adjacent_sort_transcript(
        values[0].tolist(),
        trace.target_tokens,
        vocabulary,
    )
    assert perfect.completed
    assert perfect.observations_valid
    assert perfect.final_values == (1, 2, 3)
    perfect_metrics = generated_adjacent_no_tool_metrics(
        values,
        torch.tensor([trace.target_tokens]),
        vocabulary,
        [trace],
    )
    assert perfect_metrics["exact_match"] == 1.0
    assert perfect_metrics["observation_token_accuracy"] == 1.0

    corrupted_tokens = list(trace.target_tokens)
    corrupted_tokens[1] = vocabulary.value_token(9)
    corrupted = replay_adjacent_sort_transcript(
        values[0].tolist(),
        corrupted_tokens,
        vocabulary,
    )
    assert corrupted.completed
    assert corrupted.valid_execution
    assert not corrupted.observations_valid
    assert corrupted.final_values == (1, 2, 3)

    metrics = generated_adjacent_no_tool_metrics(
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


def test_auto_advance_trace_exposes_boundaries_without_move_actions() -> None:
    vocabulary = AutoAdvanceSortVocabulary()
    trace = generate_auto_advance_sort_trace([3, 1, 2], vocabulary)

    assert trace.final_values == (1, 2, 3)
    assert vocabulary.render_tokens(trace.target_tokens) == (
        "<READ_PAIR> <PAIR> <PAIR_3_1> "
        "<SWAP> <PAIR> <PAIR_3_2> "
        "<SWAP> <CHANGED> <PAIR_1_2> "
        "<KEEP> <UNCHANGED> <NONE> "
        "<DONE>"
    )
    assert [vocabulary.action_name(token) for token in trace.action_tokens] == [
        "READ_PAIR",
        "SWAP",
        "SWAP",
        "KEEP",
        "DONE",
    ]
    action_positions = [
        index
        for index, predicted in enumerate(trace.target_prediction_mask)
        if predicted
    ]
    observation_counts = [
        (
            action_positions[index + 1]
            if index + 1 < len(action_positions)
            else len(trace.target_tokens)
        )
        - action_position
        - 1
        for index, action_position in enumerate(action_positions)
    ]
    assert observation_counts == [2, 2, 2, 2, 0]


def test_auto_advance_machine_sorts_random_lists_with_shorter_action_traces() -> None:
    auto_vocabulary = AutoAdvanceSortVocabulary(symbol_count=5)
    original_vocabulary = AdjacentSortVocabulary(symbol_count=5)
    generator = torch.Generator().manual_seed(53)

    for length in range(1, 41):
        values = torch.randint(
            0,
            5,
            (length,),
            generator=generator,
        ).tolist()
        auto_trace = generate_auto_advance_sort_trace(values, auto_vocabulary)
        original_trace = generate_adjacent_sort_trace(values, original_vocabulary)
        assert list(auto_trace.final_values) == sorted(values)
        assert len(auto_trace.action_tokens) <= len(original_trace.action_tokens)


def test_auto_advance_machine_rejects_wrong_comparison_decision() -> None:
    vocabulary = AutoAdvanceSortVocabulary()
    machine = AutoAdvanceSortMachine([2, 0, 1], vocabulary)
    machine.step(vocabulary.action_token("READ_PAIR"))

    observations = machine.step(vocabulary.action_token("KEEP"))

    assert observations == (
        vocabulary.observation_token("INVALID"),
        vocabulary.observation_token("NONE"),
    )
    assert machine.finished
    assert not machine.valid
    assert machine.last_error == "expected SWAP, received KEEP"


def test_auto_advance_tool_and_no_tool_batches_use_the_same_transcript() -> None:
    vocabulary = AutoAdvanceSortVocabulary()
    tool_batch = make_auto_advance_sort_batch(
        4,
        6,
        generator=torch.Generator().manual_seed(61),
        vocabulary=vocabulary,
    )
    no_tool_batch = make_auto_advance_sort_batch(
        4,
        6,
        generator=torch.Generator().manual_seed(61),
        vocabulary=vocabulary,
        supervise_observations=True,
    )

    torch.testing.assert_close(tool_batch.token_ids, no_tool_batch.token_ids)
    assert tool_batch.traces == no_tool_batch.traces
    assert tool_batch.labels.ne(IGNORE_INDEX).sum() < no_tool_batch.labels.ne(
        IGNORE_INDEX
    ).sum()


def test_scripted_policy_completes_auto_advance_execution() -> None:
    vocabulary = AutoAdvanceSortVocabulary()
    batch = make_auto_advance_sort_batch(
        4,
        8,
        generator=torch.Generator().manual_seed(67),
        vocabulary=vocabulary,
    )
    model = ScriptedAdjacentPolicy(
        tuple(trace.action_tokens for trace in batch.traces),
        vocabulary,
    )

    rollouts = generate_auto_advance_sort_rollouts(
        model,  # type: ignore[arg-type]
        batch,
        vocabulary,
    )

    assert all(rollout.completed for rollout in rollouts)
    assert all(
        list(rollout.final_values) == sorted(values)
        for rollout, values in zip(rollouts, batch.values.tolist())
    )
    metrics = generated_auto_advance_sort_metrics(
        batch.values,
        rollouts,
        vocabulary,
        batch.traces,
    )
    assert metrics["exact_match"] == 1.0


def test_auto_advance_no_tool_replay_checks_boundary_observations() -> None:
    vocabulary = AutoAdvanceSortVocabulary()
    values = torch.tensor([[3, 1, 2]])
    trace = generate_auto_advance_sort_trace(values[0].tolist(), vocabulary)

    perfect = replay_auto_advance_sort_transcript(
        values[0].tolist(),
        trace.target_tokens,
        vocabulary,
    )
    assert perfect.completed
    assert perfect.observations_valid
    perfect_metrics = generated_auto_advance_no_tool_metrics(
        values,
        torch.tensor([trace.target_tokens]),
        vocabulary,
        [trace],
    )
    assert perfect_metrics["exact_match"] == 1.0

    corrupted_tokens = list(trace.target_tokens)
    changed_index = corrupted_tokens.index(vocabulary.observation_token("CHANGED"))
    corrupted_tokens[changed_index] = vocabulary.observation_token("UNCHANGED")
    corrupted = replay_auto_advance_sort_transcript(
        values[0].tolist(),
        corrupted_tokens,
        vocabulary,
    )
    assert corrupted.completed
    assert not corrupted.observations_valid

    executed = execute_auto_advance_sort_actions(
        values[0].tolist(),
        trace.action_tokens,
        vocabulary,
    )
    assert executed.completed
    assert executed.final_values == (1, 2, 3)
