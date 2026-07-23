from __future__ import annotations

import torch

from list_sorting_transformer.data import (
    IGNORE_INDEX,
    make_local_window_sort_batch,
)
from list_sorting_transformer.evaluation import (
    generate_local_window_sort_rollouts,
)
from list_sorting_transformer.local_window_sort import (
    LocalWindowSortMachine,
    LocalWindowSortRollout,
    LocalWindowSortTrace,
    LocalWindowTransition,
    generate_local_window_sort_trace,
)
from list_sorting_transformer.metrics import (
    generated_local_window_sort_metrics,
)
from list_sorting_transformer.tokens import LocalWindowSortVocabulary


class ScriptedLocalWindowPolicy:
    def __init__(
        self,
        traces: tuple,
        vocabulary: LocalWindowSortVocabulary,
    ) -> None:
        self.vocab_size = vocabulary.size
        target_rows = []
        for trace in traces:
            targets = []
            for transition in trace.transitions:
                targets.append(transition.action_token)
                targets.extend(transition.response_tokens)
            target_rows.append(tuple(targets))
        self.target_rows = tuple(target_rows)

    def forward_with_state(
        self,
        token_ids: torch.Tensor,
        state: tuple[int, ...] | None = None,
    ):
        if state is None:
            indices = tuple(0 for _ in self.target_rows)
        else:
            indices = tuple(
                index + token_ids.shape[1]
                for index in state
            )
        logits = torch.full(
            (*token_ids.shape, self.vocab_size),
            -1_000.0,
            device=token_ids.device,
        )
        for row_index, target_row in enumerate(self.target_rows):
            target_index = min(indices[row_index], len(target_row) - 1)
            logits[row_index, -1, target_row[target_index]] = 1_000.0
        return logits, indices


def test_local_window_tracks_pointer_and_boundaries() -> None:
    vocabulary = LocalWindowSortVocabulary()
    machine = LocalWindowSortMachine([3, 1, 2], vocabulary)

    assert vocabulary.render_tokens(machine.initial_window_tokens()) == (
        "<WINDOW> <INITIAL> <PASS_CLEAN> <LEFT_EDGE> "
        "<PTR> 3 1 2 <WINDOW_END>"
    )

    event, response = machine.step(vocabulary.action_token("SWAP"))
    assert event == "SWAP"
    assert vocabulary.render_tokens(response) == (
        "<WINDOW> <ADVANCE> <PASS_CHANGED> 1 "
        "<PTR> 3 2 <ACTIVE_END> <WINDOW_END>"
    )

    event, response = machine.step(vocabulary.action_token("SWAP"))
    assert event == "RESET"
    assert vocabulary.render_tokens(response) == (
        "<WINDOW> <NEW_PASS> <PASS_CLEAN> <LEFT_EDGE> "
        "<PTR> 1 2 <ACTIVE_END> <WINDOW_END>"
    )


def test_atomic_pair_window_preserves_width_and_other_slots() -> None:
    separate_vocabulary = LocalWindowSortVocabulary()
    atomic_vocabulary = LocalWindowSortVocabulary(pair_encoding="atomic")
    separate_machine = LocalWindowSortMachine(
        [3, 1, 2],
        separate_vocabulary,
    )
    atomic_machine = LocalWindowSortMachine([3, 1, 2], atomic_vocabulary)

    separate_window = separate_machine.initial_window_tokens()
    atomic_window = atomic_machine.initial_window_tokens()

    assert len(separate_window) == len(atomic_window) == 9
    assert separate_window[:5] == atomic_window[:5]
    assert separate_window[7:] == atomic_window[7:]
    assert atomic_vocabulary.token_pair(atomic_window[5]) == (3, 1)
    assert atomic_window[6] == atomic_vocabulary.pair_end_token
    assert atomic_vocabulary.render_tokens(atomic_window) == (
        "<WINDOW> <INITIAL> <PASS_CLEAN> <LEFT_EDGE> "
        "<PTR> <PAIR_3_1> <PAIR_END> 2 <WINDOW_END>"
    )


def test_atomic_pair_scripted_policy_handles_tool_and_no_tool() -> None:
    vocabulary = LocalWindowSortVocabulary(pair_encoding="atomic")
    batch = make_local_window_sort_batch(
        8,
        9,
        generator=torch.Generator().manual_seed(107),
        vocabulary=vocabulary,
        tool_events=(),
    )
    model = ScriptedLocalWindowPolicy(batch.traces, vocabulary)

    for tool_events in (
        ("KEEP", "SWAP", "RESET", "FINISH"),
        (),
    ):
        rollouts = generate_local_window_sort_rollouts(
            model,  # type: ignore[arg-type]
            batch,
            vocabulary,
            tool_events=tool_events,
        )
        metrics = generated_local_window_sort_metrics(
            batch.values,
            rollouts,
            batch.traces,
        )
        assert metrics["exact_match"] == 1.0
        assert metrics["window_exact_match"] == 1.0


def test_wrong_comparison_decision_continues_execution() -> None:
    vocabulary = LocalWindowSortVocabulary()
    machine = LocalWindowSortMachine([3, 1, 2], vocabulary)

    event, response = machine.step(vocabulary.action_token("KEEP"))

    assert event == "KEEP"
    assert machine.valid
    assert not machine.halted
    assert machine.array == [3, 1, 2]
    assert vocabulary.render_tokens(response) == (
        "<WINDOW> <ADVANCE> <PASS_CLEAN> 3 "
        "<PTR> 1 2 <ACTIVE_END> <WINDOW_END>"
    )


def test_local_window_trace_sorts_random_lists_with_quadratic_tokens() -> None:
    vocabulary = LocalWindowSortVocabulary(symbol_count=5)
    generator = torch.Generator().manual_seed(83)

    for length in range(1, 41):
        values = torch.randint(
            0,
            vocabulary.symbol_count,
            (length,),
            generator=generator,
        ).tolist()
        trace = generate_local_window_sort_trace(values, vocabulary)
        assert list(trace.final_values) == sorted(values)
        assert all(
            len(transition.response_tokens) in {0, 9}
            for transition in trace.transitions
        )
        assert (
            sum(
                1 + len(transition.response_tokens)
                for transition in trace.transitions
            )
            <= 10 * (length * (length - 1) // 2 + 1) + 1
        )


def test_local_window_tool_mask_only_changes_response_supervision() -> None:
    vocabulary = LocalWindowSortVocabulary()
    all_tool = make_local_window_sort_batch(
        16,
        8,
        generator=torch.Generator().manual_seed(89),
        vocabulary=vocabulary,
        tool_events=("KEEP", "SWAP", "RESET", "FINISH"),
    )
    no_tool = make_local_window_sort_batch(
        16,
        8,
        generator=torch.Generator().manual_seed(89),
        vocabulary=vocabulary,
        tool_events=(),
    )
    boundary_tool = make_local_window_sort_batch(
        16,
        8,
        generator=torch.Generator().manual_seed(89),
        vocabulary=vocabulary,
        tool_events=("RESET", "FINISH"),
    )

    torch.testing.assert_close(all_tool.token_ids, no_tool.token_ids)
    torch.testing.assert_close(all_tool.token_ids, boundary_tool.token_ids)
    assert all_tool.traces == no_tool.traces
    for row_index, trace in enumerate(all_tool.traces):
        label_index = all_tool.prompt_length - 1
        for transition in trace.transitions:
            assert int(
                all_tool.labels[row_index, label_index]
            ) == transition.action_token
            assert int(
                no_tool.labels[row_index, label_index]
            ) == transition.action_token
            label_index += 1
            if transition.response_tokens:
                response_end = label_index + len(transition.response_tokens)
                assert torch.all(
                    all_tool.labels[row_index, label_index:response_end].eq(
                        IGNORE_INDEX
                    )
                )
                torch.testing.assert_close(
                    no_tool.labels[row_index, label_index:response_end],
                    torch.tensor(transition.response_tokens),
                )
                expected_boundary_labels = (
                    torch.full(
                        (len(transition.response_tokens),),
                        IGNORE_INDEX,
                    )
                    if transition.response_event in {"RESET", "FINISH"}
                    else torch.tensor(transition.response_tokens)
                )
                torch.testing.assert_close(
                    boundary_tool.labels[
                        row_index,
                        label_index:response_end,
                    ],
                    expected_boundary_labels,
                )
                label_index = response_end


def test_local_window_scripted_policy_handles_assistance_modes() -> None:
    vocabulary = LocalWindowSortVocabulary()
    batch = make_local_window_sort_batch(
        8,
        9,
        generator=torch.Generator().manual_seed(97),
        vocabulary=vocabulary,
        tool_events=(),
    )
    model = ScriptedLocalWindowPolicy(batch.traces, vocabulary)

    modes = (
        ("KEEP", "SWAP", "RESET", "FINISH"),
        ("KEEP", "RESET", "FINISH"),
        ("SWAP", "RESET", "FINISH"),
        ("RESET", "FINISH"),
        (),
    )
    for tool_events in modes:
        rollouts = generate_local_window_sort_rollouts(
            model,  # type: ignore[arg-type]
            batch,
            vocabulary,
            tool_events=tool_events,
        )
        metrics = generated_local_window_sort_metrics(
            batch.values,
            rollouts,
            batch.traces,
        )
        assert metrics["exact_match"] == 1.0
        assert metrics["window_exact_match"] == 1.0
        assert metrics["window_token_accuracy"] == 1.0


def test_rollout_generates_a_window_after_a_wrong_decision() -> None:
    vocabulary = LocalWindowSortVocabulary()
    batch = make_local_window_sort_batch(
        1,
        5,
        generator=torch.Generator().manual_seed(101),
        vocabulary=vocabulary,
        tool_events=(),
    )
    values = batch.values[0].tolist()
    machine = LocalWindowSortMachine(values, vocabulary)
    transitions = []
    actions = []
    first_action = machine.expected_action()
    wrong_action = vocabulary.action_token(
        "KEEP"
        if vocabulary.action_name(first_action) == "SWAP"
        else "SWAP"
    )

    for _ in range(100):
        if machine.halted:
            break
        action = wrong_action if not actions else machine.expected_action()
        event, response = machine.step(action)
        transitions.append(
            LocalWindowTransition(
                action_token=action,
                response_event=event,
                response_tokens=response,
            )
        )
        actions.append(action)
    assert machine.halted
    scripted_trace = LocalWindowSortTrace(
        input_values=tuple(values),
        initial_window_tokens=batch.traces[0].initial_window_tokens,
        transitions=tuple(transitions),
        action_tokens=tuple(actions),
        final_values=tuple(machine.array),
    )
    model = ScriptedLocalWindowPolicy((scripted_trace,), vocabulary)

    rollout = generate_local_window_sort_rollouts(
        model,  # type: ignore[arg-type]
        batch,
        vocabulary,
        tool_events=(),
    )[0]

    assert rollout.action_tokens[0] == wrong_action
    assert rollout.valid_execution
    assert rollout.completed
    assert rollout.generated_window_tokens
    assert (
        rollout.generated_window_tokens[0]
        == rollout.expected_window_tokens[0]
    )
    metrics = generated_local_window_sort_metrics(
        batch.values,
        [rollout],
        batch.traces,
    )
    assert metrics["valid_syntax"] == 1.0
    assert metrics["window_exact_match"] == 1.0
    assert metrics["trace_exact_match"] == 0.0


def test_looping_policy_stops_at_the_action_budget() -> None:
    vocabulary = LocalWindowSortVocabulary()
    batch = make_local_window_sort_batch(
        1,
        3,
        generator=torch.Generator().manual_seed(103),
        vocabulary=vocabulary,
        tool_events=(),
    )
    values = batch.values[0].tolist()
    machine = LocalWindowSortMachine(values, vocabulary)
    transitions = []
    actions = []
    for _ in range(5):
        action = vocabulary.action_token("SWAP")
        event, response = machine.step(action)
        transitions.append(
            LocalWindowTransition(
                action_token=action,
                response_event=event,
                response_tokens=response,
            )
        )
        actions.append(action)
    scripted_trace = LocalWindowSortTrace(
        input_values=tuple(values),
        initial_window_tokens=batch.traces[0].initial_window_tokens,
        transitions=tuple(transitions),
        action_tokens=tuple(actions),
        final_values=tuple(machine.array),
    )
    model = ScriptedLocalWindowPolicy((scripted_trace,), vocabulary)

    rollout = generate_local_window_sort_rollouts(
        model,  # type: ignore[arg-type]
        batch,
        vocabulary,
        tool_events=(),
        max_actions=5,
    )[0]

    assert rollout.valid_execution
    assert not rollout.completed
    assert rollout.timed_out
    assert len(rollout.action_tokens) == 5
    assert len(rollout.generated_window_tokens) == 5


def test_local_window_metrics_require_generated_windows_to_match() -> None:
    vocabulary = LocalWindowSortVocabulary()
    values = torch.tensor([[3, 1, 2]])
    trace = generate_local_window_sort_trace(
        values[0].tolist(),
        vocabulary,
    )
    expected_window = trace.transitions[0].response_tokens
    corrupted_window = list(expected_window)
    corrupted_window[1] = vocabulary.window_token("FINISHED")
    rollout = LocalWindowSortRollout(
        action_tokens=trace.action_tokens,
        final_values=trace.final_values,
        completed=True,
        valid_execution=True,
        timed_out=False,
        generated_window_tokens=(tuple(corrupted_window),),
        expected_window_tokens=(expected_window,),
        required_window_count=1,
        tool_window_count=2,
    )

    metrics = generated_local_window_sort_metrics(
        values,
        [rollout],
        [trace],
    )

    assert metrics["sorted"] == 1.0
    assert metrics["execution_completed"] == 1.0
    assert metrics["window_exact_match"] == 0.0
    assert metrics["exact_match"] == 0.0


def test_local_window_metrics_count_unreached_windows_as_incorrect() -> None:
    vocabulary = LocalWindowSortVocabulary()
    values = torch.tensor([[3, 1, 2]])
    trace = generate_local_window_sort_trace(
        values[0].tolist(),
        vocabulary,
    )
    rollout = LocalWindowSortRollout(
        action_tokens=(),
        final_values=tuple(values[0].tolist()),
        completed=False,
        valid_execution=False,
        timed_out=False,
        generated_window_tokens=(),
        expected_window_tokens=(),
        required_window_count=sum(
            bool(transition.response_tokens)
            for transition in trace.transitions
        ),
        tool_window_count=0,
    )

    metrics = generated_local_window_sort_metrics(
        values,
        [rollout],
        [trace],
    )

    assert metrics["window_exact_match"] == 0.0
    assert metrics["window_token_accuracy"] == 0.0
    assert metrics["window_transition_exact_fraction"] == 0.0
    assert metrics["generated_window_count"] == 0.0
