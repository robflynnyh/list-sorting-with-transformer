"""Bubble sort with a fixed-width local window around the active pointer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .tokens import LocalWindowSortVocabulary


WINDOW_TOKEN_LENGTH = 9
WINDOW_TOOL_EVENTS = ("KEEP", "SWAP", "RESET", "FINISH")


@dataclass(frozen=True)
class LocalWindowTransition:
    """One model action and its canonical next-window response."""

    action_token: int
    response_event: str | None
    response_tokens: tuple[int, ...]


@dataclass(frozen=True)
class LocalWindowSortTrace:
    """The deterministic local-window transcript for one input list."""

    input_values: tuple[int, ...]
    initial_window_tokens: tuple[int, ...]
    transitions: tuple[LocalWindowTransition, ...]
    action_tokens: tuple[int, ...]
    final_values: tuple[int, ...]


@dataclass(frozen=True)
class LocalWindowSortRollout:
    """An interactive rollout with optional model-generated windows."""

    action_tokens: tuple[int, ...]
    final_values: tuple[int, ...]
    completed: bool
    valid_execution: bool
    timed_out: bool
    generated_window_tokens: tuple[tuple[int, ...], ...]
    expected_window_tokens: tuple[tuple[int, ...], ...]
    required_window_count: int
    tool_window_count: int


class LocalWindowSortMachine:
    """Bubble-sort executor that exposes only a constant-size local window."""

    __slots__ = (
        "array",
        "cursor",
        "active_end",
        "changed",
        "ready_to_finish",
        "completed",
        "valid",
        "last_error",
        "vocabulary",
    )

    def __init__(
        self,
        values: Sequence[int],
        vocabulary: LocalWindowSortVocabulary,
    ) -> None:
        if not values:
            raise ValueError("local-window sort requires a non-empty input")
        self.array = [int(value) for value in values]
        for value in self.array:
            vocabulary.value_token(value)
        self.cursor = 0
        self.active_end = len(self.array) - 1
        self.changed = False
        self.ready_to_finish = len(self.array) == 1
        self.completed = False
        self.valid = True
        self.last_error: str | None = None
        self.vocabulary = vocabulary

    @property
    def halted(self) -> bool:
        return self.completed or not self.valid

    def expected_action(self) -> int:
        if self.halted:
            raise RuntimeError("halted machines do not have a next action")
        if self.ready_to_finish:
            return self.vocabulary.action_token("DONE")
        name = (
            "SWAP"
            if self.array[self.cursor] > self.array[self.cursor + 1]
            else "KEEP"
        )
        return self.vocabulary.action_token(name)

    def _slot(self, index: int) -> int:
        if index < 0:
            return self.vocabulary.window_token("LEFT_EDGE")
        if index > self.active_end:
            return self.vocabulary.window_token("ACTIVE_END")
        return self.vocabulary.value_token(self.array[index])

    def window_tokens(self, kind: str) -> tuple[int, ...]:
        """Render left context, current pair, and one right-lookahead slot."""

        if self.ready_to_finish:
            values = [
                (
                    self.vocabulary.value_token(self.array[index])
                    if index < len(self.array)
                    else self.vocabulary.window_token("ACTIVE_END")
                )
                for index in range(3)
            ]
            tokens = (
                self.vocabulary.window_token("WINDOW"),
                self.vocabulary.window_token("FINISHED"),
                self.vocabulary.window_token("PASS_CLEAN"),
                self.vocabulary.window_token("LEFT_EDGE"),
                self.vocabulary.window_token("NO_PTR"),
                *values,
                self.vocabulary.window_token("WINDOW_END"),
            )
        else:
            pass_name = "PASS_CHANGED" if self.changed else "PASS_CLEAN"
            tokens = (
                self.vocabulary.window_token("WINDOW"),
                self.vocabulary.window_token(kind),
                self.vocabulary.window_token(pass_name),
                self._slot(self.cursor - 1),
                self.vocabulary.window_token("PTR"),
                self._slot(self.cursor),
                self._slot(self.cursor + 1),
                self._slot(self.cursor + 2),
                self.vocabulary.window_token("WINDOW_END"),
            )
        if len(tokens) != WINDOW_TOKEN_LENGTH:
            raise RuntimeError("local window must have fixed token width")
        return tuple(tokens)

    def initial_window_tokens(self) -> tuple[int, ...]:
        kind = "FINISHED" if self.ready_to_finish else "INITIAL"
        return self.window_tokens(kind)

    def step(
        self,
        action_token: int,
    ) -> tuple[str | None, tuple[int, ...]]:
        """Apply one action and return its response category and window."""

        if self.halted:
            raise RuntimeError("cannot execute an action after halt")

        if self.ready_to_finish:
            expected = self.vocabulary.action_token("DONE")
            if int(action_token) != expected:
                try:
                    received = self.vocabulary.action_name(int(action_token))
                except ValueError:
                    received = f"token {int(action_token)}"
                self.last_error = f"expected DONE, received {received}"
                self.valid = False
                return None, ()
            self.completed = True
            return None, ()

        try:
            action_name = self.vocabulary.action_name(int(action_token))
        except ValueError:
            action_name = ""
        if action_name not in {"KEEP", "SWAP"}:
            received = action_name or f"token {int(action_token)}"
            self.last_error = (
                f"expected KEEP or SWAP, received {received}"
            )
            self.valid = False
            return None, ()

        if action_name == "SWAP":
            self.array[self.cursor], self.array[self.cursor + 1] = (
                self.array[self.cursor + 1],
                self.array[self.cursor],
            )
            self.changed = True

        if self.cursor + 1 < self.active_end:
            self.cursor += 1
            return action_name, self.window_tokens("ADVANCE")
        if not self.changed:
            self.ready_to_finish = True
            return "FINISH", self.window_tokens("FINISHED")

        self.active_end = max(1, self.active_end - 1)
        self.cursor = 0
        self.changed = False
        return "RESET", self.window_tokens("NEW_PASS")


def generate_local_window_sort_trace(
    values: Sequence[int],
    vocabulary: LocalWindowSortVocabulary,
) -> LocalWindowSortTrace:
    """Generate a canonical action/window transcript."""

    machine = LocalWindowSortMachine(values, vocabulary)
    initial_window = machine.initial_window_tokens()
    transitions: list[LocalWindowTransition] = []
    actions: list[int] = []
    while not machine.halted:
        action = machine.expected_action()
        event, response = machine.step(action)
        transitions.append(
            LocalWindowTransition(
                action_token=action,
                response_event=event,
                response_tokens=response,
            )
        )
        actions.append(action)

    expected = sorted(int(value) for value in values)
    if not machine.completed or machine.array != expected:
        raise RuntimeError("reference local-window machine did not sort input")
    return LocalWindowSortTrace(
        input_values=tuple(int(value) for value in values),
        initial_window_tokens=initial_window,
        transitions=tuple(transitions),
        action_tokens=tuple(actions),
        final_values=tuple(machine.array),
    )
