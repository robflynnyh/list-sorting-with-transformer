"""Executor-assisted bubble sort expressed as local adjacent-pair operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .tokens import (
    ADJACENT_SORT_ACTIONS,
    AUTO_ADVANCE_SORT_ACTIONS,
    AdjacentSortVocabulary,
    AutoAdvanceSortVocabulary,
)


_ACTION_INDEX = {
    name: index for index, name in enumerate(ADJACENT_SORT_ACTIONS)
}

_READ_PAIR = 0
_DECIDE = 1
_MOVE_RIGHT = 2
_END_PASS = 3
_RESET = 4
_DONE = 5
_FINISHED = 6
_INVALID = 7

_PHASE_ACTION_INDEX = {
    _READ_PAIR: _ACTION_INDEX["READ_PAIR"],
    _MOVE_RIGHT: _ACTION_INDEX["RIGHT"],
    _END_PASS: _ACTION_INDEX["END_PASS"],
    _RESET: _ACTION_INDEX["RESET"],
    _DONE: _ACTION_INDEX["DONE"],
}

_AUTO_ACTION_INDEX = {
    name: index for index, name in enumerate(AUTO_ADVANCE_SORT_ACTIONS)
}
_AUTO_READ_PAIR = 0
_AUTO_DECIDE = 1
_AUTO_DONE = 2
_AUTO_FINISHED = 3
_AUTO_INVALID = 4
_AUTO_PHASE_ACTION_INDEX = {
    _AUTO_READ_PAIR: _AUTO_ACTION_INDEX["READ_PAIR"],
    _AUTO_DONE: _AUTO_ACTION_INDEX["DONE"],
}


@dataclass(frozen=True)
class AdjacentSortTrace:
    """A canonical adjacent-action transcript and its sorted machine state."""

    input_values: tuple[int, ...]
    target_tokens: tuple[int, ...]
    target_prediction_mask: tuple[bool, ...]
    action_tokens: tuple[int, ...]
    final_values: tuple[int, ...]


@dataclass(frozen=True)
class AdjacentSortRollout:
    """The outcome of interactively executing model-predicted actions."""

    action_tokens: tuple[int, ...]
    final_values: tuple[int, ...]
    completed: bool
    valid_execution: bool
    timed_out: bool


@dataclass(frozen=True)
class AdjacentSortTranscriptRollout:
    """Offline replay of an adjacent transcript generated without a tool."""

    generated_tokens: tuple[int, ...]
    action_tokens: tuple[int, ...]
    final_values: tuple[int, ...]
    completed: bool
    valid_execution: bool
    syntax_valid: bool
    observations_valid: bool
    observation_correct: int
    timed_out: bool


class AdjacentSortMachine:
    """Strict bubble-sort machine with one cursor and local pair observations."""

    __slots__ = (
        "array",
        "cursor",
        "active_end",
        "changed",
        "phase",
        "valid",
        "completed",
        "last_error",
        "vocabulary",
    )

    def __init__(
        self,
        values: Sequence[int],
        vocabulary: AdjacentSortVocabulary,
    ) -> None:
        if not values:
            raise ValueError("adjacent sort requires a non-empty input")
        self.array = [int(value) for value in values]
        for value in self.array:
            vocabulary.value_token(value)
        self.cursor = 0
        self.active_end = len(self.array) - 1
        self.changed = False
        self.phase = _READ_PAIR if len(self.array) > 1 else _DONE
        self.valid = True
        self.completed = False
        self.last_error: str | None = None
        self.vocabulary = vocabulary

    @property
    def finished(self) -> bool:
        return self.phase in {_FINISHED, _INVALID}

    def _action(self, name: str) -> int:
        return self.vocabulary.action_token_offset + _ACTION_INDEX[name]

    def _observation(self, name: str) -> int:
        return self.vocabulary.observation_token(name)

    def _pair(self) -> tuple[int, int]:
        return (
            self.vocabulary.value_token(self.array[self.cursor]),
            self.vocabulary.value_token(self.array[self.cursor + 1]),
        )

    def expected_action(self) -> int:
        """Return the sole canonical next action for the current state."""

        if self.phase == _DECIDE:
            name = (
                "SWAP"
                if self.array[self.cursor] > self.array[self.cursor + 1]
                else "KEEP"
            )
            return self._action(name)
        try:
            action_index = _PHASE_ACTION_INDEX[self.phase]
        except KeyError as error:
            raise RuntimeError("finished machines do not have a next action") from error
        return self.vocabulary.action_token_offset + action_index

    def step(self, action_token: int) -> tuple[int, ...]:
        """Execute one action and return zero, one, or two observation tokens."""

        if self.finished:
            raise RuntimeError("cannot execute an action after the machine finishes")
        expected = self.expected_action()
        if int(action_token) != expected:
            try:
                received_name = self.vocabulary.action_name(int(action_token))
            except ValueError:
                received_name = f"token {int(action_token)}"
            self.last_error = (
                f"expected {self.vocabulary.action_name(expected)}, "
                f"received {received_name}"
            )
            self.valid = False
            self.phase = _INVALID
            return (self._observation("INVALID"),)

        if self.phase == _READ_PAIR:
            self.phase = _DECIDE
            return self._pair()
        if self.phase == _DECIDE:
            if action_token == self._action("SWAP"):
                self.array[self.cursor], self.array[self.cursor + 1] = (
                    self.array[self.cursor + 1],
                    self.array[self.cursor],
                )
                self.changed = True
            observations = self._pair()
            self.phase = (
                _END_PASS
                if self.cursor + 1 == self.active_end
                else _MOVE_RIGHT
            )
            return observations
        if self.phase == _MOVE_RIGHT:
            self.cursor += 1
            self.phase = _DECIDE
            return self._pair()
        if self.phase == _END_PASS:
            if self.changed:
                self.phase = _RESET
                return (self._observation("CHANGED"),)
            self.phase = _DONE
            return (self._observation("UNCHANGED"),)
        if self.phase == _RESET:
            self.active_end = max(1, self.active_end - 1)
            self.cursor = 0
            self.changed = False
            self.phase = _DECIDE
            return self._pair()
        if self.phase == _DONE:
            self.phase = _FINISHED
            self.completed = True
            return ()
        raise RuntimeError(f"unhandled adjacent-sort phase: {self.phase}")


class AutoAdvanceSortMachine:
    """Bubble-sort machine whose executor owns cursor and pass transitions."""

    __slots__ = (
        "array",
        "cursor",
        "active_end",
        "changed",
        "phase",
        "valid",
        "completed",
        "last_error",
        "vocabulary",
    )

    def __init__(
        self,
        values: Sequence[int],
        vocabulary: AutoAdvanceSortVocabulary,
    ) -> None:
        if not values:
            raise ValueError("auto-advance sort requires a non-empty input")
        self.array = [int(value) for value in values]
        for value in self.array:
            vocabulary.value_token(value)
        self.cursor = 0
        self.active_end = len(self.array) - 1
        self.changed = False
        self.phase = _AUTO_READ_PAIR if len(self.array) > 1 else _AUTO_DONE
        self.valid = True
        self.completed = False
        self.last_error: str | None = None
        self.vocabulary = vocabulary

    @property
    def finished(self) -> bool:
        return self.phase in {_AUTO_FINISHED, _AUTO_INVALID}

    def _action(self, name: str) -> int:
        return self.vocabulary.action_token_offset + _AUTO_ACTION_INDEX[name]

    def _observation(self, name: str) -> int:
        return self.vocabulary.observation_token(name)

    def _pair(self) -> int:
        return self.vocabulary.pair_token(
            self.array[self.cursor],
            self.array[self.cursor + 1],
        )

    def expected_action(self) -> int:
        """Return the next comparison decision without exposing positions."""

        if self.phase == _AUTO_DECIDE:
            name = (
                "SWAP"
                if self.array[self.cursor] > self.array[self.cursor + 1]
                else "KEEP"
            )
            return self._action(name)
        try:
            action_index = _AUTO_PHASE_ACTION_INDEX[self.phase]
        except KeyError as error:
            raise RuntimeError("finished machines do not have a next action") from error
        return self.vocabulary.action_token_offset + action_index

    def step(self, action_token: int) -> tuple[int, ...]:
        """Execute one decision and automatically expose the next decision."""

        if self.finished:
            raise RuntimeError("cannot execute an action after the machine finishes")
        expected = self.expected_action()
        if int(action_token) != expected:
            try:
                received_name = self.vocabulary.action_name(int(action_token))
            except ValueError:
                received_name = f"token {int(action_token)}"
            self.last_error = (
                f"expected {self.vocabulary.action_name(expected)}, "
                f"received {received_name}"
            )
            self.valid = False
            self.phase = _AUTO_INVALID
            return (
                self._observation("INVALID"),
                self._observation("NONE"),
            )

        if self.phase == _AUTO_READ_PAIR:
            self.phase = _AUTO_DECIDE
            return (self._observation("PAIR"), self._pair())
        if self.phase == _AUTO_DECIDE:
            if action_token == self._action("SWAP"):
                self.array[self.cursor], self.array[self.cursor + 1] = (
                    self.array[self.cursor + 1],
                    self.array[self.cursor],
                )
                self.changed = True
            if self.cursor + 1 < self.active_end:
                self.cursor += 1
                return (self._observation("PAIR"), self._pair())
            if not self.changed:
                self.phase = _AUTO_DONE
                return (
                    self._observation("UNCHANGED"),
                    self._observation("NONE"),
                )
            self.active_end = max(1, self.active_end - 1)
            self.cursor = 0
            self.changed = False
            return (self._observation("CHANGED"), self._pair())
        if self.phase == _AUTO_DONE:
            self.phase = _AUTO_FINISHED
            self.completed = True
            return ()
        raise RuntimeError(f"unhandled auto-advance phase: {self.phase}")


def generate_adjacent_sort_trace(
    values: Sequence[int],
    vocabulary: AdjacentSortVocabulary,
) -> AdjacentSortTrace:
    """Turn a list into a canonical local action/observation transcript."""

    machine = AdjacentSortMachine(values, vocabulary)
    target_tokens: list[int] = []
    prediction_mask: list[bool] = []
    action_tokens: list[int] = []
    while not machine.finished:
        action = machine.expected_action()
        observations = machine.step(action)
        target_tokens.append(action)
        prediction_mask.append(True)
        action_tokens.append(action)
        target_tokens.extend(observations)
        prediction_mask.extend(False for _ in observations)

    expected = sorted(int(value) for value in values)
    if not machine.completed or machine.array != expected:
        raise RuntimeError("reference adjacent machine did not sort the input")
    return AdjacentSortTrace(
        input_values=tuple(int(value) for value in values),
        target_tokens=tuple(target_tokens),
        target_prediction_mask=tuple(prediction_mask),
        action_tokens=tuple(action_tokens),
        final_values=tuple(machine.array),
    )


def generate_auto_advance_sort_trace(
    values: Sequence[int],
    vocabulary: AutoAdvanceSortVocabulary,
) -> AdjacentSortTrace:
    """Turn a list into an executor-controlled adjacent-sort transcript."""

    machine = AutoAdvanceSortMachine(values, vocabulary)
    target_tokens: list[int] = []
    prediction_mask: list[bool] = []
    action_tokens: list[int] = []
    while not machine.finished:
        action = machine.expected_action()
        observations = machine.step(action)
        target_tokens.append(action)
        prediction_mask.append(True)
        action_tokens.append(action)
        target_tokens.extend(observations)
        prediction_mask.extend(False for _ in observations)

    expected = sorted(int(value) for value in values)
    if not machine.completed or machine.array != expected:
        raise RuntimeError("reference auto-advance machine did not sort the input")
    return AdjacentSortTrace(
        input_values=tuple(int(value) for value in values),
        target_tokens=tuple(target_tokens),
        target_prediction_mask=tuple(prediction_mask),
        action_tokens=tuple(action_tokens),
        final_values=tuple(machine.array),
    )


def execute_adjacent_sort_actions(
    values: Sequence[int],
    action_tokens: Sequence[int],
    vocabulary: AdjacentSortVocabulary,
    *,
    timed_out: bool = False,
) -> AdjacentSortRollout:
    """Execute a supplied adjacent-action stream until completion or error."""

    machine = AdjacentSortMachine(values, vocabulary)
    consumed: list[int] = []
    for action in action_tokens:
        if machine.finished:
            break
        consumed.append(int(action))
        machine.step(int(action))
    return AdjacentSortRollout(
        action_tokens=tuple(consumed),
        final_values=tuple(machine.array),
        completed=machine.completed,
        valid_execution=machine.valid,
        timed_out=timed_out and not machine.finished,
    )


def execute_auto_advance_sort_actions(
    values: Sequence[int],
    action_tokens: Sequence[int],
    vocabulary: AutoAdvanceSortVocabulary,
    *,
    timed_out: bool = False,
) -> AdjacentSortRollout:
    """Execute auto-advance actions until completion or the first error."""

    machine = AutoAdvanceSortMachine(values, vocabulary)
    consumed: list[int] = []
    for action in action_tokens:
        if machine.finished:
            break
        consumed.append(int(action))
        machine.step(int(action))
    return AdjacentSortRollout(
        action_tokens=tuple(consumed),
        final_values=tuple(machine.array),
        completed=machine.completed,
        valid_execution=machine.valid,
        timed_out=timed_out and not machine.finished,
    )


def replay_adjacent_sort_transcript(
    values: Sequence[int],
    generated_tokens: Sequence[int],
    vocabulary: AdjacentSortVocabulary,
) -> AdjacentSortTranscriptRollout:
    """Replay a fully generated transcript without feeding it tool results."""

    done_token = vocabulary.action_token("DONE")
    action_tokens = frozenset(vocabulary.action_tokens)
    machine = AdjacentSortMachine(values, vocabulary)
    raw_tokens: list[int] = []
    for raw_token in generated_tokens:
        token = int(raw_token)
        raw_tokens.append(token)
        if token == done_token:
            break

    actions: list[int] = []
    observation_correct = 0
    observations_valid = True
    syntax_valid = True
    cursor = 0
    while cursor < len(raw_tokens) and not machine.finished:
        action = raw_tokens[cursor]
        if action not in action_tokens:
            syntax_valid = False
            break
        actions.append(action)
        cursor += 1
        expected_observations = machine.step(action)
        if not machine.valid:
            break

        for expected_observation in expected_observations:
            if cursor >= len(raw_tokens):
                syntax_valid = False
                observations_valid = False
                break
            generated_observation = raw_tokens[cursor]
            cursor += 1
            expected_is_value = (
                vocabulary.value_token(0)
                <= expected_observation
                <= vocabulary.value_token(vocabulary.symbol_count - 1)
            )
            if expected_is_value:
                token_is_valid = (
                    vocabulary.value_token(0)
                    <= generated_observation
                    <= vocabulary.value_token(vocabulary.symbol_count - 1)
                )
            else:
                token_is_valid = (
                    vocabulary.observation_token_offset
                    <= generated_observation
                    < vocabulary.size
                )
            syntax_valid = syntax_valid and token_is_valid
            observation_matches = generated_observation == expected_observation
            observations_valid = observations_valid and observation_matches
            observation_correct += int(observation_matches)
        if not syntax_valid:
            break

    completed = machine.completed and machine.valid
    return AdjacentSortTranscriptRollout(
        generated_tokens=tuple(raw_tokens),
        action_tokens=tuple(actions),
        final_values=tuple(machine.array),
        completed=completed,
        valid_execution=machine.valid,
        syntax_valid=syntax_valid and completed,
        observations_valid=observations_valid and completed,
        observation_correct=observation_correct,
        timed_out=done_token not in raw_tokens,
    )


def replay_auto_advance_sort_transcript(
    values: Sequence[int],
    generated_tokens: Sequence[int],
    vocabulary: AutoAdvanceSortVocabulary,
) -> AdjacentSortTranscriptRollout:
    """Replay a generated auto-advance transcript without live tool results."""

    done_token = vocabulary.action_token("DONE")
    action_tokens = frozenset(vocabulary.action_tokens)
    machine = AutoAdvanceSortMachine(values, vocabulary)
    raw_tokens: list[int] = []
    for raw_token in generated_tokens:
        token = int(raw_token)
        raw_tokens.append(token)
        if token == done_token:
            break

    actions: list[int] = []
    observation_correct = 0
    observations_valid = True
    syntax_valid = True
    cursor = 0
    while cursor < len(raw_tokens) and not machine.finished:
        action = raw_tokens[cursor]
        if action not in action_tokens:
            syntax_valid = False
            break
        actions.append(action)
        cursor += 1
        expected_observations = machine.step(action)
        if not machine.valid:
            break

        for expected_observation in expected_observations:
            if cursor >= len(raw_tokens):
                syntax_valid = False
                observations_valid = False
                break
            generated_observation = raw_tokens[cursor]
            cursor += 1
            expected_is_value = (
                vocabulary.value_token(0)
                <= expected_observation
                <= vocabulary.value_token(vocabulary.symbol_count - 1)
            )
            if expected_is_value:
                token_is_valid = (
                    vocabulary.value_token(0)
                    <= generated_observation
                    <= vocabulary.value_token(vocabulary.symbol_count - 1)
                )
            else:
                token_is_valid = (
                    vocabulary.observation_token_offset
                    <= generated_observation
                    < vocabulary.size
                )
            syntax_valid = syntax_valid and token_is_valid
            observation_matches = generated_observation == expected_observation
            observations_valid = observations_valid and observation_matches
            observation_correct += int(observation_matches)
        if not syntax_valid:
            break

    completed = machine.completed and machine.valid
    return AdjacentSortTranscriptRollout(
        generated_tokens=tuple(raw_tokens),
        action_tokens=tuple(actions),
        final_values=tuple(machine.array),
        completed=completed,
        valid_execution=machine.valid,
        syntax_valid=syntax_valid and completed,
        observations_valid=observations_valid and completed,
        observation_correct=observation_correct,
        timed_out=done_token not in raw_tokens,
    )
