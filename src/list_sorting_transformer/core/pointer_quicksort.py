"""Executor-assisted quicksort traces using relative pointer operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .tokens import (
    POINTER_QUICKSORT_ACTIONS,
    PointerQuicksortVocabulary,
)


_ACTION_INDEX = {
    name: index for index, name in enumerate(POINTER_QUICKSORT_ACTIONS)
}

_INIT = 0
_CHECK_RANGE = 1
_LOAD_PIVOT = 2
_SET_LT = 3
_SET_SCAN = 4
_SET_GT = 5
_CHECK_SCAN = 6
_GET_SCAN = 7
_GET_PIVOT = 8
_BRANCH = 9
_SWAP_LT_SCAN = 10
_MOVE_LT = 11
_MOVE_SCAN = 12
_SWAP_SCAN_GT = 13
_MOVE_GT = 14
_PARTITION_DONE = 15
_CHECK_RIGHT = 16
_PUSH_RIGHT = 17
_CHECK_LEFT = 18
_PUSH_LEFT = 19
_CHECK_STACK = 20
_POP_RANGE = 21
_DONE = 22
_FINISHED = 23
_INVALID = 24

_PHASE_ACTION_INDEX = {
    _INIT: _ACTION_INDEX["INIT_RANGE"],
    _CHECK_RANGE: _ACTION_INDEX["CHECK_RANGE"],
    _LOAD_PIVOT: _ACTION_INDEX["LOAD_PIVOT_LO"],
    _SET_LT: _ACTION_INDEX["SET_LT_LO"],
    _SET_SCAN: _ACTION_INDEX["SET_SCAN_LO"],
    _SET_GT: _ACTION_INDEX["SET_GT_HI"],
    _CHECK_SCAN: _ACTION_INDEX["CHECK_SCAN_GT"],
    _GET_SCAN: _ACTION_INDEX["GET_SCAN"],
    _GET_PIVOT: _ACTION_INDEX["GET_PIVOT"],
    _SWAP_LT_SCAN: _ACTION_INDEX["SWAP_LT_SCAN"],
    _MOVE_LT: _ACTION_INDEX["MOVE_LT_RIGHT"],
    _MOVE_SCAN: _ACTION_INDEX["MOVE_SCAN_RIGHT"],
    _SWAP_SCAN_GT: _ACTION_INDEX["SWAP_SCAN_GT"],
    _MOVE_GT: _ACTION_INDEX["MOVE_GT_LEFT"],
    _PARTITION_DONE: _ACTION_INDEX["PARTITION_DONE"],
    _CHECK_RIGHT: _ACTION_INDEX["CHECK_RIGHT"],
    _PUSH_RIGHT: _ACTION_INDEX["PUSH_RIGHT"],
    _CHECK_LEFT: _ACTION_INDEX["CHECK_LEFT"],
    _PUSH_LEFT: _ACTION_INDEX["PUSH_LEFT"],
    _CHECK_STACK: _ACTION_INDEX["CHECK_STACK"],
    _POP_RANGE: _ACTION_INDEX["POP_RANGE"],
    _DONE: _ACTION_INDEX["DONE"],
}


@dataclass(frozen=True)
class PointerQuicksortTrace:
    """A canonical action/observation transcript and its sorted machine state."""

    input_values: tuple[int, ...]
    target_tokens: tuple[int, ...]
    target_prediction_mask: tuple[bool, ...]
    action_tokens: tuple[int, ...]
    final_values: tuple[int, ...]


@dataclass(frozen=True)
class PointerQuicksortRollout:
    """The outcome of executing actions predicted by a model."""

    action_tokens: tuple[int, ...]
    final_values: tuple[int, ...]
    completed: bool
    valid_execution: bool
    timed_out: bool


@dataclass(frozen=True)
class PointerQuicksortTranscriptRollout:
    """Offline replay of a transcript generated without executor feedback."""

    generated_tokens: tuple[int, ...]
    action_tokens: tuple[int, ...]
    final_values: tuple[int, ...]
    completed: bool
    valid_execution: bool
    syntax_valid: bool
    observations_valid: bool
    observation_correct: int
    timed_out: bool


class PointerQuicksortMachine:
    """Strict state machine for one canonical three-way quicksort program."""

    __slots__ = (
        "array",
        "stack",
        "lo",
        "hi",
        "pivot",
        "lt",
        "scan",
        "gt",
        "phase",
        "valid",
        "completed",
        "last_error",
        "vocabulary",
    )

    def __init__(
        self,
        values: Sequence[int],
        vocabulary: PointerQuicksortVocabulary,
    ) -> None:
        if not values:
            raise ValueError("pointer quicksort requires a non-empty input")
        self.array = [int(value) for value in values]
        for value in self.array:
            vocabulary.value_token(value)
        self.stack: list[tuple[int, int]] = []
        self.lo = 0
        self.hi = 0
        self.pivot = 0
        self.lt = 0
        self.scan = 0
        self.gt = 0
        self.phase = _INIT
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

    def expected_action(self) -> int:
        """Return the sole valid next action for the current machine state."""

        if self.phase == _BRANCH:
            scanned_value = self.array[self.scan]
            if scanned_value < self.pivot:
                return self._action("BRANCH_LESS")
            if scanned_value > self.pivot:
                return self._action("BRANCH_GREATER")
            return self._action("BRANCH_EQUAL")
        try:
            action_index = _PHASE_ACTION_INDEX[self.phase]
        except KeyError as error:
            raise RuntimeError("finished machines do not have a next action") from error
        return self.vocabulary.action_token_offset + action_index

    def step(self, action_token: int) -> int | None:
        """Execute one action and return the executor-provided observation."""

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
            return self._observation("INVALID")

        if self.phase == _INIT:
            self.lo = 0
            self.hi = len(self.array) - 1
            self.phase = _CHECK_RANGE
            return self._observation("OK")
        if self.phase == _CHECK_RANGE:
            if self.lo < self.hi:
                self.phase = _LOAD_PIVOT
                return self._observation("ACTIVE")
            self.phase = _CHECK_STACK
            return self._observation("SKIP")
        if self.phase == _LOAD_PIVOT:
            self.pivot = self.array[self.lo]
            self.phase = _SET_LT
            return self.vocabulary.value_token(self.pivot)
        if self.phase == _SET_LT:
            self.lt = self.lo
            self.phase = _SET_SCAN
            return self._observation("OK")
        if self.phase == _SET_SCAN:
            self.scan = self.lo
            self.phase = _SET_GT
            return self._observation("OK")
        if self.phase == _SET_GT:
            self.gt = self.hi
            self.phase = _CHECK_SCAN
            return self._observation("OK")
        if self.phase == _CHECK_SCAN:
            if self.scan <= self.gt:
                self.phase = _GET_SCAN
                return self._observation("IN_RANGE")
            self.phase = _PARTITION_DONE
            return self._observation("PAST")
        if self.phase == _GET_SCAN:
            self.phase = _GET_PIVOT
            return self.vocabulary.value_token(self.array[self.scan])
        if self.phase == _GET_PIVOT:
            self.phase = _BRANCH
            return self.vocabulary.value_token(self.pivot)
        if self.phase == _BRANCH:
            if action_token == self._action("BRANCH_LESS"):
                self.phase = _SWAP_LT_SCAN
            elif action_token == self._action("BRANCH_GREATER"):
                self.phase = _SWAP_SCAN_GT
            else:
                self.phase = _MOVE_SCAN
            return self._observation("OK")
        if self.phase == _SWAP_LT_SCAN:
            self.array[self.lt], self.array[self.scan] = (
                self.array[self.scan],
                self.array[self.lt],
            )
            self.phase = _MOVE_LT
            return self._observation("OK")
        if self.phase == _MOVE_LT:
            self.lt += 1
            self.phase = _MOVE_SCAN
            return self._observation("OK")
        if self.phase == _MOVE_SCAN:
            self.scan += 1
            self.phase = _CHECK_SCAN
            return self._observation("OK")
        if self.phase == _SWAP_SCAN_GT:
            self.array[self.scan], self.array[self.gt] = (
                self.array[self.gt],
                self.array[self.scan],
            )
            self.phase = _MOVE_GT
            return self._observation("OK")
        if self.phase == _MOVE_GT:
            self.gt -= 1
            self.phase = _CHECK_SCAN
            return self._observation("OK")
        if self.phase == _PARTITION_DONE:
            self.phase = _CHECK_RIGHT
            return self._observation("OK")
        if self.phase == _CHECK_RIGHT:
            if self.gt + 1 < self.hi:
                self.phase = _PUSH_RIGHT
                return self._observation("ACTIVE")
            self.phase = _CHECK_LEFT
            return self._observation("SKIP")
        if self.phase == _PUSH_RIGHT:
            self.stack.append((self.gt + 1, self.hi))
            self.phase = _CHECK_LEFT
            return self._observation("OK")
        if self.phase == _CHECK_LEFT:
            if self.lo < self.lt - 1:
                self.phase = _PUSH_LEFT
                return self._observation("ACTIVE")
            self.phase = _CHECK_STACK
            return self._observation("SKIP")
        if self.phase == _PUSH_LEFT:
            self.stack.append((self.lo, self.lt - 1))
            self.phase = _CHECK_STACK
            return self._observation("OK")
        if self.phase == _CHECK_STACK:
            if self.stack:
                self.phase = _POP_RANGE
                return self._observation("NONEMPTY")
            self.phase = _DONE
            return self._observation("EMPTY")
        if self.phase == _POP_RANGE:
            self.lo, self.hi = self.stack.pop()
            self.phase = _LOAD_PIVOT
            return self._observation("OK")
        if self.phase == _DONE:
            self.phase = _FINISHED
            self.completed = True
            return None
        raise RuntimeError(f"unhandled pointer quicksort phase: {self.phase}")


def generate_pointer_quicksort_trace(
    values: Sequence[int],
    vocabulary: PointerQuicksortVocabulary,
) -> PointerQuicksortTrace:
    """Turn a list into a compact canonical action/observation transcript."""

    machine = PointerQuicksortMachine(values, vocabulary)
    target_tokens: list[int] = []
    prediction_mask: list[bool] = []
    action_tokens: list[int] = []
    while not machine.finished:
        action = machine.expected_action()
        observation = machine.step(action)
        target_tokens.append(action)
        prediction_mask.append(True)
        action_tokens.append(action)
        if observation is not None:
            target_tokens.append(observation)
            prediction_mask.append(False)

    expected = sorted(int(value) for value in values)
    if not machine.completed or machine.array != expected:
        raise RuntimeError("reference pointer quicksort did not sort the input")
    return PointerQuicksortTrace(
        input_values=tuple(int(value) for value in values),
        target_tokens=tuple(target_tokens),
        target_prediction_mask=tuple(prediction_mask),
        action_tokens=tuple(action_tokens),
        final_values=tuple(machine.array),
    )


def execute_pointer_quicksort_actions(
    values: Sequence[int],
    action_tokens: Sequence[int],
    vocabulary: PointerQuicksortVocabulary,
    *,
    timed_out: bool = False,
) -> PointerQuicksortRollout:
    """Execute a supplied action stream until completion or its first error."""

    machine = PointerQuicksortMachine(values, vocabulary)
    consumed: list[int] = []
    for action in action_tokens:
        if machine.finished:
            break
        consumed.append(int(action))
        machine.step(int(action))
    return PointerQuicksortRollout(
        action_tokens=tuple(consumed),
        final_values=tuple(machine.array),
        completed=machine.completed,
        valid_execution=machine.valid,
        timed_out=timed_out and not machine.finished,
    )


def replay_pointer_quicksort_transcript(
    values: Sequence[int],
    generated_tokens: Sequence[int],
    vocabulary: PointerQuicksortVocabulary,
) -> PointerQuicksortTranscriptRollout:
    """Replay a fully generated transcript without feeding results to its model."""

    done_token = vocabulary.action_token("DONE")
    action_tokens = frozenset(vocabulary.action_tokens)
    machine = PointerQuicksortMachine(values, vocabulary)
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
        expected_observation = machine.step(action)
        if not machine.valid:
            break
        if expected_observation is None:
            break
        if cursor >= len(raw_tokens):
            syntax_valid = False
            observations_valid = False
            break

        generated_observation = raw_tokens[cursor]
        cursor += 1
        is_value = (
            vocabulary.value_token(0)
            <= generated_observation
            < vocabulary.value_token(vocabulary.symbol_count - 1) + 1
        )
        is_status = (
            vocabulary.observation_token_offset
            <= generated_observation
            < vocabulary.size
        )
        syntax_valid = syntax_valid and (is_value or is_status)
        observation_matches = generated_observation == expected_observation
        observations_valid = observations_valid and observation_matches
        observation_correct += int(observation_matches)

    completed = machine.completed and machine.valid
    return PointerQuicksortTranscriptRollout(
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
