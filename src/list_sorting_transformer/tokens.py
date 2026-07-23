"""Token definitions for comma-separated symbol-list sorting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


PAD = 0
BOS = 1
SEP = 2
EOS = 3
COMMA = 4
VALUE_OFFSET = 5

QUICKSORT_OPERATIONS = (
    "CHECK_RANGE",
    "PUSH",
    "POP",
    "LOAD_PIVOT",
    "SET_LT",
    "SET_SCAN",
    "SET_GT",
    "COMPARE",
    "SWAP",
    "INC_LT",
    "INC_SCAN",
    "DEC_GT",
    "PARTITION_DONE",
    "ARRAY",
    "DONE",
    "ANSWER",
)
QUICKSORT_MARKERS = (
    "ACTIVE",
    "SKIP",
    "LESS",
    "EQUAL",
    "GREATER",
    "IDX",
    "NEG",
)
INDEX_DIGIT_COUNT = 10

POINTER_QUICKSORT_ACTIONS = (
    "INIT_RANGE",
    "CHECK_RANGE",
    "LOAD_PIVOT_LO",
    "SET_LT_LO",
    "SET_SCAN_LO",
    "SET_GT_HI",
    "CHECK_SCAN_GT",
    "GET_SCAN",
    "GET_PIVOT",
    "BRANCH_LESS",
    "BRANCH_EQUAL",
    "BRANCH_GREATER",
    "SWAP_LT_SCAN",
    "SWAP_SCAN_GT",
    "MOVE_LT_RIGHT",
    "MOVE_SCAN_RIGHT",
    "MOVE_GT_LEFT",
    "PARTITION_DONE",
    "CHECK_RIGHT",
    "PUSH_RIGHT",
    "CHECK_LEFT",
    "PUSH_LEFT",
    "CHECK_STACK",
    "POP_RANGE",
    "DONE",
)
POINTER_QUICKSORT_OBSERVATIONS = (
    "OK",
    "ACTIVE",
    "SKIP",
    "IN_RANGE",
    "PAST",
    "NONEMPTY",
    "EMPTY",
    "INVALID",
)
ADJACENT_SORT_ACTIONS = (
    "READ_PAIR",
    "KEEP",
    "SWAP",
    "RIGHT",
    "END_PASS",
    "RESET",
    "DONE",
)
ADJACENT_SORT_OBSERVATIONS = (
    "CHANGED",
    "UNCHANGED",
    "INVALID",
)
AUTO_ADVANCE_SORT_ACTIONS = (
    "READ_PAIR",
    "KEEP",
    "SWAP",
    "DONE",
)
AUTO_ADVANCE_SORT_OBSERVATIONS = (
    "PAIR",
    "CHANGED",
    "UNCHANGED",
    "NONE",
    "INVALID",
)
LOCAL_WINDOW_SORT_ACTIONS = (
    "KEEP",
    "SWAP",
    "DONE",
)
LOCAL_WINDOW_SORT_MARKERS = (
    "WINDOW",
    "PTR",
    "NO_PTR",
    "LEFT_EDGE",
    "ACTIVE_END",
    "INITIAL",
    "ADVANCE",
    "NEW_PASS",
    "FINISHED",
    "PASS_CLEAN",
    "PASS_CHANGED",
    "WINDOW_END",
)
_POINTER_ACTION_INDEX = {
    name: index for index, name in enumerate(POINTER_QUICKSORT_ACTIONS)
}
_POINTER_OBSERVATION_INDEX = {
    name: index for index, name in enumerate(POINTER_QUICKSORT_OBSERVATIONS)
}
_ADJACENT_ACTION_INDEX = {
    name: index for index, name in enumerate(ADJACENT_SORT_ACTIONS)
}
_ADJACENT_OBSERVATION_INDEX = {
    name: index for index, name in enumerate(ADJACENT_SORT_OBSERVATIONS)
}
_AUTO_ADVANCE_ACTION_INDEX = {
    name: index for index, name in enumerate(AUTO_ADVANCE_SORT_ACTIONS)
}
_AUTO_ADVANCE_OBSERVATION_INDEX = {
    name: index for index, name in enumerate(AUTO_ADVANCE_SORT_OBSERVATIONS)
}
_LOCAL_WINDOW_ACTION_INDEX = {
    name: index for index, name in enumerate(LOCAL_WINDOW_SORT_ACTIONS)
}
_LOCAL_WINDOW_MARKER_INDEX = {
    name: index for index, name in enumerate(LOCAL_WINDOW_SORT_MARKERS)
}


@dataclass(frozen=True)
class SymbolVocabulary:
    """Render ordered token IDs as either categorical letters or numbers."""

    representation: str = "numbers"
    symbol_count: int = 10

    def __post_init__(self) -> None:
        if self.representation not in {"alphabet", "numbers"}:
            raise ValueError("representation must be 'alphabet' or 'numbers'")
        maximum = 26 if self.representation == "alphabet" else 10
        if not 2 <= self.symbol_count <= maximum:
            raise ValueError(
                f"{self.representation} symbol_count must be in [2, {maximum}]"
            )

    @property
    def size(self) -> int:
        return VALUE_OFFSET + self.symbol_count

    def value_token(self, value: int) -> int:
        if not 0 <= value < self.symbol_count:
            raise ValueError(
                f"list values must be in [0, {self.symbol_count - 1}]"
            )
        return VALUE_OFFSET + value

    def token_value(self, token: int) -> int:
        value = token - VALUE_OFFSET
        if not 0 <= value < self.symbol_count:
            raise ValueError(f"token {token} is not a value")
        return value

    def render_value(self, value: int) -> str:
        if not 0 <= value < self.symbol_count:
            raise ValueError("value is outside the vocabulary")
        if self.representation == "alphabet":
            return chr(ord("a") + value)
        return str(value)

    def encode_list(self, values: Sequence[int]) -> list[int]:
        if not values:
            raise ValueError("symbol lists must be non-empty")
        encoded = []
        for index, value in enumerate(values):
            if index:
                encoded.append(COMMA)
            encoded.append(self.value_token(int(value)))
        return encoded

    def encode_prompt(self, values: Sequence[int]) -> list[int]:
        return [BOS, *self.encode_list(values), SEP]

    def encode_target(self, values: Sequence[int]) -> list[int]:
        return [*self.encode_list(values), EOS]

    def encode_example(self, values: Sequence[int]) -> list[int]:
        sorted_values = sorted(int(value) for value in values)
        return [*self.encode_prompt(values), *self.encode_target(sorted_values)]

    def decode_list(self, tokens: Sequence[int]) -> list[int] | None:
        """Decode a generated target, requiring exact comma grammar and EOS."""

        output = []
        expect_value = True
        saw_eos = False
        for token_value in tokens:
            token = int(token_value)
            if token == EOS:
                saw_eos = True
                break
            if token == PAD:
                break
            if expect_value:
                if not VALUE_OFFSET <= token < VALUE_OFFSET + self.symbol_count:
                    return None
                output.append(self.token_value(token))
            elif token != COMMA:
                return None
            expect_value = not expect_value
        if not saw_eos or not output or expect_value:
            return None
        return output

    def render_tokens(self, tokens: Sequence[int]) -> str:
        structural = {
            PAD: "<pad>",
            BOS: "<bos>",
            SEP: "=",
            EOS: "<eos>",
            COMMA: ",",
        }
        parts = []
        for token_value in tokens:
            token = int(token_value)
            if token in structural:
                parts.append(structural[token])
            elif VALUE_OFFSET <= token < self.size:
                parts.append(self.render_value(self.token_value(token)))
            else:
                parts.append(f"<?>[{token}]")
        return "".join(parts)


@dataclass(frozen=True)
class QuicksortTraceVocabulary(SymbolVocabulary):
    """Extend the symbol vocabulary with quicksort instructions and indices."""

    @property
    def trace_token_offset(self) -> int:
        return VALUE_OFFSET + self.symbol_count

    @property
    def index_digit_offset(self) -> int:
        return self.trace_token_offset + len(QUICKSORT_OPERATIONS) + len(
            QUICKSORT_MARKERS
        )

    @property
    def size(self) -> int:
        return self.index_digit_offset + INDEX_DIGIT_COUNT

    @property
    def operation_tokens(self) -> frozenset[int]:
        return frozenset(self.trace_token(name) for name in QUICKSORT_OPERATIONS)

    def trace_token(self, name: str) -> int:
        names = QUICKSORT_OPERATIONS + QUICKSORT_MARKERS
        try:
            offset = names.index(name)
        except ValueError as error:
            raise ValueError(f"unknown quicksort trace token: {name}") from error
        return self.trace_token_offset + offset

    def index_digit_token(self, digit: int) -> int:
        if not 0 <= digit < INDEX_DIGIT_COUNT:
            raise ValueError("index digits must be in [0, 9]")
        return self.index_digit_offset + digit

    def encode_index(self, index: int) -> list[int]:
        encoded = [self.trace_token("IDX")]
        if index < 0:
            encoded.append(self.trace_token("NEG"))
        encoded.extend(
            self.index_digit_token(int(character))
            for character in str(abs(index))
        )
        return encoded

    def render_tokens(self, tokens: Sequence[int]) -> str:
        trace_names = QUICKSORT_OPERATIONS + QUICKSORT_MARKERS
        rendered = []
        for token_value in tokens:
            token = int(token_value)
            if token == PAD:
                rendered.append("<pad>")
            elif token == BOS:
                rendered.append("<bos>")
            elif token == SEP:
                rendered.append("<trace>")
            elif token == EOS:
                rendered.append("<eos>")
            elif token == COMMA:
                rendered.append(",")
            elif VALUE_OFFSET <= token < VALUE_OFFSET + self.symbol_count:
                rendered.append(self.render_value(self.token_value(token)))
            elif self.trace_token_offset <= token < self.index_digit_offset:
                rendered.append(
                    f"<{trace_names[token - self.trace_token_offset]}>"
                )
            elif self.index_digit_offset <= token < self.size:
                rendered.append(f"<I{token - self.index_digit_offset}>")
            else:
                rendered.append(f"<?>[{token}]")
        return " ".join(rendered)


@dataclass(frozen=True)
class PointerQuicksortVocabulary(SymbolVocabulary):
    """Vocabulary for executor-assisted quicksort without numeric indices."""

    @property
    def action_token_offset(self) -> int:
        return VALUE_OFFSET + self.symbol_count

    @property
    def observation_token_offset(self) -> int:
        return self.action_token_offset + len(POINTER_QUICKSORT_ACTIONS)

    @property
    def size(self) -> int:
        return self.observation_token_offset + len(
            POINTER_QUICKSORT_OBSERVATIONS
        )

    @property
    def action_tokens(self) -> tuple[int, ...]:
        return tuple(
            self.action_token_offset + index
            for index in range(len(POINTER_QUICKSORT_ACTIONS))
        )

    def action_token(self, name: str) -> int:
        try:
            index = _POINTER_ACTION_INDEX[name]
        except KeyError as error:
            raise ValueError(f"unknown pointer quicksort action: {name}") from error
        return self.action_token_offset + index

    def action_name(self, token: int) -> str:
        index = int(token) - self.action_token_offset
        if not 0 <= index < len(POINTER_QUICKSORT_ACTIONS):
            raise ValueError(f"token {token} is not a pointer quicksort action")
        return POINTER_QUICKSORT_ACTIONS[index]

    def observation_token(self, name: str) -> int:
        try:
            index = _POINTER_OBSERVATION_INDEX[name]
        except KeyError as error:
            raise ValueError(
                f"unknown pointer quicksort observation: {name}"
            ) from error
        return self.observation_token_offset + index

    def render_tokens(self, tokens: Sequence[int]) -> str:
        rendered = []
        for token_value in tokens:
            token = int(token_value)
            if token == PAD:
                rendered.append("<pad>")
            elif token == BOS:
                rendered.append("<bos>")
            elif token == SEP:
                rendered.append("<pointer_trace>")
            elif token == EOS:
                rendered.append("<eos>")
            elif token == COMMA:
                rendered.append(",")
            elif VALUE_OFFSET <= token < VALUE_OFFSET + self.symbol_count:
                rendered.append(self.render_value(self.token_value(token)))
            elif self.action_token_offset <= token < self.observation_token_offset:
                rendered.append(
                    f"<{POINTER_QUICKSORT_ACTIONS[token - self.action_token_offset]}>"
                )
            elif self.observation_token_offset <= token < self.size:
                rendered.append(
                    "<"
                    + POINTER_QUICKSORT_OBSERVATIONS[
                        token - self.observation_token_offset
                    ]
                    + ">"
                )
            else:
                rendered.append(f"<?>[{token}]")
        return " ".join(rendered)


@dataclass(frozen=True)
class AdjacentSortVocabulary(SymbolVocabulary):
    """Vocabulary for sorting through local adjacent-pair operations."""

    @property
    def action_token_offset(self) -> int:
        return VALUE_OFFSET + self.symbol_count

    @property
    def observation_token_offset(self) -> int:
        return self.action_token_offset + len(ADJACENT_SORT_ACTIONS)

    @property
    def size(self) -> int:
        return self.observation_token_offset + len(ADJACENT_SORT_OBSERVATIONS)

    @property
    def action_tokens(self) -> tuple[int, ...]:
        return tuple(
            self.action_token_offset + index
            for index in range(len(ADJACENT_SORT_ACTIONS))
        )

    def action_token(self, name: str) -> int:
        try:
            index = _ADJACENT_ACTION_INDEX[name]
        except KeyError as error:
            raise ValueError(f"unknown adjacent-sort action: {name}") from error
        return self.action_token_offset + index

    def action_name(self, token: int) -> str:
        index = int(token) - self.action_token_offset
        if not 0 <= index < len(ADJACENT_SORT_ACTIONS):
            raise ValueError(f"token {token} is not an adjacent-sort action")
        return ADJACENT_SORT_ACTIONS[index]

    def observation_token(self, name: str) -> int:
        try:
            index = _ADJACENT_OBSERVATION_INDEX[name]
        except KeyError as error:
            raise ValueError(f"unknown adjacent-sort observation: {name}") from error
        return self.observation_token_offset + index

    def render_tokens(self, tokens: Sequence[int]) -> str:
        rendered = []
        for token_value in tokens:
            token = int(token_value)
            if token == PAD:
                rendered.append("<pad>")
            elif token == BOS:
                rendered.append("<bos>")
            elif token == SEP:
                rendered.append("<adjacent_trace>")
            elif token == EOS:
                rendered.append("<eos>")
            elif token == COMMA:
                rendered.append(",")
            elif VALUE_OFFSET <= token < VALUE_OFFSET + self.symbol_count:
                rendered.append(self.render_value(self.token_value(token)))
            elif self.action_token_offset <= token < self.observation_token_offset:
                rendered.append(
                    f"<{ADJACENT_SORT_ACTIONS[token - self.action_token_offset]}>"
                )
            elif self.observation_token_offset <= token < self.size:
                rendered.append(
                    "<"
                    + ADJACENT_SORT_OBSERVATIONS[
                        token - self.observation_token_offset
                    ]
                    + ">"
                )
            else:
                rendered.append(f"<?>[{token}]")
        return " ".join(rendered)


@dataclass(frozen=True)
class AutoAdvanceSortVocabulary(SymbolVocabulary):
    """Vocabulary for adjacent sorting with executor-controlled movement."""

    @property
    def action_token_offset(self) -> int:
        return VALUE_OFFSET + self.symbol_count

    @property
    def observation_token_offset(self) -> int:
        return self.action_token_offset + len(AUTO_ADVANCE_SORT_ACTIONS)

    @property
    def pair_token_offset(self) -> int:
        return self.observation_token_offset + len(
            AUTO_ADVANCE_SORT_OBSERVATIONS
        )

    @property
    def size(self) -> int:
        return self.pair_token_offset + self.symbol_count**2

    @property
    def action_tokens(self) -> tuple[int, ...]:
        return tuple(
            self.action_token_offset + index
            for index in range(len(AUTO_ADVANCE_SORT_ACTIONS))
        )

    def action_token(self, name: str) -> int:
        try:
            index = _AUTO_ADVANCE_ACTION_INDEX[name]
        except KeyError as error:
            raise ValueError(
                f"unknown auto-advance sort action: {name}"
            ) from error
        return self.action_token_offset + index

    def action_name(self, token: int) -> str:
        index = int(token) - self.action_token_offset
        if not 0 <= index < len(AUTO_ADVANCE_SORT_ACTIONS):
            raise ValueError(f"token {token} is not an auto-advance sort action")
        return AUTO_ADVANCE_SORT_ACTIONS[index]

    def observation_token(self, name: str) -> int:
        try:
            index = _AUTO_ADVANCE_OBSERVATION_INDEX[name]
        except KeyError as error:
            raise ValueError(
                f"unknown auto-advance sort observation: {name}"
            ) from error
        return self.observation_token_offset + index

    def pair_token(self, left: int, right: int) -> int:
        self.value_token(left)
        self.value_token(right)
        return self.pair_token_offset + left * self.symbol_count + right

    def token_pair(self, token: int) -> tuple[int, int]:
        index = int(token) - self.pair_token_offset
        if not 0 <= index < self.symbol_count**2:
            raise ValueError(f"token {token} is not an auto-advance pair")
        return divmod(index, self.symbol_count)

    def render_tokens(self, tokens: Sequence[int]) -> str:
        rendered = []
        for token_value in tokens:
            token = int(token_value)
            if token == PAD:
                rendered.append("<pad>")
            elif token == BOS:
                rendered.append("<bos>")
            elif token == SEP:
                rendered.append("<auto_advance_trace>")
            elif token == EOS:
                rendered.append("<eos>")
            elif token == COMMA:
                rendered.append(",")
            elif VALUE_OFFSET <= token < VALUE_OFFSET + self.symbol_count:
                rendered.append(self.render_value(self.token_value(token)))
            elif self.action_token_offset <= token < self.observation_token_offset:
                rendered.append(
                    f"<{AUTO_ADVANCE_SORT_ACTIONS[token - self.action_token_offset]}>"
                )
            elif self.observation_token_offset <= token < self.size:
                if token >= self.pair_token_offset:
                    left, right = self.token_pair(token)
                    rendered.append(
                        f"<PAIR_{self.render_value(left)}_{self.render_value(right)}>"
                    )
                    continue
                rendered.append(
                    "<"
                    + AUTO_ADVANCE_SORT_OBSERVATIONS[
                        token - self.observation_token_offset
                    ]
                    + ">"
                )
            else:
                rendered.append(f"<?>[{token}]")
        return " ".join(rendered)


@dataclass(frozen=True)
class LocalWindowSortVocabulary(SymbolVocabulary):
    """Vocabulary for fixed-width windows around a bubble-sort pointer."""

    @property
    def separator_token(self) -> int:
        return SEP

    @property
    def action_token_offset(self) -> int:
        return VALUE_OFFSET + self.symbol_count

    @property
    def window_token_offset(self) -> int:
        return self.action_token_offset + len(LOCAL_WINDOW_SORT_ACTIONS)

    @property
    def size(self) -> int:
        return self.window_token_offset + len(LOCAL_WINDOW_SORT_MARKERS)

    @property
    def action_tokens(self) -> tuple[int, ...]:
        return tuple(
            self.action_token_offset + index
            for index in range(len(LOCAL_WINDOW_SORT_ACTIONS))
        )

    def action_token(self, name: str) -> int:
        try:
            index = _LOCAL_WINDOW_ACTION_INDEX[name]
        except KeyError as error:
            raise ValueError(
                f"unknown local-window sort action: {name}"
            ) from error
        return self.action_token_offset + index

    def action_name(self, token: int) -> str:
        index = int(token) - self.action_token_offset
        if not 0 <= index < len(LOCAL_WINDOW_SORT_ACTIONS):
            raise ValueError(
                f"token {token} is not a local-window sort action"
            )
        return LOCAL_WINDOW_SORT_ACTIONS[index]

    def window_token(self, name: str) -> int:
        try:
            index = _LOCAL_WINDOW_MARKER_INDEX[name]
        except KeyError as error:
            raise ValueError(
                f"unknown local-window marker: {name}"
            ) from error
        return self.window_token_offset + index

    def window_name(self, token: int) -> str:
        index = int(token) - self.window_token_offset
        if not 0 <= index < len(LOCAL_WINDOW_SORT_MARKERS):
            raise ValueError(f"token {token} is not a local-window marker")
        return LOCAL_WINDOW_SORT_MARKERS[index]

    def render_tokens(self, tokens: Sequence[int]) -> str:
        rendered = []
        for token_value in tokens:
            token = int(token_value)
            if token == PAD:
                rendered.append("<pad>")
            elif token == BOS:
                rendered.append("<bos>")
            elif token == SEP:
                rendered.append("<local_window_trace>")
            elif token == EOS:
                rendered.append("<eos>")
            elif token == COMMA:
                rendered.append(",")
            elif VALUE_OFFSET <= token < VALUE_OFFSET + self.symbol_count:
                rendered.append(self.render_value(self.token_value(token)))
            elif self.action_token_offset <= token < self.window_token_offset:
                rendered.append(
                    "<"
                    + LOCAL_WINDOW_SORT_ACTIONS[
                        token - self.action_token_offset
                    ]
                    + ">"
                )
            elif self.window_token_offset <= token < self.size:
                rendered.append(
                    "<"
                    + LOCAL_WINDOW_SORT_MARKERS[
                        token - self.window_token_offset
                    ]
                    + ">"
                )
            else:
                rendered.append(f"<?>[{token}]")
        return " ".join(rendered)


def make_vocabulary(
    task: str,
    *,
    representation: str,
    symbol_count: int,
) -> SymbolVocabulary:
    if task == "direct":
        return SymbolVocabulary(representation, symbol_count)
    if task == "quicksort_trace":
        return QuicksortTraceVocabulary(representation, symbol_count)
    if task == "pointer_quicksort":
        return PointerQuicksortVocabulary(representation, symbol_count)
    if task == "pointer_quicksort_no_tool":
        return PointerQuicksortVocabulary(representation, symbol_count)
    if task == "adjacent_sort":
        return AdjacentSortVocabulary(representation, symbol_count)
    if task == "adjacent_sort_no_tool":
        return AdjacentSortVocabulary(representation, symbol_count)
    if task == "adjacent_sort_auto_advance":
        return AutoAdvanceSortVocabulary(representation, symbol_count)
    if task == "adjacent_sort_auto_advance_no_tool":
        return AutoAdvanceSortVocabulary(representation, symbol_count)
    if task == "adjacent_sort_local_window":
        return LocalWindowSortVocabulary(representation, symbol_count)
    raise ValueError(f"unsupported sorting task: {task}")


DEFAULT_VOCABULARY = SymbolVocabulary()
VOCAB_SIZE = DEFAULT_VOCABULARY.size


def encode_prompt(values: Sequence[int]) -> list[int]:
    return DEFAULT_VOCABULARY.encode_prompt(values)


def encode_example(values: Sequence[int]) -> list[int]:
    return DEFAULT_VOCABULARY.encode_example(values)


def decode_digit_list(tokens: Sequence[int]) -> list[int] | None:
    return DEFAULT_VOCABULARY.decode_list(tokens)
