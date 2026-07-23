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
                if not VALUE_OFFSET <= token < self.size:
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


DEFAULT_VOCABULARY = SymbolVocabulary()
VOCAB_SIZE = DEFAULT_VOCABULARY.size


def encode_prompt(values: Sequence[int]) -> list[int]:
    return DEFAULT_VOCABULARY.encode_prompt(values)


def encode_example(values: Sequence[int]) -> list[int]:
    return DEFAULT_VOCABULARY.encode_example(values)


def decode_digit_list(tokens: Sequence[int]) -> list[int] | None:
    return DEFAULT_VOCABULARY.decode_list(tokens)
