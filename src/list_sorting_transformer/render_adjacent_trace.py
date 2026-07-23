"""Render the adjacent-pair sorting instructions for one list."""

from __future__ import annotations

import argparse

from .adjacent_sort import generate_adjacent_sort_trace
from .render_pointer_trace import parse_values
from .tokens import AdjacentSortVocabulary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("values", type=parse_values)
    parser.add_argument(
        "--representation",
        choices=("numbers", "alphabet"),
        default="numbers",
    )
    parser.add_argument("--symbol-count", type=int, default=10)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    vocabulary = AdjacentSortVocabulary(
        representation=args.representation,
        symbol_count=args.symbol_count,
    )
    trace = generate_adjacent_sort_trace(args.values, vocabulary)
    cursor = 0
    while cursor < len(trace.target_tokens):
        action = trace.target_tokens[cursor]
        rendered_action = vocabulary.render_tokens([action])
        cursor += 1
        observations = []
        while (
            cursor < len(trace.target_tokens)
            and not trace.target_prediction_mask[cursor]
        ):
            observations.append(trace.target_tokens[cursor])
            cursor += 1
        if observations:
            print(
                f"{rendered_action} -> "
                f"{vocabulary.render_tokens(observations)}"
            )
        else:
            print(rendered_action)
    rendered_result = ",".join(
        vocabulary.render_value(value) for value in trace.final_values
    )
    print(f"result: {rendered_result}")


if __name__ == "__main__":
    main()
