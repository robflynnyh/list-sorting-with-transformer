"""Render the adjacent-pair sorting instructions for one list."""

from __future__ import annotations

import argparse

from .adjacent_sort import (
    generate_adjacent_sort_trace,
    generate_auto_advance_sort_trace,
)
from .render_pointer_trace import parse_values
from .tokens import AdjacentSortVocabulary, AutoAdvanceSortVocabulary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("values", type=parse_values)
    parser.add_argument(
        "--representation",
        choices=("numbers", "alphabet"),
        default="numbers",
    )
    parser.add_argument("--symbol-count", type=int, default=10)
    parser.add_argument(
        "--auto-advance",
        action="store_true",
        help="let the executor control cursor and pass transitions",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    vocabulary_type = (
        AutoAdvanceSortVocabulary
        if args.auto_advance
        else AdjacentSortVocabulary
    )
    trace_generator = (
        generate_auto_advance_sort_trace
        if args.auto_advance
        else generate_adjacent_sort_trace
    )
    vocabulary = vocabulary_type(args.representation, args.symbol_count)
    trace = trace_generator(args.values, vocabulary)
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
