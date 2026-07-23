"""Render the pointer-machine quicksort instructions for one list."""

from __future__ import annotations

import argparse

from .pointer_quicksort import generate_pointer_quicksort_trace
from .tokens import PointerQuicksortVocabulary


def parse_values(specification: str) -> list[int]:
    try:
        values = [
            int(component.strip())
            for component in specification.split(",")
            if component.strip()
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated integers"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


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
    vocabulary = PointerQuicksortVocabulary(
        representation=args.representation,
        symbol_count=args.symbol_count,
    )
    trace = generate_pointer_quicksort_trace(args.values, vocabulary)
    cursor = 0
    while cursor < len(trace.target_tokens):
        action = trace.target_tokens[cursor]
        rendered_action = vocabulary.render_tokens([action])
        cursor += 1
        if cursor == len(trace.target_tokens):
            print(rendered_action)
            break
        observation = trace.target_tokens[cursor]
        print(
            f"{rendered_action} -> "
            f"{vocabulary.render_tokens([observation])}"
        )
        cursor += 1
    rendered_result = ",".join(
        vocabulary.render_value(value) for value in trace.final_values
    )
    print(f"result: {rendered_result}")


if __name__ == "__main__":
    main()
