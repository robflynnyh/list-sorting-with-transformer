from __future__ import annotations

from list_sorting_transformer.evaluate import parse_lengths


def test_parse_lengths_supports_ranges_and_lists() -> None:
    assert parse_lengths("2-5,8,10-11") == [2, 3, 4, 5, 8, 10, 11]
