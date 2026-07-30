from __future__ import annotations

from pathlib import Path

import pytest

from experiments.learned_backward_shortcuts.candidate_routing_diagnostic import (
    parse_candidate,
)


def test_parse_candidate_extracts_generation_and_index() -> None:
    assert parse_candidate("robust=checkpoint.pt@40@36") == (
        "robust",
        Path("checkpoint.pt"),
        40,
        36,
    )


@pytest.mark.parametrize(
    "value",
    (
        "checkpoint.pt",
        "=checkpoint.pt@40@36",
        "robust=@40@36",
        "robust=checkpoint.pt@-1@36",
        "robust=checkpoint.pt@40@-1",
    ),
)
def test_parse_candidate_rejects_invalid_specification(value: str) -> None:
    with pytest.raises(Exception):
        parse_candidate(value)
