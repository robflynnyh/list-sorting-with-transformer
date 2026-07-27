import argparse
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "learned_backward_shortcuts"
    / "on_policy_collapse_population.py"
)
SPEC = importlib.util.spec_from_file_location(
    "on_policy_collapse_population",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_events() -> None:
    assert MODULE.parse_events("12:30,4:9") == ((12, 30), (4, 9))


@pytest.mark.parametrize(
    "value",
    ("", "12", "12:x", "12:0", "-1:3", "12:3,12:4"),
)
def test_parse_events_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        MODULE.parse_events(value)


def test_dense_evaluation_steps_clamps_at_first_step() -> None:
    assert MODULE.dense_evaluation_steps(2, radius=3) == (1, 2, 3, 4, 5)
