import argparse
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "learned_backward_shortcuts"
    / "fixed_rule_horizon_sweep.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fixed_rule_horizon_sweep",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_horizons_sorts_and_deduplicates() -> None:
    assert MODULE.parse_horizons("100,0,20,20") == (0, 20, 100)


@pytest.mark.parametrize("value", ["", "-1,20", "ten,20"])
def test_parse_horizons_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        MODULE.parse_horizons(value)
