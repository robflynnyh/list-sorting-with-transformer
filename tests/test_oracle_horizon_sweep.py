import argparse
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "token_gradient_selector"
    / "oracle_horizon_sweep.py"
)
SPEC = importlib.util.spec_from_file_location(
    "oracle_horizon_sweep",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_horizons() -> None:
    assert MODULE.parse_horizons("0,10,40") == (0, 10, 40)


@pytest.mark.parametrize(
    "value",
    ("", "10,20", "0,-1", "0,20,10", "0,10,10", "0,x"),
)
def test_parse_horizons_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        MODULE.parse_horizons(value)
