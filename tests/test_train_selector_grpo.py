import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "token_gradient_selector"
    / "train_selector_grpo.py"
)
SPEC = importlib.util.spec_from_file_location(
    "train_selector_grpo",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_devices() -> None:
    assert MODULE.parse_devices("cuda:0, cuda:2") == (
        "cuda:0",
        "cuda:2",
    )


def test_parse_devices_rejects_empty_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        MODULE.parse_devices(" , ")


def test_plateau_promotes_after_patience() -> None:
    state = MODULE.PlateauState()
    assert not MODULE.update_plateau(
        state,
        1.0,
        decay=0.0,
        patience=2,
        minimum_delta=0.1,
    )
    assert not MODULE.update_plateau(
        state,
        1.05,
        decay=0.0,
        patience=2,
        minimum_delta=0.1,
    )
    assert MODULE.update_plateau(
        state,
        1.06,
        decay=0.0,
        patience=2,
        minimum_delta=0.1,
    )
