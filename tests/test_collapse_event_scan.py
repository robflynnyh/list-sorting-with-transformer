import argparse
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "learned_backward_shortcuts"
    / "collapse_event_scan.py"
)
SPEC = importlib.util.spec_from_file_location(
    "collapse_event_scan",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_generation_seeds() -> None:
    assert MODULE.parse_generation_seeds("12,3,40") == (12, 3, 40)


@pytest.mark.parametrize("value", ["", "-1,2", "2,2", "two,3"])
def test_parse_generation_seeds_rejects_invalid_values(
    value: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        MODULE.parse_generation_seeds(value)


def test_peak_events_keep_largest_losses() -> None:
    events = []
    for training_loss in (0.2, 0.8, 0.4, 1.2):
        MODULE.add_peak_event(
            events,
            event={
                "step": training_loss,
                "training_loss": training_loss,
                "trailing_100_mean": 0.1,
                "loss_excess": training_loss - 0.1,
                "loss_ratio": training_loss / 0.1,
            },
            event_count=2,
        )
    assert [event["training_loss"] for event in events] == [1.2, 0.8]


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (999, False),
        (1000, True),
        (1001, False),
        (1010, True),
        (1099, False),
        (1103, True),
    ],
)
def test_periodic_clean_evaluation_includes_final_step(
    step: int,
    expected: bool,
) -> None:
    assert MODULE.should_evaluate_clean_accuracy(
        step,
        minimum_event_step=1000,
        horizon=1103,
        evaluation_interval=10,
    ) is expected


def test_zero_interval_disables_clean_evaluation() -> None:
    assert not MODULE.should_evaluate_clean_accuracy(
        1103,
        minimum_event_step=1000,
        horizon=1103,
        evaluation_interval=0,
    )
