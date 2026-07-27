import importlib.util
import json
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "learned_backward_shortcuts"
    / "reconstruct_collapse_population_update.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reconstruct_collapse_population_update",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_fitnesses_sorts_candidate_rows(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        json.dumps({"candidate_index": 1, "fitness": -0.2})
        + "\n"
        + json.dumps({"candidate_index": 0, "fitness": 0.3})
        + "\n"
    )

    fitnesses = MODULE.load_fitnesses(path, population_size=2)

    torch.testing.assert_close(
        fitnesses,
        torch.tensor([0.3, -0.2]),
    )


def test_load_fitnesses_requires_complete_population(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        json.dumps({"candidate_index": 1, "fitness": -0.2}) + "\n"
    )
    with pytest.raises(ValueError, match="every population index"):
        MODULE.load_fitnesses(path, population_size=2)
