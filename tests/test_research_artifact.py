from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_research_artifact.py"


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_research_artifact",
        VALIDATOR,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_research_registry_and_documentation_are_complete() -> None:
    validator = load_validator()
    assert validator.validate() == []


def test_registry_has_both_research_tracks_and_all_status_classes() -> None:
    validator = load_validator()
    registry = validator.load_registry()
    tracks = {experiment["track"] for experiment in registry["experiments"]}
    statuses = {
        experiment["status"] for experiment in registry["experiments"]
    }

    assert "length_generalisation" in tracks
    assert "shortcut_learning" in tracks
    assert {"confirmed", "preliminary", "negative", "inconclusive", "archived"} <= statuses


def test_registry_owns_every_source_package() -> None:
    validator = load_validator()
    registry = validator.load_registry()
    source_packages = set(registry["source_packages"])
    experiment_sources = {
        source
        for experiment in registry["experiments"]
        for source in experiment["source_packages"]
    }

    assert source_packages == {
        "core",
        "length_generalisation",
        "shortcut_learning",
        "transfer",
    }
    assert experiment_sources == source_packages


def test_console_scripts_resolve_after_source_reorganization() -> None:
    validator = load_validator()
    assert validator.validate_console_scripts() == []
