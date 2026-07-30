#!/usr/bin/env python3
"""Validate the research inventory, evidence, and documentation contracts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "experiments" / "registry.json"
INDEX_PATH = ROOT / "docs" / "experiment-index.md"
CRITICAL_DOCS = (
    ROOT / "README.md",
    INDEX_PATH,
    ROOT / "docs" / "length-generalisation.md",
    ROOT / "docs" / "shortcut-learning.md",
    ROOT / "docs" / "adding-experiments.md",
    ROOT / "docs" / "archive.md",
    ROOT / "docs" / "core-sequence-tasks.md",
    ROOT / "docs" / "metrics.md",
    ROOT / "docs" / "rasp_transfer_report.md",
    ROOT / "docs" / "language_model_transfer_report.md",
    ROOT / "experiments" / "README.md",
    ROOT / "experiments" / "hard_attention_eggroll" / "README.md",
    ROOT / "experiments" / "sparse_attention_adam" / "README.md",
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
CRITICAL_MARKER = re.compile(r"\b(?:TODO|TBD|FIXME):", re.IGNORECASE)
REGISTRY_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_registry() -> dict[str, Any]:
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def tracked_roots(directory: str) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", directory],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    roots: set[str] = set()
    prefix = f"{directory}/"
    for line in result.stdout.splitlines():
        if not line.startswith(prefix):
            continue
        remainder = line[len(prefix) :]
        roots.add(remainder.split("/", 1)[0])
    return roots


def experiment_directories() -> set[str]:
    return {
        path.name
        for path in (ROOT / "experiments").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }


def validate_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    statuses = set(registry.get("statuses", ()))
    tracks = set(registry.get("tracks", ()))
    experiments = registry.get("experiments")
    if registry.get("schema_version") != 1:
        errors.append("registry schema_version must be 1")
    if not isinstance(experiments, list) or not experiments:
        return errors + ["registry experiments must be a non-empty list"]

    required = {
        "id",
        "track",
        "status",
        "question",
        "training_domain",
        "evaluation_domain",
        "locations",
        "artifact_roots",
        "evidence",
        "reproduce",
        "summary",
    }
    ids: set[str] = set()
    covered_directories: set[str] = set()
    covered_artifacts: set[str] = set()
    index_text = INDEX_PATH.read_text(encoding="utf-8")

    for number, experiment in enumerate(experiments, start=1):
        missing = required - set(experiment)
        if missing:
            errors.append(
                f"registry experiment {number} missing: {', '.join(sorted(missing))}"
            )
            continue
        experiment_id = experiment["id"]
        label = f"experiment {experiment_id!r}"
        if not isinstance(experiment_id, str) or not REGISTRY_ID.fullmatch(
            experiment_id
        ):
            errors.append(f"{label} has an invalid id")
        if experiment_id in ids:
            errors.append(f"duplicate registry id: {experiment_id}")
        ids.add(experiment_id)
        if experiment["status"] not in statuses:
            errors.append(f"{label} has unknown status {experiment['status']!r}")
        if experiment["track"] not in tracks:
            errors.append(f"{label} has unknown track {experiment['track']!r}")
        if not str(experiment["question"]).endswith("?"):
            errors.append(f"{label} question must end in '?'")
        if f'id="exp-{experiment_id}"' not in index_text:
            errors.append(f"{label} has no section in docs/experiment-index.md")

        for field in ("locations", "evidence"):
            values = experiment[field]
            if not isinstance(values, list) or not values:
                errors.append(f"{label} {field} must be a non-empty list")
                continue
            for value in values:
                path = ROOT / value
                if not path.exists():
                    errors.append(f"{label} missing {field[:-1]} path: {value}")
                parts = Path(value).parts
                if len(parts) >= 2 and parts[0] == "experiments":
                    covered_directories.add(parts[1])

        artifact_roots = experiment["artifact_roots"]
        if not isinstance(artifact_roots, list):
            errors.append(f"{label} artifact_roots must be a list")
        else:
            covered_artifacts.update(artifact_roots)

        reproduce = experiment["reproduce"]
        if not isinstance(reproduce, list) or not reproduce:
            errors.append(f"{label} reproduce must be a non-empty list")
        else:
            for command in reproduce:
                errors.extend(validate_command_path(label, command))

        for url in experiment.get("wandb", ()):
            if not url.startswith("https://wandb.ai/"):
                errors.append(f"{label} has invalid W&B URL: {url}")

    missing_directories = experiment_directories() - covered_directories
    if missing_directories:
        errors.append(
            "unregistered experiment directories: "
            + ", ".join(sorted(missing_directories))
        )
    missing_artifacts = tracked_roots("artifacts") - covered_artifacts
    if missing_artifacts:
        errors.append(
            "unregistered tracked artifact roots: "
            + ", ".join(sorted(missing_artifacts))
        )
    extra_artifacts = covered_artifacts - tracked_roots("artifacts")
    if extra_artifacts:
        errors.append(
            "registry artifact roots are not tracked: "
            + ", ".join(sorted(extra_artifacts))
        )
    return errors


def validate_command_path(label: str, command: str) -> list[str]:
    errors: list[str] = []
    tokens = command.split()
    candidates: list[str] = []
    for token in tokens:
        clean = token.strip("'\"")
        if clean.startswith(("experiments/", "scripts/")):
            candidates.append(clean)
    for candidate in candidates:
        if not (ROOT / candidate).exists():
            errors.append(f"{label} command references missing path: {candidate}")
    return errors


def validate_markdown() -> list[str]:
    errors: list[str] = []
    for document in CRITICAL_DOCS:
        if not document.exists():
            errors.append(f"missing critical document: {_relative(document)}")
            continue
        text = document.read_text(encoding="utf-8")
        for match in CRITICAL_MARKER.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"critical marker in {_relative(document)}:{line}: {match.group(0)}"
            )
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_text = target.split("#", 1)[0]
            if not path_text:
                continue
            resolved = (document.parent / path_text).resolve()
            if not resolved.exists():
                errors.append(
                    f"broken link in {_relative(document)}: {raw_target}"
                )
    return errors


def validate() -> list[str]:
    errors = validate_registry(load_registry())
    errors.extend(validate_markdown())
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable validation summary",
    )
    args = parser.parse_args()
    errors = validate()
    summary = {
        "status": "failed" if errors else "passed",
        "registry": _relative(REGISTRY_PATH),
        "experiment_count": len(load_registry()["experiments"]),
        "error_count": len(errors),
        "errors": errors,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    elif errors:
        print("Research artifact validation failed:")
        for error in errors:
            print(f"- {error}")
    else:
        print(
            "Research artifact validation passed "
            f"({summary['experiment_count']} experiments)."
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
