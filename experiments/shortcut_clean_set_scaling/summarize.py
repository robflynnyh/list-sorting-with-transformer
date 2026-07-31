"""Summarize endpoint clean-set scaling results without selecting on held-out data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts" / "shortcut_clean_set_scaling"
RESULT_ROOT = ROOT / "experiments" / "shortcut_clean_set_scaling" / "results"


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize_maml(run_dir: Path) -> dict[str, Any] | None:
    rows = read_rows(run_dir / "metrics.jsonl")
    endpoints = [row for row in rows if "fitness_heldout/masked_accuracy" in row]
    if not endpoints:
        return None
    config = json.loads((run_dir / "config.json").read_text())
    row = endpoints[-1]
    return {
        "method": "maml",
        "seed": config["seed"],
        "clean_examples_per_mode": config["fitness_examples"] // 2,
        "endpoint": int(row["step"]),
        "fixed_masked_accuracy": row["fitness_fixed/masked_accuracy"],
        "fixed_incorrect_accuracy": row["fitness_fixed/incorrect_accuracy"],
        "heldout_masked_accuracy": row["fitness_heldout/masked_accuracy"],
        "heldout_incorrect_accuracy": row["fitness_heldout/incorrect_accuracy"],
        "heldout_worst_mode_accuracy": min(
            row["fitness_heldout/masked_accuracy"],
            row["fitness_heldout/incorrect_accuracy"],
        ),
        "fixed_to_heldout_accuracy_gap": row["fitness_gap/accuracy"],
        "run_dir": str(run_dir.relative_to(ROOT)),
    }


def summarize_eggroll(run_dir: Path) -> dict[str, Any] | None:
    rows = read_rows(run_dir / "metrics.jsonl")
    endpoints = [
        row
        for row in rows
        if "heldout_center_rule/masked_accuracy" in row
    ]
    if not endpoints:
        return None
    config = json.loads((run_dir / "config.json").read_text())
    row = endpoints[-1]
    fixed_worst = min(
        row["center_rule/masked_accuracy"],
        row["center_rule/incorrect_accuracy"],
    )
    heldout_worst = min(
        row["heldout_center_rule/masked_accuracy"],
        row["heldout_center_rule/incorrect_accuracy"],
    )
    return {
        "method": "eggroll",
        "seed": config["seed"],
        "clean_examples_per_mode": config["fitness_examples"] // 2,
        "endpoint": int(row["generation"]) + 1,
        "fixed_masked_accuracy": row["center_rule/masked_accuracy"],
        "fixed_incorrect_accuracy": row["center_rule/incorrect_accuracy"],
        "heldout_masked_accuracy": row[
            "heldout_center_rule/masked_accuracy"
        ],
        "heldout_incorrect_accuracy": row[
            "heldout_center_rule/incorrect_accuracy"
        ],
        "heldout_worst_mode_accuracy": heldout_worst,
        "fixed_to_heldout_accuracy_gap": fixed_worst - heldout_worst,
        "run_dir": str(run_dir.relative_to(ROOT)),
    }


def main() -> None:
    summaries = []
    for method, summarize in (
        ("maml", summarize_maml),
        ("eggroll", summarize_eggroll),
    ):
        method_root = ARTIFACT_ROOT / method
        if not method_root.exists():
            continue
        run_dirs = sorted(
            path for path in method_root.iterdir() if path.is_dir()
        )
        for run_dir in run_dirs:
            if not (run_dir / "metrics.jsonl").exists():
                continue
            summary = summarize(run_dir)
            if summary is not None:
                summaries.append(summary)

    summaries.sort(
        key=lambda row: (row["method"], row["clean_examples_per_mode"])
    )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "screening_summary.json").write_text(
        json.dumps({"runs": summaries}, indent=2) + "\n"
    )
    if summaries:
        csv_path = RESULT_ROOT / "screening_summary.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)

    for row in summaries:
        print(
            f"{row['method']:7s} clean/mode={row['clean_examples_per_mode']:3d} "
            f"heldout worst={100 * row['heldout_worst_mode_accuracy']:5.1f}% "
            f"gap={100 * row['fixed_to_heldout_accuracy_gap']:+5.1f}pp"
        )


if __name__ == "__main__":
    main()
