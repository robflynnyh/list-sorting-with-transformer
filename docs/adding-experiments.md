# Adding an Experiment

New work should extend the research artifact rather than add another isolated
run script and progress log.

## Required Structure

1. Put the launcher and family-specific notes under
   `experiments/<family>/`.
2. Put reusable implementation in the matching source package:
   `core`, `length_generalisation`, `shortcut_learning`, or `transfer`.
3. Write large outputs and checkpoints under `artifacts/<family>/`; keep them
   untracked.
4. Commit only compact evidence needed to support a claim: aggregate JSON,
   CSV, small JSONL diagnostics, or a plot.
5. Add or update an entry in `experiments/registry.json`.
6. Update `docs/experiment-index.md` with the same registry ID.
7. Add the result to a track guide only when it changes the track-level
   conclusion.

## Registry Fields

Every entry must provide:

- a stable lowercase `id`;
- one of the registered tracks and statuses;
- the scientific question;
- training and evaluation domains;
- the registered `source_packages` used by the experiment;
- local locations and evidence paths;
- reproduction commands;
- a narrow result summary.

Add W&B URLs when they are part of the evidence. W&B alone is not durable
evidence for a headline quantitative claim; record a compact local summary as
well.

## Experiment Contract

- Seed all model, data, perturbation, and evaluation streams.
- State whether data are fixed, resampled, or reporting-only.
- Separate selection data from untouched reporting data.
- Log exact accuracy and prediction diversity when collapse is possible.
- Save endpoint metrics independently of best-checkpoint metrics.
- Record evaluation example counts at each length.
- Store the effective configuration with every checkpoint.
- Resume W&B and random-generator state together.
- Launch GPU work through
  `/store/store5/software/simple-gpu-schedule/with-gpu`.

## Status Changes

- Move `preliminary` to `confirmed` only after matched controls or meaningful
  replication.
- Use `negative` when the tested setup cleanly answers the stated question
  against a suitable control.
- Use `inconclusive` when optimization failure, collapse, or confounding
  prevents that answer.
- Use `archived` when a direction is superseded but its evidence still explains
  later decisions.

Never delete a negative result merely because a later approach works. Preserve
the compact evidence and explain why the direction was stopped.

## Validation

Before committing:

```bash
python scripts/validate_research_artifact.py
PYTHONPATH=src python -m pytest -q
git diff --check
```

The registry validator rejects missing evidence, undocumented experiment
directories or tracked artifact roots, unowned source modules, root-level
implementation modules, broken local documentation links, and critical
unresolved TODO markers.
