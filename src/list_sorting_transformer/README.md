# Source Layout

The Python package follows the same research structure as `experiments/`.

| Package | Responsibility |
| --- | --- |
| `core` | Shared tokens, data generation, Transformer/LSTM models, sorting and trace machines, evaluation, metrics, plots, and baseline CLIs |
| `length_generalisation` | Modular pointer pipeline, compiled pointer circuit, sparse/hard attention, length MAML, and long-length sweeps |
| `shortcut_learning` | Evolved backward rules, fixed-fitness controllers, collapse diagnostics, gradient reversal, learned selectors, and shortcut MAML |
| `transfer` | RASP-style puzzle transfer and byte language-model transfer |

Shared dependencies and the two deliberate cross-track reuse paths are:

```text
core
  ├── length_generalisation
  ├── shortcut_learning
  └── transfer

shortcut_learning router primitives ──> length_generalisation MAML
length_generalisation compiled circuit ──> transfer studies
length_generalisation MAML helpers ──> shortcut_learning MAML
```

`experiments/` owns launch policy, research configurations, progress logs, and
committed result evidence. `src/` owns reusable executable behavior. Experiment
registry entries list the source packages they use, and
`scripts/validate_research_artifact.py` checks that every tracked source module
belongs to one declared package.

The `sort-*` console command names remain stable. Python imports should use the
package-qualified paths above; the former flat module paths were removed during
the research-artifact reorganization. Existing checkpoints remain compatible
because they store configuration dictionaries and model/optimizer state
dictionaries rather than pickled project classes.
