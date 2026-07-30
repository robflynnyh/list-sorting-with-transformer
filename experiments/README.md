# Experiments

The [experiment index](../docs/experiment-index.md) is the primary navigation
page. [`registry.json`](registry.json) is its machine-checked inventory.

| Directory | Track | Current role |
| --- | --- | --- |
| `hard_attention_eggroll` | Length | Preliminary forward-only sparse-routing result |
| `sparse_attention_adam` | Length | Positional-head and normalization ablations |
| `learned_backward_length_generalization` | Length | Inconclusive evolutionary and MAML optimization studies |
| `maml_length_generalization` | Length | Negative one-step MAML studies |
| `rasp_transfer` | Cross-track | Confirmed operation-aligned compiled transfer |
| `language_model_transfer` | Cross-track | Confirmed negative transfer boundary |
| `learned_backward_shortcuts` | Shortcut | Confirmed evolved rule plus later diagnostics |
| `token_gradient_selector` | Shortcut | Confirmed oracle and negative learned selector |

Long launchers are intentionally not unified behind one command: their resource
requirements and scientific contracts differ. Every GPU launcher must acquire
resources through `with-gpu`, either directly or through its documented outer
wrapper.

For two-update executable checks, use `scripts/smoke_length.sh` and
`scripts/smoke_shortcut.sh` from the repository root.
