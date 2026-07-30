# Archive Map

Archived means retained for provenance, not erased. The implementation may
still be tested because later experiments share its data generation or model
components.

| Direction | Status | Why it is not the active path | Durable record |
| --- | --- | --- | --- |
| Direct sorting and LSTM comparison | Archived | Too many algorithmic operations change at once to diagnose length failure | [Core notebook](core-sequence-tasks.md), [experiment index](experiment-index.md#core-sorting-baselines) |
| Textual quicksort and executor traces | Archived | Trace length and observation retrieval confound algorithm learning | [Core notebook](core-sequence-tasks.md#quicksort-execution-traces), [index](experiment-index.md#algorithm-trace-representations) |
| Continuous pointer-position vector regression | Archived diagnostic | Exact vector regression was harder to interpret than categorical modular addresses | [Core notebook](core-sequence-tasks.md#pointer-position-vector-probe) |
| Rotating value vectors with RoPE | Negative ablation | It fit the training length but produced zero exact match in the long pointer-position sweep | [Ablation JSON](../artifacts/pointer_next_position_ablation/summary.json) |
| Full compiled Transformer transfer | Negative control | Unused exact-zero paths are difficult to optimize and underperform a compiled prefix | [RASP report](rasp_transfer_report.md#why-the-full-compiled-model-is-worse) |
| Compiled blocks for byte language modelling | Negative | Matched random initialization is better in all paired seeds and tested context lengths | [Language-model report](language_model_transfer_report.md) |
| One-step and QKV-only MAML for length | Negative | Transient L400 gains collapse by the endpoint | [MAML progress](../experiments/maml_length_generalization/PROGRESS.md) |
| Horizon-24 MAML for length | Negative | The router converges toward broad suppression without beating ordinary Adam | [Learned-backward length log](../experiments/learned_backward_length_generalization/PROGRESS.md#horizon-24-router-maml) |
| Successive halving for length evolution | Negative performance result | Fewer nominal candidate updates produced worse GPU utilization and slower generations | [Controller log](../experiments/learned_backward_length_generalization/PROGRESS.md#successive-halving-controller) |
| Learned token selector | Negative | The oracle operation works, but policy reward and held-out accuracy do not improve | [Selector log](../experiments/token_gradient_selector/PROGRESS.md) |
| Collapse-window-only evolution | Inconclusive | Window improvements fail complete on-policy transfer | [On-policy summary](../experiments/learned_backward_shortcuts/results/on_policy_collapse_correction_summary.json) |

Chronological `PROGRESS.md` files remain append-only scientific lab notes. They
contain superseded launches and implementation corrections that are too
detailed for the front-page narrative but useful when auditing how a result was
obtained.
