# Experiment Index

This is the human-readable view of
[`experiments/registry.json`](../experiments/registry.json). The registry is the
machine-checked inventory; this page explains why each experiment remains in
the repository.

Accuracy values are exact-match fractions unless stated otherwise. `L20`
means list length 20. Status labels follow the definitions in the
[root README](../README.md#start-here).

## Source Ownership

`experiments/` contains launch policy, configurations, progress logs, and
result evidence. Reusable implementation is grouped under
`src/list_sorting_transformer/`:

| Source package | Experiment families |
| --- | --- |
| `core` | All families; shared task generation, models, evaluation, and baseline sorting |
| `length_generalisation` | Pointer-position ablations, modular pipeline, hard/sparse attention, compiled pointer routing, and length MAML |
| `shortcut_learning` | Evolved shortcut credit, collapse routing, oracle reversal, learned selectors, shortcut MAML, and shared learned-credit machinery used by length experiments |
| `transfer` | RASP-style downstream transfer and byte language-model transfer |

Each entry in
[`experiments/registry.json`](../experiments/registry.json) declares its exact
`source_packages`. The [source README](../src/list_sorting_transformer/README.md)
lists module responsibilities. The artifact validator rejects unowned packages
and implementation modules placed directly at the package root.

## Length Generalisation

<a id="exp-core-sorting-baselines"></a>
### Core sorting baselines

| Field | Value |
| --- | --- |
| Status | Archived |
| Task | Direct list sorting with decoder Transformers and LSTMs |
| Train / eval | Online L2-20 training; historical longer-length sweeps |
| Result | Baseline representation and architecture comparison; retained as the origin of the later diagnostics |
| Evidence | [Core notebook](core-sequence-tasks.md), [number metrics](../artifacts/numbers_alternating_seed7/metrics.json), [alphabet metrics](../artifacts/alphabet_alternating_seed7/metrics.json) |
| Reproduce | `sort-transformer-train --help`; `sort-transformer-compare --help` |

These results are single-seed historical baselines. They should not be used as
the strongest evidence for a positional architecture.

<a id="exp-algorithm-trace-representations"></a>
### Algorithm-trace representations

| Field | Value |
| --- | --- |
| Status | Archived |
| Task | Textual quicksort, executor-assisted pointer traces, and no-tool traces |
| Train / eval | Online L2-20 training and longer-list evaluations |
| Result | Rich traces exposed retrieval and execution bottlenecks, motivating atomic pointer tasks; they did not provide a clean general solution |
| Evidence | [Quicksort metrics](../artifacts/quicksort_trace_numbers_seed7_accum2/metrics.json), [tool metrics](../artifacts/pointer_quicksort_numbers_seed7_accum2/metrics.json), [no-tool metrics](../artifacts/pointer_quicksort_no_tool_numbers_seed7_accum2/metrics.json) |
| Reproduce | `sort-transformer-train --task quicksort_trace --help`; `sort-pointer-trace 3,1,2` |

<a id="exp-pointer-position-ablation"></a>
### Pointer-position ablation

| Field | Value |
| --- | --- |
| Status | Preliminary |
| Task | Predict the value after a pointer while varying positional attention |
| Train / eval | L2-20 training; fixed evaluation through L400 |
| Result | At L200, alternating RoPE/NoPE reached 83.3%, versus 41.7% all-NoPE and 31.6% all-RoPE; rotating values with RoPE failed |
| Evidence | [Ablation summary](../artifacts/pointer_next_position_ablation/summary.json), [figure](../artifacts/pointer_next_position_ablation/position_ablation_exact_match.png) |
| Reproduce | `sort-pointer-position-probe --help` |

This is one seed. It establishes a useful architecture lead, not a universal
property of RoPE.

<a id="exp-modular-pointer-pipeline"></a>
### Modular pointer pipeline

| Field | Value |
| --- | --- |
| Status | Preliminary |
| Task | Autoregressively predict modular positions, retrieve values, then compare them |
| Train / eval | Each stage trains on L2-20; stage evaluations extend to L400-2,048 |
| Result | The pointer-value stage reached 98.9% at L1,000; the two-value stage reached 100% at L200, 99.95% at L400, and 33.0% at L1,000 |
| Evidence | [Pointer-value evaluation](../artifacts/pointer_value_nocurr_gn0_10k_seed7/eval_high_lengths_to_1000.json), [pointer-pair evaluation](../artifacts/pointer_pair_from_value_gn0_2000_seed7/eval_high_lengths_to_1000.json), [pipeline description](core-sequence-tasks.md#modular-position-pipeline) |
| Reproduce | `sort-pointer-position-sequence --help`; `sort-pointer-value-from-position --help`; `sort-pointer-next-value-position --help` |

The pipeline demonstrates that learned local routing operations can extrapolate
far beyond their training range. Later-stage degradation also shows that
composing individually strong operations remains difficult.

<a id="exp-hard-attention-eggroll"></a>
### Hard-attention EGGROLL

| Field | Value |
| --- | --- |
| Status | Preliminary |
| Task | Learn pointer-next directly with forward-only rank-one evolutionary updates |
| Train / eval | Curriculum reaches L20; exact top-1 checkpoint sweep through L5,000 |
| Result | One active head per layer at generation 10,000 reached 99.6% at L400, 92.2% at L2,000, and 95.3% at L5,000 |
| Evidence | [Method and table](../experiments/hard_attention_eggroll/README.md#checkpoint-length-sweep), [CSV](../experiments/hard_attention_eggroll/results/eggroll_checkpoint_length_sweep.csv), [W&B](https://wandb.ai/wobrob101/list-sorting-hard-attention-eggroll/runs/wwv04wzb) |
| Reproduce | `bash experiments/hard_attention_eggroll/run_grouped_population.sh` |

The extreme-length cells use 64 examples and one seed. The result is strong
evidence that a minimal learned route can extrapolate on this task, but needs
seed replication.

<a id="exp-sparse-attention-adam"></a>
### Sparse-attention Adam ablations

| Field | Value |
| --- | --- |
| Status | Preliminary |
| Task | Pointer-next and KEEP/SWAP comparison with sparse/dense attention and positional-head ablations |
| Train / eval | L2-20 training; final evaluations through L5,000 |
| Result | Eight mixed ALiBi/NoPE heads reached 100% at sampled L2,000 with either entmax or softmax; direct interventions show layer-1 ALiBi routing followed by layer-2 NoPE retrieval. On matched 20,000-step autoregressive traces, entmax reached 100% through sampled L5,000 with and without a final KEEP/SWAP target; softmax reached 10.9% and 46.9%, respectively. |
| Evidence | [Ablation, mechanism, and comparison report](../experiments/sparse_attention_adam/README.md), [ablation summary](../experiments/sparse_attention_adam/results/key_difference_ablation_summary.json), [mechanism JSON](../experiments/sparse_attention_adam/results/alibi_nope_mechanism.json), [comparison JSON](../experiments/sparse_attention_adam/results/pointer_compare_summary.json), [comparison W&B](https://wandb.ai/wobrob101/list-sorting-pointer-compare-alibi-nope) |
| Reproduce | See the exact softmax and entmax commands in the [experiment README](../experiments/sparse_attention_adam/README.md#keepswap-extension). |

Sparse normalization tied softmax on pointer-next but strongly improved the
harder autoregressive retrieval trace. The measured circuit and trace
comparison remain single-checkpoint, single-seed results.

<a id="exp-learned-backward-length"></a>
### Evolved backward rules for length

| Field | Value |
| --- | --- |
| Status | Inconclusive |
| Task | Evolve an attention-gradient router on length-50 fitness while the task model trains at L2-20 |
| Train / eval | L2-20 forward training; fixed L50 ranking/acceptance; L400 reporting-only |
| Result | No sustained L400 advantage; successive halving reduced nominal updates but was slower in wall-clock time |
| Evidence | [Progress log](../experiments/learned_backward_length_generalization/PROGRESS.md), [successive-halving W&B](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/x67kp718) |
| Reproduce | `bash experiments/learned_backward_length_generalization/run_controller.sh` |

The experiment does not show that length-50 fitness is fundamentally
misaligned. The evolutionary optimizer did not produce a stable enough
length-50 advantage to test that claim.

<a id="exp-maml-length"></a>
### MAML for length

| Field | Value |
| --- | --- |
| Status | Negative |
| Task | Differentiate through hypothetical short-task updates using longer-length meta losses |
| Train / eval | L2-20 task training; L50 meta objective; L400 reporting |
| Result | QKV-only MAML peaked at 56.6% L400 then ended at 14.8%; horizon-24 routing ended below ordinary Adam while broadly suppressing attention credit |
| Evidence | [One-step and QKV results](../experiments/maml_length_generalization/PROGRESS.md), [horizon-24 log](../experiments/learned_backward_length_generalization/PROGRESS.md#horizon-24-router-maml) |
| Reproduce | `bash experiments/maml_length_generalization/run.sh`; `bash experiments/learned_backward_length_generalization/run_maml_h24.sh` |

## Shortcut Learning

<a id="exp-evolved-shortcut-credit"></a>
### Evolved shortcut-resistant credit

| Field | Value |
| --- | --- |
| Status | Confirmed |
| Task | Train a pointer model on examples containing a correct answer shortcut |
| Train / eval | Unlimited resampled shortcut data; fixed scarce clean fitness; fresh clean reporting sets |
| Result | The generation-49 horizon-320 rule reached 93.8% mean worst-mode accuracy across 20 replications, versus 4.5% for ordinary backpropagation |
| Evidence | [Replication summary](../experiments/learned_backward_shortcuts/results/random_elite_g49_h320_replications_summary.json), [horizon sweep](../experiments/learned_backward_shortcuts/results/masked_horizon_sweep_summary.json), [full log](../experiments/learned_backward_shortcuts/PROGRESS.md) |
| Reproduce | `bash experiments/learned_backward_shortcuts/run_deterministic_controller.sh`; `python experiments/learned_backward_shortcuts/fixed_rule_horizon_sweep.py --help` |

The confirmed result is transfer across fresh task-model trajectories and data
from the same random-position shortcut task. Cross-task transfer is untested.

<a id="exp-shortcut-collapse-routing"></a>
### Shortcut-collapse routing

| Field | Value |
| --- | --- |
| Status | Inconclusive |
| Task | Repair later trajectory windows where a robust shortcut rule collapses |
| Train / eval | Saved collapse windows for ranking; seed-disjoint windows and full trajectories for acceptance |
| Result | Selected candidates improved some windows but failed the complete on-policy acceptance gate |
| Evidence | [Multi-seed summary](../experiments/learned_backward_shortcuts/results/collapse_window_multi_seed_summary.json), [on-policy rejection](../experiments/learned_backward_shortcuts/results/on_policy_collapse_correction_summary.json), [trimmed evolution](../experiments/learned_backward_shortcuts/results/trimmed_collapse_evolution_summary.json) |
| Reproduce | `python experiments/learned_backward_shortcuts/collapse_window_population.py --help` |

<a id="exp-oracle-gradient-reversal"></a>
### Oracle attention-gradient reversal

| Field | Value |
| --- | --- |
| Status | Confirmed |
| Task | Apply known leak-token information only during backward credit assignment |
| Train / eval | Shortcut training; fixed and fresh masked/incorrect/correct-hint sets |
| Result | With scale 4 and learning rate `1e-4`, both random-position seeds reached 100% on fixed sets and at least 99.97% on fresh 20,000-example masked/incorrect evaluations |
| Evidence | [Optimization summary](../experiments/token_gradient_selector/results/random_oracle_optimization_summary.json), [scope summary](../experiments/token_gradient_selector/results/oracle_scope_summary.json) |
| Reproduce | `python experiments/token_gradient_selector/oracle_horizon_sweep.py --help` |

The oracle proves that attention-score credit is a sufficient intervention in
this task. It does not solve how to discover the intervention.

<a id="exp-learned-token-selector"></a>
### Learned token selector

| Field | Value |
| --- | --- |
| Status | Negative |
| Task | Learn which source tokens should receive reversed attention-score gradients |
| Train / eval | Group-reward evolutionary/GRPO training; fixed and held-out shortcut conditions |
| Result | Best held-out clean accuracy remained about 19.6%; reward did not improve over the unattended continuation |
| Evidence | [Selector diagnostics](../experiments/token_gradient_selector/results/sparse_selector_diagnostics.json), [progress log](../experiments/token_gradient_selector/PROGRESS.md#unattended-continuation) |
| Reproduce | `python experiments/token_gradient_selector/train_selector_grpo.py --help` |

<a id="exp-maml-shortcut-router"></a>
### MAML shortcut router

| Field | Value |
| --- | --- |
| Status | Preliminary |
| Task | Meta-learn persistent attention-gradient suppression through short shortcut-training lookaheads |
| Train / eval | Random-position shortcut data; masked and incorrect-hint held-out data |
| Result | At horizon 24, suppression-only routing reached 98.8% masked and 91.4% incorrect-hint accuracy in one seed; signed routing was worse |
| Evidence | [MAML shortcut log](../experiments/learned_backward_shortcuts/MAML_PROGRESS.md#eight-step-lookahead-follow-up), [suppression W&B](https://wandb.ai/wobrob101/list-sorting-maml-shortcut/runs/74p2kcpy), [signed W&B](https://wandb.ai/wobrob101/list-sorting-maml-shortcut/runs/mwrdb7b6) |
| Reproduce | `python -m list_sorting_transformer.shortcut_learning.maml_shortcut_experiment --help` |

## Cross-Track Transfer

<a id="exp-rasp-transfer"></a>
### Compiled RASP-style transfer

| Field | Value |
| --- | --- |
| Status | Confirmed |
| Task | Compare random, compiled-prefix, and fully compiled initialization on four downstream tasks |
| Train / eval | Three seeds, L2-20 training, evaluations through L400 |
| Result | Compiled prefix improved L400 three-way relation from 56.6% to 95.4% and associative recall from 8.1% to 55.5%; it did not help Dyck-2 |
| Evidence | [Full report](rasp_transfer_report.md#endpoint-results), [aggregate JSON](../experiments/rasp_transfer/results/summary.json) |
| Reproduce | `bash experiments/rasp_transfer/run_matrix.sh` |

<a id="exp-language-model-transfer"></a>
### Compiled language-model transfer

| Field | Value |
| --- | --- |
| Status | Negative |
| Task | Insert compiled routing blocks into a matched byte language model |
| Train / eval | Three paired seeds, 5,000 updates at context 256; eval through context 2,048 |
| Result | Random reached 3.026 BPC versus 3.095 compiled at context 256 and remained better at every longer context |
| Evidence | [Full report](language_model_transfer_report.md#results), [aggregate JSON](../experiments/language_model_transfer/results/summary.json) |
| Reproduce | `bash experiments/language_model_transfer/run_matrix.sh` |

## Inventory Guarantees

`python scripts/validate_research_artifact.py` verifies that:

- every top-level tracked `experiments/` directory is covered by the registry;
- every top-level tracked `artifacts/` entry is assigned to an experiment;
- every experiment declares its source packages and every implementation
  module belongs to a registered package;
- every registered local evidence path exists;
- every registry ID has a section on this page;
- required documentation links resolve;
- critical documentation does not contain unresolved `TODO`, `TBD`, or
  `FIXME` markers.
