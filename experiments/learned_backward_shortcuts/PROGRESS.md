# Learned Backward Rules for Shortcut Resistance

## Research question

Can a small clean dataset train a reusable backward credit-assignment rule that
extracts the genuine signal from a much larger dataset containing a perfect
answer shortcut?

The forward model is trained only on shortcut-containing examples. EGGROLL
optimizes a shared learned backward rule using fitness measured on 512 clean
examples. The forward model is reset to a newly sampled initialization every
EGGROLL generation; only the backward rule persists.

![Learned-backward experiment progress](results/progress.png)

The figure is regenerated from all chained `metrics.jsonl` segments with
`sort-shortcut-credit-plot`. Later segments replace duplicate resume
generations, so the chart represents the actual checkpoint lineage.

## Confirmed design

- Task: retrieve the list value immediately following `<PTR>`.
- Biased input: the correct target also appears after `<LEAK>`, immediately
  before `<QUERY>`.
- Clean fitness data: 256 examples with `<LEAK> <MASK>` and 256 with a random
  incorrect leaked value.
- Values: digits 0-9.
- Lengths: sampled uniformly from 8 through 32, with no initial length
  extrapolation objective.
- Forward model: three-layer, width-128, four-head causal decoder Transformer
  with RoPE, SwiGLU, pre-norm, and no dropout.
- Forward optimizer: Adam with constant learning rate and no weight decay.
- Backward rule: one shared two-layer, width-128, four-head reverse-causal
  Transformer. It consumes saved forward activations, upstream gradients, and
  a layer identity.
- Intervention: before each complete forward Transformer block's normal
  backward operation. Its output affects that block's parameter gradients and
  all preceding layers.
- Identity anchor: a zero-initialized residual gate makes the centre backward
  rule exactly ordinary backpropagation at initialization.
- Gradient constraint: modified gradients are rescaled to preserve the
  original per-example RMS.
- Inner training: initially 10 sequential Adam updates with batch size 64. All
  candidates in a generation receive the same initialization and batches.
- Forward reset: every EGGROLL generation samples a new forward initialization;
  no forward parameters persist between generations.
- Fitness: reduction in cross-entropy on all 512 clean fitness examples.
- Initial population: 32 rank-one antithetic directions, giving 64 candidates.
  The implementation must support population growth through 1,024 candidates
  using chunks.
- EGGROLL: rank-one matrix perturbations, antithetic candidates, standard
  fitness normalization, and the paper-style direct SGD centre update.
- Horizon curriculum: start at 10 updates and increase the evolved horizon
  when progress saturates. Forward models always train from initialization for
  the complete current horizon.
- Baselines are supporting diagnostics, not the primary implementation focus.

## Evidence requirements

Before interpreting learned-backward results:

1. A centre rule with zero gates must produce exactly the same loss and forward
   parameter gradients as ordinary backpropagation.
2. Antithetic candidates must receive exact positive and negative versions of
   the same rank-one perturbations.
3. Candidate forward models and Adam states must reset every generation.
4. Candidates within one generation must see identical biased batches and the
   same clean fitness examples.
5. Ordinary Adam must be able to learn the genuine pointer task from clean
   examples, while biased-only training should expose shortcut reliance.
6. Report correct-leak, masked-leak, and incorrect-leak accuracy separately.

### Interpretation thresholds

- Random digit accuracy is `10%`.
- Because an incorrect hint is sampled uniformly from the nine values other
  than the target, a model that only learns "do not predict the hint" can reach
  `1/9 = 11.1%` incorrect-leak accuracy without reading the pointer.
- Averaged over the balanced masked/incorrect fitness set, this exclusion rule
  can improve CE by approximately
  `(log(10) - log(9)) / 2 = 0.0527` without learning retrieval.
- Consequently, neither a small positive fitness nor incorrect-leak accuracy
  near `11.1%` is evidence of shortcut resistance. A successful learned
  backward rule must drive masked accuracy above `10%` and incorrect-leak
  accuracy materially above `11.1%`, with broad prediction diversity.

## Implementation status

- [x] Design agreed.
- [x] Shortcut task and fixed clean fitness set.
- [x] Identity-preserving learned backward rule.
- [x] Paper-style EGGROLL perturbation and update.
- [x] Resetting inner rollout and horizon curriculum.
- [x] Focused tests.
- [x] CPU smoke.
- [x] GPU smoke.
- [x] First population-64 research run launched and validated.
- [ ] Analyze learning dynamics and horizon progression.

## Sources

- Sarkar et al., *Evolution Strategies at the Hyperscale*:
  <https://arxiv.org/abs/2511.16652>
- Official EGGROLL implementation:
  <https://github.com/ESHyperscale/HyperscaleES>

## Run log

### 2026-07-26: scalar CPU smoke

- Command: `PYTHONPATH=src python -m
  list_sorting_transformer.shortcut_credit_experiment` with two generations,
  four candidates, horizon two, width 32, and 16 fitness examples.
- Result: completed both generations, wrote metrics and resumable checkpoints,
  and exercised the real custom-backward hook and EGGROLL update.
- Focused tests: `6 passed`.
- Full repository tests: `154 passed`.
- Identity evidence: the zero-gate rule produced bit-exact loss and forward
  parameter gradients relative to ordinary backpropagation.
- Perturbation evidence: matrix perturbations were rank one, antithetic
  candidates averaged exactly to the centre, and the paper update moved every
  parameter along the fitter perturbation.
- Calibration finding: `sigma=0.02` was much too conservative. The first
  candidate's mean gradient cosine was about `0.9999998` and its correction RMS
  was only about `0.0005` of the ordinary gradient. Do not launch the real run
  with this value; calibrate on GPU toward approximately `0.99` cosine first.

### 2026-07-26: full-width GPU calibration

- Hardware: one RTX A4500 through `with-gpu`.
- Full architecture: forward width 128 with three layers; backward width 128
  with two layers.
- A coarse `sigma` sweep showed a sharp nonlinear transition. `0.02` was
  ineffective, `0.1` gave gradient cosine `0.9766`, and `0.3` caused the
  correction RMS to explode.
- The refined sweep selected `sigma=0.08`: at the full horizon-10 smoke it
  produced mean gradient cosine `0.9909` and correction/original RMS ratio
  `0.1025`.
- Runtime: population 8, horizon 10, batch size 64, all 512 fitness examples,
  and 128 correct-leak diagnostics took `3.76s`. The scalar implementation
  should therefore take roughly `30s` per population-64 generation at horizon
  10.

### 2026-07-26: ordinary-Adam shortcut diagnostic

One forward model was trained from initialization on correct-leak examples
using the agreed Adam settings:

| Step | Correct leak | Masked leak | Incorrect leak | Clean CE |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 15.6% | 12.5% | 12.9% | 2.5245 |
| 20 | 14.8% | 12.9% | 15.6% | 2.4028 |
| 40 | 44.5% | 20.3% | 16.8% | 2.3394 |
| 80 | 100.0% | 21.5% | 0.0% | 3.1588 |
| 160 | 100.0% | 19.1% | 0.0% | 3.9253 |
| 1,000 | 100.0% | 20.3% | 0.0% | 6.2231 |

The task exposes the intended failure cleanly: ordinary Adam learns the fixed
leak perfectly by step 80, while performance with an incorrect leak falls to
zero and clean loss continues worsening. The horizon curriculum should
therefore eventually reach at least 80 updates.

For the complementary learnability check, ordinary Adam was trained on masked
leak examples, which force it to use the pointer:

| Step | Correct leak | Masked leak | Incorrect leak |
| ---: | ---: | ---: | ---: |
| 80 | 20.3% | 17.6% | 15.7% |
| 160 | 34.7% | 21.9% | 19.9% |
| 320 | 99.5% | 99.1% | 98.7% |
| 640 | 100.0% | 100.0% | 99.4% |
| 1,000 | 100.0% | 100.0% | 96.1% |

The genuine task is therefore learnable by this architecture and optimizer.
It learns more slowly than the shortcut: clean training reaches near-perfect
retrieval around step 320, whereas biased training learns the leak by step 80.

### 2026-07-26: first population-64 research run

- Run: [`learned-backward-p64-curriculum-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/yupx3qo8)
- Configuration: population 64, `sigma=0.08`, initial horizon 10, maximum
  horizon 160, 300 generations, and outer learning rate linearly decayed from
  `0.1`.
- Generation 0 completed in `24.4s`.
- Candidate clean-loss improvement had mean `0.3372`, standard deviation
  `0.0839`, and maximum `0.4678`. The nontrivial standard deviation confirms
  that the initial population produces a usable EGGROLL ranking signal.
- For the first positive probe candidate, mean correction/original gradient RMS
  was `0.1078`, with gradient cosine `0.9903`. This is consistent with the
  preceding calibration rather than an ineffective or unstable perturbation.
  These are candidate diagnostics, not measurements of the unperturbed centre.
- Initial post-training diagnostic accuracy was close to chance at horizon 10:
  `9.0%` with a correct leak, `11.8%` with a masked leak, and `9.6%` with an
  incorrect leak. This is expected before the inner horizon reaches the point
  where ordinary Adam learns the shortcut.
- The detached run writes checkpoints every 10 generations under
  `artifacts/learned_backward_shortcuts/learned-backward-p64-curriculum-seed7/`.
- Subsequent launches also log antithetic pair-difference magnitude, the actual
  centre update RMS, and centre gate magnitudes. Those diagnostics distinguish
  a useful directional EGGROLL signal from candidate spread caused only by
  perturbation size.

#### Generation-10 estimator probe

The generation-10 checkpoint was replayed for one generation with the expanded
diagnostics:

- Antithetic pair fitness-difference RMS: `0.0853`.
- Whole-population fitness standard deviation: `0.0752`.
- Mean absolute antithetic pair difference: `0.0697`.
- Centre gate absolute mean / maximum: `0.0120 / 0.0158`.

The paired difference is at least as large as the broad candidate spread, so
the `+/-` comparisons contain a material directional ranking signal. The
centre itself is still close to ordinary backpropagation after 10 generations;
the larger correction statistics in the main run measure one perturbed probe
candidate rather than the centre.

#### Generation-30 short-horizon behavior

The horizon remained at 10 through generation 30. Fitness showed no sustained
increase, and the plateau state reached 24 stale generations out of the
required 50. More importantly, the centre gates moved back toward the identity
anchor:

| Checkpoint | Mean absolute gate | Maximum absolute gate |
| ---: | ---: | ---: |
| 10 | 0.0120 | 0.0158 |
| 20 | 0.0089 | 0.0130 |
| 30 | 0.0028 | 0.0034 |

At this short horizon the learned rule is therefore recovering ordinary
backpropagation, not growing an arbitrary correction. That is a useful
sanity result: biased Adam does not reliably exploit the leak until later, so
there is little shortcut-specific credit to correct in only 10 updates. The
important test begins after the curriculum reaches horizons 80 and above.

The PyTorch update was also checked line by line against the official EGGROLL
implementation. Both use rank-one `A @ B.T` matrix perturbations, standardize
fitness over the full population, average `score * sigma * perturbation`, and
multiply the update by `sqrt(population_size)` before the SGD step.

#### Curriculum metric correction

The first 40 generations exposed a problem in the original plateau detector.
Mean candidate fitness was correlated `0.913` with the randomly initialized
forward model's initial clean loss. This does **not** bias EGGROLL's candidate
ranking within a generation, because every candidate shares the same initial
model and the initial term cancels. It does make fitness reduction a poor
quantity for comparing progress across generations with different initial
models.

Horizon promotion now tracks negative post-training clean CE instead. This is
the actual cross-generation objective and has much lower dependence on the
initial model. The run was stopped after generation 45, then resumed from its
durable generation-40 backward-rule checkpoint with plateau state recomputed
from generations 0-39 using the corrected objective. Metrics from the six
superseded, post-checkpoint generations were preserved separately rather than
treated as part of the continued trajectory.

- Original W&B run through the checkpoint:
  <https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/yupx3qo8>
- Continued run:
  [`learned-backward-p64-clean-plateau-resume40-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/fqpavj4v)
- Migrated checkpoint:
  `checkpoint_000040_clean_plateau.pt`
- Recomputed state at generation 40: clean-loss EMA `2.6894`, best EMA
  `2.6523`, and 39 stale generations.
- New metrics explicitly log the curriculum objective, its EMA, stale
  generation count, and promotion events.
- The resumed generations 40 and 41 completed successfully. Antithetic
  pair-difference RMS remained material at `0.0811` and `0.0759`, while the
  corrected stale counter advanced exactly from 39 to 41.

#### First horizon promotion

The corrected curriculum promoted exactly at the configured boundary:

- Generation 50: horizon 10, stale count 50, promotion flag set.
- Generation 51: horizon 20, stale count reset to zero.
- Mean post-training clean CE improved from approximately `2.70` over the final
  horizon-10 generations to `2.6354` on the first horizon-20 generation.
- Antithetic pair-difference RMS remained nonzero at `0.0573`.
- Mean correct-leak, masked-leak, and incorrect-leak accuracies were all near
  chance (`10.6%`, `11.2%`, and `10.8%`). Twenty updates are therefore still
  before the shortcut-learning transition.
- Scalar runtime increased from approximately `23s` to `32s` per generation,
  substantially less than a full doubling because fixed candidate evaluation
  accounts for part of each generation.

This verifies that the curriculum advances deterministically on the corrected
objective and preserves a usable EGGROLL signal after the objective changes.

#### Horizon-20 early behavior

Across generations 51-60:

- Mean post-training clean CE: `2.6428`.
- Mean antithetic pair-difference RMS: `0.0636`.
- Mean correct-leak / masked-leak / incorrect-leak accuracy:
  `10.3% / 10.2% / 10.1%`.
- Mean centre gate magnitude remained small and ended at `0.0104`.
- Runtime averaged approximately `32s` per generation.

There is no evidence of retrieval or shortcut use at 20 updates. The useful
result at this point is that directional signal persists while the centre
remains close to its ordinary-backprop anchor.

The evaluator now also records the number of distinct value predictions and
the modal prediction fraction. The run was resumed from the unchanged
generation-60 checkpoint so those diagnostics are present before reaching the
shortcut-learning horizons:

- Continued run:
  [`learned-backward-p64-diversity-resume60-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/1fialmlh)
- Source checkpoint:
  `learned-backward-p64-clean-plateau-resume40-seed7/checkpoint_000060.pt`
- Preserved curriculum state: horizon 20 with 9 stale generations.

The first diversity-enabled generation revealed that the horizon-20 models are
still effectively collapsed:

- Mean distinct predicted digit values: `1.31 / 10`.
- Mean modal prediction fraction: `97.1%`.
- Masked / incorrect accuracy: `9.6% / 10.1%`.

Thus, the horizon-20 CE reduction is mostly the model learning to place output
mass on the digit vocabulary rather than retrieving the pointed value. This
does not invalidate CE as the EGGROLL fitness, but it prevents interpreting
the early CE gain as task learning. Prediction diversity must expand before
accuracy changes can be meaningful.

#### Packed evaluation optimization

Candidate evaluation originally grouped the variable-length fitness examples
by exact length, resulting in approximately 50 small Transformer calls per
candidate. Evaluation now right-pads mixed-length examples and reads logits at
each example's original `<QUERY>` position. Causal masking guarantees that
padding after the query cannot affect that query.

On the full architecture and all 512 fitness examples:

- Original grouped evaluation: `109.9 ms`.
- Packed evaluation: `37.7 ms`.
- Speedup: `2.9x` for the evaluation portion.
- CE difference: `1.9e-8`.
- Accuracy: exactly identical.

A focused test also compares packed query logits against separate unpadded
forwards across different lengths. The optimization will be adopted from the
next durable checkpoint; it does not alter the training batches or EGGROLL
estimator.

The generation-80 checkpoint was replayed end to end after adopting the packed
evaluator:

- Original generation time: `35.9s`.
- Packed-evaluation generation time: `27.1s` (`24.6%` faster).
- Fitness mean changed by only `3e-8`.
- Clean CE and all reported accuracies were identical at displayed precision.
- Continued run:
  [`learned-backward-p64-packed-resume80-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/hhw7gu1w)

The packed run is therefore the authoritative trajectory from generation 80.

#### Multi-GPU candidate sharding

Candidate rollouts can optionally be sharded over explicitly reserved CUDA
devices. Both signs of each antithetic direction remain on the same device, and
results are restored to canonical population order before the unchanged
EGGROLL update.

A full population-64, horizon-20 replay from checkpoint 80 showed:

- Scalar runtime: `28.9s`.
- Two-GPU runtime: `23.2s` (`19.8%` faster).
- Fitness, clean CE, every accuracy/diversity metric, pair-difference
  statistics, and captured gradient statistics: exactly identical.

The speedup is sublinear because model construction and Python/CUDA launch
overhead are still partly serialized, but the implementation provides a
validated path to larger populations. The live trajectory will adopt two-GPU
sharding from its next durable checkpoint while both devices are available.

#### Horizon 20 completion and horizon 40

Horizon 20 ended at generation 101. Across the 42 authoritative generations
with diversity metrics:

- Mean clean CE: `2.6548`.
- Mean correct / masked / incorrect accuracy:
  `9.95% / 10.16% / 9.96%`.
- Mean distinct predicted digits: `1.14 / 10`.
- Mean modal prediction fraction: `98.3%`.
- Mean antithetic pair-difference RMS: `0.0646`.

There was no retrieval or shortcut-learning transition at horizon 20. The
centre gate also remained small (`0.0053` mean magnitude).

The curriculum promoted to horizon 40 at generation 102. Over its first four
generations:

- Mean clean CE fell to `2.5413`.
- All three accuracies remained near chance.
- Mean distinct predicted digits / modal fraction:
  `1.04 / 10` and `99.7%`.
- Mean pair-difference RMS remained material at `0.0598`.

The lower CE at horizon 40 is therefore still calibration without argmax
retrieval. One-GPU runtime is approximately `50s` per generation. The next
checkpoint will enable the validated two-GPU sharding path.

Two-GPU sharding became authoritative from checkpoint 110:

- Continued run:
  [`learned-backward-p64-two-gpu-resume110-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/0zj2x13v)
- Generation-110 runtime: `38.3s`, versus approximately `49s` immediately
  before the switch.
- Preserved state: horizon 40 with the corrected stale counter advancing to 8.
- Generation-110 clean CE / pair-difference RMS: `2.5367 / 0.0691`.
- Clean accuracy remained at chance with `1.11 / 10` distinct predicted
  digits and a `98.8%` modal fraction.
