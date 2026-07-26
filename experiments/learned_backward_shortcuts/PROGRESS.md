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

#### Evolved-centre perturbation recalibration

The fixed `sigma=0.08` search radius stopped being local as the backward-rule
centre evolved. Across generations 102-120 at horizon 40, the first probe
candidate had mean gradient cosine `0.845` and correction/original-gradient
RMS ratio `0.251`. Candidate models remained collapsed throughout those 19
generations: mean clean accuracy was `10.0%`, mean clean CE was `2.5439`, and
the modal prediction fraction was `99.5%`.

Generation 110 was replayed from the same checkpoint, with the same forward
initialization and batches, across smaller fixed search radii:

| Sigma | Clean CE | Clean acc. | Masked | Wrong hint | Distinct digits | Modal fraction | Pair delta RMS | Gradient cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.080 | 2.5367 | 10.7% | 11.1% | 10.3% | 1.11 | 98.8% | 0.0691 | 0.87837 |
| 0.040 | 2.4586 | 11.9% | 12.7% | 11.0% | 2.09 | 88.8% | 0.0421 | 0.99716 |
| 0.020 | 2.4085 | 15.3% | 16.5% | 14.1% | 5.39 | 59.7% | 0.0444 | 0.99990 |
| 0.010 | 2.3807 | 17.3% | 18.1% | 16.6% | 8.41 | 36.6% | 0.0355 | 0.99997 |
| 0.005 | 2.3786 | 17.4% | 18.1% | 16.8% | 9.73 | 33.5% | 0.0266 | 0.99998 |

This reverses the initial horizon-10 calibration because the mapping from
backward-rule parameters to gradients is nonlinear and the centre is no
longer the identity initialization. At the evolved centre, `sigma=0.08`
mostly tests destructive, nonlocal backward rules rather than estimating a
useful local search direction.

`sigma=0.01` is the selected continuation point. Halving it again yields only
`0.0021` lower CE and `0.13` percentage points more clean accuracy, while
reducing antithetic pair separation by `25%` and halving the magnitude of the
paper-style EGGROLL centre update. The `sigma=0.08` continuation was stopped
at its durable generation-120 checkpoint; generations 121-124 after that
checkpoint are excluded from the authoritative lineage.

The local-search continuation is:

- Run:
  [`learned-backward-p64-sigma001-resume120-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/1mwbgpqh)
- Source: the `sigma=0.08` generation-120 checkpoint.
- First generation clean CE / accuracy: `2.3819 / 17.7%`.
- Masked / incorrect-hint accuracy: `18.1% / 17.2%`.
- Distinct predicted digits / modal fraction: `8.84 / 10` and `38.4%`.
- Antithetic pair-difference RMS: `0.0214`.
- Centre update RMS relative to centre RMS: `0.59%`.

The first local generation therefore preserves a nonzero estimator and centre
update while immediately eliminating the population-wide prediction collapse.
Multiple generations are required before this can be called evolutionary
progress rather than improved candidate locality.

Across the first ten authoritative local-search generations, 120-129:

- Mean clean CE / accuracy: `2.3794 / 15.8%`.
- Mean masked / incorrect-hint accuracy: `16.6% / 15.0%`.
- Mean correct-leak accuracy: `29.3%`.
- Mean distinct predicted digits / modal fraction:
  `7.44 / 10` and `43.1%`.
- Mean antithetic pair-difference RMS: `0.0308`.
- Mean centre-update/centre RMS: `0.81%` per generation.

This is substantially better than the collapsed `sigma=0.08` population, but
it does **not** yet beat ordinary biased Adam at horizon 40. The earlier
single-run Adam diagnostic reached `18.6%` balanced clean accuracy and clean
CE `2.3394`; the local learned-backward population reached `15.8%` and
`2.3794`. Horizon 40 is also before the baseline's full shortcut transition,
so the decisive comparison remains horizon 80, where ordinary Adam's
incorrect-hint accuracy falls to zero.

Changing sigma also shifted the candidate-objective distribution. Carrying
the old `sigma=0.08` clean-loss EMA forward made its gradual relaxation look
like repeated progress, which would delay the horizon curriculum. At the
generation-130 checkpoint, plateau state was therefore recomputed using only
the ten `sigma=0.01` generations:

- Recomputed EMA objective: `-2.3810`.
- Best EMA objective: `-2.3806`.
- Stale generations: `5`.
- Every non-plateau checkpoint field and backward-rule tensor was verified
  identical to the source checkpoint.

The corrected continuation is:

- Run:
  [`learned-backward-p64-sigma001-plateau-resume130-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/4638c6ep)
- Migrated checkpoint:
  `checkpoint_000130_sigma001_plateau.pt`.
- Generations 130-132 from the pre-migration process are superseded by this
  continuation.

#### Horizon-40 saturation and horizon-80 test

The local gradient-transformer rule continued to improve slowly at horizon 40,
but did not improve the diagnostic most relevant to shortcut resistance:

| Window | Clean CE | Clean acc. | Masked | Wrong hint | Distinct digits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generations 120-139 | 2.3738 | 16.8% | 18.2% | 15.5% | 8.00 |
| Generations 166-185 | 2.3662 | 17.5% | 19.4% | 15.5% | 8.90 |

After 88 horizon-40 generations, balanced accuracy had gained only `0.64`
percentage points and wrong-hint accuracy was unchanged. Small noisy CE
changes continued resetting the generic plateau counter, so the durable
generation-190 checkpoint was manually promoted to horizon 80. Every model,
backward-rule, and configuration tensor was verified unchanged; only the
horizon and its plateau state were reset. Generation 190 from the interrupted
horizon-40 process is superseded.

The diagnostic-enabled horizon-80 continuation is:

- Run:
  [`learned-backward-p64-h80-bestdiag-resume190-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/274oryb1)
- Migrated checkpoint: `checkpoint_000190_horizon80.pt`.
- Generation-190 runtime: `70.2s`.

The first horizon-80 population reproduces the ordinary shortcut transition:

| Measurement | Correct leak | Masked | Wrong hint | Balanced clean |
| --- | ---: | ---: | ---: | ---: |
| Population mean | 97.7% | 16.1% | 0.23% | 8.18% |
| Fittest candidate | 66.4% | 20.3% | 4.69% | 12.5% |
| Ordinary-Adam reference | 100.0% | 21.5% | 0.0% | 10.75% |

The evolved centre is not yet shortcut resistant: its surrounding population
mean is worse than ordinary Adam. However, the fittest local perturbation
already exceeds the ordinary balanced-clean reference and preserves nonzero
wrong-hint accuracy. Its lower correct-leak accuracy also shows that the gain
comes from partially blocking shortcut acquisition, rather than merely
improving calibration. Fitness spread is now large (`0.166` SD, `0.620`
maximum clean-CE improvement), providing a strong EGGROLL signal for testing
whether repeated centre updates can capture this behavior.

This candidate is promising but not a success under the predefined threshold:
its `4.69%` wrong-hint accuracy remains below both chance and the `11.1%`
wrong-hint-exclusion baseline. Subsequent generations must improve wrong-hint
accuracy without collapsing masked accuracy toward chance.

### 2026-07-26: parallel suppress-only attention router

![Suppress-only attention-router progress](results/attention_router_progress.png)

A second backward-rule family tests a narrower credit-assignment hypothesis.
It cannot synthesize gradient vectors or create new attention edges. Instead,
it can only suppress existing token-to-token routes in attention backward:

1. The side rule receives the same token IDs as the forward model, using its
   own persistent embeddings and fixed sinusoidal positions.
2. A bidirectional Transformer block contextualizes the complete input.
3. A reverse-causal attention scorer produces a separate routing map for every
   forward layer and attention head.
4. Nonnegative, identity-initialized strengths turn those maps into
   multiplicative gates in `(0, 1]`.
5. Each gate multiplies the transpose of the normal forward-attention map.
   The remaining weights are renormalized, so the intervention changes
   relative credit routing rather than overall gradient scale.
6. The routed map is used consistently for the value gradient and the
   surrogate softmax backward that produces query and key gradients.

The normal PyTorch SDPA result is retained exactly in the forward pass. At the
zero-strength centre, every routing gate is exactly one and model gradients
match ordinary backpropagation to numerical precision. With active strengths,
the forward loss remains bit-identical while model gradients change. Tests
also verify that all gates remain positive and at most one, router checkpoints
round-trip, and legacy gradient-transformer checkpoints still load.

The implementation exposed one reproducibility bug in the shared harness:
fresh backward-rule centres were constructed before the configured seed was
applied. This did not affect candidate sharing within a generation or the
checkpoint-resumed main lineage, but nominally identical fresh runs could
start from different centres. Fresh centres are now seeded explicitly.
After the fix, scalar and two-GPU horizon-40 replays matched exactly across
every reported metric and centre update.

Full-width horizon-40 calibration from the same seeded centre:

| Sigma | Mean route gate | Suppressed fraction | Pair delta RMS | Standardized fitness SD | Centre update / centre RMS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.9870 | 11.7% | 0.00050 | 0.128 | 2.53% |
| 0.20 | 0.9827 | 10.0% | 0.00177 | 0.394 | 17.31% |

`sigma=0.1` is selected. The larger radius provides more fitness separation
but makes a single centre update too large. The parallel run starts directly
at horizon 40: the established ordinary-Adam diagnostics and the main
experiment both show that horizons 10 and 20 precede shortcut learning, and
router fitness separation there is correspondingly negligible. This keeps
the side experiment focused on the first informative regime without changing
the task, model, population, or clean fitness set.

The population-64 router run is:

- Run:
  [`attention-router-p64-h40-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/qwslfp4a)
- Hardware: two-GPU candidate sharding on GPUs 2 and 3.
- Generation-0 runtime: `27.2s`.
- Clean CE / accuracy: `2.3055 / 17.1%`.
- Masked / incorrect-hint accuracy: `19.7% / 14.5%`.
- Distinct predicted digits / modal fraction: `10 / 10` and `39.0%`.
- Antithetic pair-difference RMS: `0.00057`.
- Standardized-fitness SD: `0.127`.
- Centre update / centre RMS: `2.88%`.

Thus the full population retains a directional, nonzero estimator without
prediction collapse or an excessive first update. As with the main rule,
multiple generations are needed to separate evolution from the randomly reset
forward model.

Across its first two ten-generation windows:

| Window | Clean CE | Clean acc. | Masked | Wrong hint | Centre gate magnitude |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generations 0-9 | 2.3535 | 17.8% | 20.8% | 14.8% | 0.0112 |
| Generations 10-19 | 2.3302 | 18.6% | 20.5% | 16.8% | 0.0258 |

The router therefore improved balanced clean accuracy by `0.84` percentage
points while retaining all ten output classes. Its second window approximately
matches the earlier one-run ordinary-Adam horizon-40 accuracy (`18.6%`) and
slightly improves CE (`2.3302` versus `2.3394`), but this is not yet a
matched-seed baseline comparison.

Best-candidate diagnostics were added at checkpoint 20. The diagnostic
continuation is:

- Run:
  [`attention-router-p64-resume20-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/331xexyu)
- Source checkpoint: `attention-router-p64-h40-seed7/checkpoint_000020.pt`.
- Generations 20-26 from the original process are superseded.

#### Original router's horizon-80 failure

The router continued through generation 50 and automatically promoted to
horizon 80. Its final horizon-40 window remained balanced:

| Window | Clean CE | Masked | Wrong hint | Correct leak |
| --- | ---: | ---: | ---: | ---: |
| Generations 40-50 | 2.3583 | 20.7% | 16.6% | 39.3% |
| Generations 51-59, horizon 80 | 3.1966 | 17.3% | 0.0% | 100.0% |

Every one of the 576 candidates evaluated over the first nine horizon-80
generations followed the shortcut on the wrong-hint split. The run was stopped
at its durable generation-60 checkpoint.

A matched generation-50 radius replay showed that this was not fixed by
searching moderately farther from the centre:

| Sigma | Clean CE | Masked | Wrong hint | Correct leak | Pair delta RMS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 3.2626 | 15.9% | 0.0% | 100.0% | 0.0101 |
| 0.10 | 3.2598 | 16.1% | 0.0% | 100.0% | 0.0156 |
| 0.20 | 3.2565 | 16.1% | 0.0% | 100.0% | 0.0262 |
| 0.40 | 3.2520 | 16.3% | 0.0% | 100.0% | 0.0374 |

The ranking signal grew with radius, but none of the 256 candidates resisted
the shortcut. Edge-level inspection explained why. At the generation-50
centre, the leak edge retained `97.2%`, `98.2%`, and `99.7%` of the average
other-edge gate in the three layers. Across the complete radius replay, the
most selective candidate still retained `57.5%`; no candidate reduced the
leak edge below `50%` of comparable routes.

#### Oracle routing and complete-attention correction

An oracle diagnostic suppresses only the final query's direct backward edge
to the leaked-answer token. It uses the same forward initialization, biased
batches, optimizer, and fixed clean evaluation data:

| Horizon | Backward route | Masked | Wrong hint | Correct leak | Clean CE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 40 | Ordinary Adam | 22.3% | 18.4% | 43.0% | 2.3539 |
| 40 | Q/K/V edge suppression | 21.1% | 18.4% | 26.6% | 2.3244 |
| 40 | Complete attention suppression | 20.3% | 17.2% | 26.6% | 2.3220 |
| 80 | Ordinary Adam | 15.2% | 0.0% | 100.0% | 3.2705 |
| 80 | Q/K/V edge suppression | 22.3% | 18.4% | 35.9% | 2.2736 |
| 80 | Complete attention suppression | 19.1% | 21.1% | 37.5% | 2.2973 |

Thus attention-edge suppression can prevent the shortcut transition. The
failure was in discovering a selective route, not in the basic intervention.

The diagnostic also exposed two implementation limitations:

1. The normal forward attention remains unchanged by design. Its output
   projection therefore received parameter gradients computed from an
   activation that still contained the leak, even when Q/K/V backward credit
   for that edge was suppressed.
2. `max_log_suppression=8` was only used as a final clamp. It did not scale the
   nonnegative routing strength. Because the reverse softmax has only two
   possible destinations at the penultimate leak token, realistic
   perturbations could not strongly suppress its edge to the final query.

The corrected optional route keeps the normal forward output exact but uses
the routed value mixture when computing the attention output projection's
parameter gradient. The maximum log-suppression value now also scales the
active routing strength before clamping. Tests verify that the identity centre
still matches ordinary gradients, active routing changes the output projection
gradient, and gates remain suppress-only.

Full-population horizon-40 calibration selected `sigma=0.05`:

| Sigma | Pair delta RMS | Standardized fitness SD | Centre update / RMS |
| ---: | ---: | ---: | ---: |
| 0.050 | 0.00125 | 0.303 | 3.05% |
| 0.075 | 0.00197 | 0.475 | 6.65% |

The larger radius produces more separation but more than doubles the first
centre update. The corrected run is:

- Run:
  [`attention-router-complete-p64-h40-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/onxqe0tf)
- Configuration: population 64, horizon 40, `sigma=0.05`, complete-attention
  routing, and two-GPU candidate sharding.
- Additional diagnostics report the leak-edge gate relative to the other
  final-query gates, so broad suppression is not mistaken for selective
  shortcut blocking.

Its first two windows were:

| Window | Clean CE | Masked | Wrong hint | Correct leak | Robust min split |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generations 0-9 | 2.3520 | 20.8% | 14.8% | 35.2% | 16.1% |
| Generations 10-19 | 2.3266 | 20.4% | 16.7% | 36.0% | 17.8% |

This is balanced horizon-40 behavior, and the second window improves clean CE
and wrong-hint accuracy. The original leak-relative gate metric came from the
first perturbed candidate, however, so its apparent change from `0.992` to
`0.964` does not prove that the centre became selective. A post-update
unperturbed-centre routing diagnostic was added at checkpoint 20. It measured
a leak/other gate ratio of `0.977`: the centre had learned a real but still
modest preference amid broad suppression.

The fixed `sigma=0.05` again became nonlocal as the router centre evolved.
A matched generation-20 replay gave:

| Sigma | Pair delta RMS | Standardized fitness SD | Centre update / RMS |
| ---: | ---: | ---: | ---: |
| 0.050 | 0.00779 | 0.867 | 8.75% |
| 0.025 | 0.00571 | 0.745 | 4.32% |
| 0.010 | 0.00317 | 0.466 | 1.27% |

`sigma=0.01` retains a clear directional estimator while restoring a local
centre update. The checkpoint-20 continuation is:

- Run:
  [`attention-router-complete-p64-sigma001-centerdiag-resume20-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/vk98n352)
- Source:
  `attention-router-complete-p64-h40-seed7/checkpoint_000020.pt`.
- The one-generation centre-diagnostic and radius-probe branches are excluded
  from the authoritative lineage.

![Complete attention-router progress](results/attention_router_complete_progress.png)

### 2026-07-26: robust fitness continuation

The unrestricted gradient-transformer rule's first two horizon-80
ten-generation windows were:

| Window | Clean CE | Masked | Wrong hint | Correct leak | Robust min split |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generations 200-209 | 2.8838 | 17.4% | 1.7% | 85.9% | 10.0% |
| Generations 210-219 | 2.5741 | 12.3% | 7.5% | 38.3% | 11.1% |

The mean-CE objective reduced shortcut confidence but also drove masked
accuracy toward chance. This is calibration, not stable pointer retrieval.
Some individual candidates had both masked and wrong-hint accuracy above
chance, but they were not consistently selected by mean CE.

The harness now supports a `worst_mode_ce` objective. Candidate fitness is the
reduction in the worse of masked and incorrect-hint CE, and the plateau
objective uses that same quantity. This prevents an improvement on one clean
split from hiding regression on the other. A first attempt from checkpoint
220 was stopped after two generations because its local population was already
collapsed (`1.28` and `1.45 / 10` distinct predicted digits). The controlled
continuation therefore starts from the pre-collapse generation-190
horizon-80 checkpoint; its result will be compared against the existing
mean-CE lineage from the same centre.

- Run:
  [`learned-backward-p64-h80-worstce-resume190-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/cnn6e8jz)
- Source:
  `learned-backward-p64-sigma001-plateau-resume130-seed7/checkpoint_000190_horizon80.pt`.

Across the matched generations 190-199:

| Objective | Clean CE | Masked | Wrong hint | Correct leak | Distinct digits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean clean CE | 2.9751 | 17.4% | 0.87% | 92.7% | 9.55 |
| Worst-split CE | 2.9673 | 17.2% | 0.93% | 92.1% | 9.55 |

The alternative ranking objective made no material difference. The population
did not contain sufficiently shortcut-resistant local directions for either
objective to select, so this branch was stopped at its generation-200
checkpoint.

### 2026-07-26: one shared backward suppression map

The intended constrained router uses one input-conditioned suppression map
and applies that same map to every head in every forward layer. It does not
change forward attention: the ordinary shortcut-containing forward pass and
loss remain exact, while only the backward attention routes are suppressed.
Forward suppression is therefore out of scope because it would change the
training data path rather than the learned credit-assignment rule.

The implementation now supports both architectures:

- Shared map, the default for new routing experiments: one query/key map and
  one suppression strength expanded across all three layers and four heads.
- Independent maps, retained as the existing control and for loading old
  checkpoints.

The suppression strength is projected to the nonnegative domain after each
EGGROLL update. This matters more in the shared model because it has one scalar
strength: a negative centre is functionally identical to zero under the
suppress-only `ReLU`, leaving evolution in a flat inactive region.

Fresh full-population calibration gave:

| Horizon | Sigma | Pair delta RMS | Standardized fitness SD | Centre update / RMS | Masked | Wrong hint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 0.020 | 0.00018 | 0.040 | 0.17% | 19.7% | 14.5% |
| 40 | 0.035 | 0.00074 | 0.162 | 1.19% | 19.7% | 14.5% |
| 40 | 0.050 | 0.00174 | 0.362 | 3.77% | 19.7% | 14.6% |
| 80 | 0.010 | 0.00124 | 0.261 | 0.55% | 16.4% | 0.0% |
| 80 | 0.050 | 0.05145 | 0.996 | 11.06% | 17.0% | 0.0% |

The `sigma=0.05` horizon-40 row predates projection and moved its sole
suppression strength negative, making the updated centre inactive. The
projected `sigma=0.035` probe is local while retaining measurable ranking
signal. A first continuation at outer learning rate `0.1` was stopped after
generation 2: once the suppression strength became positive, centre updates
jumped to `5-6%` and robust wrong-hint accuracy fell from `15.6%` to `9.4%`.
The replacement keeps the informative candidate radius but reduces the centre
step fivefold:

- Run:
  [`attention-router-shared-p64-s0035-outer002-worstce-seed7`](https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/j92lh3gd)
- Configuration: population 64, horizon 40, complete attention-gradient
  routing, one shared map, `sigma=0.035`, outer learning rate `0.02`, and
  worst-split CE fitness.
- GPUs 2-3 run this lineage while the independent-map control continues on
  GPUs 0-1.

The first matched window shows no shared-map advantage:

| Router, generations 0-9 | Clean CE | Masked | Wrong hint | Correct leak | Robust min split |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent maps | 2.3520 | 20.78% | 14.85% | 35.19% | 16.13% |
| One shared map | 2.3530 | 20.69% | 14.80% | 35.64% | 15.94% |

The shared centre remained stable but learned broad rather than selective
suppression: its mean unperturbed leak/other gate ratio was `0.998`, with
`1.0` meaning no preference. A second ten-generation window will test whether
selectivity appears after the shared suppression strength becomes active.

The second window improved while retaining all ten output classes:

| Shared-map window | Clean CE | Masked | Wrong hint | Correct leak | Robust min split | Centre leak ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generations 0-9 | 2.3530 | 20.69% | 14.80% | 35.64% | 15.94% | 0.998 |
| Generations 10-19 | 2.3296 | 20.44% | 16.70% | 37.73% | 17.85% | 0.988 |

Population-level routing diagnostics were added and replayed from the
generation-20 checkpoint. They average each candidate's leak-edge gate
relative to its other final-query gates over all inner steps, then compare
selectivity (`1 - relative gate`) with candidate fitness:

| Replay generation | Best ratio in population | Candidates below 0.9 | Fitness/selectivity correlation | Fittest candidate ratio |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 0.834 | 4.69% | +0.736 | 0.834 |
| 21 | 0.818 | 6.25% | +0.738 | 0.845 |

This separates two hypotheses. Selective candidates do exist in a population
of 64, and lower leak ratios are strongly associated with better worst-split
CE fitness. The remaining issue is whether repeated low-step EGGROLL updates
can accumulate this signal into the centre, especially after moving to the
harder horizon-80 problem. Directly selecting the candidate with the best
sampled minimum accuracy is less reliable: its ratios were `1.041` and
`1.003`, consistent with noisy accuracy estimates.

The independent-map control completed two horizon-80 windows before being
stopped at its durable generation-70 checkpoint:

| Independent-map horizon-80 window | Clean CE | Masked | Wrong hint | Correct leak | Robust min split |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generations 51-60 | 3.1847 | 17.15% | 0.0% | 100% | 0.0% |
| Generations 61-70 | 3.1466 | 18.70% | 0.0% | 100% | 0.0% |

This is not evidence that horizon 80 is intrinsically hopeless; it establishes
that the independent router had not escaped the shortcut after 20 generations
there. The shared router will receive a substantially longer horizon-80 phase
before it is assessed.

A population replay from the independent generation-70 checkpoint clarifies
why simply extending that particular run was unlikely to help:

| Horizon-80 independent population diagnostic | Value |
| --- | ---: |
| Candidate leak-ratio range | 0.953-0.979 |
| Candidate fraction below 0.9 | 0.0% |
| Fitness/selectivity correlation | +0.027 |
| Fittest candidate leak ratio | 0.961 |

Unlike the shared router at horizon 40, this local independent-map population
contains no strongly selective candidate and fitness is effectively unrelated
to the small selectivity differences that remain. The corresponding shared-map
diagnostic must be measured after its horizon transition; it should not be
inferred from this failed independent centre.
