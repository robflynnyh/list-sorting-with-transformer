# Learned Backward Rules for Shortcut Resistance

## Research question

Can a small clean dataset train a reusable backward credit-assignment rule that
extracts the genuine signal from a much larger dataset containing a perfect
answer shortcut?

The forward model is trained only on shortcut-containing examples. EGGROLL
optimizes a shared learned backward rule using fitness measured on 512 clean
examples. The forward model is reset to a newly sampled initialization every
EGGROLL generation; only the backward rule persists.

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
