# Learned Backward Length Generalization

## Question

Can an evolved, suppression-only attention backward rule encourage an ordinary
pointer-next Transformer to learn solutions that generalize to much longer
sequences?

## Experimental design

- Forward-training data is the normal pointer-next task. Prompts contain
  `<bos>`, values and commas, one `<PTR>`, and `<sep>`. They contain no leak,
  hint, mask, or synthetic query tokens.
- Training list lengths are sampled uniformly from 2 through 20 on every
  optimizer update. These become token-sequence lengths 6 through 42.
- Candidate fitness is improvement in mean cross-entropy on a fixed
  256-example ranking set at list length 50, or 102 input tokens.
- Elite proposals are accepted on a separate fixed 256-example set at length
  50. The ranking and acceptance sets are generated once from consecutive
  slices of the same seeded stream and remain fixed across generations.
- Held-out reporting uses one fixed 128-example set at list length 400, or 802
  input tokens. Held-out examples do not affect candidate ranking, proposal
  acceptance, centre updates, sigma, or horizon promotion.
- A separate fixed 128-example mixed-length set reports in-domain accuracy.
- The backward rule uses `suppress_renorm`. Its edge multipliers are
  nonnegative, so it can preserve or suppress ordinary attention credit but
  cannot reverse gradient signs.
- Population candidates share the same initialization and forward-training
  batches within each trajectory.
- Elite proposals are accepted only when they beat the existing centre on both
  independently initialized acceptance trajectories. Both trajectories use
  the acceptance set, never the candidate-ranking set.
- Ordinary Adam from the same initialization and batches is rerun as a
  reporting-only control.

The high-throughput controller uses a fixed 320-update inner horizon, a
two-layer forward Transformer, BF16 forward training, 10,000 outer
generations, and the full population of 64 candidates. It retains adaptive
elite counts, sigma control, strict acceptance on two independent
trajectories, and the three-GPU vectorized implementation. Length-400
reporting, centre/ordinary controls, and proposal replay run every ten
generations; they never affect optimization. Numbered checkpoints are written
every 25 generations to keep the complete run below roughly 0.5 GB.

Sigma adapts symmetrically in log-space: an accepted proposal doubles sigma,
while a rejected proposal multiplies it by 0.8. This gives an equilibrium
acceptance rate of approximately 24% and, unlike the initial three-success
streak rule, allows one accepted update to escape the minimum sigma.

## Preserved shortcut-resistance run

The signed-credit random-leak controller was stopped after completed generation
69 at horizon 1280. Its latest held-out minimum accuracy was 92.97%, and its
best recorded value was 95.31% at horizon 640.

It can be resumed from:

```text
artifacts/learned_backward_shortcuts/attention-router-signed-credit-strict-h160-p64-seed7/latest.pt
```

No files in that run directory are overwritten by this experiment.

## Validation

A one-generation GPU smoke exercised the normal pointer prompt, vectorized
population training, fixed length-50 fitness, length-400 held-out evaluation,
adaptive elite selection, and two independent acceptance trajectories. The
proposal was correctly rejected because its matched deltas were
`-0.00000024` and `0`; strict acceptance requires both to be positive. Peak
allocated memory was 0.78 GiB for a population-8 chunk.

The full repository test suite passes with 320 tests.

## Preserved exploratory length run

- W&B:
  <https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/ksbfapf2>
- Run: `pointer-next-length20-fitness50-heldout400-seed7`
- Checkpoint:
  `artifacts/learned_backward_length_generalization/pointer-next-length20-fitness50-heldout400-seed7/latest.pt`

The run was stopped after generation 69 at horizon 1280. It used a three-layer
forward model and remains independently resumable. Its artifacts are not
overwritten by the high-throughput run.

## High-throughput validation

The full repository suite passes with 323 tests. A three-generation real-GPU
smoke covered BF16, the two-layer forward model, fixed horizon mode, adaptive
P64-compatible elite selection, strict two-trajectory acceptance, and sparse
reporting. The middle sparse generation retained both acceptance trajectories
while omitting all length-400/control work and shortcut-only metric aliases.

## Superseded high-throughput run

- W&B:
  <https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/k8sj7g3n>
- Run:
  `pointer-next-length20-fitness50-heldout400-h320-2l-p64-10k-seed7-v3`
- Service:
  `list-sorting-length-generalization-h320-2l-p64-10k-seed7-v3.service`
- Output:
  `artifacts/learned_backward_length_generalization/pointer-next-length20-fitness50-heldout400-h320-2l-p64-10k-seed7-v3`

Generation 0 completed in 52.9 seconds with full reporting, 7.72 GiB peak
allocation, and an accepted elite-4 update. Generation 1 completed in 38.9
seconds on the sparse path with 2.27 GiB peak allocation and both strict
acceptance trajectories intact. The measured weighted runtime is approximately
40.3 seconds per generation. This run was stopped after generation 17 because
the asymmetric sigma rule had already reached its minimum: every rejection
halved sigma, but recovery required three consecutive accepted proposals.

## Stopped balanced-sigma run

- W&B:
  <https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/xgcg2kik>
- Run:
  `pointer-next-length20-fitness50-heldout400-h320-2l-p64-10k-seed7-v4`
- Service:
  `list-sorting-length-generalization-h320-2l-p64-10k-seed7-v4.service`

The controller was stopped after generation 742 without useful cumulative
learning. It accepted 366 of 743 proposals (49.3%), while sigma remained
healthy and usually stayed between 0.15 and 0.21. Mean length-400 accuracy
relative to ordinary Adam was -0.77 percentage points, and the final 20 full
reports averaged -1.17 points. Candidate length-50 fitness still correlated
with the length-400 objective at roughly 0.35 to 0.39, suggesting that the
ranking signal existed but proposal selection did not retain it reliably.

The next controller configuration keeps the same total of 512 fixed length-50
examples but reserves 256 for candidate ranking and 256 for proposal
acceptance. Length 400 remains reporting-only. This change has been implemented
but not relaunched.

## Horizon-24 router MAML

The suppression-only MAML method that succeeded on the random-position
shortcut task has been transferred to this task. It keeps a queue of 24
ordinary pointer-next batches at list lengths 2 through 20, differentiates a
virtual Adam trajectory through all 24 batches using the persistent model's
current Adam moments, and updates the router against a fixed 256-example
length-50 set. It then discards the virtual trajectory, commits one real
routed Adam update on the first queued batch, and shifts the queue by one.

The task model remains the established two-layer, width-128 Transformer.
Router credit is suppression-only in `[0, 1]`; signed reversal is excluded.
Evaluation reports list lengths 2, 20, 50, and 400 and includes the existing
ordinary-Adam run as a matched reference.

A two-step GPU smoke exercised the full 24-step differentiable Adam path,
length-50 fitness, length-400 evaluation, and checkpoint persistence. The
checkpoint retained exactly 24 queued batches and reported
`train/lookahead_steps=24`.

The full 10,000-step run is tracked at
<https://wandb.ai/wobrob101/list-sorting-maml/runs/y989nxya>. At the first
evaluation (step 100), router MAML and the matched ordinary reference both
reached 11.7% length-400 accuracy. The router had begun changing the backward
rule, with 31.7% of eligible attention-gradient edges below a multiplier of
0.99. Training was proceeding at approximately 1.07 persistent updates per
second.

The run was stopped after step 620 because the router collapsed toward broad
suppression without improving length generalization. At step 600 it suppressed
97.8% of eligible attention-gradient edges and reached 13.3% length-400
accuracy, below the ordinary reference's 14.8%. Raw router meta-gradient norms
oscillated between roughly 17 and 55 near the end. These norms were measured
before clipping to 1.0, so the model did not receive updates of that magnitude,
but the persistent clipping made updates behave mostly as normalized
directions.

The failure was not merely an excessive router learning rate. At step 200 the
router improved length-50 accuracy to 90.6%, versus 87.5% for ordinary Adam,
while length-400 accuracy was already worse at 12.5% versus 14.1%. The
length-50 meta objective therefore did not reliably select for the desired
length-400 behavior.
