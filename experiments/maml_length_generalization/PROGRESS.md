# One-step MAML Length Generalization

## Method

This experiment compares one-step meta-learning methods on an ordinary
pointer-next Transformer. The original MAML methods meta-update either all
forward parameters or only the QKV parameters. The router-MAML method instead
meta-learns a separate suppression-only attention-gradient router while the
forward model receives persistent ordinary updates.

Each iteration:

1. Samples one ordinary pointer-next batch with list length 2 through 20.
2. Computes one differentiable virtual SGD update on that batch.
3. Evaluates the virtually updated weights on one batch from fixed
   meta-training sets at lengths 40, 60, 70, 70, 80, 90, and 100. Each listed
   length has 256 examples; repeating 70 gives it twice the sampling weight.
4. Backpropagates the meta loss through the virtual update and applies an Adam
   meta-update to the persistent network.
5. Discards the virtual weights, recomputes the original short-batch loss from
   the meta-updated network, and applies one persistent ordinary Adam update.

The same short batch is used for the virtual and ordinary updates, but the
ordinary forward and backward pass is recomputed after the meta-update. The
meta and ordinary Adam optimizers have independent moment states.

One 64-example meta batch is used per iteration, cycling evenly through the
listed lengths. Fixed length-400 examples are reporting-only and never
influence either optimizer.

The matched `ordinary` mode omits the virtual and meta updates. It uses the
same model initialization, short-data seed, ordinary Adam configuration, and
fixed evaluation sets as the MAML run. Both modes report length-50 and
length-400 accuracy every 100 steps.

## Metrics

- `train/virtual_short_loss`: short-task loss used to construct the virtual
  update.
- `train/meta_batch_length`: length used by the current meta batch.
- `train/meta_loss_after_virtual`: differentiable mixed-length meta objective.
- `train/virtual_step_meta_loss_change`: meta loss after the virtual step minus
  its value before the step. Negative means the short update helped the current
  meta batch.
- `train/ordinary_short_loss`: recomputed loss used for the persistent ordinary
  update.
- `eval/length_N/accuracy`: fixed-set accuracy at lengths 2, 20, 50, and 400.

## Status

The two unit tests confirm that the virtual update does not mutate persistent
weights and that the MAML gradient differs from a direct length-50 gradient.
CPU and real-shape GPU smoke runs completed successfully. The production-shape
GPU smoke measured approximately 17.5 training iterations per second before
periodic evaluation overhead.

The full repository suite passes with 327 tests.

## Completed length-50-only run

- W&B:
  <https://wandb.ai/wobrob101/list-sorting-maml/runs/s8ne3b1e>
- Run: `pointer-next-maml-length20-meta50-heldout400-seed7`
- Service: `list-sorting-maml-length-generalization.service`
- Output:
  `artifacts/maml_length_generalization/pointer-next-maml-length20-meta50-heldout400-seed7`

The run completed all 10,000 steps. Length-50 accuracy reached 100% by step
300 and remained there. Length-400 accuracy peaked at 16.4% around steps
300-400, then finished at 13.3%. Training longer therefore did not improve
length-400 generalization.

## Mixed-length comparison

The next comparison uses fixed meta-training sets at
`40,60,70,70,80,90,100` and a matched ordinary-Adam run. Implementation and
validation are complete. CPU tests cover repeated-length weighting and the
ordinary reporting path. Production-shape GPU smokes cover the mixed cycle and
the maximum meta length of 100; the latter measured approximately 16.8
iterations per second. The full repository suite passes with 329 tests.

### Outcome

- Mixed MAML W&B:
  <https://wandb.ai/wobrob101/list-sorting-maml/runs/d8fce2ih>
- Ordinary W&B:
  <https://wandb.ai/wobrob101/list-sorting-maml/runs/xdjcxpsa>

Mixed MAML reached 53.9% length-400 accuracy at step 200, then collapsed to
3.9% at step 300 and remained near 4.7% through step 7000, when it was stopped.
The matched ordinary run completed 10,000 steps, finishing at 90.6% on length
50 and 15.6% on length 400. Its best length-400 result was 16.4% at step 500.

## QKV-only meta update

The next ablation restricts the persistent outer MAML update to
`blocks.*.attention.qkv.weight`. The virtual short update still differentiates
through every model parameter, and the ordinary Adam step still updates the
whole model. The completed ordinary metrics are loaded into the MAML run as
`ordinary_reference/length_50/*` and
`ordinary_reference/length_400/*`, making both curves visible in one W&B run.

Unit tests confirm that the QKV scope selects only the two fused QKV matrices.
CPU and production-shape length-100 GPU smokes pass; the latter selected 98,304
meta-updated parameters and ran at approximately 17.3 iterations per second.
The full repository suite passes with 331 tests.

### Completed QKV run

- W&B:
  <https://wandb.ai/wobrob101/list-sorting-maml/runs/hh138504>
- Service: `list-sorting-maml-qkv.service`
- Output:
  `artifacts/maml_length_generalization/pointer-next-maml-qkv-meta40-60-70x2-80-90-100-heldout400-seed7`

The run completed all 10,000 steps. It reached its best length-400 accuracy of
53.1% at step 200, then settled near 30% and finished at 29.7%. The matched
ordinary reference finished at 15.6%. Both length-50 and length-400 ordinary
reference curves are present in this W&B run.

## Persistent router MAML

The router-MAML variant keeps both the task model and attention-gradient router
across iterations. Each iteration:

1. Computes one hypothetical short-task update using router-modified attention
   gradients.
2. Evaluates the hypothetical model on the fixed mixed-length meta data.
3. Differentiates that loss through the hypothetical update and updates only
   the router.
4. Discards the hypothetical model weights.
5. Recomputes the same short batch with the updated router and applies one real
   persistent Adam update to the task model.

The router changes backward credit on existing causal attention edges but does
not change the numerical forward pass. It is suppression-only, shares one
routing map across task-model layers and heads, and begins nearly neutral. This
is not population evolution: there is one persistent router, one persistent
task model, one meta update, and one real task-model update per iteration.

Focused tests confirm that routed and ordinary forward logits are identical,
the long-task loss produces a finite nonzero router gradient through the
hypothetical model update, and both persistent states are checkpointed. A
two-step production-shape GPU smoke ran at approximately 11-15 iterations per
second. The full repository suite passes with 333 tests.

### Active router-MAML run

- W&B:
  <https://wandb.ai/wobrob101/list-sorting-maml/runs/ln82yczj>
- Service: `list-sorting-router-maml.service`
- `with-gpu` ticket: `80574c9d`
- Output:
  `artifacts/maml_length_generalization/pointer-next-router-maml-meta40-60-70x2-80-90-100-heldout400-seed7`

The run completed all 10,000 steps. It finished at 82.0% length-50 accuracy and
12.5% length-400 accuracy, below the matched ordinary results of 90.6% and
15.6%. The router remained active, with a final mean backward multiplier of
0.897 and a nonzero meta-gradient, but suppressing gradient credit alone did
not improve length generalization.

### Signed-credit ablation

The suppression-only router can reduce an attention edge's backward multiplier
from one toward zero, but cannot oppose a harmful gradient. The matched signed
ablation expands this range to `[-1, 1]`, allowing the router to preserve,
suppress, or reverse attention-score credit without changing the forward pass.
All data, model, optimizer, seed, and evaluation settings remain unchanged.
Focused tests and a production-shape GPU smoke confirm that signed router
meta-gradients flow through the hypothetical update. The full repository suite
passes with 334 tests.
