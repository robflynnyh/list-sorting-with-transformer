# One-step MAML Length Generalization

## Method

This experiment trains one ordinary pointer-next Transformer. There is no
learned backward network.

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
