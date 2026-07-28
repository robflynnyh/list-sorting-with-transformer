# One-step MAML Length Generalization

## Method

This experiment trains one ordinary pointer-next Transformer. There is no
learned backward network.

Each iteration:

1. Samples one ordinary pointer-next batch with list length 2 through 20.
2. Computes one differentiable virtual SGD update on that batch.
3. Evaluates the virtually updated weights on one batch from a fixed
   256-example length-50 meta-training set.
4. Backpropagates the length-50 loss through the virtual update and applies an
   Adam meta-update to the persistent network.
5. Discards the virtual weights, recomputes the original short-batch loss from
   the meta-updated network, and applies one persistent ordinary Adam update.

The same short batch is used for the virtual and ordinary updates, but the
ordinary forward and backward pass is recomputed after the meta-update. The
meta and ordinary Adam optimizers have independent moment states.

Length 50 is meta-training data. Fixed length-400 examples are reporting-only
and never influence either optimizer.

## Metrics

- `train/virtual_short_loss`: short-task loss used to construct the virtual
  update.
- `train/meta_length50_loss_after_virtual`: differentiable length-50 objective.
- `train/virtual_step_meta_loss_change`: length-50 loss after the virtual step
  minus its value before the step. Negative means the short update helped the
  current length-50 batch.
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

## Active run

- W&B:
  <https://wandb.ai/wobrob101/list-sorting-maml/runs/s8ne3b1e>
- Run: `pointer-next-maml-length20-meta50-heldout400-seed7`
- Service: `list-sorting-maml-length-generalization.service`
- Output:
  `artifacts/maml_length_generalization/pointer-next-maml-length20-meta50-heldout400-seed7`

The run uses one GPU through `with-gpu`. At step 300, fixed-set accuracies were
100% at lengths 2, 20, and 50, and 16.4% at reporting-only length 400. The
measured training rate is approximately 18.9 iterations per second before
periodic evaluation overhead.
