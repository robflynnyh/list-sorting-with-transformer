# Persistent Router MAML Shortcut Diagnostic

This diagnostic applies the persistent one-step router-MAML method to the
existing random-position shortcut task.

## Design

- The task model trains only on effectively unlimited correct-leak examples.
- The leak is inserted at a random list position.
- The task model and router both persist across steps; neither is reset.
- Each step computes one hypothetical router-mediated task-model update.
- The hypothetical model is evaluated on one balanced minibatch from a fixed
  512-example fitness set containing 256 masked and 256 incorrect leaks.
- Only the router receives the resulting meta-gradient.
- The hypothetical model is discarded.
- The same correct-leak batch is recomputed with the updated router and used
  for one persistent task-model Adam update.
- A separate fixed 512-example masked/incorrect set is reporting-only.
- The initial router is suppression-only and nearly identical to ordinary
  backpropagation. Gradient reversal is deliberately excluded from this first
  diagnostic.

The comparison includes an ordinary-Adam run with the same model
initialization, biased training stream, fixed evaluation sets, and optimizer.

## Success criterion

Correct-leak accuracy alone is not success because the model can copy the
shortcut. The router must improve both held-out masked-leak accuracy above 10%
and held-out incorrect-leak accuracy materially above the 11.1% exclusion
baseline, while retaining diverse value predictions.

## Seed-7 outcome

- Router MAML:
  <https://wandb.ai/wobrob101/list-sorting-maml-shortcut/runs/0w751sl2>
- Ordinary Adam:
  <https://wandb.ai/wobrob101/list-sorting-maml-shortcut/runs/hz3v3yoo>

Both runs completed 2,000 persistent task-model updates.

| Method | Correct leak | Masked held-out | Incorrect held-out |
| --- | ---: | ---: | ---: |
| Ordinary Adam | 100.0% | 22.3% | 0.0% |
| Router MAML | 100.0% | 21.9% | 0.0% |

Router MAML therefore did not learn shortcut resistance. Its final mean
backward multiplier was `0.955`, so it learned only mild suppression, and its
meta-gradient norm fell to approximately `1.8e-5` after the task model had
already fit the shortcut. The matched result supports a signal-starvation
failure: once correct-leak training loss approaches zero, the hypothetical
one-step update is too small for the clean fitness loss to teach the router.

## Eight-step lookahead follow-up

The next run keeps both persistent networks but replaces the one-step virtual
update with a receding eight-step lookahead. At each persistent step:

1. Simulate differentiable Adam on the current batch and seven forthcoming
   biased batches, starting from the task model's current Adam moments.
2. Evaluate the eight-step virtual model on the next fixed clean-fitness
   minibatch and update only the router.
3. Discard the virtual trajectory.
4. Commit only the current batch as one real persistent Adam update.
5. Shift the eight-batch window by one.

The virtual trajectory uses the same learning rate, Adam equations, and
differentiable global gradient clipping as the persistent optimizer.

The suppression-only horizon-24 run reached 99.2% masked, 91.4% incorrect,
and 100% correct-hint held-out accuracy by step 500:
<https://wandb.ai/wobrob101/list-sorting-maml-shortcut/runs/74p2kcpy>.
A matched signed-credit run will test whether allowing attention-gradient
multipliers in `[-1, 1]` improves on suppression in `[0, 1]`.

The signed run is tracked at
<https://wandb.ai/wobrob101/list-sorting-maml-shortcut/runs/mwrdb7b6>. At step
100 it reached 27.0% masked, 24.6% incorrect, and 52.0% correct-hint held-out
accuracy, slightly below suppression at the same step. It had not yet used
negative multipliers: the reversed-edge fraction was zero.
