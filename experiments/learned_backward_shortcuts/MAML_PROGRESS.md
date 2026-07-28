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
