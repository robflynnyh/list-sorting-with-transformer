# Metrics Reference

This document describes the metrics written to `metrics.json` and W&B.
Accuracy, match, syntax, and fraction metrics range from `0.0` to `1.0`.

## Steps and Logging

`step` is the optimizer-update number. With gradient accumulation, one
optimizer update contains `gradient_accumulation_steps` forward and backward
passes followed by one `optimizer.step()`.

Training metrics are logged every `log_interval` optimizer updates. W&B's
automatic `_step` counts calls to `wandb.log()` rather than optimizer updates,
so charts should use the project metric `step` as their x-axis.

## Training Metrics

| Metric | Meaning |
| --- | --- |
| `train/loss` | Mean teacher-forced cross-entropy over predicted outputs. Inputs and padding are excluded; executor observations are also excluded for `pointer_quicksort`. Lower is better. |
| `train/token_accuracy` | Teacher-forced next-token accuracy over the same included tokens. Correct preceding target tokens are supplied to the model. |
| `train/length` | Mean problem length across the accumulated microbatches in this optimizer update. |
| `train/minimum_length` | Shortest accumulated microbatch length in this optimizer update. |
| `train/maximum_length` | Longest accumulated microbatch length in this optimizer update. |
| `train/learning_rate` | Learning rate used by the optimizer update. |
| `train/gradient_norm` | Global gradient norm returned before clipping. The default clipping threshold is `1.0`, so the logged value can exceed `1.0`. |
| `train/elapsed_seconds` | Wall-clock training time since this process started. |

Older runs created before independent microbatch-length sampling may contain
only `train/length`. In those runs, all accumulated microbatches used that same
length.

## Evaluation Prefixes

Metrics under `eval/length_N/` evaluate newly generated examples of list length
`N`. With the default training range of 2-20, lengths 2, 11, and 20 are
in-domain checks, while lengths 25 and 40 test length extrapolation.

## Free-Generation Metrics

These metrics evaluate autoregressive generation without supplying the correct
intermediate tokens.

| Metric | Meaning |
| --- | --- |
| `exact_match` | The final answer exactly equals the correctly sorted input. For `pointer_quicksort`, the executor must also reach a valid `DONE`. |
| `trace_exact_match` | Every generated quicksort event or pointer action exactly matches the deterministic reference trace. |
| `full_exact_match` | Both execution trace and final result are exact. |
| `operation_prefix_fraction` | Fraction of reference operations generated correctly before the first incorrect operation. |
| `valid_syntax` | The answer follows the required grammar. For `pointer_quicksort`, this means the executor reached `DONE` without an invalid action. |
| `correct_length` | A syntactically valid answer contains the same number of values as the input. |
| `sorted` | A syntactically valid answer is nondecreasing. It may still contain the wrong values. |
| `multiset_preserved` | A syntactically valid answer contains exactly the input values with the same multiplicities. It need not be ordered. |
| `trace_syntax_valid` | Generated trace tokens are structurally valid. For pointer traces, all emitted actions must have been accepted by the executor, although execution may be incomplete. |
| `target_token_accuracy` | Positional token accuracy of the freely generated final answer, or positional action accuracy for `pointer_quicksort`. |
| `full_target_token_accuracy` | Positional token accuracy over the freely generated trace and answer. It equals action accuracy for `pointer_quicksort`. |
| `execution_completed` | Pointer executor reached `DONE` without an invalid action. Emitted only for `pointer_quicksort`. |
| `timed_out` | Pointer executor did not finish within the canonical maximum action count. Emitted only for `pointer_quicksort`. |

## Teacher-Forced Evaluation

| Metric | Meaning |
| --- | --- |
| `teacher_forced_loss` | Cross-entropy when every correct preceding trace and answer token is supplied. |
| `teacher_forced_token_accuracy` | Next-token accuracy under the same teacher-forced condition. |

Teacher-forced metrics measure whether local next-token prediction has been
learned. For pointer traces, only model actions are scored; gold executor
observations remain in context. These metrics do not show whether the model can
execute a complete trace autonomously, because free generation can compound an
early error.

## Recommended Dashboard

For the default experiment, the clearest primary charts are:

- `exact_match` at lengths 20, 25, and 40 for sorting performance;
- `trace_exact_match` at lengths 20, 25, and 40 for strict execution;
- `operation_prefix_fraction` at lengths 20, 25, and 40 for partial progress;
- `teacher_forced_loss` alongside free-generation metrics to distinguish local
  learning from rollout failure.

Do not interpret `sorted` alone as task success: a model can emit a short or
incorrect but monotonic sequence. Likewise, high teacher-forced token accuracy
does not imply high complete-trace exact match.
