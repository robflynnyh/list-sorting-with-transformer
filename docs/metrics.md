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
| `train/loss` | Mean teacher-forced cross-entropy over predicted outputs. Inputs and padding are excluded. Executor-supplied responses are excluded, while responses assigned to the model are included. Lower is better. |
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

These metrics evaluate autoregressive rollouts under the task's configured
executor assistance.

| Metric | Meaning |
| --- | --- |
| `exact_match` | The final answer exactly equals the correctly sorted input. Executor-assisted machine runs must reach a valid `DONE`; partial- and no-tool runs must also generate their assigned observations or windows correctly. |
| `trace_exact_match` | Every generated quicksort event, machine action, and model-assigned observation or window exactly matches the deterministic reference trace. A locally executable but noncanonical `KEEP` or `SWAP` therefore makes this zero even if execution continues. |
| `full_exact_match` | Both execution trace and final result are exact. |
| `operation_prefix_fraction` | Fraction of reference operations generated correctly before the first incorrect operation. |
| `valid_syntax` | The answer follows the required grammar. For executor-assisted tasks, this means the machine reached an in-phase `DONE`. An incorrect but executable `KEEP` or `SWAP` does not make the syntax invalid. |
| `correct_length` | A syntactically valid answer contains the same number of values as the input. |
| `sorted` | A syntactically valid answer is nondecreasing. It may still contain the wrong values. |
| `multiset_preserved` | A syntactically valid answer contains exactly the input values with the same multiplicities. It need not be ordered. |
| `trace_syntax_valid` | Generated trace tokens are structurally valid. For local-window sorting, either `KEEP` or `SWAP` is accepted while a pair is active, irrespective of which one the reference trace uses. |
| `target_token_accuracy` | Positional token accuracy of the freely generated final answer, or positional action accuracy for a machine task. |
| `full_target_token_accuracy` | Positional accuracy over the complete generated target. It equals action accuracy for fully executor-assisted runs and includes every model-generated observation or window for partial- and no-tool runs. |
| `execution_completed` | Offline or interactive executor reached `DONE` without an invalid action. |
| `observation_token_accuracy` | Positional accuracy of observations assigned to the model. Missing observations after an early failure count as incorrect. |
| `observation_exact_match` | Every model-generated observation was correct. For fully no-tool tasks, action replay must also reach `DONE`. |
| `window_token_accuracy` | Positional accuracy across fixed-width local windows generated along the model's actual execution path. If a rollout terminates before attempting any window despite the reference requiring one, the score is zero rather than vacuously perfect. |
| `window_transition_exact_fraction` | Fraction of model-generated windows that exactly match the executor window. |
| `window_exact_match` | Every local window encountered along the model's actual execution path was generated exactly. This measures state tracking separately from whether its `KEEP` and `SWAP` decisions matched the reference. |
| `generated_window_count` | Mean number of local windows generated by the model per example. |
| `tool_window_count` | Mean number of local windows supplied by the executor per example. |
| `timed_out` | Machine decoding did not produce an in-phase `DONE` within its action budget. This force-stops looping policies. |

## Teacher-Forced Evaluation

| Metric | Meaning |
| --- | --- |
| `teacher_forced_loss` | Cross-entropy when every correct preceding trace and answer token is supplied. |
| `teacher_forced_token_accuracy` | Next-token accuracy under the same teacher-forced condition. |

Teacher-forced metrics measure whether local next-token prediction has been
learned. For fully executor-assisted tasks, only model actions are scored and
gold executor observations remain in context. Partial-assistance tasks also
score the observations or windows assigned to the model; fully no-tool tasks
score all actions and responses. These metrics do not show whether the model
can execute a complete trace autonomously, because free generation can
compound an early error.

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
