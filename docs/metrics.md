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
| `train/loss` | Mean teacher-forced cross-entropy over generated trace and answer tokens. Input and padding tokens are excluded. Lower is better. |
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
| `exact_match` | The final answer exactly equals the correctly sorted input. Trace correctness is not required. |
| `trace_exact_match` | Every generated quicksort event and argument exactly matches the deterministic reference trace. |
| `full_exact_match` | The entire generated trace and final answer exactly match the reference target. |
| `operation_prefix_fraction` | Fraction of reference operations generated correctly before the first incorrect operation. |
| `valid_syntax` | The answer after `<ANSWER>` follows the value/comma grammar and ends in `<eos>`. |
| `correct_length` | A syntactically valid answer contains the same number of values as the input. |
| `sorted` | A syntactically valid answer is nondecreasing. It may still contain the wrong values. |
| `multiset_preserved` | A syntactically valid answer contains exactly the input values with the same multiplicities. It need not be ordered. |
| `trace_syntax_valid` | The output contains operation-framed trace content followed by an `<ANSWER>` marker. This is weaker than semantic trace correctness. |
| `target_token_accuracy` | Positional token accuracy of the freely generated final answer. |
| `full_target_token_accuracy` | Positional token accuracy over the freely generated trace and answer. Insertions or omissions shift later positions, making this deliberately strict. |

## Teacher-Forced Evaluation

| Metric | Meaning |
| --- | --- |
| `teacher_forced_loss` | Cross-entropy when every correct preceding trace and answer token is supplied. |
| `teacher_forced_token_accuracy` | Next-token accuracy under the same teacher-forced condition. |

Teacher-forced metrics measure whether local next-token prediction has been
learned. They do not show whether the model can execute a complete trace
autonomously, because free generation can compound an early error.

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
