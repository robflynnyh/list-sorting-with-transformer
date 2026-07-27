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
- Candidate fitness is improvement in mean cross-entropy on one fixed
  512-example set at list length 50, or 102 input tokens.
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
  independently initialized acceptance trajectories. Both use the same fixed
  length-50 fitness set.
- Ordinary Adam from the same initialization and batches is rerun as the
  generation-level control.

The initial controller starts at horizon 160 and uses the same automatic
performance-plateau horizon extension, adaptive elite counts, sigma control,
and three-GPU vectorized population implementation as the shortcut-resistance
experiment.

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

## Active run

- W&B:
  <https://wandb.ai/wobrob101/list-sorting-learned-backward/runs/ksbfapf2>
- Run: `pointer-next-length20-fitness50-heldout400-seed7`
- Service: `list-sorting-length-generalization-seed7-v3.service`
- GPUs: 0, 1, and 2 through `with-gpu`; GPU 3 remains available.

The first two horizon-160 generations took 37.9 and 36.3 seconds, with
7.78 GiB peak allocation per worker. Generation 0 rejected a mixed-sign
proposal. Generation 1 accepted elite count 2 because both independent
acceptance deltas were positive (`+0.0163` and `+0.4578`). At the identity
rule, ordinary training reached 79–80% on the fixed length-50 set, 100% on the
mixed length-2-through-20 set, and 8.6–11.7% on the fixed length-400 set.
