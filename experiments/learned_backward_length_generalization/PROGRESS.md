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

## Active high-throughput run

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
40.3 seconds per generation, or 4.7 days for 10,000 generations.
