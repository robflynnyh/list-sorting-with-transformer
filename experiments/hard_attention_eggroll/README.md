# Hard-attention forward EGGROLL

This standalone experiment asks whether EGGROLL can directly learn a sparse
Transformer on the clean pointer-next task. It does not use the learned
backward-rule or shortcut experiments.

## Model

- Two-layer, four-head decoder Transformer.
- Every head selects exactly one causal source token with hard top-1 attention.
- Split 64-dimensional content and 64-dimensional position inputs.
- Absolute position is the concatenation of fixed Fourier residue codes for
  moduli `2,3,5,7,11,13,17,19`.
- Every example receives an independent absolute offset sampled uniformly from
  `[-1_000_000, 1_000_000]`.
- No RoPE or other stream-position encoding is applied.

The positional configuration matches the modular random-offset setup used by
the compiled pointer/RASP experiments.

## Optimizer

Each generation samples 32 rank-one directions and evaluates both signs, for a
population of 64 candidates. All candidates see the same examples and position
offsets. Candidate matrix weights are not materialized: the batched forward
pass applies each rank-one perturbation in factorized form.

The reward is negative cross-entropy, standardized across the population. Its
EGGROLL estimate updates the persistent center model with plain SGD and no
weight decay, matching the reference optimizer default.

`--update-rule elite_centroid` provides the earlier elite alternative. It
keeps only the fitter sign from each antithetic pair, selects the eight best
unique directions by default, and moves the center by `learning_rate` times
the mean candidate displacement. Thus `--learning-rate 0.03` means a 3% move
toward the selected elite centroid.

## Run

```bash
experiments/hard_attention_eggroll/run_clean_pointer.sh
```

The launcher always acquires one GPU through `with-gpu`. Set `GPU_POOL`,
`RUN_NAME`, or `GENERATIONS` in the environment to override its defaults.
Long-length center evaluations are processed in batches of 128 examples to
bound attention memory without changing the fixed evaluation set.

Useful W&B metrics:

- `eval/length_N/accuracy`: center-model accuracy at each fixed test length.
- `eval/in_domain_accuracy_mean`: mean over lengths at or below 20.
- `eval/out_of_domain_accuracy_mean`: mean over lengths above 20.
- `population/loss_std`: variation in candidate fitness.
- `routing/antithetic_disagreement_fraction`: fraction of selected attention
  edges that differ between the positive and negative member of a pair.
- `optimization/parameter_update_rms`: RMS size of the actual persistent
  center-model update.
- `optimization/update_to_parameter_rms_ratio`: update RMS divided by current
  parameter RMS.
- `optimization/elite_positive_fraction`: fraction of selected unique elite
  directions using the positive antithetic sign.
- `train/prediction_mode_fraction`: collapse diagnostic for center predictions.
