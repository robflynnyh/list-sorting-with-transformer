# Shortcut Clean-Set Scaling

This experiment asks how many unique clean pointer problems are needed to
learn shortcut-resistant credit assignment. It compares the strongest
suppression-only router configurations for persistent MAML and EGGROLL.

## Controlled Variable

`CLEAN_EXAMPLES_PER_MODE` is the number of unique examples in each of the
fixed masked-hint and incorrect-hint sets used for optimization. A value of
one therefore means one masked and one incorrect example. The sweep uses:

```text
1, 2, 4, 8, 16, 24, 64, 256 examples per mode
```

The 256-per-mode condition reproduces the clean-data budget used by the
original successful runs. All conditions retain:

- unlimited resampled correct-hint forward-training data;
- the same model, router, optimizer, horizon, and random seed;
- suppression-only `suppress_renorm` credit routing, with no gradient
  reversal;
- 2,048 fresh reporting-only examples per mode;
- identical reporting examples between clean-set sizes.

The small clean set remains fixed throughout each run. Reporting examples
never affect MAML updates, EGGROLL candidate ranking, proposal acceptance,
sigma adaptation, or stopping.

## Screening Protocol

The screening sweep uses one seed (`7`) to identify the transition region.
The 256-per-mode MAML reference runs for the original 2,000 persistent
updates. It converged by step 200, so subsequent MAML conditions use 1,000
updates with the same 24-step lookahead; ambiguous boundary conditions can be
extended afterward. EGGROLL runs for 60 generations at the original
160-update inner horizon,
using population 64, adaptive elite-centroid updates, and strict matched
two-trajectory acceptance.

The primary endpoint is the minimum of fresh masked-hint and incorrect-hint
accuracy. Fixed-set accuracy and the fixed-to-held-out gap diagnose
overfitting. Endpoint accuracy is primary; selecting the best held-out
checkpoint would leak reporting data into model selection.

Once the screening sweep finishes, replicate only the smallest successful
size and the adjacent failing size across additional seeds. Do not interpret
the one-seed screen as a confirmed sample-complexity threshold.

For the adjacent failed/successful EGGROLL conditions (two and four clean
examples per mode in the seed-7 screen), the longer-horizon
audit first selects among saved checkpoints using only the fixed clean
fitness set at horizon 160. It then evaluates the selected rule at horizon
320 on fresh forward initializations and fresh held-out examples. Selection
never uses held-out accuracy. The audit also runs matched ordinary shortcut
training and direct masked training controls:

```bash
with-gpu 2 --num 1 -- \
  bash experiments/shortcut_clean_set_scaling/run_horizon_320_audits.sh
```

## Run

All GPU work must go through `with-gpu`:

```bash
bash experiments/shortcut_clean_set_scaling/launch_sweep.sh
```

Override `SIZES`, `METHODS`, or `SEED` to run a subset. Runs are logged to the
[`list-sorting-shortcut-clean-set-scaling`](https://wandb.ai/wobrob101/list-sorting-shortcut-clean-set-scaling)
W&B project and written under `artifacts/shortcut_clean_set_scaling/`.

Summarize completed runs with:

```bash
python experiments/shortcut_clean_set_scaling/summarize.py
```
