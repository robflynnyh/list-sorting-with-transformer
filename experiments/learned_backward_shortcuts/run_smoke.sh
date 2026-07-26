#!/usr/bin/env bash
set -euo pipefail

python -m list_sorting_transformer.shortcut_credit_experiment \
  --run-name learned-backward-shortcuts-smoke \
  --generations 2 \
  --population-size 4 \
  --horizon 2 \
  --max-horizon 2 \
  --batch-size 4 \
  --fitness-examples 16 \
  --fitness-batch-size 8 \
  --d-model 32 \
  --backward-d-model 32 \
  --forward-layers 2 \
  --backward-layers 1 \
  --heads 4 \
  --device "${DEVICE:-cpu}" \
  "$@"
