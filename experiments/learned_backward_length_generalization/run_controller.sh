#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-pointer-next-length20-fitness50-heldout400-h320-2l-p64-10k-seed7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/learned_backward_length_generalization}"
GPU_DEVICES="${GPU_DEVICES:-cuda:0,cuda:1,cuda:2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

exec "${PYTHON_BIN}" -m list_sorting_transformer.shortcut_learning.shortcut_credit_experiment \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_ROOT}" \
  --generations 10000 \
  --population-size 64 \
  --horizon 320 \
  --max-horizon 320 \
  --horizon-promotion-mode fixed \
  --batch-size 64 \
  --fitness-examples 256 \
  --acceptance-fitness-examples 256 \
  --fitness-batch-size 64 \
  --correct-eval-examples 128 \
  --heldout-examples 128 \
  --report-interval 10 \
  --task-variant pointer_next_length \
  --min-length 2 \
  --max-length 20 \
  --fitness-length 50 \
  --heldout-length 400 \
  --forward-learning-rate 3e-4 \
  --forward-training-precision bf16 \
  --sigma 0.21 \
  --outer-learning-rate 0.007 \
  --outer-update-rule elite_centroid \
  --elite-interpolation 0.5 \
  --elite-backtracking \
  --adaptive-elite-counts 1,2,4,8 \
  --elite-rejection-sigma-decay 0.8 \
  --elite-min-sigma 0.00328125 \
  --elite-acceptance-patience 1 \
  --elite-acceptance-sigma-growth 2 \
  --elite-acceptance-trajectories 2 \
  --candidate-ranking-trajectories 1 \
  --d-model 128 \
  --backward-d-model 128 \
  --backward-rule-type attention_router \
  --routing-credit-mode suppress_renorm \
  --shared-routing-map \
  --fitness-objective mean_clean_ce \
  --forward-layers 2 \
  --backward-layers 2 \
  --heads 4 \
  --seed 7 \
  --checkpoint-interval 25 \
  --device cuda:0 \
  --candidate-devices "${GPU_DEVICES}" \
  --vectorized-population \
  --vectorized-chunk-size 16 \
  --wandb \
  --wandb-project list-sorting-learned-backward \
  --wandb-entity wobrob101 \
  "$@"
