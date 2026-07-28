#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-attention-router-performance-curriculum-h160-p64-seed7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/learned_backward_shortcuts}"
GPU_DEVICES="${GPU_DEVICES:-cuda:0,cuda:1,cuda:2}"
ROUTING_CREDIT_MODE="${ROUTING_CREDIT_MODE:-suppress_renorm}"
PYTHON_BIN="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

exec "${PYTHON_BIN}" -u -m list_sorting_transformer.shortcut_credit_experiment \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_ROOT}" \
  --generations 200 \
  --population-size 64 \
  --horizon 160 \
  --max-horizon 1280 \
  --horizon-multiplier 2 \
  --horizon-promotion-mode performance_plateau \
  --horizon-score-window 8 \
  --horizon-min-generations 20 \
  --horizon-max-generations 30 \
  --horizon-failed-extension-limit 2 \
  --plateau-patience 5 \
  --plateau-min-delta 0.01 \
  --batch-size 64 \
  --fitness-examples 512 \
  --fitness-batch-size 64 \
  --correct-eval-examples 128 \
  --min-length 8 \
  --max-length 32 \
  --leak-placement random_list \
  --forward-learning-rate 3e-4 \
  --sigma 0.21 \
  --outer-learning-rate 0.007 \
  --outer-update-rule elite_centroid \
  --elite-interpolation 0.5 \
  --elite-backtracking \
  --adaptive-elite-counts 1,2,4,8 \
  --elite-rejection-sigma-decay 0.5 \
  --elite-min-sigma 0.00328125 \
  --elite-acceptance-patience 3 \
  --elite-acceptance-sigma-growth 2 \
  --elite-acceptance-trajectories 2 \
  --candidate-ranking-trajectories 1 \
  --d-model 128 \
  --backward-d-model 128 \
  --backward-rule-type attention_router \
  --routing-credit-mode "${ROUTING_CREDIT_MODE}" \
  --shared-routing-map \
  --fitness-objective worst_mode_ce \
  --forward-layers 3 \
  --backward-layers 2 \
  --heads 4 \
  --seed 7 \
  --checkpoint-interval 1 \
  --device cuda:0 \
  --candidate-devices "${GPU_DEVICES}" \
  --vectorized-population \
  --vectorized-chunk-size 16 \
  --wandb \
  --wandb-project list-sorting-learned-backward \
  --wandb-entity wobrob101 \
  "$@"
