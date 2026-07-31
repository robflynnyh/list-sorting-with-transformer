#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CLEAN_EXAMPLES_PER_MODE" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLEAN_EXAMPLES_PER_MODE="$1"
SEED="${SEED:-7}"
PYTHON_BIN="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
RUN_NAME="eggroll-clean${CLEAN_EXAMPLES_PER_MODE}-per-mode-seed${SEED}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

exec "${PYTHON_BIN}" -u -m \
  list_sorting_transformer.shortcut_learning.shortcut_credit_experiment \
  --run-name "${RUN_NAME}" \
  --output-dir artifacts/shortcut_clean_set_scaling/eggroll \
  --generations 60 \
  --population-size 64 \
  --horizon 160 \
  --max-horizon 160 \
  --horizon-promotion-mode fixed \
  --batch-size 64 \
  --fitness-examples "$((2 * CLEAN_EXAMPLES_PER_MODE))" \
  --fitness-batch-size 64 \
  --heldout-fitness-examples 4096 \
  --correct-eval-examples 512 \
  --report-interval 5 \
  --control-report-interval 1 \
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
  --routing-credit-mode suppress_renorm \
  --shared-routing-map \
  --fitness-objective worst_mode_ce \
  --forward-layers 3 \
  --backward-layers 2 \
  --heads 4 \
  --seed "${SEED}" \
  --checkpoint-interval 10 \
  --device cuda:0 \
  --candidate-devices cuda:0 \
  --vectorized-population \
  --vectorized-chunk-size 22 \
  --wandb \
  --wandb-project list-sorting-shortcut-clean-set-scaling \
  --wandb-entity wobrob101
