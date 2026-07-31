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
RUN_NAME="maml-clean${CLEAN_EXAMPLES_PER_MODE}-per-mode-seed${SEED}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

exec "${PYTHON_BIN}" -u -m \
  list_sorting_transformer.shortcut_learning.maml_shortcut_experiment \
  --run-name "${RUN_NAME}" \
  --output-dir artifacts/shortcut_clean_set_scaling/maml \
  --method router_maml \
  --steps 2000 \
  --lookahead-steps 24 \
  --batch-size 64 \
  --min-length 8 \
  --max-length 32 \
  --fitness-examples "$((2 * CLEAN_EXAMPLES_PER_MODE))" \
  --fitness-batch-size 32 \
  --heldout-fitness-examples 4096 \
  --heldout-fitness-batch-size 64 \
  --eval-examples 512 \
  --eval-batch-size 64 \
  --ordinary-learning-rate 3e-4 \
  --router-learning-rate 3e-4 \
  --gradient-clip 1 \
  --d-model 128 \
  --layers 3 \
  --heads 4 \
  --router-d-model 128 \
  --router-heads 4 \
  --router-credit-mode suppress_renorm \
  --router-initial-gate 1e-3 \
  --router-minimum-gate 1e-6 \
  --log-interval 10 \
  --eval-interval 100 \
  --checkpoint-interval 500 \
  --seed "${SEED}" \
  --device cuda:0 \
  --wandb \
  --wandb-project list-sorting-shortcut-clean-set-scaling \
  --wandb-entity wobrob101 \
  --wandb-group maml-seed${SEED}

