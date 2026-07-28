#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-pointer-next-maml-length20-meta50-heldout400-seed7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/maml_length_generalization}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

exec "${PYTHON_BIN}" -m list_sorting_transformer.maml_length_generalization \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_ROOT}" \
  --steps 10000 \
  --batch-size 64 \
  --min-length 2 \
  --max-length 20 \
  --meta-length 50 \
  --meta-examples 256 \
  --meta-batch-size 64 \
  --heldout-length 400 \
  --eval-examples 128 \
  --eval-batch-size 32 \
  --inner-learning-rate 3e-4 \
  --meta-learning-rate 3e-4 \
  --ordinary-learning-rate 3e-4 \
  --gradient-clip 1 \
  --d-model 128 \
  --layers 2 \
  --heads 4 \
  --log-interval 10 \
  --eval-interval 100 \
  --checkpoint-interval 500 \
  --seed 7 \
  --device cuda \
  --wandb \
  --wandb-project list-sorting-maml \
  --wandb-entity wobrob101 \
  "$@"
