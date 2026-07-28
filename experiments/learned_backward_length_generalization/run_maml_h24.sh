#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-pointer-next-router-maml-h24-meta50-heldout400-seed7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/maml_length_generalization}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

exec "${WITH_GPU}" any -- env PYTHONPATH="${PYTHONPATH}" \
  "${PYTHON_BIN}" -u -m list_sorting_transformer.maml_length_generalization \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_ROOT}" \
  --method router_maml \
  --steps 10000 \
  --lookahead-steps 24 \
  --batch-size 64 \
  --min-length 2 \
  --max-length 20 \
  --meta-lengths 50 \
  --meta-examples 256 \
  --meta-batch-size 64 \
  --heldout-length 400 \
  --eval-examples 128 \
  --eval-batch-size 32 \
  --ordinary-learning-rate 3e-4 \
  --router-learning-rate 3e-4 \
  --gradient-clip 1 \
  --router-credit-mode suppress_renorm \
  --d-model 128 \
  --layers 2 \
  --heads 4 \
  --router-d-model 128 \
  --router-heads 4 \
  --log-interval 10 \
  --eval-interval 100 \
  --checkpoint-interval 500 \
  --ordinary-reference-metrics \
    artifacts/maml_length_generalization/pointer-next-ordinary-length20-heldout400-seed7/metrics.jsonl \
  --seed 7 \
  --device cuda \
  --wandb \
  --wandb-project list-sorting-maml \
  --wandb-entity wobrob101 \
  --wandb-group router-maml-h24-length-generalization \
  "$@"
