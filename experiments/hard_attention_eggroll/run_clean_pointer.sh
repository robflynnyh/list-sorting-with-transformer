#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
GPU_POOL="${GPU_POOL:-all}"
RUN_NAME="${RUN_NAME:-clean-pointer-top1-eggroll-seed7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/hard_attention_eggroll}"
GENERATIONS="${GENERATIONS:-20000}"

cd "${ROOT}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

exec "${WITH_GPU}" "${GPU_POOL}" -- env PYTHONPATH="${ROOT}/src" \
  "${PYTHON_BIN}" -u \
  -m list_sorting_transformer.hard_attention_eggroll \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_ROOT}" \
  --generations "${GENERATIONS}" \
  --population-size 64 \
  --population-chunk-size 16 \
  --batch-size 256 \
  --train-min-length 2 \
  --train-max-length 20 \
  --eval-lengths 2,5,10,20,40,100,400 \
  --eval-examples 1024 \
  --eval-batch-size 128 \
  --d-model 128 \
  --layers 2 \
  --heads 4 \
  --attention-mode top1 \
  --position-moduli 2,3,5,7,11,13,17,19 \
  --position-offset-min -1000000 \
  --position-offset-max 1000000 \
  --sigma 0.005 \
  --learning-rate 0.3 \
  --weight-decay 0 \
  --fitness-shaping zscore \
  --log-interval 10 \
  --eval-interval 100 \
  --checkpoint-interval 1000 \
  --seed 7 \
  --device cuda \
  --wandb \
  --wandb-project list-sorting-hard-attention-eggroll \
  --wandb-entity wobrob101 \
  "$@"
