#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
GPU_POOL="${GPU_POOL:-all}"
RUN_NAME="${RUN_NAME:-pointer-next-asentmax-nape-adam-20k-seed7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/sparse_attention_adam}"
STEPS="${STEPS:-20000}"

cd "${ROOT}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

exec "${WITH_GPU}" "${GPU_POOL}" -- \
  env PYTHONPATH="${ROOT}/src" \
  "${PYTHON_BIN}" -u \
  -m list_sorting_transformer.sparse_attention_adam \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_ROOT}" \
  --steps "${STEPS}" \
  --batch-size 256 \
  --train-min-length 2 \
  --train-max-length 20 \
  --eval-lengths 2,5,10,20,40,100,400,1000,2000 \
  --final-eval-lengths 5000 \
  --eval-examples 512 \
  --long-eval-examples 64 \
  --long-eval-min-length 1000 \
  --final-eval-examples 64 \
  --learning-rate 4e-4 \
  --weight-decay 0 \
  --warmup-steps 1000 \
  --precision bfloat16 \
  --log-interval 50 \
  --eval-interval 500 \
  --checkpoint-interval 1000 \
  --seed 7 \
  --device cuda \
  --wandb \
  --wandb-project list-sorting-sparse-attention-adam \
  --wandb-entity wobrob101 \
  "$@"
