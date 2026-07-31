#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
GPU_POOL="${GPU_POOL:-all}"
PYTHON_BIN="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/sparse_attention_adam}"
STEPS="${STEPS:-5000}"
RUN_NAME="${RUN_NAME:-pointer-compare-softmax-nape-fixed-seed4}"

if [[ "${1:-}" != "--worker" ]]; then
  exec "${WITH_GPU}" "${GPU_POOL}" -- "$0" --worker
fi

cd "${ROOT}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

exec env PYTHONPATH="${ROOT}/src" \
  "${PYTHON_BIN}" -u \
  -m list_sorting_transformer.length_generalisation.sparse_attention_adam \
  --task pointer_compare \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_ROOT}" \
  --steps "${STEPS}" \
  --batch-size 128 \
  --train-min-length 2 \
  --train-max-length 20 \
  --eval-lengths 2,5,10,20,40,100,400 \
  --final-eval-lengths 1000,2000,5000 \
  --eval-examples 512 \
  --long-eval-examples 64 \
  --long-eval-min-length 1000 \
  --final-eval-examples 64 \
  --eval-batch-size 128 \
  --layers 2 \
  --d-model 128 \
  --heads 8 \
  --alibi-heads 4 \
  --ffn-multiplier 4 \
  --attention-normalizer softmax \
  --scaling-mode none \
  --architecture standard \
  --input-position-mode nape_only \
  --value-input-mode embedding \
  --learning-rate 4e-4 \
  --weight-decay 0 \
  --warmup-steps 20000 \
  --minimum-lr-ratio 0 \
  --gradient-clip 1 \
  --precision bfloat16-true \
  --optimizer-name adamw \
  --log-interval 100 \
  --eval-interval 1000 \
  --checkpoint-interval 1000 \
  --seed 4 \
  --device cuda \
  --wandb \
  --wandb-project list-sorting-pointer-compare-alibi-nope \
  --wandb-entity wobrob101
