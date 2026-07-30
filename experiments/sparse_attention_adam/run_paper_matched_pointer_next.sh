#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
GPU_POOL="${GPU_POOL:-all}"
RUN_NAME="${RUN_NAME:-pointer-next-asentmax-paper-matched-312k-seed4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/sparse_attention_adam}"
STEPS="${STEPS:-312500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"

cd "${ROOT}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

exec "${WITH_GPU}" "${GPU_POOL}" -- \
  env PYTHONPATH="${ROOT}/src" \
  "${PYTHON_BIN}" -u \
  -m list_sorting_transformer.sparse_attention_adam \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_ROOT}" \
  --steps "${STEPS}" \
  --batch-size 128 \
  --train-min-length 2 \
  --train-max-length 20 \
  --eval-lengths 2,5,10,20,40,100,400,1000,2000 \
  --final-eval-lengths 5000 \
  --eval-examples 512 \
  --long-eval-examples 64 \
  --long-eval-min-length 1000 \
  --final-eval-examples 64 \
  --d-model 256 \
  --layers 2 \
  --heads 8 \
  --ffn-multiplier 4 \
  --alibi-heads 4 \
  --architecture paper_gemma2 \
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
  --eval-interval "${EVAL_INTERVAL}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --seed 4 \
  --device cuda \
  --wandb \
  --wandb-project list-sorting-sparse-attention-adam \
  --wandb-entity wobrob101 \
  "$@"
