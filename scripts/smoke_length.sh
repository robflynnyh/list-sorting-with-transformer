#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
GPU_POOL="${GPU_POOL:-all}"
PYTHON_BIN="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/.scratch/research-artifact-smoke/length}"

cd "${ROOT}"

exec "${WITH_GPU}" "${GPU_POOL}" -- \
  env PYTHONPATH="${ROOT}/src" \
  "${PYTHON_BIN}" -u -m list_sorting_transformer.sparse_attention_adam \
  --run-name research-artifact-length-smoke \
  --output-dir "${OUTPUT_DIR}" \
  --steps 2 \
  --batch-size 8 \
  --train-min-length 2 \
  --train-max-length 4 \
  --eval-lengths 2,4 \
  --final-eval-lengths "" \
  --eval-examples 8 \
  --long-eval-examples 8 \
  --final-eval-examples 8 \
  --eval-batch-size 8 \
  --d-model 128 \
  --layers 1 \
  --heads 4 \
  --alibi-heads 2 \
  --attention-normalizer softmax \
  --scaling-mode none \
  --warmup-steps 1 \
  --precision float32 \
  --log-interval 1 \
  --eval-interval 1 \
  --checkpoint-interval 2 \
  --seed 7 \
  --device cuda
