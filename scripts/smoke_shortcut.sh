#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
GPU_POOL="${GPU_POOL:-all}"
PYTHON_BIN="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/.scratch/research-artifact-smoke/shortcut}"

cd "${ROOT}"

exec "${WITH_GPU}" "${GPU_POOL}" -- \
  env PYTHONPATH="${ROOT}/src" \
  "${PYTHON_BIN}" -u -m list_sorting_transformer.shortcut_credit_experiment \
  --run-name research-artifact-shortcut-smoke \
  --output-dir "${OUTPUT_DIR}" \
  --generations 2 \
  --population-size 4 \
  --horizon 2 \
  --max-horizon 2 \
  --batch-size 4 \
  --fitness-examples 16 \
  --fitness-batch-size 8 \
  --correct-eval-examples 8 \
  --heldout-examples 8 \
  --d-model 32 \
  --backward-d-model 32 \
  --forward-layers 2 \
  --backward-layers 1 \
  --heads 4 \
  --checkpoint-interval 1 \
  --device cuda
