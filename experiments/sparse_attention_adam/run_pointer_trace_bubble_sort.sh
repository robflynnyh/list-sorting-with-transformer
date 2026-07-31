#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
GPU_POOL="${GPU_POOL:-all}"
PYTHON_BIN="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/artifacts/sparse_attention_adam/pointer-compare-trace-entmax15-nape-fixed-seed4/checkpoint_step_20000.pt}"
OUTPUT="${OUTPUT:-${ROOT}/experiments/sparse_attention_adam/results/pointer_trace_bubble_sort_summary.json}"
LENGTHS="${LENGTHS:-2,3,5,10,20,40,100}"
EXAMPLES="${EXAMPLES:-64}"
LONG_EXAMPLES="${LONG_EXAMPLES:-16}"
LONG_MIN_LENGTH="${LONG_MIN_LENGTH:-100}"
SEED="${SEED:-20260730}"

if [[ "${1:-}" != "--worker" ]]; then
  exec "${WITH_GPU}" "${GPU_POOL}" -- "$0" --worker
fi

cd "${ROOT}"
exec env PYTHONPATH="${ROOT}/src" \
  "${PYTHON_BIN}" -u \
  -m list_sorting_transformer.length_generalisation.pointer_trace_bubble_sort \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  --lengths "${LENGTHS}" \
  --examples "${EXAMPLES}" \
  --long-examples "${LONG_EXAMPLES}" \
  --long-min-length "${LONG_MIN_LENGTH}" \
  --seed "${SEED}" \
  --device cuda
