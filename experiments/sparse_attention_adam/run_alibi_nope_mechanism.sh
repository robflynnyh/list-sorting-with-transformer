#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
GPU_POOL="${GPU_POOL:-all}"
PYTHON_BIN="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/artifacts/sparse_attention_adam/ablate-i-width128-heads8-softmax-nape-no-scaling-seed4/checkpoint_step_5000.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-${ROOT}/experiments/sparse_attention_adam/results/alibi_nope_mechanism.json}"
OUTPUT_PLOT="${OUTPUT_PLOT:-${ROOT}/experiments/sparse_attention_adam/results/alibi_nope_head_ablation.svg}"
EXAMPLES="${EXAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"

if [[ "${1:-}" != "--worker" ]]; then
  exec "${WITH_GPU}" "${GPU_POOL}" -- "$0" --worker
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  printf 'Missing checkpoint: %s\n' "${CHECKPOINT}" >&2
  printf '%s\n' \
    'Generate ablation I with: ABLATION_PHASE=fourth bash experiments/sparse_attention_adam/run_key_difference_ablations.sh' >&2
  exit 1
fi

cd "${ROOT}"
exec env PYTHONPATH="${ROOT}/src" \
  "${PYTHON_BIN}" -u \
  experiments/sparse_attention_adam/analyze_alibi_nope_mechanism.py \
  --checkpoint "${CHECKPOINT}" \
  --output-json "${OUTPUT_JSON}" \
  --output-plot "${OUTPUT_PLOT}" \
  --lengths 20 100 400 \
  --examples "${EXAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda
