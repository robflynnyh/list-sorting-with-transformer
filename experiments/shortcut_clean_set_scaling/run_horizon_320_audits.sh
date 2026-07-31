#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

for size in 2 8; do
  run_dir="artifacts/shortcut_clean_set_scaling/eggroll/eggroll-clean${size}-per-mode-seed7"
  output="experiments/shortcut_clean_set_scaling/results/eggroll-clean${size}-h320.json"
  "${PYTHON_BIN}" -u \
    experiments/shortcut_clean_set_scaling/evaluate_selected_horizon.py \
    --run-dir "${run_dir}" \
    --selection-horizon 160 \
    --evaluation-horizon 320 \
    --selection-replicates 2 \
    --evaluation-replicates 5 \
    --evaluation-examples 4096 \
    --device cuda:0 \
    --output "${output}"
done
