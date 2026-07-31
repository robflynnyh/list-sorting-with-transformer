#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIZES="${SIZES:-256 24 8 2 1 4 16 64}"
METHODS="${METHODS:-maml eggroll}"

cd "${ROOT}"
for size in ${SIZES}; do
  for method in ${METHODS}; do
    echo "Starting ${method} with ${size} clean examples per mode" >&2
    if [[ "${method}" == "maml" && "${size}" == "256" ]]; then
      MAML_STEPS=2000 \
        bash "experiments/shortcut_clean_set_scaling/run_${method}.sh" "${size}"
    else
      bash "experiments/shortcut_clean_set_scaling/run_${method}.sh" "${size}"
    fi
  done
done

"${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}" \
  experiments/shortcut_clean_set_scaling/summarize.py
