#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIZES="${SIZES:-256 24 8 2 1 4 16 64}"
METHODS="${METHODS:-maml eggroll}"

cd "${ROOT}"
for method in ${METHODS}; do
  for size in ${SIZES}; do
    if [[ "${method}" == "maml" ]]; then
      expected_steps="${MAML_STEPS:-1000}"
      if [[ "${size}" == "256" ]]; then
        expected_steps=2000
      fi
      checkpoint="artifacts/shortcut_clean_set_scaling/maml/maml-clean${size}-per-mode-seed${SEED:-7}/checkpoint_$(printf '%06d' "${expected_steps}").pt"
    else
      checkpoint="artifacts/shortcut_clean_set_scaling/eggroll/eggroll-clean${size}-per-mode-seed${SEED:-7}/checkpoint_000060.pt"
    fi
    if [[ -f "${checkpoint}" ]]; then
      echo "Skipping completed ${method} condition at clean/mode=${size}" >&2
      continue
    fi
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
