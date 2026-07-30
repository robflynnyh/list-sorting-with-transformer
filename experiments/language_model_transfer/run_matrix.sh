#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${ROOT}/artifacts/language_model_transfer}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/experiments/language_model_transfer/results}"
GPU_POOL="${GPU_POOL:-0-3}"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
STEPS="${STEPS:-5000}"
SEEDS="${SEEDS:-7 17 29}"

mkdir -p "${ARTIFACT_ROOT}" "${RESULT_ROOT}"

pids=()
for seed in ${SEEDS}; do
  for initialization in random compiled_middle compiled_middle_frozen; do
    run_name="${initialization}_seed${seed}"
    output_directory="${ARTIFACT_ROOT}/${run_name}"
    if [[ -s "${output_directory}/metrics.json" ]] \
      && [[ -s "${output_directory}/checkpoint.pt" ]]; then
      echo "Skipping completed ${run_name}"
      continue
    fi
    mkdir -p "${output_directory}"
    "${WITH_GPU}" "${GPU_POOL}" -- \
      env PYTHONPATH="${ROOT}/src" \
      python -m list_sorting_transformer.transfer.language_model_transfer run \
        --initialization "${initialization}" \
        --seed "${seed}" \
        --steps "${STEPS}" \
        --output-directory "${output_directory}" \
        >"${output_directory}/run.log" 2>&1 &
    pids+=("$!")
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one language-model transfer run failed" >&2
  exit "${status}"
fi

PYTHONPATH="${ROOT}/src" \
python -m list_sorting_transformer.transfer.language_model_transfer summarize \
  --input-root "${ARTIFACT_ROOT}" \
  --output-directory "${RESULT_ROOT}"
