#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
RUNNER="${ROOT}/experiments/learned_backward_shortcuts/run_deterministic_controller.sh"
RUN_NAME="${RUN_NAME:-attention-router-signed-function-diverse-balanced-sigma-h160-p64-seed7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/learned_backward_shortcuts}"

exec "${WITH_GPU}" any --num 3 -- env \
  RUN_NAME="${RUN_NAME}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  ROUTING_CREDIT_MODE=signed \
  GPU_DEVICES=cuda:0,cuda:1,cuda:2 \
  bash "${RUNNER}" \
    --direction-sampler function_diverse \
    --direction-candidate-multiplier 4 \
    --direction-probe-examples 8 \
    --direction-signature-size 1024 \
    "$@"
