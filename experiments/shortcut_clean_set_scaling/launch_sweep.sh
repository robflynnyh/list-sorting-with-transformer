#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"

cd "${ROOT}"
exec "${WITH_GPU}" all --num 1 -- \
  bash experiments/shortcut_clean_set_scaling/run_sweep.sh
