#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POPULATION_SIZE="${POPULATION_SIZE:-8192}"
UNIQUE_EXAMPLES="${UNIQUE_EXAMPLES:-8}"
POPULATION_CHUNK_SIZE="${POPULATION_CHUNK_SIZE:-4096}"
POPULATION_PRECISION="${POPULATION_PRECISION:-bfloat16}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"

exec "${ROOT}/experiments/hard_attention_eggroll/run_clean_pointer.sh" \
  --population-size "${POPULATION_SIZE}" \
  --population-chunk-size "${POPULATION_CHUNK_SIZE}" \
  --batch-size "${UNIQUE_EXAMPLES}" \
  --population-data-mode grouped \
  --population-precision "${POPULATION_PRECISION}" \
  --attention-mode dense \
  --update-rule paper_standardized \
  --learning-rate "${LEARNING_RATE}" \
  --curriculum \
  --curriculum-accuracy-threshold 0.70 \
  --curriculum-success-checks 3 \
  --curriculum-check-interval 500 \
  --curriculum-examples 1024 \
  --curriculum-initial-top-k 20 \
  "$@"
