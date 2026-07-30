#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
WITH_GPU="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
GPU_POOL="${GPU_POOL:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/sparse_attention_adam}"
STEPS="${STEPS:-5000}"
ABLATION_PHASE="${ABLATION_PHASE:-all}"

if [[ "${1:-}" != "--worker" ]]; then
  exec "${WITH_GPU}" "${GPU_POOL}" -- "$0" --worker
fi

cd "${ROOT}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

COMMON_ARGS=(
  --output-dir "${OUTPUT_ROOT}"
  --steps "${STEPS}"
  --train-min-length 2
  --train-max-length 20
  --eval-lengths 2,5,10,20,40,100,400,1000,2000
  --final-eval-lengths 5000
  --eval-examples 512
  --long-eval-examples 64
  --long-eval-min-length 1000
  --final-eval-examples 64
  --layers 2
  --ffn-multiplier 4
  --learning-rate 4e-4
  --weight-decay 0
  --gradient-clip 1
  --log-interval 100
  --eval-interval "${STEPS}"
  --checkpoint-interval "${STEPS}"
  --seed 4
  --device cuda
  --wandb
  --wandb-project list-sorting-sparse-attention-ablation
  --wandb-entity wobrob101
)

run_variant() {
  local run_name="$1"
  shift
  env PYTHONPATH="${ROOT}/src" \
    "${PYTHON_BIN}" -u \
    -m list_sorting_transformer.sparse_attention_adam \
    --run-name "${run_name}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

# A-D form the cumulative ablation from the previous setup to the successful
# paper-matched setup. E-G separate the remaining architecture/capacity change.
if [[ "${ABLATION_PHASE}" == "all" || "${ABLATION_PHASE}" == "first" ]]; then
  run_variant ablate-a-old-recipe-old-model-seed4 \
  --batch-size 256 \
  --d-model 128 \
  --heads 4 \
  --alibi-heads 2 \
  --architecture standard \
  --input-position-mode modular \
  --value-input-mode embedding_plus_scalar \
  --warmup-steps 1000 \
  --minimum-lr-ratio 0.1 \
    --precision bfloat16 \
    --optimizer-name adam

  run_variant ablate-b-paper-recipe-old-model-seed4 \
  --batch-size 128 \
  --d-model 128 \
  --heads 4 \
  --alibi-heads 2 \
  --architecture standard \
  --input-position-mode modular \
  --value-input-mode embedding_plus_scalar \
  --warmup-steps 20000 \
  --minimum-lr-ratio 0 \
    --precision bfloat16-true \
    --optimizer-name adamw

  run_variant ablate-c-paper-recipe-nape-input-seed4 \
  --batch-size 128 \
  --d-model 128 \
  --heads 4 \
  --alibi-heads 2 \
  --architecture standard \
  --input-position-mode nape_only \
  --value-input-mode embedding \
  --warmup-steps 20000 \
  --minimum-lr-ratio 0 \
    --precision bfloat16-true \
    --optimizer-name adamw

  run_variant ablate-d-paper-recipe-nape-large-standard-seed4 \
  --batch-size 128 \
  --d-model 256 \
  --heads 8 \
  --alibi-heads 4 \
  --architecture standard \
  --input-position-mode nape_only \
  --value-input-mode embedding \
  --warmup-steps 20000 \
  --minimum-lr-ratio 0 \
    --precision bfloat16-true \
    --optimizer-name adamw \
    --final-eval-lengths ""
fi

if [[ "${ABLATION_PHASE}" == "all" || "${ABLATION_PHASE}" == "second" ]]; then
  # E: Change only the standard block to Gemma2 style at the small capacity.
  run_variant ablate-e-paper-recipe-nape-small-gemma-seed4 \
    --batch-size 128 \
    --d-model 128 \
    --heads 4 \
    --alibi-heads 2 \
    --architecture paper_gemma2 \
    --input-position-mode nape_only \
    --value-input-mode embedding \
    --warmup-steps 20000 \
    --minimum-lr-ratio 0 \
    --precision bfloat16-true \
    --optimizer-name adamw \
    --final-eval-lengths ""

  # F: Increase width while retaining four heads and the standard block.
  run_variant ablate-f-paper-recipe-nape-width256-heads4-standard-seed4 \
    --batch-size 128 \
    --d-model 256 \
    --heads 4 \
    --alibi-heads 2 \
    --architecture standard \
    --input-position-mode nape_only \
    --value-input-mode embedding \
    --warmup-steps 20000 \
    --minimum-lr-ratio 0 \
    --precision bfloat16-true \
    --optimizer-name adamw \
    --final-eval-lengths ""

  # G: Increase head count while retaining d_model=128 and the standard block.
  run_variant ablate-g-paper-recipe-nape-width128-heads8-standard-seed4 \
    --batch-size 128 \
    --d-model 128 \
    --heads 8 \
    --alibi-heads 4 \
    --architecture standard \
    --input-position-mode nape_only \
    --value-input-mode embedding \
    --warmup-steps 20000 \
    --minimum-lr-ratio 0 \
    --precision bfloat16-true \
    --optimizer-name adamw \
    --final-eval-lengths ""
fi

if [[ "${ABLATION_PHASE}" == "all" || "${ABLATION_PHASE}" == "third" ]]; then
  # H: Change only the successful small eight-head model's normalizer.
  run_variant ablate-h-paper-recipe-nape-width128-heads8-softmax-seed4 \
    --batch-size 128 \
    --d-model 128 \
    --heads 8 \
    --alibi-heads 4 \
    --attention-normalizer softmax \
    --architecture standard \
    --input-position-mode nape_only \
    --value-input-mode embedding \
    --warmup-steps 20000 \
    --minimum-lr-ratio 0 \
    --precision bfloat16-true \
    --optimizer-name adamw \
    --final-eval-lengths ""
fi
