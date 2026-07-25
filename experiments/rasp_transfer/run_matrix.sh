#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

tasks=(
    following_value
    three_way_relation
    associative_recall
    dyck_2_completion
)
initializations=(random compiled_prefix compiled_full)
seeds=(7 17 29)
output_root="artifacts/rasp_transfer"
pids=()

cleanup() {
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

for task in "${tasks[@]}"; do
    for initialization in "${initializations[@]}"; do
        for seed in "${seeds[@]}"; do
            steps=2000
            if [[ "$task" == "associative_recall" ]]; then
                steps=10000
            fi
            output_directory="${output_root}/${task}/${initialization}/seed_${seed}"
            metrics="${output_directory}/metrics.json"
            log="${output_directory}/run.log"
            if [[ -f "$metrics" ]]; then
                echo "Skipping completed ${task}/${initialization}/seed_${seed}"
                continue
            fi
            mkdir -p "$output_directory"
            with-gpu any --idle-seconds 1 -- \
                env PYTHONPATH=src python -u -m \
                list_sorting_transformer.rasp_transfer train \
                --task "$task" \
                --initialization "$initialization" \
                --seed "$seed" \
                --steps "$steps" \
                --output-directory "$output_directory" \
                >"$log" 2>&1 &
            pids+=("$!")
        done
    done
done

failed=0
for pid in "${pids[@]:-}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
exit "$failed"
