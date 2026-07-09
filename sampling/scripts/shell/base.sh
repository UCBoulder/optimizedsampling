#!/bin/bash
# Method x budget x seed sweep with no initial set. Env vars, e.g.:
#   DATASET=togo CFG=../configs/togo/RIDGE.yaml METHODS="poprisk" BUDGETS="100 500" ./base.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET:?}" "${CFG:?}"
SCRIPT="${SCRIPT:-train_ridge.py}"
SEEDS="${SEEDS:-1 42 123 456 789 1234 5678 9101 1213 1415}"
METHODS="${METHODS:-random_unit}"
BUDGETS="${BUDGETS:-100}"
UTIL_LAMBDAS="${UTIL_LAMBDAS:-0.5}"
GROUP_PATH="${GROUP_PATH:-}"
GROUP_TYPE="${GROUP_TYPE:-}"
UNIT_ASSIGNMENT_PATH="${UNIT_ASSIGNMENT_PATH:-}"
REGION_ASSIGNMENT_PATH="${REGION_ASSIGNMENT_PATH:-}"
POINTS_PER_UNIT="${POINTS_PER_UNIT:-}"
ALPHA="${ALPHA:-10}"

for METHOD in $METHODS; do
    UTILS="0.5"; [[ "$METHOD" == poprisk* ]] && UTILS="$UTIL_LAMBDAS"
    for BUDGET in $BUDGETS; do
        for SEED in $SEEDS; do
            for UTIL in $UTILS; do
                ARGS=(--cfg "$CFG" --exp-name "${DATASET}_method_${METHOD}_budget_${BUDGET}_seed_${SEED}" \
                      --sampling_fn "$METHOD" --budget "$BUDGET" --seed "$SEED" --util_lambda "$UTIL")
                if [[ -n "$UNIT_ASSIGNMENT_PATH" ]]; then
                    IN="$POINTS_PER_UNIT"; OUT=$((POINTS_PER_UNIT + ALPHA))
                    ARGS+=(--cost_func region_aware_unit_cost --cost_name "cluster_based_c1_${IN}_c2_${OUT}" \
                           --unit_assignment_path "$UNIT_ASSIGNMENT_PATH" --unit_type cluster --points_per_unit "$POINTS_PER_UNIT" \
                           --region_assignment_path "$REGION_ASSIGNMENT_PATH" --in_region_unit_cost "$IN" --out_of_region_unit_cost "$OUT" --alpha "$ALPHA")
                else
                    ARGS+=(--cost_func uniform --cost_name uniform)
                fi
                [[ "$METHOD" == poprisk* && -n "$GROUP_PATH" ]] && ARGS+=(--group_assignment_path "$GROUP_PATH" --group_type "$GROUP_TYPE")
                python "$SCRIPT" "${ARGS[@]}"
            done
        done
    done
done
