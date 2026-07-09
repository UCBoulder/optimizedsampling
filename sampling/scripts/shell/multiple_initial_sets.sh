#!/bin/bash
# "Multiple initial sets" sweep (Fig. 4): initial-set size x cost function x method x budget x seed.
# Env vars, e.g.:
#   DATASET=togo CFG=../configs/togo/RIDGE.yaml GROUP_PATH=... GROUP_TYPE=image_clusters_3 \
#   UNIT_ASSIGNMENT_PATH=... REGION_ASSIGNMENT_PATH=... POINTS_PER_UNIT=25 \
#   SIZES="0 500 1000 1500 2000 2500" \
#   INIT_NAME_TEMPLATE='multiple/cluster_sampling/2_strata_desired_25ppc_{size}_size' \
#   ID_PATH_TEMPLATE='.../sample_region_canton_25ppc_{size}_size_seed_1.pkl' \
#   ./multiple_initial_sets.sh

set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET:?}" "${CFG:?}" "${GROUP_PATH:?}" "${GROUP_TYPE:?}" "${UNIT_ASSIGNMENT_PATH:?}" \
  "${REGION_ASSIGNMENT_PATH:?}" "${POINTS_PER_UNIT:?}" "${INIT_NAME_TEMPLATE:?}" "${ID_PATH_TEMPLATE:?}"
SCRIPT="${SCRIPT:-train_ridge.py}"
SEEDS="${SEEDS:-1 42 123 456 789}"
METHODS="${METHODS:-poprisk}"
BUDGETS="${BUDGETS:-100 200 300 400 500 600 700 800 900 1000}"
SIZES="${SIZES:-0}"
COST_FNS="${COST_FNS:-cluster_based uniform}"
ALPHA="${ALPHA:-10}"

for SIZE in $SIZES; do
    if [[ "$SIZE" == 0 ]]; then
        INIT_NAME="multiple/cluster_sampling/empty_initial_set"; ID_PATH=""
    else
        INIT_NAME="${INIT_NAME_TEMPLATE//\{size\}/$SIZE}"; ID_PATH="${ID_PATH_TEMPLATE//\{size\}/$SIZE}"
    fi
    for COST_FN in $COST_FNS; do
        for METHOD in $METHODS; do
            for BUDGET in $BUDGETS; do
                for SEED in $SEEDS; do
                    SEEDED_ID_PATH="${ID_PATH//\{seed\}/$SEED}"
                    ARGS=(--cfg "$CFG" \
                        --exp-name "${DATASET}_${INIT_NAME//\//_}_cost_${COST_FN}_method_${METHOD}_budget_${BUDGET}_seed_${SEED}" \
                        --sampling_fn "$METHOD" --budget "$BUDGET" --seed "$SEED" --initial_set_str "$INIT_NAME")
                    [[ -n "$SEEDED_ID_PATH" ]] && ARGS+=(--id_path "$SEEDED_ID_PATH")
                    if [[ "$COST_FN" == cluster_based ]]; then
                        IN="$POINTS_PER_UNIT"; OUT=$((POINTS_PER_UNIT + ALPHA))
                        ARGS+=(--cost_func region_aware_unit_cost --cost_name "cluster_based_c1_${IN}_c2_${OUT}" \
                            --unit_assignment_path "$UNIT_ASSIGNMENT_PATH" --unit_type cluster --points_per_unit "$POINTS_PER_UNIT" \
                            --region_assignment_path "$REGION_ASSIGNMENT_PATH" --in_region_unit_cost "$IN" --out_of_region_unit_cost "$OUT" --alpha "$ALPHA")
                    else
                        ARGS+=(--cost_func uniform --cost_name uniform)
                    fi
                    [[ "$METHOD" == poprisk* ]] && ARGS+=(--group_assignment_path "$GROUP_PATH" --group_type "$GROUP_TYPE")
                    python "$SCRIPT" "${ARGS[@]}"
                done
            done
        done
    done
done
