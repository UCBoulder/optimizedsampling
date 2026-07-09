#!/bin/bash
# Cost-sensitivity sweep (Fig. 5): alpha x group assignment x method x budget x seed,
# at a fixed initial-set size. Env vars, e.g.:
#   DATASET=togo_ph_h20 CFG=../configs/togo/RIDGE.yaml \
#   GROUP_TYPES="regions image_clusters_3" \
#   GROUP_PATHS="../../0_data/groups/togo/region_assignments_dict.pkl ../../0_data/groups/togo/image_3_cluster_assignments.pkl" \
#   UNIT_ASSIGNMENT_PATH=... REGION_ASSIGNMENT_PATH=... POINTS_PER_UNIT=25 \
#   INIT_NAME='cluster_sampling/2_strata_desired_25ppc_500_size' \
#   ID_PATH_TEMPLATE='.../sample_region_canton_25ppc_500_size_seed_{seed}.pkl' \
#   ALPHAS="0 1 5 10 15 20 25 30" \
#   ./cost_sensitivity.sh

set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET:?}" "${CFG:?}" "${GROUP_TYPES:?}" "${GROUP_PATHS:?}" "${UNIT_ASSIGNMENT_PATH:?}" \
  "${REGION_ASSIGNMENT_PATH:?}" "${POINTS_PER_UNIT:?}" "${INIT_NAME:?}" "${ID_PATH_TEMPLATE:?}"
SCRIPT="${SCRIPT:-train_ridge.py}"
SEEDS="${SEEDS:-1 42 123 456 789}"
METHODS="${METHODS:-random_unit poprisk}"
BUDGETS="${BUDGETS:-100}"
UTIL_LAMBDAS="${UTIL_LAMBDAS:-0.5}"
ALPHAS="${ALPHAS:-0 1 5 10 15 20 25 30}"

read -ra TYPES <<< "$GROUP_TYPES"
read -ra PATHS <<< "$GROUP_PATHS"

for ALPHA in $ALPHAS; do
    IN="$POINTS_PER_UNIT"; OUT=$((POINTS_PER_UNIT + ALPHA))
    for i in "${!TYPES[@]}"; do
        GROUP_TYPE="${TYPES[$i]}"; GROUP_PATH="${PATHS[$i]}"
        for METHOD in $METHODS; do
            UTILS="0.5"; [[ "$METHOD" == poprisk* ]] && UTILS="$UTIL_LAMBDAS"
            for BUDGET in $BUDGETS; do
                for SEED in $SEEDS; do
                    ID_PATH="${ID_PATH_TEMPLATE//\{seed\}/$SEED}"
                    for UTIL in $UTILS; do
                        ARGS=(--cfg "$CFG" \
                            --exp-name "${DATASET}_${INIT_NAME//\//_}_method_${METHOD}_${GROUP_TYPE}_alpha_${ALPHA}_budget_${BUDGET}_seed_${SEED}" \
                            --sampling_fn "$METHOD" --budget "$BUDGET" --seed "$SEED" --util_lambda "$UTIL" \
                            --initial_set_str "$INIT_NAME" --id_path "$ID_PATH" \
                            --cost_func region_aware_unit_cost --cost_name "cluster_based_c1_${IN}_c2_${OUT}" \
                            --unit_assignment_path "$UNIT_ASSIGNMENT_PATH" --unit_type cluster --points_per_unit "$POINTS_PER_UNIT" \
                            --region_assignment_path "$REGION_ASSIGNMENT_PATH" --in_region_unit_cost "$IN" --out_of_region_unit_cost "$OUT" --alpha "$ALPHA")
                        [[ "$METHOD" == poprisk* ]] && ARGS+=(--group_assignment_path "$GROUP_PATH" --group_type "$GROUP_TYPE")
                        python "$SCRIPT" "${ARGS[@]}"
                    done
                done
            done
        done
    done
done
