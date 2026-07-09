#!/bin/bash
# Representative-sampling sweep: initial-set size x method x budget x seed, grouped by
# GROUP_TYPE (states / image_clusters_N / nlcd). Env vars, e.g.:
#   DATASET=usavars_population CFG=../configs/usavars/population.yaml \
#   GROUP_PATH=.../state_assignments_dict.pkl GROUP_TYPE=states \
#   UNIT_ASSIGNMENT_PATH=.../county_assignments_dict.pkl REGION_ASSIGNMENT_PATH=.../state_assignments_dict.pkl \
#   POINTS_PER_UNIT=10 SIZES=100 \
#   INIT_NAME_TEMPLATE='cluster_sampling/5_fixedstrata_10ppc_{size}_size' \
#   ID_PATH_TEMPLATE='.../sample_state_combined_county_id_10ppc_{size}_size_seed_{seed}.pkl' \
#   ./rep_sampling.sh

set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET:?}" "${CFG:?}" "${GROUP_PATH:?}" "${GROUP_TYPE:?}" "${UNIT_ASSIGNMENT_PATH:?}" \
  "${REGION_ASSIGNMENT_PATH:?}" "${POINTS_PER_UNIT:?}" "${INIT_NAME_TEMPLATE:?}" "${ID_PATH_TEMPLATE:?}"
SCRIPT="${SCRIPT:-train_ridge.py}"
SEEDS="${SEEDS:-1 42 123 456 789 1234 5678 9101 1213 1415}"
METHODS="${METHODS:-random_unit greedycost poprisk poprisk_avg}"
BUDGETS="${BUDGETS:-50 100 200 300 400 500 1000}"
UTIL_LAMBDAS="${UTIL_LAMBDAS:-0.01 0.1 0.5 0.9 0.99}"
SIZES="${SIZES:-100}"
ALPHA="${ALPHA:-10}"

for SIZE in $SIZES; do
    INIT_NAME="${INIT_NAME_TEMPLATE//\{size\}/$SIZE}"
    for METHOD in $METHODS; do
        UTILS="0.5"; [[ "$METHOD" == poprisk* ]] && UTILS="$UTIL_LAMBDAS"
        for BUDGET in $BUDGETS; do
            for SEED in $SEEDS; do
                ID_PATH="${ID_PATH_TEMPLATE//\{size\}/$SIZE}"; ID_PATH="${ID_PATH//\{seed\}/$SEED}"
                IN="$POINTS_PER_UNIT"; OUT=$((POINTS_PER_UNIT + ALPHA))
                for UTIL in $UTILS; do
                    ARGS=(--cfg "$CFG" \
                        --exp-name "${DATASET}_${INIT_NAME//\//_}_method_${METHOD}_${GROUP_TYPE}_budget_${BUDGET}_seed_${SEED}" \
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
