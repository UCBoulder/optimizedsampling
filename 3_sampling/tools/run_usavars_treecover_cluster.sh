# === run_setting3.sh ===

#!/bin/bash
source "$(dirname "$0")/run_usavars_treecover.sh"

run_setting3() {
    local seed="$1"
    DATASET="usavars"
    ALPHAS=(10)

    for SIZE in 100 200 300; do
        INIT_NAME_EXP="cluster_sampling_5_fixedstrata_10ppc_${SIZE}_size"
        INIT_NAME="cluster_sampling/5_fixedstrata_10ppc_${SIZE}_size"
        INIT_SET_IDS="/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/treecover/cluster_sampling/fixedstrata_Alabama_01-Colorado_08-Montana_30-New York_36-Ohio_39/sample_state_combined_county_id_10ppc_${SIZE}_size_seed_${seed}.pkl"

        for COST_FN in "cluster_based"; do
            echo "⚙️ Running Setting 3 with COST_FN=${COST_FN}"

            for ALPHA in "${ALPHAS[@]}"; do
                for METHOD in "${METHODS[@]}"; do
                    for BUDGET in "${BUDGETS[@]}"; do
                        for UTIL_LAMBDA in "${UTIL_LAMBDAS[@]}"; do

                            COST_FN_NAME="uniform"
                            COST_ARRAY_PATH=""
                            UNIT_ASSIGNMENT_PATH=""
                            UNIT_COST_PATH=""
                            UNIT_TYPE=""
                            POINTS_PER_UNIT=""

                            if [ "$COST_FN" == "cluster_based" ]; then
                                UNIT_ASSIGNMENT_PATH="/home/libe2152/optimizedsampling/0_data/groups/usavars/treecover/county_assignments_dict.pkl"
                                UNIT_TYPE="cluster"
                                POINTS_PER_UNIT=10
                                IN_REGION_UNIT_COST=$POINTS_PER_UNIT
                                OUT_OF_REGION_UNIT_COST=$((POINTS_PER_UNIT+ALPHA))
                                COST_FN_NAME="region_aware_unit_cost"
                                COST_FN_WITH_SPECIFICS="cluster_based_c1_${IN_REGION_UNIT_COST}_c2_${OUT_OF_REGION_UNIT_COST}"
                                REGION_ASSIGNMENT_PATH="/home/libe2152/optimizedsampling/0_data/groups/usavars/treecover/state_assignments_dict.pkl"
                            fi

                            EXP_NAME="usavars_treecover_${INIT_NAME_EXP}_cost_${COST_FN_WITH_SPECIFICS}_method_${METHOD}_budget_${BUDGET}_seed_${seed}"
                            if [ "$METHOD" == "poprisk" ]; then
                                EXP_NAME="usavars_treecover_${INIT_NAME_EXP}_cost_${COST_FN_WITH_SPECIFICS}_method_${METHOD}_${GROUP_TYPE}_${UTIL_LAMBDA}_budget_${BUDGET}_seed_${seed}"
                            fi
                            
                            CMD="python $SCRIPT \
                                --cfg \"$CFG_FILE\" \
                                --exp-name \"$EXP_NAME\" \
                                --sampling_fn \"$METHOD\" \
                                --util_lambda \"$UTIL_LAMBDA\" \
                                --budget $BUDGET \
                                --initial_set_str \"$INIT_NAME\" \
                                --id_path \"$INIT_SET_IDS\" \
                                --seed $seed \
                                --cost_func \"$COST_FN_NAME\" \
                                --cost_name \"$COST_FN_WITH_SPECIFICS\""

                            if [ "$COST_FN" == "cluster_based" ]; then
                                CMD+=" --unit_assignment_path \"$UNIT_ASSIGNMENT_PATH\""
                                CMD+=" --unit_type \"$UNIT_TYPE\""
                                CMD+=" --points_per_unit \"$POINTS_PER_UNIT\""
                                CMD+=" --region_assignment_path \"$REGION_ASSIGNMENT_PATH\""
                                CMD+=" --in_region_unit_cost \"$IN_REGION_UNIT_COST\""
                                CMD+=" --out_of_region_unit_cost \"$OUT_OF_REGION_UNIT_COST\""
                                CMD+=" --alpha \"$ALPHA\""
                            fi

                            CMD+=" $(append_method_flags $METHOD)"
                            run_experiment "$CMD" "$EXP_NAME" "$DATASET" "$METHOD" "$BUDGET" "$seed" "$INIT_NAME" "$COST_FN"
                        done
                    done
                done
            done
        done
    done
}

# Loop over seeds
for SEED in "${SEEDS[@]}"; do
    run_setting3 "$SEED"
done
