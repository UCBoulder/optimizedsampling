# === run_setting3.sh ===

#!/bin/bash
source "$(dirname "$0")/run_togo.sh"

run_setting3() {
    local seed="$1"
    DATASET="togo_ph_h20"
    ALPHAS=(0 1 10 15 20 25)

    for SIZE in 500; do
        INIT_NAME_EXP="cluster_sampling_2_strata_desired_25ppc_${SIZE}_size"
        INIT_NAME="cluster_sampling/2_strata_desired_25ppc_${SIZE}_size"
        INIT_SET_IDS="/home/libe2152/optimizedsampling/0_data/initial_samples/togo/cluster_sampling/fixedstrata_kara-plateaux/sample_region_canton_25ppc_${SIZE}_size_seed_${seed}.pkl"
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
                                UNIT_ASSIGNMENT_PATH="/home/libe2152/optimizedsampling/0_data/groups/togo/canton_assignments_dict.pkl"
                                UNIT_TYPE="cluster"
                                POINTS_PER_UNIT=25
                                IN_REGION_UNIT_COST=$POINTS_PER_UNIT
                                OUT_OF_REGION_UNIT_COST=$((POINTS_PER_UNIT+ALPHA))
                                COST_FN_NAME="region_aware_unit_cost"
                                COST_FN_WITH_SPECIFICS="cluster_based_c1_${IN_REGION_UNIT_COST}_c2_${OUT_OF_REGION_UNIT_COST}"
                                REGION_ASSIGNMENT_PATH="/home/libe2152/optimizedsampling/0_data/groups/togo/region_assignments_dict.pkl"
                            fi


                            if [[ "$METHOD" == "poprisk" || "$METHOD" == "poprisk_avg" ]]; then
                                EXP_NAME="${DATASET}_${INIT_NAME_EXP}_cost_${COST_FN_WITH_SPECIFICS}_method_${METHOD}_${UTIL_LAMBDA}_${GROUP_TYPE}_budget_${BUDGET}_seed_${seed}"
                            else
                                EXP_NAME="${DATASET}_${INIT_NAME_EXP}_cost_${COST_FN_WITH_SPECIFICS}_method_${METHOD}_budget_${BUDGET}_seed_${seed}"
                            fi
                            CMD="python $SCRIPT \
                                --cfg \"$CFG_FILE\" \
                                --exp-name \"$EXP_NAME\" \
                                --sampling_fn \"$METHOD\" \
                                --group_type \"$GROUP_TYPE\" \
                                --group_assignment_path \"$GROUP_ASSIGNMENT_PATH\" \
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
