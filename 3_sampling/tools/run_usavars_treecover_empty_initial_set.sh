# === run_setting1.sh ===

#!/bin/bash
source "$(dirname "$0")/run_usavars_treecover.sh"

run_setting1() {
    local seed="$1"
    INIT_NAME="empty_initial_set"
    DATASET="usavars"
    UTIL_LAMBDA=0.1

    for COST_FN in "uniform" "cluster_based" "convenience_based"; do
        echo "⚙️ Running Setting 1 with COST_FN=${COST_FN}"

        for METHOD in "${METHODS[@]}"; do
            for BUDGET in "${BUDGETS[@]}"; do

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
                    COST_FN_NAME="region_aware_unit_cost"
                    COST_FN_WITH_SPECIFICS="cluster_based_c1_10_c2_15"
                    REGION_ASSIGNMENT_PATH="/home/libe2152/optimizedsampling/0_data/groups/usavars/treecover/state_assignments_dict.pkl"
                    IN_REGION_UNIT_COST=10
                    OUT_OF_REGION_UNIT_COST=15
                elif [ "$COST_FN" == "convenience_based" ]; then
                    COST_ARRAY_PATH="/home/libe2152/optimizedsampling/0_data/costs/usavars/treecover/convenience_costs/distance_based_costs_top50_urban.pkl"
                    COST_FN_NAME="pointwise_by_array"
                    COST_FN_WITH_SPECIFICS="convenience_based_top50_urban"
                fi

                EXP_NAME="usavars_treecover_${INIT_NAME}_cost_${COST_FN_WITH_SPECIFICS}_method_${METHOD}_budget_${BUDGET}_seed_${seed}"
                if [ "$METHOD" == "poprisk" ]; then
                    EXP_NAME="usavars_treecover_${INIT_NAME}_cost_${COST_FN_WITH_SPECIFICS}_method_${METHOD}_${UTIL_LAMBDA}_budget_${BUDGET}_seed_${seed}"
                fi

                CMD="python $SCRIPT \
                    --cfg \"$CFG_FILE\" \
                    --exp-name \"$EXP_NAME\" \
                    --sampling_fn \"$METHOD\" \
                    --budget $BUDGET \
                    --initial_set_str \"$INIT_NAME\" \
                    --seed $seed \
                    --cost_func \"$COST_FN_NAME\" \
                    --cost_name \"$COST_FN\""

                if [ "$COST_FN" == "convenience_based" ]; then
                    CMD+=" --cost_array_path \"$COST_ARRAY_PATH\""
                elif [ "$COST_FN" == "cluster_based" ]; then
                    CMD+=" --unit_assignment_path \"$UNIT_ASSIGNMENT_PATH\""
                    CMD+=" --unit_type \"$UNIT_TYPE\""
                    CMD+=" --points_per_unit \"$POINTS_PER_UNIT\""
                    CMD+=" --region_assignment_path \"$REGION_ASSIGNMENT_PATH\""
                    CMD+=" --in_region_unit_cost \"$IN_REGION_UNIT_COST\""
                    CMD+=" --out_of_region_unit_cost \"$OUT_OF_REGION_UNIT_COST\""
                fi

                CMD+=" $(append_method_flags $METHOD)"
                run_experiment "$CMD" "$EXP_NAME" "$DATASET" "$METHOD" "$BUDGET" "$seed" "$INIT_NAME" "$COST_FN"
            done
        done
    done
}

# Loop call example
for SEED in "${SEEDS[@]}"; do
    run_setting1 "$SEED"
done
