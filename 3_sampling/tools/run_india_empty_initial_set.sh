# === run_setting1.sh ===

#!/bin/bash
source "$(dirname "$0")/run_india.sh"

run_setting1() {
    local seed="$1"
    INIT_NAME="empty_initial_set"
    DATASET="india"
    GROUP_TYPE="urban_rural"

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
                    UNIT_ASSIGNMENT_PATH="/home/libe2152/optimizedsampling/0_data/groups/india_secc/district_assignments_dict.pkl"
                    UNIT_TYPE="cluster"
                    POINTS_PER_UNIT=5
                    COST_FN_NAME="region_aware_unit_cost"
                    REGION_ASSIGNMENT_PATH="/home/libe2152/optimizedsampling/0_data/groups/india_secc/state_assignments_dict.pkl"
                    IN_REGION_UNIT_COST=50
                    OUT_OF_REGION_UNIT_COST=100
                elif [ "$COST_FN" == "convenience_based" ]; then
                    COST_ARRAY_PATH="/home/libe2152/optimizedsampling/0_data/costs/india_secc/convenience_costs/distance_based_costs_top10_urban.pkl"
                    COST_FN_NAME="pointwise_by_array"
                fi

                if [[ "$METHOD" == "poprisk" || "$METHOD" == "match_population_proportion" ]]; then
                    EXP_NAME="india_${INIT_NAME}_cost_${COST_FN}_method_${METHOD}_${GROUP_TYPE}_budget_${BUDGET}_seed_${seed}"
                else
                    EXP_NAME="india_${INIT_NAME}_cost_${COST_FN}_method_${METHOD}_budget_${BUDGET}_seed_${seed}"
                fi
                CMD="python $SCRIPT \
                    --cfg \"$CFG_FILE\" \
                    --exp-name \"$EXP_NAME\" \
                    --sampling_fn \"$METHOD\" \
                    --group_type \"$GROUP_TYPE\" \
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
