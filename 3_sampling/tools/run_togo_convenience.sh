# === run_setting2.sh ===

#!/bin/bash
source "$(dirname "$0")/run_togo.sh"

run_setting2() {
    local seed="$1"
    INIT_NAME_EXP="convenience_sampling"
    INIT_NAME="convenience_sampling/urban_based_top10_urban_1000_points"
    INIT_SET_IDS="/home/libe2152/optimizedsampling/0_data/initial_samples/india_secc/convenience_sampling/urban_based/IDS_top10_urban_1000_points_probabilistic_1000_size_seed_${seed}.pkl"
    GROUP_TYPE="urban_rural"
    DATASET="india"
    GROUP_PATH="/home/libe2152/optimizedsampling/0_data/groups/india_secc/urban_rural_groups.pkl"
    UTIL_LAMBDA=0.1

    for COST_FN in "convenience_based" "uniform"; do
        echo "⚙️ Running Setting 2 with POINTS_PER_CLUSTER=${POINTS_PER_CLUSTER}, COST_FN=${COST_FN}"

        for METHOD in "${METHODS[@]}"; do
            for BUDGET in "${BUDGETS[@]}"; do

                COST_FN_NAME="uniform"
                COST_ARRAY_PATH=""

                if [ "$COST_FN" == "convenience_based" ]; then
                    COST_ARRAY_PATH="/home/libe2152/optimizedsampling/0_data/costs/india_secc/convenience_costs/distance_based_costs_top10_urban.pkl"
                    COST_FN_NAME="pointwise_by_array"
                fi

                EXP_NAME="india_${INIT_NAME_EXP}_cost_${COST_FN}_method_${METHOD}_budget_${BUDGET}_seed_${seed}"
                if [ "$METHOD" == "poprisk" ]; then
                    EXP_NAME="india_${INIT_NAME_EXP}_cost_${COST_FN}_method_${METHOD}_${GROUP_TYPE}_${UTIL_LAMBDA}_budget_${BUDGET}_seed_${seed}"
                fi
                CMD="python $SCRIPT \
                    --cfg \"$CFG_FILE\" \
                    --exp-name \"$EXP_NAME\" \
                    --sampling_fn \"$METHOD\" \
                    --group_type \"$GROUP_TYPE\" \
                    --budget $BUDGET \
                    --initial_set_str \"$INIT_NAME\" \
                    --id_path \"$INIT_SET_IDS\" \
                    --seed $seed \
                    --cost_func \"$COST_FN_NAME\" \
                    --cost_name \"$COST_FN\""

                if [ "$COST_FN" == "convenience_based" ]; then
                    CMD+=" --cost_array_path \"$COST_ARRAY_PATH\""
                fi

                CMD+=" $(append_method_flags $METHOD)"
                run_experiment "$CMD" "$EXP_NAME" "$DATASET" "$METHOD" "$BUDGET" "$seed" "${INIT_NAME}_${POINTS_PER_CLUSTER}ppc" "$COST_FN"
            done
        done
    done
}

# Loop call example
for SEED in "${SEEDS[@]}"; do
    run_setting2 "$SEED"
done
