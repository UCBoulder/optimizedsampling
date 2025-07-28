# === run_setting2.sh ===

#!/bin/bash
source "$(dirname "$0")/run_usavars_population.sh"

run_setting2() {
    local seed="$1"
    DATASET="usavars"

    for SIZE in 100 200 300 400 500 1000; do
        INIT_NAME_EXP="convenience_sampling_top20_urban_${SIZE}_points"
        INIT_NAME="convenience_sampling/top20_urban_${SIZE}_points"
        INIT_SET_IDS="/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/population/convenience_sampling/urban_based/IDS_top20_urban_${SIZE}_points_probabilistic_${SIZE}_size_seed_${seed}.pkl"

        for COST_FN in "convenience_based"; do
            echo "⚙️ Running Setting 2 with POINTS_PER_CLUSTER=${POINTS_PER_CLUSTER}, COST_FN=${COST_FN}"

            for METHOD in "${METHODS[@]}"; do
                for BUDGET in "${BUDGETS[@]}"; do
                    for UTIL_LAMBDAS in "${UTIL_LAMBDAS[@]}"; do

                        COST_FN_NAME="uniform"
                        COST_ARRAY_PATH=""

                        if [ "$COST_FN" == "convenience_based" ]; then
                            COST_ARRAY_PATH="/home/libe2152/optimizedsampling/0_data/costs/usavars/population/convenience_costs/linear_distance_km_based_costs_top20_urban_0.01.pkl"
                            COST_FN_NAME="pointwise_by_array"
                            COST_FN_WITH_SPECIFICS='convenience_based_capped_5_linear_top20_urban_0.01'
                        fi

                        EXP_NAME="usavars_population_${INIT_NAME_EXP}_cost_${COST_FN_WITH_SPECIFICS}_method_${METHOD}_budget_${BUDGET}_seed_${seed}"

                        if [ "$METHOD" == "poprisk" ]; then
                            EXP_NAME="usavars_population_${INIT_NAME_EXP}_cost_${COST_FN_WITH_SPECIFICS}_method_${METHOD}_${UTIL_LAMBDA}_${GROUP_TYPE}_budget_${BUDGET}_seed_${seed}"
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

                        if [ "$COST_FN" == "convenience_based" ]; then
                            CMD+=" --cost_array_path \"$COST_ARRAY_PATH\""
                        fi

                        CMD+=" $(append_method_flags $METHOD)"
                        run_experiment "$CMD" "$EXP_NAME" "$DATASET" "$METHOD" "$BUDGET" "$seed" "${INIT_NAME}_${POINTS_PER_CLUSTER}ppc" "$COST_FN"
                    done
                done
            done
        done
    done
}

# Loop call example
for SEED in "${SEEDS[@]}"; do
    run_setting2 "$SEED"
done