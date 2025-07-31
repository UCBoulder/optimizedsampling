# === run_setting3.sh ===

#!/bin/bash
source "$(dirname "$0")/run_togo.sh"

run_setting3() {
    local seed="$1"
    SEEDS=(1 42 123 456 789)
    METHODS=("poprisk")
    BUDGETS=(100 200 300 400 500 600 700 800 900 1000)
    DATASET="togo"

    for SIZE in 500 1000 1500 2000 2500; do
        for PPC in 20; do
            if [ "$SIZE" -eq 0 ]; then
                INIT_NAME_EXP="empty_initial_set"
                INIT_NAME="multiple/cluster_sampling/empty_initial_set"
                INIT_SET_IDS=None
            else
                INIT_NAME_EXP="multiple_cluster_sampling_2_strata_desired_25ppc_${SIZE}_size"
                INIT_NAME="multiple/cluster_sampling/2_strata_desired_25ppc_${SIZE}_size"
                INIT_SET_IDS="/home/libe2152/optimizedsampling/0_data/initial_samples/togo/cluster_sampling/fixedstrata_kara-plateaux/sample_region_canton_25ppc_${SIZE}_size_seed_1.pkl"
            fi
            for COST_FN in "cluster_based"; do
                echo "⚙️ Running Setting 3 with COST_FN=${COST_FN}"

                for METHOD in "${METHODS[@]}"; do
                    for BUDGET in "${BUDGETS[@]}"; do

                        COST_FN_NAME="uniform"
                        COST_FN_NAME_EXP="${COST_FN}"
                        COST_ARRAY_PATH=""
                        UNIT_ASSIGNMENT_PATH=""
                        UNIT_COST_PATH=""
                        UNIT_TYPE=""
                        POINTS_PER_UNIT=""

                        if [ "$COST_FN" == "cluster_based" ]; then
                            UNIT_ASSIGNMENT_PATH="/home/libe2152/optimizedsampling/0_data/groups/togo/canton_assignments_dict.pkl"
                            IN_REGION_UNIT_COST=25
                            OUT_OF_REGION_UNIT_COST=35
                            UNIT_TYPE="cluster"
                            POINTS_PER_UNIT=25
                            COST_FN_NAME="region_aware_unit_cost"
                            COST_FN_WITH_SPECIFICS="cluster_based_c1_${IN_REGION_UNIT_COST}_c2_${OUT_OF_REGION_UNIT_COST}"
                            REGION_ASSIGNMENT_PATH="/home/libe2152/optimizedsampling/0_data/groups/togo/region_assignments_dict.pkl"
                        fi


                        if [[ "$METHOD" == "poprisk" || "$METHOD" == "poprisk_mod" ]]; then
                            EXP_NAME="togo_${INIT_NAME_EXP}_cost_${COST_FN_WITH_SPECIFICS}_method_${METHOD}_${GROUP_TYPE}_budget_${BUDGET}_seed_${seed}"
                        else
                            EXP_NAME="togo_${INIT_NAME_EXP}_cost_${COST_FN_WITH_SPECIFICS}_method_${METHOD}_budget_${BUDGET}_seed_${seed}"
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
                        fi

                        CMD+=" $(append_method_flags $METHOD)"
                        run_experiment "$CMD" "$EXP_NAME" "$DATASET" "$METHOD" "$BUDGET" "$seed" "$INIT_NAME" "$COST_FN"
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
