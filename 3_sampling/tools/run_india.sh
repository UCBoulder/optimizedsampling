# === run_common.sh ===

#!/bin/bash

# === Shared Config ===
CFG_FILE="/home/libe2152/optimizedsampling/3_sampling/configs/india_secc/RIDGE.yaml"
SCRIPT="train.py"
EMAIL="liviabetti1@gmail.com"
SEND_EMAIL_PY="send_email.py"
LOGFILE="completed_experiments_india.log"
FAILED_LOG="failed_experiments_india.log"

ROOT_DIR="/share/india_secc"
SIM_MATRIX_PATH="/share/india_secc/similarity_matrix.npz"
DIST_MATRIX_PATH=""

GROUP_TYPE="dists_from_top20_urban_tiers"
GROUP_PATH="/home/libe2152/optimizedsampling/0_data/groups/india_secc/dist_from_top20urban_area_tiers.pkl"

SEEDS=(1 42 123 456 789 1234 5678 9101 1213 1415)
METHODS=("random" "poprisk_mod" "poprisk")
BUDGETS=(200 400 600 800 1000 2000)

# === Helpers ===
send_error_email() {
    local exp_name="$1"; local dataset="$2"; local method="$3"; local budget="$4"; local seed="$5"; local init_name="$6"; local cost_name="$7"
    local subject="AL Experiment Error: $exp_name"
    local body="Experiment failed: $exp_name\nDataset: $dataset\nMethod: $method\nBudget: $budget\nSeed: $seed\nInitial: $init_name\nCost: $cost_name"
    echo "📧 Sending failure email for: $exp_name"
    python "$SEND_EMAIL_PY" "$subject" "$body" "$EMAIL"
}

run_experiment() {
    local cmd="$1"
    local exp_name="$2"
    local dataset="$3"
    local method="$4"
    local budget="$5"
    local seed="$6"
    local init_name="$7"
    local cost_name="$8"

    if grep -Fxq "$exp_name" "$LOGFILE"; then
        echo "⏭️  Skipping already completed experiment: $exp_name"
        return
    fi

    echo "▶️ Running: $exp_name"
    eval "$cmd"
    local status=$?

    if [[ $status -ne 0 ]]; then
        echo "❌ Error in experiment: $exp_name"
        echo "$exp_name" >> "$FAILED_LOG"
        send_error_email "$exp_name" "$dataset" "$method" "$budget" "$seed" "$init_name" "$cost_name"
    else
        echo "$exp_name" >> "$LOGFILE"
    fi
}

append_method_flags() {
    local method="$1"
    local cmd=""

    if [ "$method" == "similarity" ]; then
        cmd+=" --similarity_matrix_path \"$SIM_MATRIX_PATH\""
    elif [ "$method" == "diversity" ]; then
        cmd+=" --distance_matrix_path \"$DIST_MATRIX_PATH\""
    fi

    if [[ "$method" == "poprisk" || "$method" == "poprisk_mod" ]]; then
        UTIL_LAMBDA=0.5
        cmd+=" --util_lambda 0.5 --group_assignment_path \"$GROUP_PATH\" --group_type \"$GROUP_TYPE\""
    elif [ "$method" == "match_population_proportion" ]; then
        cmd+=" --group_assignment_path \"$GROUP_PATH\" --group_type \"$GROUP_TYPE\""
    fi

    echo "$cmd"
}