# === run_common.sh ===

#!/bin/bash

# === Shared Config ===
CFG_FILE="/home/libe2152/optimizedsampling/3_sampling/configs/usavars/RIDGE_TC.yaml"
SCRIPT="train.py"
EMAIL="liviabetti1@gmail.com"
SEND_EMAIL_PY="send_email.py"
LOGFILE="completed_experiments_usavars_treecover.log"
FAILED_LOG="failed_experiments_usavars_treecover.log"

ROOT_DIR="/share/usavars"
SIM_MATRIX_PATH="/home/libe2152/optimizedsampling/0_data/cosine_similarity/usavars/treecover/cosine_similarity_train_test.npy"
TRAIN_SIM_MATRIX_PATH="/home/libe2152/optimizedsampling/0_data/cosine_similarity/usavars/treecover/cosine_similarity_train_train.npy"
SIM_PER_UNIT_PATH="/home/libe2152/optimizedsampling/0_data/cosine_similarity/usavars/treecover/county_cosine_similarity_train_train.npy"

GROUP_PATH="../../0_data/groups/usavars/treecover/state_assignments_dict.pkl"
GROUP_TYPE="states"

SEEDS=(1 42 123 456 789 1234 5678 9101 1213 1415)
METHODS=("random" "random_unit" "greedycost" "poprisk" "poprisk_avg")
BUDGETS=(50 100 200 300 400 500 1000)

UTIL_LAMBDAS=(0.5 1.0)

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
        echo "Error in experiment: $exp_name"
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
        cmd+=" --train_similarity_matrix_path \"$TRAIN_SIM_MATRIX_PATH\" --similarity_per_unit_path \"$SIM_PER_UNIT_PATH\""
    fi

    if [ "$method" == "poprisk" ]; then
        cmd+=" --group_assignment_path \"$GROUP_PATH\" --group_type \"$GROUP_TYPE\""
    elif [ "$method" == "poprisk_avg" ]; then
        cmd+=" --group_assignment_path \"$GROUP_PATH\" --group_type \"$GROUP_TYPE\""
    fi

    echo "$cmd"
}