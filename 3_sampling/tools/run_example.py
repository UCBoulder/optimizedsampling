import subprocess

def run_one_example():
    # === Configuration ===
    seed = 1
    dataset = "usavars"
    size = 100
    cost_fn = "cluster_based"  # or "uniform"
    method = "diversity"             # or "random", "greedycost", "similarity"
    budget = 100
    util_lambda = 0.5

    script = "train.py"
    cfg_file = "/home/libe2152/optimizedsampling/3_sampling/configs/usavars/RIDGE_POP.yaml"

    init_name_exp = f"cluster_sampling_5_fixedstrata_10ppc_{size}_size"
    init_name = f"cluster_sampling/5_fixedstrata_10ppc_{size}_size"
    id_path = (
        f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/population/"
        f"cluster_sampling/fixedstrata_Idaho_16-Louisiana_22-Mississippi_28-New Mexico_35-Pennsylvania_42/"
        f"sample_state_combined_county_id_10ppc_{size}_size_seed_{seed}.pkl"
    )

    print(f"⚙️ Running Setting 3 with COST_FN={cost_fn}")
    cmd = [
        "python", script,
        "--cfg", cfg_file,
        "--sampling_fn", method,    
        "--budget", str(budget),
        "--initial_set_str", init_name,
        "--id_path", id_path,
        "--seed", str(seed),
    ]

    if cost_fn == "uniform":
        cost_fn_name = "uniform"
        cost_fn_with_specifics = "uniform"
    elif cost_fn == "cluster_based":
        unit_assignment_path = "/home/libe2152/optimizedsampling/0_data/groups/usavars/population/county_assignments_dict.pkl"
        region_assignment_path = "/home/libe2152/optimizedsampling/0_data/groups/usavars/population/state_assignments_dict.pkl"
        cost_fn_name = "region_aware_unit_cost"
        cost_fn_with_specifics = "cluster_based_c1_10_c2_20"
        cmd += [
            "--cost_func", cost_fn_name,
            "--cost_name", cost_fn_with_specifics,
            "--unit_assignment_path", unit_assignment_path,
            "--unit_type", "cluster",
            "--points_per_unit", "10",
            "--region_assignment_path", region_assignment_path,
            "--in_region_unit_cost", "10",
            "--out_of_region_unit_cost", "20",
        ]
    else:
        raise ValueError(f"Unknown cost_fn: {cost_fn}")

    if cost_fn == "uniform":
        cmd += [
            "--cost_func", cost_fn_name,
            "--cost_name", cost_fn_with_specifics,
        ]

    # Set experiment name
    if method == "poprisk":
        exp_name = f"{dataset}_population_{init_name_exp}_cost_{cost_fn_with_specifics}_method_{method}_{util_lambda}_budget_{budget}_seed_{seed}"
        cmd += ["--util_lambda", str(util_lambda)]
    else:
        exp_name = f"{dataset}_population_{init_name_exp}_cost_{cost_fn_with_specifics}_method_{method}_budget_{budget}_seed_{seed}"

    cmd += ["--exp-name", exp_name]

    # Method-specific flags
    if method == "similarity":
        sim_matrix_path = "/home/libe2152/optimizedsampling/0_data/cosine_similarity/usavars/population/cosine_similarity_train_test.npy"
        cmd += ["--similarity_matrix_path", sim_matrix_path]

    elif method == "diversity":
        train_sim_matrix_path = "/home/libe2152/optimizedsampling/0_data/cosine_similarity/usavars/population/cosine_similarity_train_train.npy"
        sim_per_unit_path = "/home/libe2152/optimizedsampling/0_data/cosine_similarity/usavars/population/county_cosine_similarity_train_train.npy"
        cmd += [
            "--train_similarity_matrix_path", train_sim_matrix_path,
            "--similarity_per_unit_path", sim_per_unit_path,
        ]

    if method.startswith("poprisk") or method == "match_population_proportion":
        group_path = "/home/libe2152/optimizedsampling/0_data/groups/usavars/population/NLCD_cluster_assignments_8_dict.pkl"
        group_type = "dists_from_top20_urban_tiers"
        cmd += ["--group_assignment_path", group_path, "--group_type", group_type]

    # Run
    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_one_example()

