import subprocess

def run_one_example():
    # === Configuration ===
    seed = 1
    dataset = "usavars"
    size = 100
    cost_fn = "convenience_based"  # or "uniform"
    method = "poprisk_mod"             # or "random", "greedycost", "similarity"
    budget = 100
    util_lambda = 0.5

    script = "train.py"
    cfg_file = "/home/libe2152/optimizedsampling/3_sampling/configs/usavars/RIDGE_POP.yaml"
    init_name_exp = f"convenience_sampling_top20_urban_{size}_points"
    init_name = f"convenience_sampling/top20_urban_{size}_points"
    id_path = f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/population/convenience_sampling/urban_based/IDS_top20_urban_{size}_points_probabilistic_{size}_size_seed_{seed}.pkl"

    # === Cost settings ===
    if cost_fn == "convenience_based":
        cost_array_path = "/home/libe2152/optimizedsampling/0_data/costs/usavars/population/convenience_costs/distance_based_costs_top20_urban_0.01.pkl"
        cost_fn_name = "pointwise_by_array"
        cost_fn_specifics = "convenience_based_top20_urban_0.01"
    else:
        cost_array_path = ""
        cost_fn_name = "uniform"
        cost_fn_specifics = "uniform"

    group_path = "/home/libe2152/optimizedsampling/0_data/groups/usavars/population/dist_from_top20urban_area_tiers.pkl"
    group_type = "dists_from_top20_urban_tiers"

    # === Experiment name ===
    if method == "poprisk":
        util_lambda = 1.0
        exp_name = f"{dataset}_population_{init_name_exp}_cost_{cost_fn_specifics}_method_{method}_{util_lambda}_{group_type}_budget_{budget}_seed_{seed}"
    else:
        exp_name = f"{dataset}_population_{init_name_exp}_cost_{cost_fn_specifics}_method_{method}_budget_{budget}_seed_{seed}"

    # === Base command ===
    cmd = [
        "python", script,
        "--cfg", cfg_file,
        "--exp-name", exp_name,
        "--sampling_fn", method,
        "--budget", str(budget),
        "--initial_set_str", init_name,
        "--id_path", id_path,
        "--seed", str(seed),
        "--cost_func", cost_fn_name,
        "--cost_name", cost_fn_specifics,
    ]

    if cost_fn == "convenience_based":
        cmd += ["--cost_array_path", cost_array_path]

    # === Method-specific flags ===
    if method == "similarity":
        sim_matrix_path = "/home/libe2152/optimizedsampling/0_data/cosine_similarity/usavars/population/cosine_similarity_train_test.npy"
        cmd += ["--similarity_matrix_path", sim_matrix_path]
    elif method == "diversity":
        dist_matrix_path = "/home/libe2152/optimizedsampling/0_data/cosine_distance/usavars/population/cosine_distance.npy"
        dist_per_unit_path = "/home/libe2152/optimizedsampling/0_data/cosine_distance/usavars/population/avg_cosine_distance_per_county.npy"
        cmd += [
            "--distance_matrix_path", dist_matrix_path,
            "--distance_per_unit_path", dist_per_unit_path,
        ]
    if method in ["poprisk", "match_population_proportion"]:
        group_path = "/home/libe2152/optimizedsampling/0_data/groups/usavars/population/NLCD_cluster_assignments_8_dict.pkl"
        cmd += ["--group_assignment_path", group_path, "--group_type", group_type]

    if method in ["poprisk", "match_population_proportion", "poprisk_mod"]:
        cmd += ["--group_assignment_path", group_path, "--group_type", group_type]

    if method.startswith("poprisk"):
        cmd += ["--util_lambda", str(util_lambda)]

    # === Run the command ===
    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_one_example()

