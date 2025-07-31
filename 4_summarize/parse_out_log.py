import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

LOG_FILENAME = "stdout.log"
BASE_DIR = "../0_output"

# Regex patterns for extracting metrics
initial_r2_re = re.compile(r"Initial R² score: (-?[0-9.]+)")
updated_r2_re = re.compile(r"Updated R² score: (-?[0-9.]+)")
size_re = re.compile(r"Selected (\d+) new samples")
seed_re = re.compile(r"_seed_\d+")
initial_set_size_re = re.compile(r"_(\d+)_size")

DATASET_NAMES = {
    "INDIA_SECC", "USAVARS_POP", "USAVARS_TC", "TOGO_PH_H2O"
}

def extract_log_metrics(content):
    """Extract metrics from log content."""
    initial_r2 = updated_r2 = size = None
    if (m := initial_r2_re.search(content)):
        initial_r2 = float(m.group(1))
    if (m := updated_r2_re.search(content)):
        updated_r2 = float(m.group(1))
    if (m := size_re.search(content)):
        size = float(m.group(1))
    return initial_r2, updated_r2, size

def extract_dataset_index(path_parts):
    return next((i for i, part in enumerate(path_parts) if part in DATASET_NAMES), None) #make sure this does what I think

def parse_initial_set_and_cost(path_parts, dataset_idx):
    if path_parts[dataset_idx + 1] == "empty_initial_set":
        init_set_base = "empty_initial_set"
        cost_type = path_parts[dataset_idx + 2]
        method_base_idx = dataset_idx + 3
    else:
        post_dataset_parts = path_parts[dataset_idx + 1 : dataset_idx + 5]
        if any(p in {"multiple", "multiple_cluster_sampling"} for p in post_dataset_parts):
            init_set_idx = dataset_idx + 3
        else:
            init_set_idx = dataset_idx + 2
        init_set_full = path_parts[init_set_idx]
        init_set_base = seed_re.sub("", init_set_full)
        cost_type = path_parts[init_set_idx + 1]
        method_base_idx = init_set_idx + 2
    return init_set_base, cost_type, method_base_idx

def parse_method_and_budget(path_parts, method_base_idx):
    if "opt" in path_parts:
        idx = path_parts.index("opt")
        method = path_parts[idx + 1]
        if method.startswith("poprisk"):
            group_type = path_parts[idx + 2]
            budget = path_parts[idx + 3].replace("budget_", "")
            lambda_val = path_parts[idx + 4].replace("util_lambda_", "")
            method = f"{method}_{group_type}_{lambda_val}"
        else:
            budget = path_parts[idx + 2].replace("budget_", "")
    else:
        method = path_parts[method_base_idx]
        budget = path_parts[method_base_idx + 1].replace("budget_", "")

    return method, budget

def parse_path(root):
    path_parts = root.split("/")
    dataset_idx = extract_dataset_index(path_parts)
    if dataset_idx is None:
        raise ValueError("Dataset name not found in path")

    dataset = path_parts[dataset_idx]
    seed = path_parts[-1].split("_")[1]

    init_set_base, cost_type, method_base_idx = parse_initial_set_and_cost(path_parts, dataset_idx)
    method, budget = parse_method_and_budget(path_parts, method_base_idx)

    return (dataset, init_set_base, cost_type, budget), method, seed

def process_log_file(log_path, root):
    try:
        with open(log_path) as f:
            content = f.read()
        initial_r2, updated_r2, size = extract_log_metrics(content)
        key, method, seed = parse_path(root)
        return key, method, seed, initial_r2, updated_r2, size
    except Exception as e:
        print(f"Failed to parse {log_path}: {e}")
        return None

def should_process_path(log_path, log_filename):
    return os.path.isfile(log_path) and os.path.basename(log_path) == log_filename

def scan_directory(base_dir, log_filename):
    for root, _, files in os.walk(base_dir):
        if log_filename in files:
            yield os.path.join(root, log_filename), root

def aggregate_results(base_dir=BASE_DIR, log_filename=LOG_FILENAME, dataset=None, initial_set=None,
                      sampling_type=None, cost_type=None, budget_list=None, multiple=None):
    results = defaultdict(lambda: {
        "initial_r2": [], 
        "methods": defaultdict(list),
        "method_seeds": defaultdict(set),
        "seeds": set(),
        "size": defaultdict(list)
    })

    if dataset and initial_set and sampling_type and cost_type:
        if multiple:
            base_dir = os.path.join(base_dir, dataset, "multiple")
        else:
            base_dir = os.path.join(base_dir, dataset)
        base_dir = os.path.join(base_dir, sampling_type, initial_set, cost_type)

    for log_path, root in scan_directory(base_dir, log_filename):
        parsed = process_log_file(log_path, root)
        if parsed is None:
            continue
        key, method, seed, initial_r2, updated_r2, size = parsed
        (_, _, _, budget) = key

        if budget_list is not None and int(budget) not in budget_list:
            print(f"{budget} not in budget_list...")
            continue

        if initial_r2 is not None:
            results[key]["initial_r2"].append((seed, initial_r2))
            results[key]["seeds"].add(seed)

        if updated_r2 is not None:
            results[key]["methods"][method].append((seed, updated_r2))
            results[key]["method_seeds"][method].add(seed)

            if size is not None:
                results[key]["size"][method].append((seed, size))

    for key, val in list(results.items()):
        # === Clean initial_r2 ===
        seed_to_r2 = {}
        for s, r in val["initial_r2"]:
            try:
                seed = int(s)
            except ValueError:
                continue
            if seed in EXPECTED_SEEDS:
                if seed not in seed_to_r2:
                    seed_to_r2[seed] = r
                else:
                    print(f"Warning: Duplicate seed {seed} in initial_r2 for {key}, ignoring subsequent entries.")

        val["initial_r2"] = [
            seed_to_r2[seed] for seed in sorted(EXPECTED_SEEDS)
            if seed in seed_to_r2
        ]

        clean_methods = {}
        clean_sizes = {}
        delta_r2 = {}

        for method in list(val["methods"].keys()):
            entries = val["methods"][method]
            seed_to_r2 = {}
            for s, r in entries:
                try:
                    seed = int(s)
                except ValueError:
                    continue
                if seed in EXPECTED_SEEDS:
                    if seed not in seed_to_r2:
                        seed_to_r2[seed] = r
                    else:
                        print(f"Warning: Duplicate seed {seed} in method '{method}' for {key}, ignoring duplicate.")

            if not EXPECTED_SEEDS.issubset(seed_to_r2.keys()):
                missing = EXPECTED_SEEDS - seed_to_r2.keys()
                print(f"Skipping method {method} for {key} due to missing seeds: {missing}")
                continue

            updated_r2_vals = [seed_to_r2[seed] for seed in sorted(EXPECTED_SEEDS)]
            clean_methods[method] = updated_r2_vals

            # Clean size values similarly
            if method in val["size"]:
                size_entries = val["size"][method]
                seed_to_size = {}
                for s, sz in size_entries:
                    try:
                        seed = int(s)
                    except ValueError:
                        continue
                    if seed in EXPECTED_SEEDS and seed not in seed_to_size:
                        seed_to_size[seed] = sz
                    elif seed in seed_to_size:
                        print(f"Warning: Duplicate seed {seed} in size for method '{method}' in {key}, ignoring duplicate.")
                if EXPECTED_SEEDS.issubset(seed_to_size.keys()):
                    clean_sizes[method] = [seed_to_size[seed] for seed in sorted(EXPECTED_SEEDS)]

            # Compute delta R²
            if len(updated_r2_vals) == len(val["initial_r2"]):
                delta_r2[method] = [
                    updated - init for init, updated in zip(val["initial_r2"], updated_r2_vals)
                ]
            else:
                print(f"Mismatch in length for {key}, method {method}: skipping delta_r2")

        val["methods"] = clean_methods
        val["size"] = clean_sizes
        val["delta_r2"] = delta_r2

        if not val["methods"]:
            print(f"Removing entire experiment for {key} as no methods had complete seeds.")
            del results[key]


    return results


def build_filtered_df(results_dict, dataset, init_set, cost_type, alpha=None):
    rows = []
    for (ds, iset, ctype, budget), data in results_dict.items():
        if (ds, iset, ctype) != (dataset, init_set, cost_type):
            continue

        row = {
            "dataset": ds,
            "initial_set": iset,
            "cost_type": ctype,
            "budget": int(budget),
        }

        match = initial_set_size_re.search(iset)
        if match:
            row["initial_set_size"] = int(match.group(1))
        else:
            print(f"Warning: Could not extract size from initial_set: {iset}")
            row["initial_set_size"] = None
            
        if alpha is not None:
            row['alpha'] = float(alpha)

        arr = np.array(data["initial_r2"])
        row["initial_r2_mean"] = round(arr.mean(), 2)
        row["initial_r2_std"] = round(arr.std(), 2)

        for method, vals in data["methods"].items():
            if vals:
                arr = np.array(vals)
                row[f"{method}_updated_r2_mean"] = round(arr.mean(), 2)
                row[f"{method}_updated_r2_std"] = round(arr.std(), 2)

                delta_vals = data.get("delta_r2", {}).get(method, [])
                if delta_vals:
                    darr = np.array(delta_vals)
                    row[f"{method}_delta_r2_mean"] = round(darr.mean(), 2)
                    row[f"{method}_delta_r2_se"] = np.std(darr, ddof=1) / np.sqrt(np.size(darr))

        rows.append(row)
    return pd.DataFrame(rows).sort_values("budget")

def save_csv(df, dataset, init_set=None, cost_type=None, out_dir="results", multiple=False):
    os.makedirs(out_dir, exist_ok=True)
    
    name_parts = ["aggregated_r2", dataset]
    if init_set:
        name_parts.append(str(init_set))
    if cost_type:
        name_parts.append(str(cost_type))
    
    filename = "_".join(name_parts) + ".csv"

    if multiple:
        out_dir = os.path.join(out_dir, 'multiple')

    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"CSV saved to: {path}")

if __name__ == "__main__":
    initial_set_strs = {
        'USAVARS_POP': '{num_strata}_fixedstrata_{ppc}ppc_{size}_size',
        'USAVARS_TC': '{num_strata}_fixedstrata_{ppc}ppc_{size}_size',
        'INDIA_SECC': '{num_strata}_state_district_desired_{ppc}ppc_{size}_size',
        'TOGO_PH_H2O': '{num_strata}_strata_desired_{ppc}ppc_{size}_size'
    }

    PPC = {
        "USAVARS_POP": 10,
        "USAVARS_TC": 10,
        "INDIA_SECC": 20,
        "TOGO_PH_H2O": 25,
    }

    NUM_STRATA = {
        "USAVARS_POP": 5,
        "USAVARS_TC": 5,
        "INDIA_SECC": 10,
        "TOGO_PH_H2O": 2,
    }

    import argparse

    parser = argparse.ArgumentParser(description="Run sampling evaluation script")
    parser.add_argument("--multiple", type=lambda x: x.lower() == 'true', default=False,
                    help="Whether to use multiple experiments")

    args = parser.parse_args()
    multiple = args.multiple

    if multiple:
        EXPECTED_SEEDS = set([1, 42, 123, 456, 789]) # edit this...
    else:
        EXPECTED_SEEDS = set([1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415]) # edit this...

    for dataset in DATASET_NAMES:
        dataset_dfs = []
        multiple_dfs = []
        initial_set_sizes = []
        initial_r2_means = []
        initial_r2_stds = []

        ppc = PPC[dataset]
        num_strata = NUM_STRATA[dataset]

        seen_initial_sets = set()
        for size in range(100, 5100, 100):
            
            initial_set = initial_set_strs[dataset].format(num_strata=num_strata, ppc=ppc, size=size)
            sampling_type = 'cluster_sampling'
            for alpha in [0, 1, 5, 10, 15, 20, 25, 30]:
                in_region_cost = ppc
                out_of_region_cost = in_region_cost + alpha
                cost_type = f"cluster_based_c1_{in_region_cost}_c2_{out_of_region_cost}"
                budget_list = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000]
                try:
                    results = aggregate_results(dataset=dataset, initial_set=initial_set, sampling_type=sampling_type, cost_type=cost_type, budget_list=budget_list, multiple=multiple)
                    keys = set((ds, iset, ctype) for (ds, iset, ctype, _) in results)

                    for ds, iset, ctype in sorted(keys):
                        df = build_filtered_df(results, ds, iset, ctype, alpha)
                        if df.empty:
                            print(f"Skipping empty table for: {ds}, {iset}, {ctype}")
                            continue
                        save_csv(df, ds, iset, ctype, multiple=multiple)
                        dataset_dfs.append(df)
                        multiple_dfs.append(df)

                        # Save initial R² stats only once per (dataset, init_set)
                        key = (ds, iset)
                        if key not in seen_initial_sets:
                            initial_set_sizes.append(size)
                            initial_r2_means.append(np.unique(df['initial_r2_mean']))
                            initial_r2_stds.append(np.unique(df['initial_r2_std']))
                            seen_initial_sets.add(key)

                except Exception as e:
                    x=1


            if dataset_dfs and not multiple:
                print(len(dataset_dfs))
                combined_df = pd.concat(dataset_dfs, ignore_index=True, join="inner")

                save_csv(combined_df, dataset, initial_set, multiple=multiple)
                print(f"Saved combined results for {dataset}")
                dataset_dfs = []

        if multiple and multiple_dfs:
            combined_df = pd.concat(multiple_dfs, ignore_index=True, join="outer")

            save_csv(combined_df, dataset, multiple=multiple)
            print(f"Saved combined results for {dataset}")
            

        # Save initial R² summary
        if len(initial_set_sizes) > 0:
            print(initial_set_sizes)
            summary_df = pd.DataFrame({
                'initial_set_size': initial_set_sizes,
                'initial_r2_mean': [vals[0] if len(vals) == 1 else np.nan for vals in initial_r2_means],
                'initial_r2_std': [vals[0] if len(vals) == 1 else np.nan for vals in initial_r2_stds]
            })
            summary_path = f"results/multiple/{dataset}_initial_r2_summary.csv" if multiple else f"results/{dataset}_initial_r2_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved initial R² summary to {summary_path}")

            # Reset tracking lists
            initial_set_sizes = []
            initial_r2_means = []
            initial_r2_stds = []