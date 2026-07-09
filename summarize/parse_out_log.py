import os
import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict

LOG_FILENAME = "stdout.log"
BASE_DIR = "../0_output"

initial_r2_re = re.compile(r"Initial R² score: (-?[0-9.]+)")
updated_r2_re = re.compile(r"Updated R² score: (-?[0-9.]+)")
initial_mse_re = re.compile(r"Initial MSE: (-?[0-9.]+)")
updated_mse_re = re.compile(r"Updated MSE: (-?[0-9.]+)")
initial_rmse_re = re.compile(r"Initial RMSE: (-?[0-9.]+)")
updated_rmse_re = re.compile(r"Updated RMSE: (-?[0-9.]+)")
initial_mae_re = re.compile(r"Initial MAE: (-?[0-9.]+)")
updated_mae_re = re.compile(r"Updated MAE: (-?[0-9.]+)")
size_re = re.compile(r"Selected (\d+) new samples")
seed_re = re.compile(r"_seed_\d+")
initial_set_size_re = re.compile(r"_(\d+)_size")

DATASET_NAMES = {"TOGO_PH_H2O"}
METRICS = ['r2', 'mse', 'rmse', 'mae']


def extract_log_metrics(content):
    metrics = {}
    patterns = {
        'initial_r2': initial_r2_re, 'updated_r2': updated_r2_re,
        'initial_mse': initial_mse_re, 'updated_mse': updated_mse_re,
        'initial_rmse': initial_rmse_re, 'updated_rmse': updated_rmse_re,
        'initial_mae': initial_mae_re, 'updated_mae': updated_mae_re,
        'size': size_re
    }
    for key, pattern in patterns.items():
        if (m := pattern.search(content)):
            metrics[key] = float(m.group(1))
    return metrics


def parse_path(root):
    path_parts = root.split("/")

    dataset_idx = next((i for i, part in enumerate(path_parts) if part in DATASET_NAMES), None)
    if dataset_idx is None:
        raise ValueError("Dataset name not found in path")

    dataset = path_parts[dataset_idx]
    seed = path_parts[-1].split("_")[1]

    if path_parts[dataset_idx + 1] == "empty_initial_set":
        init_set_base = "empty_initial_set"
        cost_type = path_parts[dataset_idx + 2]
        method_base_idx = dataset_idx + 3
    else:
        post_dataset_parts = path_parts[dataset_idx + 1 : dataset_idx + 6]
        if any(p in {"multiple", "multiple_cluster_sampling"} for p in post_dataset_parts):
            init_set_idx = dataset_idx + 4
        else:
            init_set_idx = dataset_idx + 3

        init_set_full = path_parts[init_set_idx]
        init_set_base = seed_re.sub("", init_set_full)
        cost_type = path_parts[init_set_idx + 1]
        method_base_idx = init_set_idx + 2

    if "opt" in path_parts:
        idx = path_parts.index("opt")
        method = path_parts[idx + 1]
        if method.startswith("poprisk"):
            group_type = path_parts[idx + 2]
            lambda_val = path_parts[idx + 4].replace("util_lambda_", "")
            method = f"{method}_{group_type}_{lambda_val}"
            budget = path_parts[idx + 3].replace("budget_", "")
        else:
            budget = path_parts[idx + 2].replace("budget_", "")
    else:
        method = path_parts[method_base_idx]
        budget = path_parts[method_base_idx + 1].replace("budget_", "")

    return (dataset, init_set_base, cost_type, budget), method, seed


def process_log_file(log_path, root):
    try:
        with open(log_path) as f:
            content = f.read()
        metrics = extract_log_metrics(content)
        key, method, seed = parse_path(root)
        return key, method, seed, metrics
    except Exception:
        return None


def aggregate_results(base_dir=BASE_DIR, log_filename=LOG_FILENAME, dataset=None,
                     initial_set=None, sampling_type=None, cost_type=None,
                     budget_list=None, multiple=None, model='ridge'):
    results = defaultdict(lambda: {
        **{f"initial_{metric}": [] for metric in METRICS},
        "methods": defaultdict(list),
        "method_seeds": defaultdict(set),
        "seeds": set(),
        "size": defaultdict(list),
        **{f"{metric}_methods": defaultdict(list) for metric in METRICS}
    })

    if dataset and initial_set and sampling_type and cost_type:
        path_parts = [base_dir, dataset, model]
        if multiple:
            path_parts.append("multiple")
        path_parts.extend([sampling_type, initial_set, cost_type])
        base_dir = os.path.join(*path_parts)

    for root, _, files in os.walk(base_dir):
        if log_filename not in files:
            continue

        log_path = os.path.join(root, log_filename)
        parsed = process_log_file(log_path, root)
        if parsed is None:
            continue

        key, method, seed, metrics = parsed
        (_, _, _, budget) = key

        if budget_list is not None and int(budget) not in budget_list:
            continue

        results[key]["seeds"].add(seed)

        for metric in METRICS:
            if f'initial_{metric}' in metrics:
                results[key][f"initial_{metric}"].append((seed, metrics[f'initial_{metric}']))

        for metric in METRICS:
            if f'updated_{metric}' in metrics:
                results[key][f"{metric}_methods"][method].append((seed, metrics[f'updated_{metric}']))
                results[key]["method_seeds"][method].add(seed)

        if 'size' in metrics:
            results[key]["size"][method].append((seed, metrics['size']))

    for key, val in list(results.items()):
        for metric in METRICS:
            seed_to_val = {}
            for s, v in val[f"initial_{metric}"]:
                try:
                    seed = int(s)
                except ValueError:
                    continue
                if seed in EXPECTED_SEEDS and seed not in seed_to_val:
                    seed_to_val[seed] = v

            val[f"initial_{metric}"] = [
                seed_to_val[seed] for seed in sorted(EXPECTED_SEEDS)
                if seed in seed_to_val
            ]

        clean_methods = {}
        clean_sizes = {}
        delta_metrics = {metric: {} for metric in METRICS}

        for metric in METRICS:
            for method in list(val[f"{metric}_methods"].keys()):
                entries = val[f"{metric}_methods"][method]
                seed_to_val = {}

                for s, v in entries:
                    try:
                        seed = int(s)
                    except ValueError:
                        continue
                    if seed in EXPECTED_SEEDS and seed not in seed_to_val:
                        seed_to_val[seed] = v

                if not EXPECTED_SEEDS.issubset(seed_to_val.keys()):
                    continue

                updated_vals = [seed_to_val[seed] for seed in sorted(EXPECTED_SEEDS)]

                if metric not in clean_methods:
                    clean_methods[metric] = {}
                clean_methods[metric][method] = updated_vals

                if len(updated_vals) == len(val[f"initial_{metric}"]):
                    if metric == 'r2':
                        delta_metrics[metric][method] = [
                            updated - init for init, updated in zip(val[f"initial_{metric}"], updated_vals)
                        ]
                    else:
                        delta_metrics[metric][method] = [
                            init - updated for init, updated in zip(val[f"initial_{metric}"], updated_vals)
                        ]

        for method in set(m for metric_methods in clean_methods.values() for m in metric_methods.keys()):
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

                if EXPECTED_SEEDS.issubset(seed_to_size.keys()):
                    clean_sizes[method] = [seed_to_size[seed] for seed in sorted(EXPECTED_SEEDS)]

        val["methods"] = clean_methods
        val["size"] = clean_sizes
        val["delta_metrics"] = delta_metrics

        if not any(clean_methods.values()):
            del results[key]

    return results


def build_filtered_df(results_dict, dataset, init_set, cost_type, alpha=None,
                     initial_metrics_lookup=None):
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

        if iset == "empty_initial_set":
            row["initial_set_size"] = 0
        else:
            match = initial_set_size_re.search(iset)
            row["initial_set_size"] = int(match.group(1)) if match else None

        if alpha is not None:
            row['alpha'] = float(alpha)

        for metric in METRICS:
            arr = np.array(data[f"initial_{metric}"])
            if len(arr) > 0:
                row[f"initial_{metric}_mean"] = round(arr.mean(), 2)
                row[f"initial_{metric}_std"] = round(arr.std(), 2)
                row[f"initial_{metric}_se"] = np.std(arr, ddof=1) / np.sqrt(np.size(arr))

        for metric in METRICS:
            if metric in data["methods"]:
                for method, vals in data["methods"][metric].items():
                    if vals:
                        arr = np.array(vals)
                        row[f"{method}_updated_{metric}_mean"] = round(arr.mean(), 2)
                        row[f"{method}_updated_{metric}_std"] = round(arr.std(), 2)
                        row[f"{method}_updated_{metric}_se"] = np.std(arr, ddof=1) / np.sqrt(np.size(arr))

                        delta_vals = data["delta_metrics"][metric].get(method, [])
                        if delta_vals:
                            darr = np.array(delta_vals)
                            row[f"{method}_delta_{metric}_mean"] = round(darr.mean(), 2)
                            row[f"{method}_delta_{metric}_se"] = np.std(darr, ddof=1) / np.sqrt(np.size(darr))

        if initial_metrics_lookup is not None and row["initial_set_size"] is not None:
            target_size = row["initial_set_size"] + row["budget"]

            if target_size in initial_metrics_lookup:
                default_metrics = initial_metrics_lookup[target_size]
                for metric in METRICS:
                    if f"initial_{metric}_mean" in default_metrics:
                        row[f"default_updated_{metric}_mean"] = default_metrics[f"initial_{metric}_mean"]
                        row[f"default_updated_{metric}_std"] = default_metrics[f"initial_{metric}_std"]
                        row[f"default_updated_{metric}_se"] = default_metrics[f"initial_{metric}_se"]

                        if metric == 'r2':
                            row[f"default_delta_{metric}_mean"] = default_metrics[f"initial_{metric}_mean"] - row[f"initial_{metric}_mean"]
                        else:
                            row[f"default_delta_{metric}_mean"] = row[f"initial_{metric}_mean"] - default_metrics[f"initial_{metric}_mean"]

                        row[f"default_delta_{metric}_se"] = default_metrics[f"initial_{metric}_se"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("budget")


def save_csv(df, dataset, init_set=None, cost_type=None, out_dir="results", multiple=False):
    os.makedirs(out_dir, exist_ok=True)

    name_parts = ["aggregated_metrics", dataset]
    if init_set:
        name_parts.append(str(init_set))
    if cost_type:
        name_parts.append(str(cost_type))

    filename = "_".join(name_parts) + ".csv"

    if multiple:
        out_dir = os.path.join(out_dir, 'multiple')
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, filename), index=False)


def read_initial_metrics_from_json(dataset, initial_set, base_dir=BASE_DIR,
                                   model='ridge', sampling_type='cluster_sampling',
                                   multiple=False, expected_seeds=None):
    path_parts = [base_dir, dataset, model]
    if multiple:
        path_parts.append("multiple")
    path_parts.extend([sampling_type, initial_set])
    initial_set_dir = os.path.join(*path_parts)

    if not os.path.exists(initial_set_dir):
        return {}

    metrics_by_seed = {metric: {} for metric in METRICS}

    for filename in os.listdir(initial_set_dir):
        if not filename.startswith("initial_set_metrics_seed_") or not filename.endswith(".json"):
            continue

        seed_str = filename.replace("initial_set_metrics_seed_", "").replace(".json", "")
        try:
            seed = int(seed_str)
        except ValueError:
            continue

        if expected_seeds is not None and seed not in expected_seeds:
            continue

        json_path = os.path.join(initial_set_dir, filename)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            for metric in METRICS:
                if metric in data:
                    metrics_by_seed[metric][seed] = data[metric]
        except Exception:
            pass

    result = {}
    for metric in METRICS:
        if metrics_by_seed[metric]:
            sorted_seeds = sorted(metrics_by_seed[metric].keys())
            result[metric] = [metrics_by_seed[metric][seed] for seed in sorted_seeds]

    return result


def collect_initial_metrics_for_dataset(dataset, initial_set_strs, PPC, NUM_STRATA,
                                       multiple=False, expected_seeds=None):
    initial_metrics_by_set = {}
    ppc = PPC[dataset]
    num_strata = NUM_STRATA[dataset]
    sampling_type = 'cluster_sampling'

    for size in range(0, 5100, 50):
        if size == 0:
            initial_set = "empty_initial_set"
        else:
            initial_set = initial_set_strs[dataset].format(num_strata=num_strata, ppc=ppc, size=size)

        try:
            metrics = read_initial_metrics_from_json(
                dataset, initial_set, multiple=multiple, expected_seeds=expected_seeds
            )
            if not metrics:
                continue

            if initial_set == "empty_initial_set":
                set_size = 0
            else:
                match = initial_set_size_re.search(initial_set)
                set_size = int(match.group(1)) if match else None

            if set_size is not None:
                key = (dataset, initial_set)
                if key not in initial_metrics_by_set:
                    initial_metrics_by_set[key] = {}
                if set_size not in initial_metrics_by_set[key]:
                    initial_metrics_by_set[key][set_size] = metrics
        except Exception:
            pass

    return initial_metrics_by_set


def build_initial_metrics_lookup(initial_metrics_by_set):
    initial_metrics_lookup = {}
    for key, size_data in initial_metrics_by_set.items():
        for size, metrics in size_data.items():
            if size not in initial_metrics_lookup:
                initial_metrics_lookup[size] = {}
            for metric in METRICS:
                if metric in metrics and len(metrics[metric]) > 0:
                    arr = np.array(metrics[metric])
                    initial_metrics_lookup[size][f"initial_{metric}_mean"] = round(arr.mean(), 2)
                    initial_metrics_lookup[size][f"initial_{metric}_std"] = round(arr.std(), 2)
                    initial_metrics_lookup[size][f"initial_{metric}_se"] = np.std(arr, ddof=1) / np.sqrt(len(arr))
    return initial_metrics_lookup


def process_dataset_configurations(dataset, initial_set_strs, PPC, NUM_STRATA,
                                   initial_metrics_lookup, multiple=False,
                                   alphas=None, budget_list=None):
    if alphas is None:
        alphas = [10]
    if budget_list is None:
        budget_list = range(100, 2100, 100)

    dataset_dfs = []
    multiple_dfs = []
    ppc = PPC[dataset]
    num_strata = NUM_STRATA[dataset]
    sampling_type = 'cluster_sampling'

    for size in range(0, 5100, 100):
        if size == 0:
            initial_set = "empty_initial_set"
        else:
            initial_set = initial_set_strs[dataset].format(num_strata=num_strata, ppc=ppc, size=size)

        for alpha in alphas:
            in_region_cost = ppc
            out_of_region_cost = in_region_cost + alpha
            cost_type = f"cluster_based_c1_{in_region_cost}_c2_{out_of_region_cost}"

            try:
                results = aggregate_results(
                    dataset=dataset, initial_set=initial_set, sampling_type=sampling_type,
                    cost_type=cost_type, budget_list=budget_list, multiple=multiple
                )

                keys = set((ds, iset, ctype) for (ds, iset, ctype, _) in results)

                for ds, iset, ctype in sorted(keys):
                    df = build_filtered_df(results, ds, iset, ctype, alpha, initial_metrics_lookup)
                    if df.empty:
                        continue

                    save_csv(df, ds, iset, ctype, multiple=multiple)
                    dataset_dfs.append(df)
                    multiple_dfs.append(df)

            except Exception:
                pass

        if dataset_dfs and not multiple:
            combined_df = pd.concat(dataset_dfs, ignore_index=True, join="outer")
            save_csv(combined_df, dataset, initial_set, multiple=multiple)
            dataset_dfs = []

    return dataset_dfs, multiple_dfs


def save_initial_metrics_summaries(dataset, initial_metrics_by_set, multiple=False):
    for metric in METRICS:
        sizes = []
        means = []
        stds = []
        ses = []

        for key, size_data in initial_metrics_by_set.items():
            for size, metrics in size_data.items():
                if metric in metrics and len(metrics[metric]) > 0:
                    arr = np.array(metrics[metric])
                    sizes.append(size)
                    means.append(round(arr.mean(), 2))
                    stds.append(round(arr.std(), 2))
                    ses.append(np.std(arr, ddof=1) / np.sqrt(len(arr)))

        if len(sizes) > 0:
            sorted_data = sorted(zip(sizes, means, stds, ses))
            sizes, means, stds, ses = zip(*sorted_data) if sorted_data else ([], [], [], [])

            summary_df = pd.DataFrame({
                'initial_set_size': sizes,
                f'initial_{metric}_mean': means,
                f'initial_{metric}_std': stds,
                f'initial_{metric}_se': ses
            })

            out_dir = "results/multiple" if multiple else "results"
            os.makedirs(out_dir, exist_ok=True)
            summary_df.to_csv(f"{out_dir}/{dataset}_initial_{metric}_summary.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run sampling evaluation script")
    parser.add_argument("--multiple", type=lambda x: x.lower() == 'true', default=False)
    multiple = parser.parse_args().multiple

    EXPECTED_SEEDS = set([1, 42, 123, 456, 789]) if multiple else set([1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415])

    initial_set_strs = {
        'USAVARS_POP': '{num_strata}_fixedstrata_{ppc}ppc_{size}_size',
        'USAVARS_TC': '{num_strata}_fixedstrata_{ppc}ppc_{size}_size',
        'INDIA_SECC': '{num_strata}_state_district_desired_{ppc}ppc_{size}_size',
        'TOGO_PH_H2O': '{num_strata}_strata_desired_{ppc}ppc_{size}_size'
    }

    PPC = {"USAVARS_POP": 10, "USAVARS_TC": 10, "INDIA_SECC": 20, "TOGO_PH_H2O": 25}
    NUM_STRATA = {"USAVARS_POP": 5, "USAVARS_TC": 5, "INDIA_SECC": 10, "TOGO_PH_H2O": 2}

    for dataset in DATASET_NAMES:
        initial_metrics_by_set = collect_initial_metrics_for_dataset(
            dataset, initial_set_strs, PPC, NUM_STRATA, multiple, expected_seeds=EXPECTED_SEEDS
        )
        initial_metrics_lookup = build_initial_metrics_lookup(initial_metrics_by_set)

        dataset_dfs, multiple_dfs = process_dataset_configurations(
            dataset, initial_set_strs, PPC, NUM_STRATA, initial_metrics_lookup, multiple
        )

        if multiple and multiple_dfs:
            combined_df = pd.concat(multiple_dfs, ignore_index=True, join="outer")
            save_csv(combined_df, dataset, multiple=multiple)

        save_initial_metrics_summaries(dataset, initial_metrics_by_set, multiple)
