import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict

# === CONFIGURATION ===
LOG_FILENAME = "stdout.log"
BASE_DIR = "/home/libe2152/optimizedsampling/0_output"

# === REGEX ===
initial_r2_re = re.compile(r"Initial R² score: (-?[0-9.]+)")
updated_r2_re = re.compile(r"Updated R² score: (-?[0-9.]+)")

seed_re = re.compile(r"_seed_\d+")
size_re = re.compile(r"Selected (\d+) new samples")

# === METHOD LABELS FOR LATEX ===
METHOD_LABELS = {
    "random": "Random",
    "greedycost": "Greedy Low-Cost",
    "poprisk_nlcd_0.5": "PopRisk NLCD ($\\lambda = 0.5$)",
    "poprisk_mod_dists_from_top20_urban_tiers_0.5": "PopRisk Urban Tiers ($\\lambda = 0.5$)",
    "poprisk_dists_from_top20_urban_tiers_0.5": "PopRisk Urban Tiers ($\\lambda = 0.5$)",
    "poprisk_urban_rural_0.5": "PopRisk ($\\lambda = 0.5$)",
    "poprisk_regions_0.5": "PopRisk ($\\lambda = 0.5$)",
    "poprisk_regions_1.0": "PopRisk ($\\lambda = 1.0$)",
    "poprisk_0.1": "PopRisk ($\\lambda = 0.1$)",
    "similarity": "Similarity",
    "diversity": "Diversity",
    "poprisk_mod_nlcd_0.5": "PopRisk Mod NLCD ($\\lambda = 0.5$)",
    "poprisk_mod_nlcd_1.0": "PopRisk Mod NLCD ($\\lambda = 1.0$)"
}

# === FUNCTION DEFINITIONS ===

def parse_log_file(log_path, root):
    with open(log_path, "r") as f:
        content = f.read()

    initial_r2 = updated_r2 = size = None
    if (m := initial_r2_re.search(content)):
        initial_r2 = float(m.group(1))
    if (m := updated_r2_re.search(content)):
        updated_r2 = float(m.group(1))
    if (m := size_re.search(content)):
        size = float(m.group(1))

    path_parts = root.split("/")
    dataset_names = {"INDIA_SECC", "USAVARS_POP", "USAVARS_TC", "TOGO_PH_H2O", "TOGO_P_2022_JUL_DEC_P20"}  # add other dataset names here

    # Find dataset index dynamically
    dataset_idx = next((i for i, part in enumerate(path_parts) if part in dataset_names), None)
    if dataset_idx is None:
        raise ValueError("Dataset name not found in path_parts")

    dataset = path_parts[dataset_idx]

    # Example extraction assuming fixed offsets relative to dataset_idx
    seed = path_parts[-1].split("_")[1]

    # Now adjust following logic relative to dataset_idx
    if path_parts[dataset_idx + 1] == "empty_initial_set":
        init_set_base = "empty_initial_set"
        cost_type = path_parts[dataset_idx + 2]
        method_base_idx = dataset_idx + 3
    else:
        # Handle 'multiple' or 'multiple_cluster_sampling' presence relative to dataset_idx
        # Check if those keywords are anywhere after dataset_idx, or specifically at some offset
        post_dataset_parts = path_parts[dataset_idx + 1 : dataset_idx + 5]  # slice as needed
        if any(p in {"multiple", "multiple_cluster_sampling"} for p in post_dataset_parts):
            dataset += "_MULTIPLE"
            init_set_idx = dataset_idx + 3
        else:
            init_set_idx = dataset_idx + 2

        init_set_full = path_parts[init_set_idx]
        init_set_base = seed_re.sub("", init_set_full)

    if "opt" in path_parts:
        cost_type = path_parts[init_set_idx+1]
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
        cost_type = path_parts[init_set_idx + 1]
        method_base_idx = init_set_idx + 2
        method = path_parts[method_base_idx]
        if method == "match_population_proportion":
            budget = path_parts[method_base_idx + 2].replace("budget_", "")
        else:
            budget = path_parts[method_base_idx + 1].replace("budget_", "")

    key = (dataset, init_set_base, cost_type, budget)
    return key, method, size, initial_r2, updated_r2, seed

def aggregate_results(base_dir=BASE_DIR, log_filename=LOG_FILENAME):
    results = defaultdict(lambda: {"initial_r2": [], "methods": defaultdict(list), "seeds": set(), "size": defaultdict(list)})

    for root, dirs, files in os.walk(base_dir):
        if log_filename in files:
            log_path = os.path.join(root, log_filename)
            try:
                key, method, size, initial_r2, updated_r2, seed = parse_log_file(log_path, root)
            except Exception as e:
                print(f"Failed to parse {log_path}: {e}")
                continue

            if initial_r2 is not None and seed not in results[key]["seeds"]:
                results[key]["initial_r2"].append(initial_r2)
                results[key]["seeds"].add(seed)
            if updated_r2 is not None:
                results[key]["methods"][method].append(updated_r2)
            if size is not None:
                results[key]["size"][method].append(size)
    return results

def build_filtered_df(results_dict, dataset, init_set, cost_type):
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

        if len(data["initial_r2"]) < 10 and row["dataset"] != "INDIA_SECC_MULTIPLE":
            continue

        arr = np.array(data["initial_r2"])
        row["initial_r2_mean"] = round(arr.mean(), 2)
        row["initial_r2_std"] = round(arr.std(), 2)

        for method, vals in data["methods"].items():
            if vals:
                arr = np.array(vals)
                if len(arr) < 5 and row["dataset"] != "INDIA_SECC_MULTIPLE":
                    continue
                row[f"{method}_updated_r2_mean"] = round(arr.mean(), 2)
                row[f"{method}_updated_r2_std"] = round(arr.std(), 2)

        rows.append(row)
    df = pd.DataFrame(rows).sort_values("budget")
    return df

def format_r2(mean, std):
    return f"{mean:.2f} ± {std:.2f}" if pd.notnull(mean) and pd.notnull(std) else "--"

def generate_latex_table(df, method_labels, dataset, init_set, cost_type):
    def prettify_name(s): return s.replace("_", " ").title()
    def label_for_init(init): return {
        "cluster_sampling": "Cluster Sampling",
        "empty_initial_set": "No Initial Set",
    }.get(init, "Initial Setting")
    
    def detail_for_init(init): return {
        "cluster_sampling": "$k$ Points Per Cluster; Total size $M$",
        "empty_initial_set": "No Initial Set",
    }.get(init, prettify_name(init))

    first_row = df.iloc[0]
    init_r2 = format_r2(first_row.get("initial_r2_mean"), first_row.get("initial_r2_std"))
    init_label = label_for_init(init_set)
    init_detail = detail_for_init(init_set)
    multirow_n = len(df)
    left_col_text = f"\\multirow{{{multirow_n}}}{{*}}{{\\shortstack[l]{{{init_label} \\\\ {init_detail} \\\\ (Init $R^2$ = {init_r2})}}}}"

    lines = []
    lines.append("\\begin{table}[t!]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{l" + "c" * (1 + len(method_labels)) + "}%" )
    lines.append("\\hline%")

    available_methods = [m for m in method_labels if f"{m}_updated_r2_mean" in df.columns]
    method_names = [f"\\multicolumn{{1}}{{c}}{{{method_labels[m]}}}" for m in available_methods]

    lines.append("Problem Setting & Budget & " + "&".join(method_names) + "\\\\%")
    lines.append(" &  & " + " & ".join(["($R^2$ ± Std)"] * len(method_labels)) + "\\\\%")
    lines.append("\\hline%")

    for i, (_, row) in enumerate(df.iterrows()):
        budget = int(row["budget"])
        cells = [
            format_r2(row.get(f"{m}_updated_r2_mean"), row.get(f"{m}_updated_r2_std"))
            for m in method_labels
        ]
        if i == 0:
            lines.append(f"{left_col_text} & {budget} & " + " & ".join(cells) + "\\\\%")
        else:
            lines.append(f"& {budget} & " + " & ".join(cells) + "\\\\%")

    lines.append("\\hline%")
    lines.append("\\end{tabular}%")
    lines.append(f"\\caption{{Updated $R^2$ for {dataset.upper()} with initial set \\texttt{{{init_set}}} and cost \\texttt{{{cost_type}}}.}}")
    lines.append(f"\\label{{tab:{dataset}_{init_set}_{cost_type}}}")
    lines.append("\\end{table}")

    tex_path = f"latex_table_r2/latex_table_{dataset}_{init_set}_{cost_type}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"📄 LaTeX table written to: {tex_path}")


def save_csv(df, dataset, init_set, cost_type):
    path = f"aggregated_r2/aggregated_r2_{dataset}_{init_set}_{cost_type}.csv"
    df.to_csv(path, index=False)
    print(f"📊 CSV saved to: {path}")

def generate_size_latex_table(results_dict, method_labels, dataset, init_set, cost_type):
    rows = []
    for (ds, iset, ctype, budget), data in results_dict.items():
        if (ds, iset, ctype) != (dataset, init_set, cost_type):
            continue

        row = {
            "budget": int(budget),
        }

        for method, vals in data["size"].items():
            if vals:
                arr = np.array(vals)
                row[f"{method}_size_mean"] = round(arr.mean(), 1)
                row[f"{method}_size_std"] = round(arr.std(), 1)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("budget")
    if df.empty:
        print(f"No size data found for: {dataset}, {init_set}, {cost_type}")
        return

    # Generate LaTeX table
    lines = []
    lines.append("\\begin{table}[t!]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{l" + "c" * len(method_labels) + "}%" )
    lines.append("\\hline%")

    method_names = [f"\\multicolumn{{1}}{{c}}{{{label}}}" for label in method_labels.values()]
    lines.append("Budget & " + " & ".join(method_names) + "\\\\%")
    lines.append(" & " + " & ".join(["(Size ± Std)"] * len(method_labels)) + "\\\\%")
    lines.append("\\hline%")

    for _, row in df.iterrows():
        budget = int(row["budget"])
        cells = []
        for method in method_labels:
            mean = row.get(f"{method}_size_mean")
            std = row.get(f"{method}_size_std")
            if pd.notnull(mean) and pd.notnull(std):
                cells.append(f"{mean:.1f} ± {std:.1f}")
            else:
                cells.append("--")
        lines.append(f"{budget} & " + " & ".join(cells) + "\\\\%")

    lines.append("\\hline%")
    lines.append("\\end{tabular}%")
    lines.append(f"\\caption{{Average number of selected samples for {dataset.upper()} with initial set \\texttt{{{init_set}}} and cost \\texttt{{{cost_type}}}.}}")
    lines.append(f"\\label{{tab:{dataset}_{init_set}_{cost_type}_sizes}}")
    lines.append("\\end{table}")

    tex_path = f"latex_table_sizes/latex_table_sizes_{dataset}_{init_set}_{cost_type}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"📄 Size LaTeX table written to: {tex_path}")


if __name__ == "__main__":
    results = aggregate_results()

    # Group all available (dataset, initial_set, cost_type)
    keys = set((ds, iset, ctype) for (ds, iset, ctype, _) in results)

    for dataset, init_set, cost_type in sorted(keys):
        try:
            df = build_filtered_df(results, dataset, init_set, cost_type)
            if df.empty:
                print(f"Skipping empty table for: {dataset}, {init_set}, {cost_type}")
                continue
            save_csv(df, dataset, init_set, cost_type)
            generate_latex_table(df, METHOD_LABELS, dataset, init_set, cost_type)
            generate_size_latex_table(results, METHOD_LABELS, dataset, init_set, cost_type)
        except Exception as e:
            print(e)
