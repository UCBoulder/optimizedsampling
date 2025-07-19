import os
import re
import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict
from pylatex import Document, Section, Tabular, MultiColumn, MultiRow, NoEscape

base_dir = "/home/libe2152/optimizedsampling/0_output"
log_filename = "stdout.log"

# Regex patterns
initial_r2_re = re.compile(r"Initial R² score: ([0-9.]+)")
updated_r2_re = re.compile(r"Updated R² score: ([0-9.]+)")
seed_re = re.compile(r"_seed_\d+")

# Data structure
results = defaultdict(lambda: {"initial_r2": [], "methods": defaultdict(list)})

# Walk all log files
for root, dirs, files in os.walk(base_dir):
    if log_filename in files:
        log_path = os.path.join(root, log_filename)

        with open(log_path, "r") as f:
            content = f.read()

        # Parse R² scores
        initial_r2_match = initial_r2_re.search(content)
        updated_r2_match = updated_r2_re.search(content)

        initial_r2 = float(initial_r2_match.group(1)) if initial_r2_match else None
        updated_r2 = float(updated_r2_match.group(1)) if updated_r2_match else None

        # Parse path elements
        path_parts = root.split('/')
        dataset = path_parts[5]
        init_set_full = path_parts[6]

        # Normalize initial set name (remove _seed_#)
        init_set_base = seed_re.sub("", init_set_full)

        if "cost_aware" in path_parts:
            idx = path_parts.index("cost_aware")
            method = path_parts[idx + 1]
            budget = path_parts[idx + 2].replace("budget_", "")
        else:
            idx = len(path_parts) - 3
            method = path_parts[idx]
            budget = path_parts[idx + 1].replace("budget_", "")

        key = (dataset, init_set_base, budget)

        # Collect results
        if initial_r2 is not None:
            results[key]["initial_r2"].append(initial_r2)
        if updated_r2 is not None:
            results[key]["methods"][method].append(updated_r2)

# Construct output table
rows = []
for (dataset, init_set_base, budget), data in results.items():
    row = {
        "dataset": dataset,
        "initial_set": init_set_base,
        "budget": int(budget),
    }

    if data["initial_r2"]:
        initial_r2_vals = np.array(data["initial_r2"])
        row["initial_r2_mean"] = round(initial_r2_vals.mean(), 2)
        row["initial_r2_se"] = round(initial_r2_vals.std(ddof=1) / np.sqrt(len(initial_r2_vals)), 2)

    for method, r2_vals in data["methods"].items():
        if r2_vals:
            r2_array = np.array(r2_vals)
            row[f"{method}_updated_r2_mean"] = round(r2_array.mean(), 2)
            row[f"{method}_updated_r2_se"] = round(r2_array.std(ddof=1) / np.sqrt(len(r2_array)), 2)

    rows.append(row)

# Save CSV
df = pd.DataFrame(rows)
df.sort_values(by=["dataset", "initial_set", "budget"], inplace=True)

output_path = "aggregated_r2_by_initialset_with_ste.csv"
df.to_csv(output_path, index=False)
print(f"✅ Saved to {output_path}")

df = pd.read_csv("aggregated_r2_by_initialset_with_ste.csv")
method_labels = {
    "random": "Random",
    "poprisk": "PopRisk ($\\lambda = 0.5$)",
    "matchpopprop": "Proportional Stratified",
    "stratified": "Stratified"
}


def format_r2(mean, se):
    if pd.notnull(mean) and pd.notnull(se):
        return f"{mean:.2f} ± {se:.2f}"
    else:
        return "--"

for dataset_name, group_df in df.groupby("dataset"):
    lines = []

    lines.append("\\begin{table*}[t!]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{ll" + "c" * len(method_labels) + "}%" )
    lines.append("\\hline%")
    
    # Header rows
    method_names = [f"\\multicolumn{{1}}{{c}}{{{name}}}" for name in method_labels.values()]
    lines.append("\\multirow{2}{*}{Initial Set}&\\multirow{2}{*}{Budget}&" + "&".join(method_names) + "\\\\%")
    lines.append("&&" + "&".join(["($R^2$ ± SE)"] * len(method_labels)) + "\\\\%")
    lines.append("\\hline%")

    for init_set, init_df in group_df.groupby("initial_set"):
        init_df = init_df.sort_values(by="budget")
        init_r2 = init_df.iloc[0].get("initial_r2_mean", None)
        init_se = init_df.iloc[0].get("initial_r2_se", None)

        # format multirow header
        init_r2_text = f"(Init $R^2$ = {init_r2:.4f} ± {init_se:.4f})" if pd.notnull(init_r2) else "--"
        init_label = init_set.replace("_", " ").title()
        multirow_label = f"\\multirow{{{len(init_df)}}}{{*}}{{\\shortstack[l]{{{init_label}\\\\{init_r2_text}}}}}"

        for i, (_, row) in enumerate(init_df.iterrows()):
            budget = int(row["budget"])
            r2_values = [format_r2(row.get(f"{m}_updated_r2_mean"), row.get(f"{m}_updated_r2_se")) for m in method_labels]
            line = f"{multirow_label if i == 0 else ''}&{budget}&" + "&".join(r2_values) + "\\\\%"
            lines.append(line)

        lines.append("\\hline%")

    lines.append("\\end{tabular}%")
    lines.append(f"\\caption{{Updated $R^2$ scores for the {dataset_name.upper()} Population dataset across budgets and initial sets. Initial $R^2$ shown under each setting.}}")
    lines.append(f"\\label{{{{tab:{dataset_name.lower()}_pop_r2}}}}")
    lines.append("\\end{table*}")

    tex_path = f"output_latex_table_{dataset_name}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ Saved LaTeX code for dataset '{dataset_name}' to {tex_path}")
