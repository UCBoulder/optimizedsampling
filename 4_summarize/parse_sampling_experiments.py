import os
import re
import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict

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
        row["initial_r2_mean"] = round(initial_r2_vals.mean(), 4)
        row["initial_r2_se"] = round(initial_r2_vals.std(ddof=1) / np.sqrt(len(initial_r2_vals)), 4)

    for method, r2_vals in data["methods"].items():
        if r2_vals:
            r2_array = np.array(r2_vals)
            row[f"{method}_updated_r2_mean"] = round(r2_array.mean(), 4)
            row[f"{method}_updated_r2_se"] = round(r2_array.std(ddof=1) / np.sqrt(len(r2_array)), 4)

    rows.append(row)

# Save CSV
df = pd.DataFrame(rows)
df.sort_values(by=["dataset", "initial_set", "budget"], inplace=True)

output_path = "aggregated_r2_by_initialset_with_std.csv"
df.to_csv(output_path, index=False)
print(f"✅ Saved to {output_path}")
