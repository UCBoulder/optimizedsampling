import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

METHOD_LABELS = {
    "random": "Random",
    "poprisk_urban_rural_0.5": "PopRisk ($\\lambda = 0.5$)",
    # Add other methods here if needed
}

plt.rcParams.update({'font.size': 24}) 

def format_r2(mean, std):
    return f"{mean:.2f} ± {std:.2f}" if pd.notnull(mean) and pd.notnull(std) else "--"

def plot_sample_cost_vs_r2(csv_path, method_labels, budget_limit=None, out_path="plot.png"):
    """
    Plots Sample Cost (budget + initial size) vs. R²:
    - One line per method
    - Transparent std band
    - Each method starts with a big X at its initial R²
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Empty CSV: {csv_path}")
        return

    # Parse metadata from filename
    basename = os.path.basename(csv_path).replace(".csv", "")
    _, dataset, init_set, cost_type = basename.split("_", maxsplit=3)

    # Extract initial size from initial_set field
    def extract_size(init_str):
        match = re.search(r"(\d+)_size", init_str)
        return int(match.group(1)) if match else None

    df["initial_size"] = df["initial_set"].apply(extract_size)
    df["sample_cost"] = df["budget"] + df["initial_size"]

    # Color map for methods
    method_names = list(method_labels.keys())
    color_map = cm.get_cmap("tab10", len(method_names))
    method_to_color = {m: color_map(i) for i, m in enumerate(method_names)}

    # Start plot
    plt.figure(figsize=(20, 12))

    # Plot initial R² Xs
    initial_points = df.groupby("initial_size").first().reset_index()
    plt.scatter(initial_points["initial_size"], initial_points["initial_r2_mean"],
                color='black', marker='x', s=200, linewidths=3, label="Initial $R^2$")

    # Plot each method
    for method_key in method_names:
        mean_col = f"{method_key}_updated_r2_mean"
        std_col = f"{method_key}_updated_r2_std"
        if mean_col not in df.columns:
            continue

        for init_size in sorted(df["initial_size"].unique()):
            subset = df[df["initial_size"] == init_size].copy()
            if budget_limit is not None:
                subset = subset[subset["budget"] <= budget_limit]
            if subset.empty:
                continue

            x = [init_size] + subset["sample_cost"].tolist()
            y = [subset["initial_r2_mean"].iloc[0]] + subset[mean_col].tolist()
            y_std = [0] + subset[std_col].tolist()

            label = method_labels[method_key] if init_size == df["initial_size"].min() else None
            color = method_to_color[method_key]

            plt.plot(x, y, label=label, color=color, linewidth=2)
            plt.fill_between(x, np.array(y) - np.array(y_std), np.array(y) + np.array(y_std),
                             alpha=0.2, color=color)

    # Final plot polish
    plt.xlabel("Sample Cost")
    plt.ylabel("Updated $R^2$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("india_secc_multiple.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    csv_path = "aggregated_r2/aggregated_r2_INDIA_SECC_MULTIPLE_cluster_based_c1_20_c2_30_opt.csv"
    plot_sample_cost_vs_r2(csv_path, METHOD_LABELS, budget_limit=1200)