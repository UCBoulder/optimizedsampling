import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 24}) 

METHOD_LABELS = {
    # "random": "Random",
    "poprisk_urban_rural_0.5": "PopRisk ($\\lambda = 0.5$)",
    #"poprisk_regions_0.5": "PopRisk ($\\lambda = 0.5$)",
    "poprisk_image_clusters_3_0.5": "PopRisk Img ($\\lambda = 0.5$)"
    # Add other methods here if needed
}

plt.rcParams.update({'font.size': 24}) 

def format_r2(mean, std):
    return f"{mean:.2f} ± {std:.2f}" if pd.notnull(mean) and pd.notnull(std) else "--"

def plot_sample_cost_vs_r2(csv_path, initial_set_r2_csv_path, method_labels, init_set_lower_limit=None, budget_upper_limit=None, dataset_name=None, out_path="plot.png"):
    """
    Plots Sample Cost (budget + initial size) vs. R²:
    - One line per method
    - Transparent std band
    - Each method starts with a big X at its initial R²
    - Plots grey error band + line connecting initial R²s
    """
    df = pd.read_csv(csv_path)
    df_init = pd.read_csv(initial_set_r2_csv_path)  # has columns: initial_set_size, initial_r2_mean, initial_r2_std

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
    plt.figure(figsize=(16, 8))

    # Plot initial R² line and std band (from df_init)
    if not df_init.empty:
        sorted_init = df_init.sort_values("initial_set_size")

        if init_set_lower_limit is not None:
            sorted_init = sorted_init[sorted_init['initial_set_size'] >= init_set_lower_limit]

        x_init = sorted_init["initial_set_size"].values
        y_init = sorted_init["initial_r2_mean"].values
        y_std = sorted_init["initial_r2_std"].values

        plt.plot(x_init, y_init, color="grey", linestyle="--", linewidth=2, label="Initial $R^2$ Trend")
        plt.fill_between(x_init, y_init - y_std, y_init + y_std, color="lightgrey", alpha=0.5)

    initial_points = df.groupby("initial_size").first().reset_index()
    if init_set_lower_limit is not None:
        initial_points = initial_points[initial_points['initial_size'] >= init_set_lower_limit]
    plt.scatter(initial_points["initial_size"][:-1], initial_points["initial_r2_mean"][:-1],
                color='black', marker='x', s=200, linewidths=3, label="Initial $R^2$")

    # Plot each method
    for method_key in method_names:
        mean_col = f"{method_key}_updated_r2_mean"
        std_col = f"{method_key}_updated_r2_std"
        if mean_col not in df.columns:
            continue

        for init_size in sorted(df["initial_size"].unique()):
            subset = df[df["initial_size"] == init_size].copy()
            if budget_upper_limit is not None:
                subset = subset[subset["budget"] <= budget_upper_limit]
            if init_set_lower_limit is not None:
                subset = subset[subset["initial_size"] >= init_set_lower_limit]
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
    plt.xlabel("Sample Cost", fontsize=24)
    plt.ylabel("Updated $R^2$", fontsize=24)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_multiple.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    budget_limit = {
        "INDIA_SECC": 1200,
        "TOGO_PH_H2O": 600
    }

    init_set_limit = {
        "INDIA_SECC": 2000,
        "TOGO_PH_H2O": 500
    }
    for dataset_name in ["INDIA_SECC", "TOGO_PH_H2O"]:
        csv_path = f"results/multiple/aggregated_r2_{dataset_name}.csv"
        initial_set_r2_csv_path = f"results/{dataset_name}_initial_r2_summary.csv"
        plot_sample_cost_vs_r2(csv_path, initial_set_r2_csv_path, METHOD_LABELS, dataset_name=dataset_name, init_set_lower_limit=init_set_limit[dataset_name], budget_upper_limit=budget_limit[dataset_name])