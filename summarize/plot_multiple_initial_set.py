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
    "poprisk_states_0.5": "Admin-Rep",
    "poprisk_image_clusters_3_0.5": "Image-Rep"
}

plt.rcParams.update({'font.size': 24}) 

def format_r2(mean, se):
    return f"{mean:.2f} ± {se:.2f}" if pd.notnull(mean) and pd.notnull(se) else "--"

def plot_sample_cost_vs_r2(csv_path, initial_set_r2_csv_path, method_labels, init_set_lower_limit=None, init_set_upper_limit=None, budget_upper_limit=None, dataset_name=None, out_path="plot.png"):
    """
    Plots Sample Cost (budget + initial size) vs. R²:
    - One line per method
    - Transparent se band
    - Each method starts with a big X at its initial R²
    - Plots grey error band + line connecting initial R²s
    """
    df = pd.read_csv(csv_path)
    df_init = pd.read_csv(initial_set_r2_csv_path)  # has columns: initial_set_size, initial_r2_mean, initial_r2_se

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

    # Plot initial R² line and se band (from df_init)
    if not df_init.empty:
        sorted_init = df_init.sort_values("initial_set_size")

        if init_set_lower_limit is not None:
            sorted_init = sorted_init[sorted_init['initial_set_size'] >= init_set_lower_limit]

        if init_set_upper_limit is not None:
            sorted_init = sorted_init[sorted_init['initial_set_size'] <= init_set_upper_limit]

        x_init = sorted_init["initial_set_size"].values
        y_init = sorted_init["initial_r2_mean"].values
        y_se = sorted_init["initial_r2_se"].values

        plt.plot(x_init, y_init, color="grey", linestyle="--", linewidth=2, label="$R^2$ trend over multiple random initial samples")
        plt.fill_between(x_init, y_init - y_se, y_init + y_se, color="lightgrey", alpha=0.5)

    initial_points = df.groupby("initial_size").first().reset_index()
    if init_set_lower_limit is not None:
        initial_points = initial_points[initial_points['initial_size'] >= init_set_lower_limit]

    if init_set_upper_limit is not None:
        initial_points = initial_points[initial_points['initial_size'] < init_set_upper_limit]
    plt.scatter(initial_points["initial_size"], initial_points["initial_r2_mean"],
                color='black', marker='x', s=200, linewidths=3, label="Single initial sample")

    # Plot each method
    for method_key in method_names:
        mean_col = f"{method_key}_updated_r2_mean"
        se_col = f"{method_key}_updated_r2_se"
        if mean_col not in df.columns:
            continue

        first_plot = True
        for init_size in sorted(df["initial_size"].unique()):
            subset = df[df["initial_size"] == init_size].copy()
            if budget_upper_limit is not None:
                subset = subset[subset["budget"] <= budget_upper_limit]
            if init_set_lower_limit is not None:
                subset = subset[subset["initial_size"] >= init_set_lower_limit]
            if init_set_upper_limit is not None:
                subset = subset[subset["initial_size"] < init_set_upper_limit]
            if subset.empty:
                continue

            x = [init_size] + subset["sample_cost"].tolist()
            y = [subset["initial_r2_mean"].iloc[0]] + subset[mean_col].tolist()
            y_se = [0] + subset[se_col].tolist()

            label = f"$R^2$ trend over augmented samples" if first_plot else None #method_labels[method_key] if init_size == df["initial_size"].min() else None
            color = method_to_color[method_key]

            plt.plot(x, y, label=label, color=color, linewidth=2)
            plt.fill_between(x, np.array(y) - np.array(y_se), np.array(y) + np.array(y_se),
                             alpha=0.2, color=color)
            first_plot = False

    # Final plot polish
    plt.xlabel("Total Sample Cost", fontsize=24)
    plt.ylabel("$R^2$", fontsize=24)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_multiple.png", dpi=300)
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import re

def plot_stacked_augmented_r2_trends(csv_paths, initial_r2_paths, method_labels,
                                     budget_limits=None,
                                     init_set_lower_limits=None,
                                     init_set_upper_limits=None,
                                     output_path="stacked_plot.png"):
    """
    Creates a stacked plot with one subplot per dataset.
    
    Parameters:
    - csv_paths: dict mapping dataset_name -> csv_path
    - initial_r2_paths: dict mapping dataset_name -> initial_r2_csv_path
    - method_labels: dict of method labels
    - budget_limits: dict mapping dataset_name -> budget limit (optional)
    - init_set_lower_limits: dict mapping dataset_name -> lower limit (optional)
    - init_set_upper_limits: dict mapping dataset_name -> upper limit (optional)
    """
    
    num_datasets = len(csv_paths)
    fig, axes = plt.subplots(num_datasets, 1, figsize=(16, 6 * num_datasets),
                             sharex=False, sharey=False)

    if num_datasets == 1:
        axes = [axes]  # Make it iterable

    # Color map for methods
    method_names = list(method_labels.keys())
    color_map = cm.get_cmap("tab10", len(method_names))
    method_to_color = {m: color_map(0) for i, m in enumerate(method_names)}

    # Helper function to extract initial size
    def extract_size(init_str):
        match = re.search(r"(\d+)_size", init_str)
        return int(match.group(1)) if match else None

    for idx, (dataset_name, csv_path) in enumerate(csv_paths.items()):
        ax = axes[idx]
        
        # Load data for this dataset
        df = pd.read_csv(csv_path)
        df_init = pd.read_csv(initial_r2_paths[dataset_name])
        
        if df.empty:
            print(f"Empty CSV: {csv_path}")
            continue
            
        # Process the main dataframe
        df["initial_size"] = df["initial_set"].apply(extract_size)
        df["sample_cost"] = df["budget"] + df["initial_size"]
        
        # Get limits for this dataset
        budget_limit = budget_limits.get(dataset_name) if budget_limits else None
        init_lower = init_set_lower_limits.get(dataset_name) if init_set_lower_limits else None
        init_upper = init_set_upper_limits.get(dataset_name) if init_set_upper_limits else None
        
        # Apply filters
        if init_lower is not None:
            df = df[df["initial_size"] >= init_lower]
        if init_upper is not None:
            df = df[df["initial_size"] < init_upper]
        if budget_limit is not None:
            df = df[df["budget"] <= budget_limit]
            
        # Plot initial R² trend line and se band (from df_init)
        if not df_init.empty:
            sorted_init = df_init.sort_values("initial_set_size")
            
            if init_lower is not None:
                sorted_init = sorted_init[sorted_init['initial_set_size'] >= init_lower]
            if init_upper is not None:
                sorted_init = sorted_init[sorted_init['initial_set_size'] <= init_upper]
                
            x_init = sorted_init["initial_set_size"].values
            y_init = sorted_init["initial_r2_mean"].values
            y_se = sorted_init["initial_r2_se"].values

            ax.plot(x_init, y_init, color="grey", linestyle="--", linewidth=2, 
                   label="$R^2$ trend over multiple random initial samples")
            ax.fill_between(x_init, y_init - y_se, y_init + y_se, 
                           color="lightgrey", alpha=0.5)

        # Plot single initial sample points
        initial_points = df.groupby("initial_size").first().reset_index()
        ax.scatter(initial_points["initial_size"], initial_points["initial_r2_mean"],
                  color='black', marker='x', s=200, linewidths=3, 
                  label="Single initial sample")

        # Plot each method
        for method_key in METHOD_LABELS.keys():
            mean_col = f"{method_key}_updated_r2_mean"
            se_col = f"{method_key}_updated_r2_se"
            if mean_col not in df.columns:
                continue

            first_plot = True
            for init_size in sorted(df["initial_size"].unique()):
                subset = df[df["initial_size"] == init_size].copy()
                if budget_limit is not None:
                    subset = subset[subset["budget"] <= budget_limit]
                if init_lower is not None:
                    subset = subset[subset["initial_size"] >= init_lower]
                if init_upper is not None:
                    subset = subset[subset["initial_size"] < init_upper]
                if subset.empty:
                    continue

                x = [init_size] + subset["sample_cost"].tolist()
                y = [subset["initial_r2_mean"].iloc[0]] + subset[mean_col].tolist()
                y_se = [0] + subset[se_col].tolist()

                label = f"$R^2$ trend over augmented samples" if first_plot else None
                color = method_to_color[method_key]

                ax.plot(x, y, label=label, color=color, linewidth=2)
                ax.fill_between(x, np.array(y) - np.array(y_se), 
                               np.array(y) + np.array(y_se),
                               alpha=0.2, color=color)
                first_plot = False

        # Subplot formatting with (a), (b) labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes, 
                fontsize=24, fontweight='bold', va='top', ha='left')
        ax.set_ylabel("$R^2$", fontsize=24)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Only show legend on first subplot
        if idx == 1:
            ax.legend(fontsize=24, loc='lower right')

    # Set x-label only on bottom subplot
    axes[-1].set_xlabel("Total Sample Cost", fontsize=24)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# if __name__ == "__main__":
    # budget_limit = {
    #     "INDIA_SECC": 1200,
    #     "TOGO_PH_H2O": 600
    # }

    # init_set_lower_limit = {
    #     "INDIA_SECC": 2000,
    #     "TOGO_PH_H2O": 500
    # }
    # init_set_upper_limit = {
    #     "INDIA_SECC": 6000,
    #     "TOGO_PH_H2O": 3000
    # }

    # for dataset_name in ["INDIA_SECC", "TOGO_PH_H2O"]:
    #     csv_path = f"results/multiple/aggregated_r2_{dataset_name}.csv"
    #     initial_set_r2_csv_path = f"results/{dataset_name}_initial_r2_summary.csv"
    #     plot_sample_cost_vs_r2(csv_path, initial_set_r2_csv_path, METHOD_LABELS, dataset_name=dataset_name, init_set_lower_limit=init_set_lower_limit[dataset_name], init_set_upper_limit=init_set_upper_limit[dataset_name], budget_upper_limit=budget_limit[dataset_name])

# Example usage with your data structure:
if __name__ == "__main__":
    
    csv_paths = {
        "INDIA_SECC": "results/multiple/aggregated_r2_INDIA_SECC.csv",
        "TOGO_PH_H2O": "results/multiple/aggregated_r2_TOGO_PH_H2O.csv"
    }

    initial_r2_paths = {
        "INDIA_SECC": "results/INDIA_SECC_initial_r2_summary.csv",
        "TOGO_PH_H2O": "results/TOGO_PH_H2O_initial_r2_summary.csv"
    }

    budget_limits = {
        "INDIA_SECC": 1200,
        "TOGO_PH_H2O": 600
    }

    init_set_lower_limits = {
        "INDIA_SECC": 2000,
        "TOGO_PH_H2O": 500
    }

    init_set_upper_limits = {
        "INDIA_SECC": 6000,
        "TOGO_PH_H2O": 3000
    }

    # Create the stacked plot
    plot_stacked_augmented_r2_trends(
        csv_paths=csv_paths,
        initial_r2_paths=initial_r2_paths,
        method_labels=METHOD_LABELS,
        budget_limits=budget_limits,
        init_set_lower_limits=init_set_lower_limits,
        init_set_upper_limits=init_set_upper_limits,
        output_path="comparison_stacked.png"
    )