import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os

# AAAI column width in inches
COLUMNWIDTH_PT = 220
COLUMNWIDTH_IN = COLUMNWIDTH_PT / 72.27  # ≈ 3.31 inches

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams.update({
    "font.size": 10,        # main text / title
    "axes.titlesize": 10,
    "axes.labelsize": 10,    # axis labels slightly smaller
    "xtick.labelsize": 9,   # tick labels
    "ytick.labelsize": 9,
    "legend.fontsize": 9,   # legend
})


METHOD_LABELS = {
    "poprisk_states_0.5": "Admin-Rep",
    "poprisk_image_clusters_3_0.5": "Image-Rep"
}

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

        plt.plot(x_init, y_init, color="grey", linestyle="--", linewidth=1.5, label="$R^2$ trend over multiple random initial samples")
        plt.fill_between(x_init, y_init - y_se, y_init + y_se, color="lightgrey", alpha=0.5)

    initial_points = df.groupby("initial_size").first().reset_index()
    if init_set_lower_limit is not None:
        initial_points = initial_points[initial_points['initial_size'] >= init_set_lower_limit]

    if init_set_upper_limit is not None:
        initial_points = initial_points[initial_points['initial_size'] < init_set_upper_limit]
    plt.scatter(initial_points["initial_size"], initial_points["initial_r2_mean"],
                color='black', marker='x', s=75, linewidths=2, label="Single initial sample")

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

def break_axis_at_zero_x(ax, x1, x2, xmin=0):
    import matplotlib.patches as mpatches
    from matplotlib.transforms import blended_transform_factory
    """
    Break x-axis at zero - efficient version without fill_betweenx.
    
    Parameters:
    - ax: matplotlib axis
    - x1: start position of break in data coordinates
    - x2: end position of break in data coordinates  
    - xmin: minimum x value (default 0)
    """
    ax.set_xlim(xmin, None)
    
    d = 0.015  # size of break markers (in axes coordinates)
    
    # Use simple line markers instead of fill_betweenx
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                  linestyle="none", color='k', mew=1.5, clip_on=False,
                  transform=ax.transData)
    
    # Get the y-axis limits to position markers at the bottom
    ylim = ax.get_ylim()
    y_pos = ylim[0]  # Bottom of the plot in data coordinates
    
    # Add break markers at bottom
    ax.plot([x1], [y_pos], **kwargs, zorder=10)
    ax.plot([x2], [y_pos], **kwargs, zorder=10)
    
    # Add a small white rectangle patch at the axis to cover the gap
    # This is much more efficient than fill_betweenx
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    rect = mpatches.Rectangle((x1, 0), x2-x1, 0.02, 
                               transform=trans,
                               facecolor='white', 
                               edgecolor='none',
                               clip_on=False, 
                               zorder=9)
    ax.add_patch(rect)
    
    return ax

def plot_stacked_augmented_r2_trends(csv_paths, initial_r2_paths, method_labels,
                                     budget_limits=None,
                                     init_set_lists=None,
                                     r2_lower_bound=None,
                                     output_path="stacked_plot.png",
                                     layout="stacked"):
    """
    Creates a plot with one subplot per dataset.
    
    Parameters:
    - csv_paths: dict mapping dataset_name -> csv_path
    - initial_r2_paths: dict mapping dataset_name -> initial_r2_csv_path
    - method_labels: dict of method labels
    - budget_limits: dict mapping dataset_name -> budget limit (optional)
    - init_set_lists: dict mapping dataset_name -> list of initial set sizes (optional)
    - r2_lower_bound: dict mapping dataset_name -> lower y-axis bound (optional)
    - output_path: path to save the output figure
    - layout: "stacked" for vertical subplots (default) or "side_by_side" for horizontal subplots
    """
    
    num_datasets = len(csv_paths)
    
    # Create subplots based on layout preference
    if layout == "side_by_side":
        # side-by-side layout (horizontal)
        fig, axes = plt.subplots(1, num_datasets, figsize=(COLUMNWIDTH_IN*num_datasets, 2.5), 
                                sharex=False, sharey=False)
    else:  # stacked (default)
        # stacked layout (vertical)
        fig, axes = plt.subplots(num_datasets, 1, figsize=(COLUMNWIDTH_IN, 2.5*num_datasets), 
                                sharex=False, sharey=False)

    if num_datasets == 1:
        axes = [axes]  # Make it iterable

    # Color map for methods
    method_names = list(method_labels.keys())
    color_map = cm.get_cmap("tab10", len(method_names))
    method_to_color = {m: color_map(0) for i, m in enumerate(method_names)}

    # Helper function to extract initial size
    def extract_size(init_str):
        if init_str == "empty_initial_set":
            return 0
        match = re.search(r"(\d+)_size", init_str)
        return int(match.group(1)) if match else None

    for idx, (dataset_name, csv_path) in enumerate(csv_paths.items()):
        ax = axes[idx]
        #x2 = 400 if dataset_name == "TOGO_PH_H2O" else 1000
        #ax = break_axis_at_zero_x(ax, 10, x2)
        
        # Load data for this dataset
        df = pd.read_csv(csv_path)
        df_init = pd.read_csv(initial_r2_paths[dataset_name])
        
        if df.empty:
            print(f"Empty CSV: {csv_path}")
            continue
            
        # Load empty initial set data from separate CSV file
        # File naming: aggregated_metrics_{dataset_name}_empty_initial_set_cluster_based_c1_{c1}_c2_{c2}.csv
        if dataset_name == "INDIA_SECC":
            empty_csv_path = f"results/multiple/aggregated_metrics_{dataset_name}_empty_initial_set_cluster_based_c1_20_c2_30.csv"
        elif dataset_name == "TOGO_PH_H2O":
            empty_csv_path = f"results/multiple/aggregated_metrics_{dataset_name}_empty_initial_set_cluster_based_c1_25_c2_35.csv"
        else:
            empty_csv_path = None
        
        df_empty = pd.DataFrame()  # Initialize as empty
        if empty_csv_path and os.path.exists(empty_csv_path):
            df_empty = pd.read_csv(empty_csv_path)
            
        # Process the main dataframe
        df["initial_size"] = df["initial_set"].apply(extract_size)
        df["sample_cost"] = df["budget"] + df["initial_size"]
        
        # Get limits for this dataset
        budget_limit = budget_limits.get(dataset_name) if budget_limits else None
        init_set_list = init_set_lists.get(dataset_name) if init_set_lists else None
        
        # Apply filters
        if init_set_list is not None:
            df = df[df["initial_size"].isin(init_set_list)]
        if budget_limit is not None:
            df = df[df["budget"] <= budget_limit]
            if not df_empty.empty:
                df_empty = df_empty[df_empty["budget"] <= budget_limit]

        # Plot initial R² trend line and se band (from df_init)
        if not df_init.empty:
            sorted_init = df_init.sort_values("initial_set_size")
            
            # Get the full data before filtering for extension points
            x_init_full = sorted_init["initial_set_size"].values.tolist()
            y_init_full = sorted_init["initial_r2_mean"].values.tolist()
            y_se_full = sorted_init["initial_r2_se"].values.tolist()
            
            if init_set_list is not None:
                sorted_init = sorted_init[sorted_init['initial_set_size'].isin(init_set_list)]
            
            # Extend gray line to include additional points: 1000 for India, 250 for Togo
            x_init = sorted_init["initial_set_size"].values.tolist()
            y_init = sorted_init["initial_r2_mean"].values.tolist()
            y_se = sorted_init["initial_r2_se"].values.tolist()
            
            # Add extension points from full data if not already included
            if dataset_name == "INDIA_SECC" and 1000 not in x_init:
                if 1000 in x_init_full:
                    idx = x_init_full.index(1000)
                    x_init.insert(0, 1000)
                    y_init.insert(0, y_init_full[idx])
                    y_se.insert(0, y_se_full[idx])
            elif dataset_name == "TOGO_PH_H2O" and 250 not in x_init:
                if 250 in x_init_full:
                    idx = x_init_full.index(250)
                    x_init.insert(0, 250)
                    y_init.insert(0, y_init_full[idx])
                    y_se.insert(0, y_se_full[idx])
            
            x_init = np.array(x_init)
            y_init = np.array(y_init)
            y_se = np.array(y_se)

            ax.plot(x_init, y_init, color="grey", linestyle="--", linewidth=1.5, 
                   label="Trend (random init)") #$R^2$ trend over multiple random initial samples
            ax.fill_between(x_init, y_init - y_se, y_init + y_se, 
                           color="lightgrey", alpha=0.5)

        # Plot single initial sample points
        initial_points = df.groupby("initial_size").first().reset_index()
        ax.scatter(initial_points["initial_size"], initial_points["initial_r2_mean"],
                  color='black', marker='x', s=40, linewidths=1.5, 
                  label="Single initial sample")

        # Plot empty_initial_set augmentation (lines starting at 0)
        if not df_empty.empty:
            for method_key in METHOD_LABELS.keys():
                mean_col = f"{method_key}_updated_r2_mean"
                se_col = f"{method_key}_updated_r2_se"
                if mean_col not in df_empty.columns:
                    continue
                
                # Sort by budget for proper line plotting
                df_empty_sorted = df_empty.sort_values("budget")
                
                # Filter out NaN values
                df_empty_sorted = df_empty_sorted[df_empty_sorted[mean_col].notna()]
                
                if df_empty_sorted.empty:
                    continue
                
                x_empty = df_empty_sorted["budget"].tolist()  # Start at first budget, sample_cost = budget for empty
                y_empty = df_empty_sorted[mean_col].tolist()  # Start at first R² value
                y_se_empty = df_empty_sorted[se_col].fillna(0).tolist()
                
                color = method_to_color[method_key]
                ax.plot(x_empty, y_empty, color=color, linewidth=1.5)
                ax.fill_between(x_empty, np.array(y_empty) - np.array(y_se_empty), 
                               np.array(y_empty) + np.array(y_se_empty),
                               alpha=0.2, color=color)

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
                if init_set_list is not None:
                    subset = subset[subset["initial_size"].isin(init_set_list)]
                if subset.empty:
                    continue

                x = [init_size] + subset["sample_cost"].tolist()
                y = [subset["initial_r2_mean"].iloc[0]] + subset[mean_col].tolist()
                y_se = [0] + subset[se_col].tolist()

                label = f"Trend (augmented)" if first_plot else None #$R^2$ trend over augmented samples
                color = method_to_color[method_key]

                ax.plot(x, y, label=label, color=color, linewidth=1.5)
                ax.fill_between(x, np.array(y) - np.array(y_se), 
                               np.array(y) + np.array(y_se),
                               alpha=0.2, color=color)
                first_plot = False

        # Subplot formatting with (a), (b) labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes, 
                fontsize=10, fontweight='bold', va='top', ha='left')

        ax.set_ylabel("$R^2$", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Set y-axis limits: bottom=0.05, top varies by dataset
        y_bottom = 0.00
        if dataset_name == "INDIA_SECC":
            y_top = 0.375
        elif dataset_name == "TOGO_PH_H2O":
            y_top = 0.25
        else:
            y_top = 0.3  # default
        ax.set_ylim(bottom=y_bottom, top=y_top)
        
        # Set x-label for all subplots in side-by-side layout
        if layout == "side_by_side":
            ax.set_xlabel("Total Sample Cost", fontsize=10)
    
    # Legend positioning based on layout
    if layout == "side_by_side":
        # Place legend in bottom right corner of second (b) subplot (index 1)
        if len(axes) > 1:
            axes[1].legend(fontsize=8, loc='lower right', frameon=True,
                          handletextpad=0.5, handlelength=1.0, columnspacing=0.4)
        elif len(axes) == 1:
            axes[0].legend(fontsize=8, loc='lower right', frameon=True,
                          handletextpad=0.5, handlelength=1.0, columnspacing=0.4)
    else:
        # Show legend on second subplot for stacked (original behavior)
        if len(axes) > 1:
            axes[1].legend(fontsize=10, loc='lower right')
        elif len(axes) == 1:
            axes[0].legend(fontsize=10, loc='lower right')
    
    # Set x-label only on bottom subplot for stacked layout
    if layout == "stacked":
        axes[-1].set_xlabel("Total Sample Cost", fontsize=24)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
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
        "INDIA_SECC": "results/multiple/aggregated_metrics_INDIA_SECC.csv",
        "TOGO_PH_H2O": "results/multiple/aggregated_metrics_TOGO_PH_H2O.csv"
    }

    initial_r2_paths = {
        "INDIA_SECC": "results/INDIA_SECC_initial_r2_summary.csv",
        "TOGO_PH_H2O": "results/TOGO_PH_H2O_initial_r2_summary.csv"
    }

    budget_limits = {
        "INDIA_SECC": 1500,
        "TOGO_PH_H2O": 600
    }

    init_set_lists = {
        "INDIA_SECC": [2000, 3000, 4000, 5000],
        "TOGO_PH_H2O": [100, 500, 1000, 1500, 2000, 2500, 3000]
    }

    r2_lower_bound = {
        "INDIA_SECC": 0.05,
        "TOGO_PH_H2O": 0.05
    }

    # Create the plot (use layout="side_by_side" for horizontal subplots or layout="stacked" for vertical)
    plot_stacked_augmented_r2_trends(
        csv_paths=csv_paths,
        initial_r2_paths=initial_r2_paths,
        method_labels=METHOD_LABELS,
        budget_limits=budget_limits,
        init_set_lists=init_set_lists,
        r2_lower_bound=r2_lower_bound,
        output_path="comparison_stacked.pdf",
        layout="side_by_side"  # Change to "stacked" for vertical layout
    )