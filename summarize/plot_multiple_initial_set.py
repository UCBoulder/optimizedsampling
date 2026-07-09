import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os

COLUMNWIDTH_PT = 220
COLUMNWIDTH_IN = COLUMNWIDTH_PT / 72.27

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


METHOD_LABELS = {
    "poprisk_states_0.5": "Admin-Rep",
    "poprisk_image_clusters_3_0.5": "Image-Rep"
}

def format_r2(mean, se):
    return f"{mean:.2f} ± {se:.2f}" if pd.notnull(mean) and pd.notnull(se) else "--"


def plot_stacked_augmented_r2_trends(csv_paths, initial_r2_paths, method_labels,
                                     budget_limits=None,
                                     init_set_lists=None,
                                     output_path="stacked_plot.png",
                                     layout="stacked"):
    num_datasets = len(csv_paths)

    if layout == "side_by_side":
        fig, axes = plt.subplots(1, num_datasets, figsize=(COLUMNWIDTH_IN*num_datasets, 2.5),
                                sharex=False, sharey=False)
    else:
        fig, axes = plt.subplots(num_datasets, 1, figsize=(COLUMNWIDTH_IN, 2.5*num_datasets),
                                sharex=False, sharey=False)

    if num_datasets == 1:
        axes = [axes]

    method_names = list(method_labels.keys())
    color_map = cm.get_cmap("tab10", len(method_names))
    method_to_color = {m: color_map(0) for i, m in enumerate(method_names)}

    def extract_size(init_str):
        if init_str == "empty_initial_set":
            return 0
        match = re.search(r"(\d+)_size", init_str)
        return int(match.group(1)) if match else None

    for idx, (dataset_name, csv_path) in enumerate(csv_paths.items()):
        ax = axes[idx]

        df = pd.read_csv(csv_path)
        df_init = pd.read_csv(initial_r2_paths[dataset_name])

        if df.empty:
            continue

        if dataset_name == "INDIA_SECC":
            empty_csv_path = f"results/multiple/aggregated_metrics_{dataset_name}_empty_initial_set_cluster_based_c1_20_c2_30.csv"
        elif dataset_name == "TOGO_PH_H2O":
            empty_csv_path = f"results/multiple/aggregated_metrics_{dataset_name}_empty_initial_set_cluster_based_c1_25_c2_35.csv"
        else:
            empty_csv_path = None

        df_empty = pd.DataFrame()
        if empty_csv_path and os.path.exists(empty_csv_path):
            df_empty = pd.read_csv(empty_csv_path)

        df["initial_size"] = df["initial_set"].apply(extract_size)
        df["sample_cost"] = df["budget"] + df["initial_size"]

        budget_limit = budget_limits.get(dataset_name) if budget_limits else None
        init_set_list = init_set_lists.get(dataset_name) if init_set_lists else None

        if init_set_list is not None:
            df = df[df["initial_size"].isin(init_set_list)]
        if budget_limit is not None:
            df = df[df["budget"] <= budget_limit]
            if not df_empty.empty:
                df_empty = df_empty[df_empty["budget"] <= budget_limit]

        if not df_init.empty:
            sorted_init = df_init.sort_values("initial_set_size")

            x_init_full = sorted_init["initial_set_size"].values.tolist()
            y_init_full = sorted_init["initial_r2_mean"].values.tolist()
            y_se_full = sorted_init["initial_r2_se"].values.tolist()

            if init_set_list is not None:
                sorted_init = sorted_init[sorted_init['initial_set_size'].isin(init_set_list)]

            x_init = sorted_init["initial_set_size"].values.tolist()
            y_init = sorted_init["initial_r2_mean"].values.tolist()
            y_se = sorted_init["initial_r2_se"].values.tolist()

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
                   label="Trend (random init)")
            ax.fill_between(x_init, y_init - y_se, y_init + y_se,
                           color="lightgrey", alpha=0.5)

        initial_points = df.groupby("initial_size").first().reset_index()
        ax.scatter(initial_points["initial_size"], initial_points["initial_r2_mean"],
                  color='black', marker='x', s=40, linewidths=1.5,
                  label="Single initial sample")

        if not df_empty.empty:
            for method_key in METHOD_LABELS.keys():
                mean_col = f"{method_key}_updated_r2_mean"
                se_col = f"{method_key}_updated_r2_se"
                if mean_col not in df_empty.columns:
                    continue

                df_empty_sorted = df_empty.sort_values("budget")
                df_empty_sorted = df_empty_sorted[df_empty_sorted[mean_col].notna()]

                if df_empty_sorted.empty:
                    continue

                x_empty = df_empty_sorted["budget"].tolist()
                y_empty = df_empty_sorted[mean_col].tolist()
                y_se_empty = df_empty_sorted[se_col].fillna(0).tolist()

                color = method_to_color[method_key]
                ax.plot(x_empty, y_empty, color=color, linewidth=1.5)
                ax.fill_between(x_empty, np.array(y_empty) - np.array(y_se_empty),
                               np.array(y_empty) + np.array(y_se_empty),
                               alpha=0.2, color=color)

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

                label = "Trend (augmented)" if first_plot else None
                color = method_to_color[method_key]

                ax.plot(x, y, label=label, color=color, linewidth=1.5)
                ax.fill_between(x, np.array(y) - np.array(y_se),
                               np.array(y) + np.array(y_se),
                               alpha=0.2, color=color)
                first_plot = False

        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='left')

        ax.set_ylabel("$R^2$", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        y_bottom = 0.00
        if dataset_name == "INDIA_SECC":
            y_top = 0.375
        elif dataset_name == "TOGO_PH_H2O":
            y_top = 0.25
        else:
            y_top = 0.3
        ax.set_ylim(bottom=y_bottom, top=y_top)

        if layout == "side_by_side":
            ax.set_xlabel("Total Sample Cost", fontsize=10)

    if layout == "side_by_side":
        if len(axes) > 1:
            axes[1].legend(fontsize=8, loc='lower right', frameon=True,
                          handletextpad=0.5, handlelength=1.0, columnspacing=0.4)
        elif len(axes) == 1:
            axes[0].legend(fontsize=8, loc='lower right', frameon=True,
                          handletextpad=0.5, handlelength=1.0, columnspacing=0.4)
    else:
        if len(axes) > 1:
            axes[1].legend(fontsize=10, loc='lower right')
        elif len(axes) == 1:
            axes[0].legend(fontsize=10, loc='lower right')

    if layout == "stacked":
        axes[-1].set_xlabel("Total Sample Cost", fontsize=24)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

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

    plot_stacked_augmented_r2_trends(
        csv_paths=csv_paths,
        initial_r2_paths=initial_r2_paths,
        method_labels=METHOD_LABELS,
        budget_limits=budget_limits,
        init_set_lists=init_set_lists,
        output_path="comparison_stacked.pdf",
        layout="side_by_side"
    )
