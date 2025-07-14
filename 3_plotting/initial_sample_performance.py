import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Color Universal Design (CUD) palette for colorblind-friendly consistency
SAMPLING_TYPE_COLORS = {
    "Cluster": "#E69F00",       # orange
    "Convenience Probabilistic": "#56B4E9",   # sky blue
    "Random": "#009E73",        # green
    "Convenience Deterministic": "#F0E442",  # yellow
    # "": "#0072B2",      # blue
    # "": "#D55E00",       # vermillion
}

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_r2_boxplot_across_methods(csv_dict, sample_size, save_path=None, rotation=30):
    """
    Plot R² boxplots across sampling methods using a dictionary of CSV paths,
    using predefined colorblind-friendly colors from SAMPLING_TYPE_COLORS.

    Args:
        csv_dict (dict): Keys are sampling type labels (e.g., "Cluster"), values are CSV file paths.
        sample_size (int): Sample size used in the experiments.
        save_path (str, optional): If provided, saves the plot to this path.
        rotation (int): Rotation angle for x-axis labels (default 30).
    """
    # Combine and label data
    dfs = []
    for label, path in csv_dict.items():
        try:
            df = pd.read_csv(path)
            if 'r2' not in df.columns:
                raise ValueError(f"'r2' column not found in {path}")
            df['sampling_type'] = label
            dfs.append(df)
        except Exception as e:
            print(f"[Error] Could not read {path}: {e}")

    if not dfs:
        print("No data to plot.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    # Pull consistent colors from global color map
    palette = {label: SAMPLING_TYPE_COLORS.get(label, "#CCCCCC") for label in csv_dict.keys()}

    # Plotting setup
    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    plt.figure(figsize=(8, 6), dpi=300)

    ax = sns.boxplot(
        data=combined_df,
        x='sampling_type',
        y='r2',
        palette=palette,
        width=0.6,
        linewidth=1.5,
        fliersize=3,
        boxprops=dict(alpha=0.9),
    )

    # Styling
    ax.set_title(f"R² Scores by Sampling Method\n(Sample Size = {sample_size})", fontsize=14, pad=15)
    ax.set_xlabel("Sampling Method", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)

    # Rotate x labels with alignment to avoid overlap
    ax.tick_params(axis='x', labelrotation=45, labelsize=9)

    ax.set_ylim(bottom=-0.1)

    sns.despine(offset=10, trim=True)
    ax.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    for label in ['population', 'treecover']:
        for sample_size in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            for points_per_cluster in [2, 5, 10, 25]:
                plot_r2_boxplot_across_methods(
                csv_dict={
                    "Cluster": f"/home/libe2152/optimizedsampling/0_results/usavars/{label}/summaries/filtered_cluster_sampling_sample_size_{sample_size}_ppc_{points_per_cluster}.csv",
                    "Convenience\nProbabilistic": f"/home/libe2152/optimizedsampling/0_results/usavars/{label}/summaries/filtered_convenience_sampling_probabilistic_sample_size_{sample_size}.csv",
                    "Convenience\nDeterministic": f"/home/libe2152/optimizedsampling/0_results/usavars/{label}/summaries/filtered_convenience_sampling_deterministic_sample_size_{sample_size}.csv",
                    "Random": f"/home/libe2152/optimizedsampling/0_results/usavars/{label}/summaries/filtered_random_sampling_sample_size_{sample_size}.csv",
                },
                sample_size=sample_size,
                save_path= f"/home/libe2152/optimizedsampling/0_results/usavars/{label}/plots/r2_boxplot_sample{sample_size}_ppc{points_per_cluster}.png",
                rotation=30
            )