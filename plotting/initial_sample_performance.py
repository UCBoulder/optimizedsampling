import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SAMPLING_TYPE_COLORS = {
    "Cluster": "#E69F00",
    "Convenience\n(Urban)": "#56B4E9",
    "Random": "#009E73",
}

def plot_r2_boxplot_across_methods(csv_dict, sample_size, save_path=None, set_y_lim=False):
    dfs = []
    for label, path in csv_dict.items():
        df = pd.read_csv(path)
        df['sampling_type'] = label
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)

    palette = {label: SAMPLING_TYPE_COLORS.get(label, "#CCCCCC") for label in csv_dict}

    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    plt.figure(figsize=(8, 6), dpi=300)

    ax = sns.boxplot(
        data=combined_df, x='sampling_type', y='r2', palette=palette,
        width=0.6, linewidth=1.5, fliersize=3, boxprops=dict(alpha=0.9),
    )
    ax.set_title(f"R2 Scores by Sampling Method\n(Sample Size = {sample_size})", fontsize=14, pad=15)
    ax.set_xlabel("Sampling Method", fontsize=12)
    ax.set_ylabel("R2 Score", fontsize=12)
    ax.tick_params(axis='x', labelrotation=45, labelsize=9)
    if set_y_lim:
        ax.set_ylim(bottom=-0.1)

    sns.despine(offset=10, trim=True)
    ax.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Path to usavars results dir")
    parser.add_argument("--labels", nargs="+", default=["population", "treecover"])
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[100, 200, 300])
    parser.add_argument("--points-per-cluster", nargs="+", type=int, default=[10])
    args = parser.parse_args()

    for label in args.labels:
        label_dir = f"{args.results_dir}/{label}"
        for sample_size in args.sample_sizes:
            for ppc in args.points_per_cluster:
                plot_r2_boxplot_across_methods(
                    csv_dict={
                        "Convenience\n(Urban)": f"{label_dir}/summaries/filtered_convenience_sampling_urban_based_sample_size_{sample_size}.csv",
                        "Cluster": f"{label_dir}/summaries/filtered_cluster_sampling_sample_size_{sample_size}_ppc_{ppc}.csv",
                        "Random": f"{label_dir}/summaries/filtered_random_sampling_sample_size_{sample_size}.csv",
                    },
                    sample_size=sample_size,
                    save_path=f"{label_dir}/plots/r2_boxplot_sample{sample_size}_ppc{ppc}.png",
                )
