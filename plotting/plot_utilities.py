import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def extract_base_sampling_type(long_sampling_type):
    for base_type in ['random_sampling', 'convenience_sampling', 'cluster_sampling']:
        if long_sampling_type.startswith(base_type):
            return base_type
    return None

def merge_utilities_with_r2(utilities_df, summaries_dir):
    utilities_df['base_sampling_type'] = utilities_df['sampling_type'].apply(extract_base_sampling_type)
    merged_parts = []
    for (sampling_type, sample_size), group_df in utilities_df.groupby(['base_sampling_type', 'size']):
        summary_csv_path = os.path.join(summaries_dir, f"{sampling_type}_r2_scores.csv")
        if not os.path.exists(summary_csv_path):
            continue
        r2_df = pd.read_csv(summary_csv_path)
        r2_df_filtered = r2_df[r2_df['size'] == sample_size].copy()
        r2_df_filtered['base_sampling_type'] = sampling_type
        group_df = group_df.drop(columns=['sampling_type'])
        merged = group_df.merge(r2_df_filtered, on=['initial_set', 'size', 'base_sampling_type'], how='inner')
        merged_parts.append(merged)
    merged_df = pd.concat(merged_parts, ignore_index=True)
    return merged_df.rename(columns={'base_sampling_type': 'sampling_type'})

def compute_rho_labels(df, utility_col):
    labels = {}
    for name, group in df.groupby('sampling_type'):
        if group[utility_col].nunique() > 1:
            rho, _ = spearmanr(group[utility_col], group['r2'])
            labels[name] = f"{name} (ρ={rho:.2f})"
        else:
            labels[name] = f"{name} (ρ=N/A)"
    return labels

def plot_utility_vs_r2(df, utility_col, title=None, save_path=None, set_y_lim=False):
    rho_labels = compute_rho_labels(df, utility_col)
    df['sampling_type_rho'] = df['sampling_type'].map(rho_labels)
    overall_rho = spearmanr(df[utility_col], df['r2'])[0] if df[utility_col].nunique() > 1 else None

    plt.figure(figsize=(8, 6), dpi=300)
    sns.scatterplot(data=df, x=utility_col, y='r2', hue='sampling_type_rho', palette='Set2', s=70, alpha=0.8, edgecolor='k')
    plt.xlabel(f"Utility ({utility_col})")
    plt.ylabel("R² Score")
    plt.title(title or (f"Utility vs R² (ρ={overall_rho:.2f})" if overall_rho is not None else "Utility vs R²"))
    if set_y_lim:
        plt.ylim(bottom=0)
    plt.legend(title='Sampling Type')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_top_percent_utility_vs_r2(df, utility_col, top_percent=0.1, **kwargs):
    n = int(len(df) * top_percent)
    plot_utility_vs_r2(df.nlargest(n, 'r2').copy(), utility_col, **kwargs)

def run_utility_r2_pipeline(dir, label, utilities_csv_path, r2_dir, utility_cols, top_percent=0.1):
    utilities_df = pd.read_csv(utilities_csv_path)
    merged_df = merge_utilities_with_r2(utilities_df, dir)
    set_y_lim = (label == "population")

    for utility in utility_cols:
        plot_utility_vs_r2(
            merged_df, utility_col=utility,
            save_path=os.path.join(r2_dir, f"plots/r2_{utility}_utility_scatterplot_{label}.png"),
            set_y_lim=set_y_lim
        )
        plot_top_percent_utility_vs_r2(
            merged_df, utility_col=utility, top_percent=top_percent,
            save_path=os.path.join(r2_dir, f"plots/r2_{utility}_utility_scatterplot_top_{int(top_percent*100)}_percent_{label}.png"),
            set_y_lim=False
        )

DEFAULT_UTILITY_METRICS = [
    'size', 'pop_risk_nlcd_0.5', 'pop_risk_nlcd_0', 'pop_risk_nlcd_0.01',
    'pop_risk_nlcd_0.1', 'pop_risk_nlcd_0.9', 'pop_risk_nlcd_0.99',
    'pop_risk_nlcd_1', 'pop_risk_image_clusters_8_0.5', 'pop_risk_image_clusters_8_0', 'pop_risk_image_clusters_8_0.01',
    'pop_risk_image_clusters_8_0.1', 'pop_risk_image_clusters_8_0.9', 'pop_risk_image_clusters_8_0.99',
    'pop_risk_image_clusters_8_1'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Path to usavars results dir")
    parser.add_argument("--summaries-dir", required=True, help="Path to dir containing *_r2_scores.csv summary files")
    parser.add_argument("--labels", nargs="+", default=["population", "treecover"])
    parser.add_argument("--utility-metrics", nargs="+", default=DEFAULT_UTILITY_METRICS)
    args = parser.parse_args()

    for label in args.labels:
        base_dir = f"{args.results_dir}/{label}"
        utilities_csv = os.path.join(base_dir, "utilities.csv")
        os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
        run_utility_r2_pipeline(args.summaries_dir, label, utilities_csv, base_dir, args.utility_metrics)
