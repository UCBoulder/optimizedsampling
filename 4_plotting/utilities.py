import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def load_and_merge_utilities_r2(utilities_csv_path, summaries_dir):
    """
    Load utilities CSV and corresponding R² summaries,
    merge on sampled_path, size, and sampling_type.
    
    Args:
        utilities_csv_path (str): Path to utilities CSV file.
        summaries_dir (str): Directory where filtered summary CSVs reside.
                             Files are named like filtered_{sampling_type}_sample_size_{size}.csv.
                             
    Returns:
        pd.DataFrame: Merged DataFrame with columns including:
                      sampled_path, sampling_type, size, utility columns..., r2
    """
    # Load utilities CSV
    utilities_df = pd.read_csv(utilities_csv_path)
    
    # Check required columns present
    for col in ['sampled_path', 'sampling_type', 'size']:
        if col not in utilities_df.columns:
            raise ValueError(f"Column '{col}' missing from utilities CSV")

    # Prepare list for per sampling_type and sample_size r2 DataFrames
    merged_parts = []

    # Get unique (sampling_type, sample_size) pairs
    groups = utilities_df.groupby(['sampling_type', 'size'])

    for (sampling_type, sample_size), group_df in groups:
        summary_file = os.path.join(
            summaries_dir,
            f"filtered_{sampling_type}_sample_size_{sample_size}.csv"
        )
        if not os.path.exists(summary_file):
            print(f"[Warning] Summary file not found: {summary_file}, skipping this group.")
            continue

        r2_df = pd.read_csv(summary_file)
        # Check for required columns
        if 'filename' not in r2_df.columns or 'r2' not in r2_df.columns:
            raise ValueError(f"Summary CSV {summary_file} missing 'filename' or 'r2' columns")

        # Merge utilities with r2 on sampled_path and filename
        merged = group_df.merge(
            r2_df,
            left_on='sampled_path',
            right_on='filename',
            how='inner',
            suffixes=('', '_r2')
        )

        merged_parts.append(merged)

    if not merged_parts:
        raise RuntimeError("No data merged; check files and keys")

    merged_df = pd.concat(merged_parts, ignore_index=True)
    return merged_df



def plot_utility_vs_r2(merged_df, utility_col='size', title=None, save_path=None, set_y_lim=False):
    """
    Scatter plot utility vs. r2 colored by sampling_type.
    Annotates plot with Spearman's ρ per sampling_type.

    Args:
        merged_df (pd.DataFrame): DataFrame with 'sampling_type', utility_col, 'r2'
        utility_col (str): Column for x-axis (utility)
        title (str, optional): Plot title
        save_path (str, optional): Save path
        set_y_lim (bool): Whether to force y-axis to start at 0
    """
    sns.set(style="white", context="notebook", font_scale=1.2)
    plt.figure(figsize=(8, 6), dpi=300)

    # Compute Spearman's rho for each sampling type
    spearman_rhos = {}
    for sampling_type in merged_df['sampling_type'].unique():
        sub_df = merged_df[merged_df['sampling_type'] == sampling_type]
        if sub_df[utility_col].nunique() > 1:
            rho, _ = spearmanr(sub_df[utility_col], sub_df['r2'])
            spearman_rhos[sampling_type] = round(rho, 2)
        else:
            spearman_rhos[sampling_type] = None

    # Build legend labels with ρ
    merged_df['sampling_label'] = merged_df['sampling_type'].apply(
        lambda t: f"{t} (ρ={spearman_rhos[t]})" if spearman_rhos[t] is not None else t
    )

    ax = sns.scatterplot(
        data=merged_df,
        x=utility_col,
        y='r2',
        hue='sampling_label',
        palette='Set2',
        s=70,
        alpha=0.8,
        edgecolor='k'
    )

    ax.set_xlabel(f"Utility ({utility_col})", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.grid(False)

    if title:
        ax.set_title(title, fontsize=14, pad=15)
    else:
        ax.set_title(f"Utility ({utility_col}) vs. R² by Sampling Type", fontsize=14, pad=15)

    if set_y_lim:
        ax.set_ylim(bottom=0)

    plt.legend(title='Sampling Type\n(Spearman ρ)', loc='best', fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

# === Usage Example ===
if __name__ == '__main__':
    for label in ['population', 'treecover']:
        utilities_csv_path = f"/home/libe2152/optimizedsampling/0_results/usavars/{label}/utilities.csv"
        results_dir = f"/home/libe2152/optimizedsampling/0_results/usavars/{label}"
        summaries_dir = os.path.join(results_dir, "summaries")
        plot_dir = os.path.join(results_dir, "plots")

        os.makedirs(plot_dir, exist_ok=True)

        set_y_lim = (label == "population")
        merged_df = load_and_merge_utilities_r2(utilities_csv_path, summaries_dir)

        for utility in ['size', 'pop_risk_0.5', 'pop_risk_0', 'pop_risk_0.01', 'pop_risk_0.1', 'pop_risk_0.9', 'pop_risk_0.99', 'pop_risk_1']:

            plot_utility_vs_r2(
                merged_df,
                utility_col=utility,
                save_path=os.path.join(plot_dir, f"r2_{utility}_utility_scatterplot.png"),
                set_y_lim=set_y_lim
            )