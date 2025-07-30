import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def load_utilities_csv(path):
    df = pd.read_csv(path)
    required_cols = ['sampled_path', 'sampling_type', 'size']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df

def get_summary_path(summaries_dir, sampling_type, sample_size):
    if sampling_type.startswith('cluster_sampling'):
        sampling_type = 'cluster_sampling'
    
    sample_size = ((sample_size + 99) // 100) * 100

    # Base pattern to match potential ppc suffix
    pattern = os.path.join(
        summaries_dir,
        f"filtered_{sampling_type}_sample_size_{sample_size}*.csv"
    )
    
    matches = glob.glob(pattern)
    if not matches:
        return None
    elif len(matches) > 1:
        print(f"[Warning] Multiple summary files matched pattern: {pattern}. Using first match:\n  {matches[0]}")
    
    return matches[0]


def merge_utilities_with_r2(utilities_df, summaries_dir):
    merged_parts = []
    groups = utilities_df.groupby(['sampling_type', 'size'])

    for (sampling_type, sample_size), group_df in groups:
        summary_path = get_summary_path(summaries_dir, sampling_type, sample_size)
        if not os.path.exists(summary_path):
            print(f"[Warning] Missing summary: {summary_path}")
            continue

        r2_df = pd.read_csv(summary_path)
        from IPython import embed; embed()
        if 'r2' not in r2_df:
            raise ValueError(f"Missing 'filename' or 'r2' in {summary_path}")

        merged = group_df.merge(
            r2_df, left_on='sampled_path', right_on='filename', how='inner'
        )
        merged_parts.append(merged)

    if not merged_parts:
        raise RuntimeError("No merged data. Check summary files.")

    return pd.concat(merged_parts, ignore_index=True)

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
    sns.scatterplot(
        data=df,
        x=utility_col, y='r2',
        hue='sampling_type_rho',
        palette='Set2', s=70, alpha=0.8, edgecolor='k'
    )
    plt.xlabel(f"Utility ({utility_col})")
    plt.ylabel("R² Score")
    if title:
        plt.title(title)
    elif overall_rho is not None:
        plt.title(f"Utility vs R² (ρ={overall_rho:.2f})")
    else:
        plt.title("Utility vs R²")
    if set_y_lim:
        plt.ylim(bottom=0)
    plt.legend(title='Sampling Type')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

def plot_top_percent_utility_vs_r2(df, utility_col, top_percent=0.1, **kwargs):
    n = int(len(df) * top_percent)
    top_df = df.nlargest(n, 'r2').copy()
    plot_utility_vs_r2(top_df, utility_col, **kwargs)

def run_utility_r2_pipeline(label, utility_cols, top_percent=0.1):
    base_dir = f"/home/libe2152/optimizedsampling/0_results/usavars/{label}"
    utilities_csv_path = os.path.join(base_dir, "utilities.csv")
    summaries_dir = os.path.join(base_dir, "summaries")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    set_y_lim = (label == "population")
    utilities_df = load_utilities_csv(utilities_csv_path)
    merged_df = merge_utilities_with_r2(utilities_df, summaries_dir)

    for utility in utility_cols:
        plot_utility_vs_r2(
            merged_df, utility_col=utility,
            save_path=os.path.join(plot_dir, f"r2_{utility}_utility_scatterplot.png"),
            set_y_lim=set_y_lim
        )
        plot_top_percent_utility_vs_r2(
            merged_df, utility_col=utility, top_percent=top_percent,
            save_path=os.path.join(plot_dir, f"r2_{utility}_utility_scatterplot_top_{int(top_percent*100)}_percent.png"),
            set_y_lim=False
        )

if __name__ == '__main__':
    utility_metrics = [
        'size', 'pop_risk_nlcd_0.5', 'pop_risk_nlcd_0', 'pop_risk_nlcd_0.01',
        'pop_risk_nlcd_0.1', 'pop_risk_nlcd_0.9', 'pop_risk_nlcd_0.99',
        'pop_risk_nlcd_1', 'similarity', 'diversity'
    ]

    for label in ['population', 'treecover']:
        run_utility_r2_pipeline(label, utility_metrics)
