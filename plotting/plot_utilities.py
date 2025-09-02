import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def extract_base_sampling_type(long_sampling_type):
    for base_type in ['random_sampling', 'convenience_sampling', 'cluster_sampling']:
        if long_sampling_type.startswith(base_type):
            return base_type
    return None


def load_utilities_csv(path):
    print(f"[INFO] Loading utilities CSV from: {path}")
    df = pd.read_csv(path)
    required_cols = ['initial_set', 'sampling_type', 'size']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Missing required column in utilities CSV: {col}")
    print(f"[INFO] Utilities CSV loaded with {len(df)} rows")
    return df

def load_all_r2_scores(r2_dir):
    r2_files = glob.glob(os.path.join(r2_dir, "*_r2_scores.csv"))
    all_dfs = []
    print(f"[INFO] Found {len(r2_files)} R² score files in {r2_dir}")
    for path in r2_files:
        filename = os.path.basename(path)
        if not filename.endswith("_r2_scores.csv"):
            print(f"[WARNING] Skipping unexpected file: {filename}")
            continue
        sampling_type = filename.replace("_r2_scores.csv", "")
        print(f"[INFO] Loading R² scores for sampling_type: {sampling_type}")
        df = pd.read_csv(path)
        if 'initial_set' not in df.columns or 'r2' not in df.columns:
            print(f"[WARNING] Missing 'initial_set' or 'r2' columns in {filename}, skipping.")
            continue
        df['sampling_type'] = sampling_type
        all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError(f"[ERROR] No valid R² score files loaded from {r2_dir}")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"[INFO] Loaded R² scores for {combined_df['sampling_type'].nunique()} sampling types, total rows: {len(combined_df)}")
    return combined_df

def merge_utilities_with_r2(utilities_df, summaries_dir):
    utilities_df['base_sampling_type'] = utilities_df['sampling_type'].apply(extract_base_sampling_type)

    merged_parts = []

    groups = utilities_df.groupby(['base_sampling_type', 'size'])

    for (sampling_type, sample_size), group_df in groups:
        summary_csv_path = os.path.join(summaries_dir, f"{sampling_type}_r2_scores.csv")
        if not os.path.exists(summary_csv_path):
            print(f"[Warning] Missing summary file: {summary_csv_path}")
            continue

        r2_df = pd.read_csv(summary_csv_path)
        r2_df_filtered = r2_df[r2_df['size'] == sample_size].copy()

        r2_df_filtered['base_sampling_type'] = sampling_type

        group_df = group_df.drop(columns=['sampling_type'])

        merged = group_df.merge(
            r2_df_filtered,
            on=['initial_set', 'size', 'base_sampling_type'],
            how='inner'
        )
        merged_parts.append(merged)

    if not merged_parts:
        raise RuntimeError("No merged data. Check summary files and matching columns.")

    merged_df = pd.concat(merged_parts, ignore_index=True)

    merged_df = merged_df.rename(columns={'base_sampling_type': 'sampling_type'})

    #print(f"[INFO] Merged data includes sampling types:\n{merged_df['sampling_type'].value_counts()}")

    return merged_df


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
        print(f"[INFO] Saved plot: {save_path}")
    else:
        plt.show()


def plot_top_percent_utility_vs_r2(df, utility_col, top_percent=0.1, **kwargs):
    n = int(len(df) * top_percent)
    top_df = df.nlargest(n, 'r2').copy()
    print(f"[INFO] Plotting top {top_percent*100:.1f}% ({n} samples) for utility '{utility_col}'")
    plot_utility_vs_r2(top_df, utility_col, **kwargs)

def run_utility_r2_pipeline(dir, label, utilities_csv_path, r2_dir, utility_cols, top_percent=0.1):
    print(f"[INFO] Starting utility vs R² pipeline for label: {label}")

    utilities_df = load_utilities_csv(utilities_csv_path)
    # r2_df = load_all_r2_scores(r2_dir)

    merged_df = merge_utilities_with_r2(utilities_df, dir)
    print("Merged dataframe sampling types count:")
    print(merged_df['sampling_type'].value_counts())

    
    set_y_lim = (label == "population")
    
    for utility in utility_cols:
        print(f"[INFO] Plotting utility vs R² for: {utility}")
        plot_utility_vs_r2(
            merged_df,
            utility_col=utility,
            save_path=os.path.join(r2_dir, f"plots/r2_{utility}_utility_scatterplot_{label}.png"),
            set_y_lim=set_y_lim
        )
        plot_top_percent_utility_vs_r2(
            merged_df,
            utility_col=utility,
            top_percent=top_percent,
            save_path=os.path.join(r2_dir, f"plots/r2_{utility}_utility_scatterplot_top_{int(top_percent*100)}_percent_{label}.png"),
            set_y_lim=False
        )
    print(f"[INFO] Completed utility vs R² pipeline for label: {label}")

if __name__ == "__main__":
    utility_metrics = [
        'size', 'pop_risk_nlcd_0.5', 'pop_risk_nlcd_0', 'pop_risk_nlcd_0.01',
        'pop_risk_nlcd_0.1', 'pop_risk_nlcd_0.9', 'pop_risk_nlcd_0.99',
        'pop_risk_nlcd_1', 'pop_risk_image_clusters_8_0.5', 'pop_risk_image_clusters_8_0', 'pop_risk_image_clusters_8_0.01',
        'pop_risk_image_clusters_8_0.1', 'pop_risk_image_clusters_8_0.9', 'pop_risk_image_clusters_8_0.99',
        'pop_risk_image_clusters_8_1'
    ]

    for label in ['population', 'treecover']:
        base_dir = f"../../0_results/usavars/{label}"
        utilities_csv = os.path.join(base_dir, "utilities.csv")
        r2_scores_dir = base_dir

        plots_dir = os.path.join(r2_scores_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        summaries_dir = "../0_results/usavars/population" 

        run_utility_r2_pipeline(summaries_dir, label, utilities_csv, r2_scores_dir, utility_metrics)