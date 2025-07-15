import pandas as pd
import os

def summarize_cluster_sampling_r2(csv_path, sample_size, **kwargs):
    """
    Summarize cluster sampling R2 for fixed sample_size and optional points_per_cluster,
    averaging over seeds.

    Returns a DataFrame with columns:
    ['points_per_cluster', 'sample_size', 'mean_r2', 'std_r2', 'num_seeds']
    """
    df = pd.read_csv(csv_path)
    
    # Filter by sample_size
    filtered = df[df['sample_size'] == sample_size]

    if filtered.empty:
        print("No matching data found.")
        return None

    # summary_df = pd.DataFrame({
    #     'sample_size': [sample_size],
    #     'points_per_cluster': [points_per_cluster],
    #     'mean_r2': [filtered['r2'].mean()],
    #     'std_r2': [filtered['r2'].std()],
    #     'num_seeds': [filtered['r2'].count()],
    # })

    # # Reorder columns
    # cols = ['points_per_cluster', 'sample_size', 'mean_r2', 'std_r2', 'num_seeds']
    # summary_df = summary_df[[c for c in cols if c in summary_df.columns]]

    return filtered

def summarize_convenience_sampling_r2(csv_path, sample_size, **kwargs):
    """
    Summarize convenience sampling R2 for fixed sample_size and method,
    averaging over seeds.

    Returns a DataFrame with columns:
    ['sample_size', 'method', 'mean_r2', 'std_r2', 'num_seeds']
    """
    method = kwargs.get('method', None)
    assert method is not None, 'Need to specify method'

    df = pd.read_csv(csv_path)

    # Filter by sample_size and method
    filtered = df[(df['sample_size'] == sample_size) & (df['method'] == method)]

    if filtered.empty:
        print("No matching data found.")
        return None

    summary_df = pd.DataFrame({
        'sample_size': [sample_size],
        'method': [method],
        'mean_r2': [filtered['r2'].mean()],
        'std_r2': [filtered['r2'].std()],
        'num_seeds': [filtered['r2'].count()],
    })

    return filtered #, summary_df

def summarize_random_sampling_r2(csv_path, sample_size, **kwargs):
    """
    Summarize convenience sampling R2 for fixed sample_size and method,
    averaging over seeds.

    Returns a DataFrame with columns:
    ['sample_size', 'method', 'mean_r2', 'std_r2', 'num_seeds']
    """

    df = pd.read_csv(csv_path)

    # Filter by sample_size and method
    filtered = df[(df['sample_size'] == sample_size)]

    if filtered.empty:
        print("No matching data found.")
        return None

    summary_df = pd.DataFrame({
        'sample_size': [sample_size],
        'mean_r2': [filtered['r2'].mean()],
        'std_r2': [filtered['r2'].std()],
        'num_seeds': [filtered['r2'].count()],
    })

    return filtered #, summary_df


def save_r2_summary(csv_path, sample_size, summary_csv_dir, sampling_type, **kwargs):
    """
    General function to save summary data (DataFrame) returned by summary_fn to CSV.
    Appends if file exists, else creates.
    
    The saved filename includes sample_size and optionally points_per_cluster.

    Parameters
    ----------
    csv_path : str
        Path to the original detailed CSV.
    sample_size : int
        Total sample size filter.
    summary_csv_dir : str
        Directory to save summary CSV files.
    summary_fn : callable
        Function with signature accepting csv_path, sample_size, and optionally other kwargs.
    **kwargs :
        Additional keyword arguments to pass to summary_fn (e.g., points_per_cluster).
    """
    os.makedirs(summary_csv_dir, exist_ok=True)

    if sampling_type not in SUMMARY_FN_MAP:
        raise ValueError(f"Unknown sampling_type '{sampling_type}'. Valid types: {list(SUMMARY_FN_MAP.keys())}")

    summary_fn = SUMMARY_FN_MAP[sampling_type]

    #filtered, summary_df = summary_fn(csv_path, sample_size, **kwargs)
    filtered = summary_fn(csv_path, sample_size, **kwargs)
    if filtered is None:
        from IPython import embed; embed()
    filtered = filtered.drop(columns=['Unnamed: 0'], errors='ignore')

    # if summary_df is None or summary_df.empty:
    #     print("No summary data to save.")
    #     return

    points_per_cluster = kwargs.get('points_per_cluster', None)
    ppc_str = f"_ppc_{points_per_cluster}" if points_per_cluster is not None else ""
    method = kwargs.get('method', None)
    method_str = f"_{method}" if method is not None else ""

    filename = f"filtered_{sampling_type}{method_str}_sample_size_{sample_size}{ppc_str}.csv"
    filtered_csv_path = os.path.join(summary_csv_dir, filename)

    filtered.to_csv(filtered_csv_path, index=False)

    print(f"Saved all filtered to {filtered_csv_path}")

    # filename = f"summary_{sampling_type}{method_str}_sample_size_{sample_size}{ppc_str}.csv"
    # summary_csv_path = os.path.join(summary_csv_dir, filename)

    # summary_df.to_csv(summary_csv_path, index=False)

    # print(f"Saved summary to {summary_csv_path}")

SUMMARY_FN_MAP = {
    'cluster_sampling': summarize_cluster_sampling_r2,
    'convenience_sampling': summarize_convenience_sampling_r2,
    'random_sampling': summarize_random_sampling_r2
}

if __name__ == "__main__":
    for label in ['population', 'treecover']:
        summary_csv_dir = f"/home/libe2152/optimizedsampling/0_results/usavars/{label}/summaries"

        for sample_size in range(100, 1100, 100):
            sampling_type = "cluster_sampling"
            csv_path = f"/home/libe2152/optimizedsampling/0_results/usavars/{label}/{sampling_type}_r2_scores.csv"
            # for points_per_cluster in [2, 5, 10, 25]:
            save_r2_summary(csv_path, sample_size, summary_csv_dir, sampling_type) #points_per_cluster=points_per_cluster)

            sampling_type = 'convenience_sampling'
            csv_path = f"/home/libe2152/optimizedsampling/0_results/usavars/{label}/{sampling_type}_r2_scores.csv"
            for method in ['deterministic', 'probabilistic']:
                save_r2_summary(csv_path, sample_size, summary_csv_dir, sampling_type, method=method)

            sampling_type = 'random_sampling'
            csv_path = f"/home/libe2152/optimizedsampling/0_results/usavars/{label}/{sampling_type}_r2_scores.csv"
            save_r2_summary(csv_path, sample_size, summary_csv_dir, sampling_type)