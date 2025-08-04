import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import matplotlib.ticker as mticker\

mpl.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 24}) 

tab10 = plt.get_cmap("tab10")

# Group method keys by display name
METHOD_NAMES = {
    "random_unit": "Random Clusters",
    "poprisk_regions_0.5": "Admin-Rep",
    "poprisk_image_clusters_3_0.5": "Image-Rep",
    "poprisk_states_0.5": "Admin-Rep",
    #"poprisk_image_clusters_8_0.5": "Image-Rep",
}

# Unique display names in desired color order
display_name_order = [
    "Image-Rep",        # tab10(0)
    "Random Clusters",  # tab10(1)
    "Admin-Rep",        # tab10(2)
]

# Assign color per display name
DISPLAY_NAME_COLORS = {
    name: tab10(i) for i, name in enumerate(display_name_order)
}

# Build final METHOD_COLORS using display names
METHOD_COLORS = {
    method_key: DISPLAY_NAME_COLORS[display_name]
    for method_key, display_name in METHOD_NAMES.items()
}

# Example dataset names (unchanged)
DATASET_NAMES = {
    "INDIA_SECC", "USAVARS_POP", "USAVARS_TC", "TOGO_PH_H2O"
}


def load_and_filter_csv(csv_path, budget):
    """
    Load CSV and filter rows by specified budget.
    """
    df = pd.read_csv(csv_path)
    df_filtered = df[df['budget'] == budget]
    if df_filtered.empty:
        raise ValueError(f"No rows found with budget = {budget}")
    return df_filtered

def get_delta_r2_columns(df):
    """
    Get columns that contain mean R² scores for each method.
    """
    return [col for col in df.columns if col.endswith("_delta_r2_mean")]

def get_delta_r2_se_columns(df):
    """
    Get columns that contain mean R² scores for each method.
    """
    return [col for col in df.columns if col.endswith("_delta_r2_se")]

def extract_method_name(col_name):
    """
    Strip column suffix to get the method name.
    """
    return col_name.replace("_delta_r2_mean", "")

def plot_delta_r2_vs_alpha(df, r2_mean_cols, r2_se_cols, budget, save_path=None, y_lim=None):
    """
    Plot alpha vs delta R² score with shaded standard error bands for each method.
    """
    # Set Seaborn style for a clean aesthetic
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Color, marker, and linestyle settings
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X'] * 2
    linestyles = ['--', '-', '-.', ':'] * 3

    for idx, (mean_col, se_col) in enumerate(zip(r2_mean_cols, r2_se_cols)):
        method_name = extract_method_name(mean_col)
        if method_name not in METHOD_NAMES.keys():
            continue
        print(method_name)

        method_name_str = METHOD_NAMES[method_name]
        alpha_vals = df['alpha']
        mean_vals = df[mean_col]
        se_vals = df[se_col]

        if method_name == "greedycost":
            subset = alpha_vals != 0
            alpha_vals = alpha_vals[subset]
            mean_vals = mean_vals[subset]
            se_vals = se_vals[subset]

        # Line plot
        ax.plot(
            alpha_vals, mean_vals,
            label=method_name_str,
            marker=markers[idx],
            linestyle=linestyles[idx],
            color=METHOD_COLORS[method_name],
            linewidth=4,
            markersize=6
        )
        print('plotted')

        #Shaded standard error band
        ax.fill_between(
            alpha_vals,
            mean_vals - se_vals,
            mean_vals + se_vals,
            color=METHOD_COLORS[method_name],
            alpha=0.1
        )

    ax.set_xlabel("Cost Difference", fontsize=24)
    ax.set_ylabel("Δ R²", fontsize=24)

    ax.tick_params(labelsize=20)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    if y_lim:
        ax.set_ylim(y_lim)

    ax.legend(
        fontsize=20,
        title_fontsize=20,
        loc='upper right',
        #bbox_to_anchor=(1.02, 0.5),
        frameon=True
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        fig.show()

    fig.close('all')


def visualize_delta_r2_from_csv(csv_path, budget, save_path=None):
    """
    Top-level function to load, process, and plot CSV data.
    """
    df = load_and_filter_csv(csv_path, budget)
    r2_cols = get_delta_r2_columns(df)
    se_cols = get_delta_r2_se_columns(df)
    plot_delta_r2_vs_alpha(df, r2_cols, se_cols, budget, save_path=save_path)

if __name__ == "__main__":
    plt.close('all')

    initial_set_strs = {
        'USAVARS_POP': '{num_strata}_fixedstrata_{ppc}ppc_{size}_size',
        'USAVARS_TC': '{num_strata}_fixedstrata_{ppc}ppc_{size}_size',
        'INDIA_SECC': '{num_strata}_state_district_desired_{ppc}ppc_{size}_size',
        'TOGO_PH_H2O': '{num_strata}_strata_desired_{ppc}ppc_{size}_size'
    }

    for dataset in DATASET_NAMES:
        for ppc in [10, 20, 25]:
            for size in [100, 200, 300, 500, 2000]:
                for num_strata in [2, 5, 10]:
                    initial_set = initial_set_strs[dataset].format(num_strata=num_strata, ppc=ppc, size=size)
                    sampling_type = 'cluster_sampling'
                    for budget in [100, 500, 1000, 2000]:

                        try:
                            csv_path = f"results/aggregated_r2_{dataset}_{initial_set}.csv"

                            visualize_delta_r2_from_csv(csv_path, budget=budget, save_path=f"plots/{dataset}_{initial_set}_budget_{budget}.png")
                        except Exception as e:
                            print(e)