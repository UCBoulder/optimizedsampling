import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_sample_cost_vs_r2(csv_path, title=None):
    """
    Plots sample cost vs. R^2 from a CSV, with:
    - Big X at each initial set R^2
    - Lines per method (same color per method)
    - Each line starts at its respective initial size
    """
    df = pd.read_csv(csv_path)

    # Extract numeric initial size from 'initial_set'
    def extract_size(init_str):
        match = re.search(r'(\d+)_size', init_str)
        return int(match.group(1)) if match else None

    df["initial_size"] = df["initial_set"].apply(extract_size)
    df["sample_cost"] = df["budget"] + df["initial_size"]

    # Identify methods
    method_cols = [col for col in df.columns if col.endswith("_updated_r2_mean")]
    method_names = [col.replace("_updated_r2_mean", "") for col in method_cols]

    # Color mapping per method
    color_map = cm.get_cmap("tab10", len(method_names))
    method_to_color = {method: color_map(i) for i, method in enumerate(method_names)}
    method_to_label_plotted = {method: False for method in method_names}

    plt.figure(figsize=(10, 6))

    # Plot big X for initial R^2
    initial_points = df.groupby("initial_size").first().reset_index()
    plt.scatter(initial_points["initial_size"], initial_points["initial_r2_mean"],
                color='black', marker='x', s=100, label="Initial $R^2$")

    # Plot each method's trajectory for each initial set
    for init_set in df["initial_set"].unique():
        subset = df[df["initial_set"] == init_set].copy()
        init_size = subset["initial_size"].iloc[0]
        init_r2 = subset["initial_r2_mean"].iloc[0]

        for method in method_names:
            if f"{method}_updated_r2_mean" not in subset.columns:
                continue

            x = [init_size] + subset["sample_cost"].tolist()
            y = [init_r2] + subset[f"{method}_updated_r2_mean"].tolist()

            label = method if not method_to_label_plotted[method] else None
            method_to_label_plotted[method] = True

            plt.plot(x, y, marker='o', label=label, color=method_to_color[method])

    # Final touches
    plt.xlabel("Sample Cost (initial size + budget)")
    plt.ylabel("$R^2$ Score")
    plt.title(title) if title is not None else None
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('india_secc_multiple.png')

if __name__ == "__main__":
    csv_path = "aggregated_r2/aggregated_r2_INDIA_SECC_MULTIPLE_10_state_district_desired_20ppc.csv"
    plot_sample_cost_vs_r2(csv_path, title=None)