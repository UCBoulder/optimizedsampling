import dill
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def stratify_points(gdf_points, strata_col):
    """
    Stratify GeoDataFrame by a specified column (e.g., 'STATEFP' or 'URBAN_RURAL').
    Returns a dict mapping strata -> GeoDataFrame of points.
    """
    strata_values = gdf_points[strata_col].unique()
    return {val: gdf_points[gdf_points[strata_col] == val] for val in strata_values}

def determine_clusters_and_points(total_sample_size, strata_dict, points_per_cluster):
    """
    Given a desired total number of points and strata, allocate clusters equally.
    Returns dict: stratum -> number of clusters to sample
    """
    n_strata = len(strata_dict)
    sample_per_stratum = total_sample_size // n_strata
    num_clusters_per_stratum = sample_per_stratum // points_per_cluster

    if num_clusters_per_stratum < 1:
        print(
            f"WARNING: Not enough sample size ({total_sample_size}) for "
            f"{points_per_cluster} points per cluster across {n_strata} strata. "
            f"Reducing points_per_cluster to {sample_per_stratum}."
        )
        points_per_cluster = sample_per_stratum
        num_clusters_per_stratum = 1

    clusters_per_stratum = {
        stratum: num_clusters_per_stratum
        for stratum in strata_dict
    }

    return clusters_per_stratum, points_per_cluster

def sample_clusters(gdf_points, cluster_col, n_clusters):
    """
    Sample clusters with probability proportional to the number of points in each cluster.
    """
    counts = gdf_points[cluster_col].value_counts()
    probs = counts / counts.sum()
    n_clusters = min(n_clusters, len(probs))
    sampled = probs.sample(n=n_clusters, weights=probs, replace=False, random_state=42)
    return sampled.index.tolist()

def sample_points_within_clusters(gdf_points, cluster_col, sampled_clusters, points_per_cluster):
    """
    Sample fixed number of points within each selected cluster.
    """
    sampled_points_list = []
    for cluster in sampled_clusters:
        cluster_points = gdf_points[gdf_points[cluster_col] == cluster]
        if len(cluster_points) <= points_per_cluster:
            sampled = cluster_points
        else:
            sampled = cluster_points.sample(n=points_per_cluster, random_state=42)
        sampled_points_list.append(sampled)
    return pd.concat(sampled_points_list)

def cluster_sampling(
    gdf_points,
    total_sample_size,
    strata_col,
    cluster_col,
    points_per_cluster=25
):
    """
    General-purpose cluster sampling pipeline.

    Parameters:
    - gdf_points: GeoDataFrame with necessary strata and cluster columns
    - strata_col: column to stratify by (e.g. 'STATEFP')
    - cluster_col: cluster ID column (e.g. 'COUNTYFP')
    - total_sample_size: total number of points desired
    - points_per_cluster: fixed number of points sampled per cluster

    Returns:
    - GeoDataFrame of sampled points
    """
    strata_dict = stratify_points(gdf_points, strata_col)
    clusters_per_stratum, points_per_cluster = determine_clusters_and_points(
        total_sample_size, strata_dict, points_per_cluster
    )

    all_sampled_points = []
    for stratum, gdf_stratum in strata_dict.items():
        n_clusters = clusters_per_stratum[stratum]
        sampled_clusters = sample_clusters(gdf_stratum, cluster_col, n_clusters)
        sampled_points = sample_points_within_clusters(
            gdf_stratum, cluster_col, sampled_clusters, points_per_cluster
        )
        all_sampled_points.append(sampled_points)

    return pd.concat(all_sampled_points).reset_index(drop=True), points_per_cluster

def plot(label, all_points, selected_points, desired_sample_size, points_per_cluster, strata_str, cluster_str):
    fig, ax = plt.subplots(figsize=(12, 10))

    world = gpd.read_file("../boundaries/us_states_provinces/ne_110m_admin_1_states_provinces.shp", engine="pyogrio")
    exclude_states = ["Alaska", "Hawaii", "Puerto Rico"]
    contiguous_us = world[~world["name"].isin(exclude_states)]
    contiguous_outline = contiguous_us.dissolve()

    contiguous_outline.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=4, alpha=0.8)

    all_points.plot(ax=ax, color='#cccccc', markersize=5, label='All Points', zorder=1, alpha=0.6)

    legend_label = f'Sampled Points ({len(selected_points):,})'
    selected_points.plot(ax=ax, color='#d62728', markersize=5, label=legend_label, zorder=3, alpha=0.8)

    title_str = f'Cluster Sampling\n(Stratified by {strata_str}, clusters by {cluster_str})\n({points_per_cluster} points per cluster; {desired_sample_size} desired sample size)'
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    ax.legend(loc='lower left', fontsize=10, title='Legend', title_fontsize=11, frameon=True)

    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    save_str = f'{label}/cluster_sampling/plots/{strata_str}_strata_{cluster_str}_clusters_{points_per_cluster}_points_per_cluster_{desired_sample_size}_size.png' 
    plt.savefig(save_str, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    strata_col = "STATEFP"
    cluster_col = "COUNTYFP"

    for label in ["income", "population", "treecover"]:
        for sample_size in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            for desired_points_per_cluster in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
                gdf_path = f"{label}/gdf_counties_2015.geojson"
                gdf_points = gpd.read_file(gdf_path)

                if strata_col == "STATEFP":
                    gdf_points.loc[gdf_points["STATEFP"] == '11', "STATEFP"] = '24' #replace DoC with Marylan

                pd_sampled, points_per_cluster = cluster_sampling(gdf_points, sample_size, strata_col, cluster_col, points_per_cluster=desired_points_per_cluster)

                if points_per_cluster < desired_points_per_cluster:
                    break

                sampled_ids = pd_sampled['id'].tolist()

                strata_name = "state" if strata_col == "STATEFP" else strata_col.lower()
                cluster_name = "county" if cluster_col == "COUNTYFP" else cluster_col.lower()
                out_path = f"{label}/cluster_sampling/IDS_{strata_name}_strata_{cluster_name}_clusters_{points_per_cluster}_points_per_cluster_{sample_size}_size.pkl"

                plot(
                    label=label,
                    all_points=gdf_points,
                    selected_points=pd_sampled,
                    desired_sample_size=sample_size,
                    points_per_cluster=points_per_cluster,
                    strata_str=strata_name,
                    cluster_str=cluster_name
                )

                with open(out_path, "wb") as f:
                    dill.dump(sampled_ids, f)

                print(f"Saved {len(sampled_ids)} sampled IDs for '{label}' to {out_path}")
