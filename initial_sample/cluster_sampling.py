import dill
import geopandas as gpd
import pandas as pd
import numpy as np

def stratify_points(gdf_points, strata_col):
    """
    Stratify GeoDataFrame by a specified column (e.g., 'STATEFP' or 'URBAN_RURAL').
    Returns a dict mapping strata -> GeoDataFrame of points.
    """
    strata_values = gdf_points[strata_col].unique()
    return {val: gdf_points[gdf_points[strata_col] == val] for val in strata_values}

def determine_clusters_per_stratum(total_sample_size, strata_dict, points_per_cluster):
    """
    Given a desired total number of points and strata, allocate clusters equally.
    Returns dict: stratum -> number of clusters to sample
    """
    n_strata = len(strata_dict)
    sample_per_stratum = total_sample_size // n_strata
    clusters_per_stratum = {
        stratum: max(1, sample_per_stratum // points_per_cluster)
        for stratum in strata_dict
    }
    return clusters_per_stratum

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
    clusters_per_stratum = determine_clusters_per_stratum(
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

    return pd.concat(all_sampled_points).reset_index(drop=True)

if __name__ == "__main__":
    strata_col = "STATEFP"
    cluster_col = "COUNTYFP"
    # sample_size = 1000
    points_per_cluster = 25

    for sample_size in [100, 500, 1000]:
        for label in ["population", "income", "treecover"]:
            gdf_path = f"{label}/gdf_counties_2015.geojson"
            gdf_points = gpd.read_file(gdf_path)

            if strata_col == "STATEFP":
                gdf_points = gdf_points[gdf_points["STATEFP"] != '11']

            pd_sampled = cluster_sampling(gdf_points, sample_size, strata_col, cluster_col, points_per_cluster=points_per_cluster)

            sampled_ids = pd_sampled['id'].tolist()

            strata_name = "state" if strata_col == "STATEFP" else strata_col.lower()
            cluster_name = "county" if cluster_col == "COUNTYFP" else cluster_col.lower()
            out_path = f"{label}/IDS_{strata_name}_strata_{cluster_name}_clusters_{points_per_cluster}_points_per_cluster_{sample_size}_size.pkl"
            with open(out_path, "wb") as f:
                dill.dump(sampled_ids, f)

            print(f"Saved {len(sampled_ids)} sampled IDs for '{label}' to {out_path}")
