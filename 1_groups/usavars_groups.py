import pickle
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

def assign_groups_by_farthest_group(distances, gdf_points, id_col):
    id_array = gdf_points[id_col].astype(str).to_numpy()
    distances = np.array(distances)

    # Step 1: Assign group 3 to distances >= 400
    group_dict = {}
    distance_dict = {}

    far_mask = distances >= 50
    close_mask = ~far_mask

    # Assign group 3 to far points
    for id_, dist, is_far in zip(id_array, distances, far_mask):
        distance_dict[id_] = dist
        if is_far:
            group_dict[id_] = 1
        else:
            group_dict[id_] = 0

    return group_dict, distance_dict

def assign_groups_by_id_logspace(distances, gdf_points, id_col):
    id_array = gdf_points[id_col].astype(str).to_numpy()
    
    log_dists = np.log(distances)

    X1_log = np.percentile(log_dists, 33)
    X2_log = np.percentile(log_dists, 66)

    X1 = np.exp(X1_log)
    X2 = np.exp(X2_log)

    distance_dict = dict(zip(id_array, distances))

    group_dict = {}
    for id_, dist in distance_dict.items():
        if dist < X1:
            group = 0
        elif dist < X2:
            group = 1
        else:
            group = 2
        group_dict[id_] = group

    return group_dict, distance_dict, X1, X2

def make_group_assignment(df, columns, id_col):
    """
    Create a dict mapping from row ID to a combined group ID like 'COUNTY_NAME_COUNTYFP'.

    Parameters:
        df (pd.DataFrame): DataFrame with 'id' column and grouping columns
        columns (list of str): Columns to join for the group ID

    Returns:
        dict: Mapping from df['id'] to joined string from selected columns
    """
    group_ids = df[columns].astype(str).agg("_".join, axis=1)
    return dict(zip(df[id_col], group_ids))

def save_dict_to_pkl(d, filepath):
    """
    Save a dictionary to a pickle file.

    Parameters:
        d (dict): Dictionary to save
        filepath (str): Full path to the output .pkl file
    """
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(d, f)

import os
import matplotlib.pyplot as plt
import geopandas as gpd

def plot_group_assignments(
    gdf_points,
    group_dict,
    dataset,
    group_type,
    id_col='id',
    title=None,
    point_size=5,
    cmap='tab10',
    country_shape_file=None,
    country_name=None,
    exclude_names=None
):
    """
    Plots GeoDataFrame points with different colors based on group_dict, with optional country boundary overlay.

    Args:
        gdf_points (GeoDataFrame): GeoDataFrame containing point geometries.
        group_dict (dict): Dictionary mapping ID (int or str) to group number.
        dataset (str): Name of dataset (used for saving).
        group_type (str): Group type identifier (used for saving).
        id_col (str): Column name for IDs.
        title (str): Optional title for the plot.
        point_size (int): Marker size.
        cmap (str): Colormap name.
        country_shape_file (str): Path to country shapefile (optional).
        country_name (str): Filter by country name (optional).
        exclude_names (list): Exclude by subregion names (optional).

    Returns:
        fig: The matplotlib figure.
    """
    # Prepare group assignments
    gdf = gdf_points.copy()
    gdf['group'] = gdf[id_col].astype(str).map({str(k): v for k, v in group_dict.items()})
    gdf = gdf.dropna(subset=['group'])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Optional country boundary
    if country_shape_file:
        country = gpd.read_file(country_shape_file, engine="pyogrio")
        if country_name:
            country = country[country['NAME'] == country_name]
        if exclude_names:
            country = country[~country['NAME'].isin(exclude_names)]
        country = country.to_crs('EPSG:4326')
        country.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=1, alpha=0.8)

    # Plot groups
    unique_groups = sorted(gdf['group'].unique())
    cmap = plt.get_cmap(cmap, len(unique_groups)+5)

    for i, group in enumerate(unique_groups):
        group_gdf = gdf[gdf['group'] == group]
        group_gdf.plot(ax=ax, markersize=point_size, color=cmap(i*2), label=f"Group {group}", zorder=2)

    ax.legend(title="Groups", fontsize=10, title_fontsize=11)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{dataset}_{group_type}.png", dpi=300)
    plt.close()
    print(f"Saved plot to plots/{dataset}_{group_type}.png")


if __name__ == "__main__":
    distance_path = "/home/libe2152/optimizedsampling/0_data/distances/usavars/population/distance_km_to_all_urban.pkl"

    import dill
    with open(distance_path, "rb") as f:
        arrs = dill.load(f)

    gdf_points = gpd.read_file(f"/share/india_secc/MOSAIKS/train_shrugs_with_admins.geojson")
    gdf_points = gdf_points.to_crs("EPSG:4326")
    id_col='condensed_shrug_id'

    # distances = [arrs['distances_to_urban_area'][str(id)] for id in gdf_points[id_col]]
    # group_dict, _ = assign_groups_by_farthest_group(distances, gdf_points, id_col='id')

    filepath = "/home/libe2152/optimizedsampling/0_data/groups/india_secc/urban_rural_groups.pkl"
    # save_dict_to_pkl(group_dict, filepath)

    with open(filepath, "rb") as f:
        group_dict = dill.load(f)

    country_shape_file = '../0_data/boundaries/world/ne_10m_admin_0_countries.shp'
    country_name = 'India'

    plot_group_assignments(gdf_points, group_dict, dataset='india', group_type='urban_rural', id_col=id_col, country_shape_file=country_shape_file, country_name=country_name)
    print(group_dict)


