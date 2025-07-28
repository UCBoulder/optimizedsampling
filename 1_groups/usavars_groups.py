import pickle
import geopandas as gpd
import pandas as pd

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

if __name__ == "__main__":
    distance_path = "/home/libe2152/optimizedsampling/0_data/distances/usavars/population/distance_km_to_all_urban.pkl"

    import dill
    with open(distance_path, "rb") as f:
        arrs = dill.load(f)

    gdf_points = gpd.read_file(f"/home/libe2152/optimizedsampling/0_data/admin_gdfs/usavars/population/gdf_counties_2015.geojson")
    gdf_points = gdf_points.to_crs("EPSG:4326")
    id_col='id'

    distances = [arrs['distances_to_urban_area'][str(id)] for id in gdf_points[id_col]]
    group_dict, _ = assign_groups_by_farthest_group(distances, gdf_points, id_col='id')

    filepath = "/home/libe2152/optimizedsampling/0_data/groups/usavars/population/dist_from_all_urban_area_tiers.pkl"
    save_dict_to_pkl(group_dict, filepath)
    print(group_dict)


