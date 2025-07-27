import pickle
import geopandas as gpd
import pandas as pd

import numpy as np

import numpy as np

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
    import geopandas as gpd
    from pathlib import Path

    gdf = gpd.read_file("/home/libe2152/optimizedsampling/0_data/admin_gdfs/togo/gdf_adm3.geojson")

    ADMIN_IDS = {
        'admin_1': 'region',
        'admin_2': 'prefecture',
        'admin_3': 'canton'
    }

    id_col = 'id'
    output_dir = Path("/home/libe2152/optimizedsampling/0_data/groups/togo")
    output_dir.mkdir(parents=True, exist_ok=True)

    for admin_col, admin_name in ADMIN_IDS.items():
        if admin_col not in gdf.columns:
            print(f"⚠️ Warning: Column '{admin_col}' not found in GeoDataFrame. Skipping.")
            continue

        assignment_dict = make_group_assignment(gdf, [admin_col], id_col)
        output_path = output_dir / f"{admin_name}_assignments_dict.pkl"
        save_dict_to_pkl(assignment_dict, output_path)
        print(f"✅ Saved: {output_path}")

    admin3_to_admin2 = gdf.groupby("admin_3")["admin_2"].nunique()
    violations_3_to_2 = admin3_to_admin2[admin3_to_admin2 > 1]

    # Check that each admin_2 maps to only one admin_1
    admin2_to_admin1 = gdf.groupby("admin_2")["admin_1"].nunique()
    violations_2_to_1 = admin2_to_admin1[admin2_to_admin1 > 1]

    # Print results
    if len(violations_3_to_2):
        print(f"❌ Violations in admin_3 → admin_2: {len(violations_3_to_2)}")
        print(violations_3_to_2)
        from IPython import embed; embed()
    else:
        print("✅ All admin_3 codes uniquely map to a single admin_2")

    if len(violations_2_to_1):
        print(f"❌ Violations in admin_2 → admin_1: {len(violations_2_to_1)}")
        print(violations_2_to_1)
    else:
        print("✅ All admin_2 codes uniquely map to a single admin_1")



