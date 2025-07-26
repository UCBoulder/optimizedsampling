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
    import pickle
    import numpy as np

    N = 5
    n = 100
    subset_seed = 42

    for label in ['population', 'treecover']:
        feature_path = f"/home/libe2152/optimizedsampling/0_data/features/usavars/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl"

        with open(feature_path, "rb") as f:
            arrs = pickle.load(f)

        ids = np.array(arrs['ids_train'])
        num_ids = len(ids)

        total_needed = N * n
        if total_needed > num_ids:
            raise ValueError(f"Requested {total_needed} IDs (N={N}, n={n}), but only {num_ids} available.")

        rng = np.random.default_rng(subset_seed)
        shuffled_ids = rng.permutation(ids)

        selected_ids = shuffled_ids[:total_needed]
        leftover_ids = set(shuffled_ids[total_needed:])

        split_ids = np.split(selected_ids, N)

        id_to_subset = {str(id_): 0 for id_ in ids}

        for subset_idx, id_subset in enumerate(split_ids, start=1):
            for id_ in id_subset:
                id_to_subset[str(id_)] = subset_idx

        # Save to pickle
        out_path = f"/home/libe2152/optimizedsampling/0_data/groups/usavars/{label}/{N}_subset_size_{n}_assignments.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(id_to_subset, f)

        print(f"Saved subset assignment for {label} to {out_path}")

