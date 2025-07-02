import os
import dill
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
import shapely

def save_ids_with_assignments(gdf, ids, label, type_str, col_name):
    matched_rows = gdf[gdf['id'].isin(ids)]

    unmatched_ids = set(ids) - set(matched_rows['id'])
    if unmatched_ids:
        print(f"Warning: {len(unmatched_ids)} ID(s) not found in GeoDataFrame.")

    assignments = list(matched_rows.set_index('id').loc[ids][col_name])

    os.makedirs(label, exist_ok=True)

    with open(f"{label}/{type_str}.pkl", 'wb') as f:
        dill.dump({"ids": ids, "assignments": assignments}, f)
    print("Saved")

if __name__ == "__main__":
    for label in ["income", "population", "treecover"]:
        year=2015

        gdf_states = gpd.read_file(f"/home/libe2152/optimizedsampling/initial_sample/{label}/gdf_states_{year}.geojson")

        with open(f"../data/int/feature_matrices/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl", "rb") as f:
            arrs = dill.load(f)
        
        invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
        ids_train = arrs['ids_train']
        valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
        ids = ids_train[valid_idxs]

        save_ids_with_assignments(gdf_states, ids, label, "state", "STATE_NAME")
