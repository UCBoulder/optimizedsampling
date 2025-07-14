import os
import dill
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from geopy.distance import geodesic
from shapely.strtree import STRtree

def compute_or_load_distances_to_urban(gdf_points, gdf_urban_top):

    out_path = f"{label}/convenience_sampling/distance_to_top{N_urban}_urban.pkl"

    if os.path.exists(out_path):
        print(f"Loading precomputed distances from {out_path}...")
        with open(out_path, "rb") as f:
            out_dict = dill.load(f)
        return out_dict['distances_to_urban_area']


    print("Computing distances to nearest urban areas using spatial index...")
    assert gdf_points.crs.to_string() == "EPSG:4326"
    assert gdf_urban_top.crs.to_string() == "EPSG:4326"

    #precompute R-tree index and geometry mapping
    urban_geoms = gdf_urban_top.geometry.values
    tree = STRtree(urban_geoms)
    geom_to_index = {id(geom): i for i, geom in enumerate(urban_geoms)}

    def arc_distance_to_urban(point):
        #query for nearest candidate urban polygon
        nearest_geom_idx = tree.nearest(point)
        nearest_geom = tree.geometries.take(nearest_geom_idx)
        nearest_pt = nearest_points(point, nearest_geom)[1]
        return geodesic((point.y, point.x), (nearest_pt.y, nearest_pt.x)).meters

    distances = gdf_points.geometry.apply(arc_distance_to_urban).to_numpy()
    print("Finished computing distances.")
    return distances

def dist_to_cost(distances, scale='sqrt', alpha=0.01, epsilon=1e-6):
    """
    Convert distances to cost and normalize to match uniform cost scale.
    """
    print(f"Converting distances to costs using scale: {scale}, alpha: {alpha}")
    distances = np.array(distances)
    if scale == 'linear':
        raw_costs = 1 + alpha * distances
    elif scale == 'log':
        raw_costs = 1 + alpha * np.log1p(distances + epsilon)
    elif scale == 'sqrt':
        raw_costs = 1 + alpha * np.sqrt(distances)
    else:
        raise ValueError("Unsupported scale type")

    # Normalize so that mean cost is 1 (or another target)
    # normalized_costs = raw_costs / np.mean(raw_costs) * normalize_to_mean
    # print(f"Mean raw cost: {np.mean(raw_costs):.4f}, after normalization: {np.mean(normalized_costs):.4f}")
    return raw_costs


def save_dist_array(ids, dists, out_path):
    """
    Save dictionary with 'ids' and 'costs' as a dill pickle.
    """
    out_dict = {
        'ids': np.array(ids),
        'distances_to_urban_area': np.array(dists)
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        dill.dump(out_dict, f)

    print(f"Saved dist array with {len(ids)} points to {out_path}")

def save_cost_array(ids, costs, out_path):
    """
    Save dictionary with 'ids' and 'costs' as a dill pickle.
    """
    out_dict = {
        'ids': np.array(ids),
        'costs': np.array(costs)
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        dill.dump(out_dict, f)

    print(f"Saved cost array with {len(ids)} points to {out_path}")

# === Example Usage ===
if __name__ == "__main__":
    N_urban = 50

    for label in ["population", "income", "treecover"]:
        print(f"\n=== Processing label: {label} ===")

        # Load training data
        print("Loading training data...")
        with open(f"../data/int/feature_matrices/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl", "rb") as f:
            arrs = dill.load(f)

        invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
        ids_train = arrs['ids_train']
        valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
        ids = ids_train[valid_idxs]
        latlons = arrs['latlons_train'][valid_idxs]

        print(f"Loaded {len(ids)} valid points.")

        points = [Point(lon, lat) for lat, lon in latlons]
        gdf_points = gpd.GeoDataFrame({'id': ids}, geometry=points, crs="EPSG:4326")

        # Load urban area polygons
        print("Loading urban areas...")
        urban_areas_path = "/home/libe2152/optimizedsampling/boundaries/us_urban_area_census_2020/tl_2020_us_uac20_with_pop.shp"
        gdf_urban = gpd.read_file(urban_areas_path).to_crs("EPSG:4326")
        gdf_urban_top = gdf_urban.nlargest(N_urban, 'POP').copy()
        print(f"Selected top {N_urban} urban areas by population.")

        # Compute distance-based cost
        dists = compute_or_load_distances_to_urban(gdf_points, gdf_urban_top)
        out_path = f"{label}/convenience_sampling/distance_to_top{N_urban}_urban.pkl"
        save_dist_array(ids, dists, out_path)
        print(f"Min/Max/Mean distance (m): {dists.min():.2f} / {dists.max():.2f} / {dists.mean():.2f}")

        costs = dist_to_cost(dists, scale='sqrt', alpha=0.01)
        print(f"Cost stats -> Min: {costs.min():.4f}, Max: {costs.max():.4f}, Mean: {costs.mean():.4f}")

        # Save
        out_path = f"{label}/convenience_sampling/distance_based_costs_top{N_urban}_urban.pkl"
        save_cost_array(ids, costs, out_path)
