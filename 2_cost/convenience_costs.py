import os
import dill
import numpy as np
import geopandas as gpd

from sklearn.neighbors import BallTree
from shapely.geometry import Point

from sklearn.neighbors import BallTree
import numpy as np

def compute_exact_geodesic_nn_balltree(points_geom, urban_geom):
    """
    Compute geodesic distances (in kilometers) from input geometries to their nearest
    urban geometry using BallTree with haversine metric.

    All geometries must be in EPSG:4326.
    If inputs are not Points (e.g. Polygon or MultiPolygon), their centroids are used.
    """
    def ensure_points(geoms):
        """Convert Polygon/MultiPolygon to centroid if necessary."""
        return [geom.centroid if geom.geom_type != "Point" else geom for geom in geoms]

    # Ensure all are points
    points_geom_clean = ensure_points(points_geom)
    urban_geom_clean = ensure_points(urban_geom)

    # Extract lat/lon and convert to radians
    points_coords = np.array([[pt.y, pt.x] for pt in points_geom_clean])
    urban_coords = np.array([[pt.y, pt.x] for pt in urban_geom_clean])

    points_rad = np.radians(points_coords)
    urban_rad = np.radians(urban_coords)

    # BallTree NN query with haversine distance
    tree = BallTree(urban_rad, metric='haversine')
    dist_rad, _ = tree.query(points_rad, k=1)

    # Convert to kilometers
    dist_km = dist_rad.flatten() * 6371.0  # Earth radius in km

    return dist_km


def compute_or_load_distances_to_urban(gdf_points, gdf_urban_top, dist_path, id_col="id"):
    if os.path.exists(dist_path):
        print(f"Loading precomputed distances from {dist_path}...")
        with open(dist_path, "rb") as f:
            dist_dict = dill.load(f)
        return dist_dict['distances_to_urban_area']

    print("Computing distances to nearest urban areas using BallTree...")
    assert gdf_points.crs.to_string() == "EPSG:4326"
    assert gdf_urban_top.crs.to_string() == "EPSG:4326"

    # Project to CEA for accurate centroid computation
    cea_crs = "+proj=cea"
    gdf_points_proj = gdf_points.to_crs(cea_crs)
    gdf_urban_proj = gdf_urban_top.to_crs(cea_crs)

    # Compute centroids (only if not Point)
    def get_centroids(gdf_orig, gdf_proj):
        return gdf_proj.geometry.centroid if not all(gdf_orig.geom_type == "Point") else gdf_orig.geometry

    point_geoms = get_centroids(gdf_points, gdf_points_proj).to_crs("EPSG:4326")
    urban_geoms = get_centroids(gdf_urban_top, gdf_urban_proj).to_crs("EPSG:4326")

    distances = compute_exact_geodesic_nn_balltree(
        point_geoms,
        urban_geoms
    )
    id_array = gdf_points[id_col].astype(str).to_numpy()
    distance_dict = dict(zip(id_array, distances))

    os.makedirs(os.path.dirname(dist_path), exist_ok=True)
    with open(dist_path, "wb") as f:
        dill.dump({"distances_to_urban_area": distance_dict}, f)

    print(f"Saved distances for {len(distance_dict)} points to {dist_path}")
    return distance_dict


def compute_or_load_cluster_centroid_distances_to_urban(gdf_clusters, gdf_urban_top, dist_path):
    if os.path.exists(dist_path):
        print(f"Loading precomputed cluster distances from {dist_path}...")
        with open(dist_path, "rb") as f:
            dist_dict = dill.load(f)
        return dist_dict['distances_to_urban_area']

    print("Computing cluster centroid distances using BallTree...")

    assert gdf_clusters.crs.to_string() == "EPSG:4326"
    assert gdf_urban_top.crs.to_string() == "EPSG:4326"

    # Project to projected CRS for centroid accuracy
    cluster_centroids = (
        gdf_clusters.to_crs("+proj=cea").geometry.centroid.to_crs("EPSG:4326")
    )
    urban_centroids = (
        gdf_urban_top.to_crs("+proj=cea").geometry.centroid.to_crs("EPSG:4326")
    )

    distances = compute_exact_geodesic_nn_balltree(
        cluster_centroids,
        urban_centroids
    )

    index_strs = gdf_clusters.index.astype(str).to_numpy()
    distance_dict = dict(zip(index_strs, distances))

    os.makedirs(os.path.dirname(dist_path), exist_ok=True)
    with open(dist_path, "wb") as f:
        dill.dump({"distances_to_urban_area": distance_dict}, f)

    print(f"Saved cluster centroid distances for {len(distance_dict)} clusters to {dist_path}")
    return distance_dict


def dist_to_cost(distances, cap, scale='sqrt', alpha=0.1, epsilon=1e-6):
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
    
    raw_costs = np.array([min(cap, cost) for cost in raw_costs])

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

def save_cluster_dist_array(cluster_ids, distances, out_path):
    """
    Save dictionary with cluster IDs and distances as a dill pickle.
    """
    out_dict = {
        'cluster_ids': np.array(cluster_ids),
        'distances_to_urban_area': np.array(distances)
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        dill.dump(out_dict, f)

    print(f"Saved cluster distance array with {len(cluster_ids)} entries to {out_path}")

def save_cluster_cost_array(cluster_ids, costs, out_path):
    """
    Save dictionary with cluster IDs and costs as a dill pickle.
    """
    out_dict = {
        'cluster_ids': np.array(cluster_ids),
        'costs': np.array(costs)
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        dill.dump(out_dict, f)

    print(f"Saved cluster cost array with {len(cluster_ids)} entries to {out_path}")


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
    #usavars 
    for label in ["population", "treecover"]:
        for n_urban in [20, 50]:
            gdf_points = gpd.read_file(f"/home/libe2152/optimizedsampling/0_data/admin_gdfs/usavars/{label}/gdf_counties_2015.geojson")
            gdf_points = gdf_points.to_crs("EPSG:4326")

            pop_col = 'POP'
            gdf_urban_path = '/home/libe2152/optimizedsampling/0_data/boundaries/us/us_urban_area_census_2020/tl_2020_us_uac20_with_pop.shp'
            gdf_urban = gpd.read_file(gdf_urban_path)
            gdf_urban = gdf_urban.to_crs("EPSG:4326")
            gdf_urban_top = gdf_urban.nlargest(n_urban, pop_col)

            dist_dict_all = compute_or_load_distances_to_urban(gdf_points, gdf_urban, f"/home/libe2152/optimizedsampling/0_data/distances/usavars/{label}/distance_km_to_all_urban.pkl")

            dist_dict = compute_or_load_distances_to_urban(gdf_points, gdf_urban_top, f"/home/libe2152/optimizedsampling/0_data/distances/usavars/{label}/distance_km_to_top{n_urban}_urban.pkl")
            dists_all = np.array([dist_dict_all[str(i)] for i in gdf_points['id']])
            dists = np.array([dist_dict[str(i)] for i in gdf_points['id']])
            print(f"Min/Max/Mean distance (m): {dists.min():.2f} / {dists.max():.2f} / {dists.mean():.2f}")

            cap = 7
            alpha=0.01
            costs = dist_to_cost(dists, cap=cap, scale='linear', alpha=alpha)
            print(f"Cost stats -> Min: {costs.min():.4f}, Max: {costs.max():.4f}, Mean: {costs.mean():.4f}")

            # Save
            out_path = f"/home/libe2152/optimizedsampling/0_data/costs/usavars/{label}/convenience_costs/capped_{cap}_linear_distance_km_based_costs_top{n_urban}_urban_{alpha}.pkl"
            save_cost_array(gdf_points['id'], costs, out_path)
