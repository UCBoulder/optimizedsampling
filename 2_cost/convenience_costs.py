import os
import dill
import numpy as np
import geopandas as gpd

from sklearn.neighbors import BallTree
from shapely.geometry import Point

def compute_exact_geodesic_nn_balltree(points_geom, urban_geom):
    """
    Compute geodesic distances (in meters) from input geometries to their nearest
    urban geometry using BallTree with haversine metric.

    All geometries must be in EPSG:4326.
    If inputs are not Points (e.g. Polygon or MultiPolygon), their centroids are used.
    """
    def ensure_points(geoms):
        """Convert Polygon/MultiPolygon to centroid if necessary."""
        return [geom.centroid if not geom.geom_type == "Point" else geom for geom in geoms]

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

    # Convert to meters
    dist_m = dist_rad.flatten() * 6371000  # Earth radius in meters

    return dist_m


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
    # for N_urban in [10, 20, 50]:

    #     for label in ["population", "treecover"]:
    #         print(f"\n=== Processing label: {label} ===")

    #         # Load training data
    #         print("Loading training data...")
    #         with open(f"/home/libe2152/optimizedsampling/0_data/features/usavars/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl", "rb") as f:
    #             arrs = dill.load(f)

    #         invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
    #         ids_train = arrs['ids_train']
    #         valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
    #         ids = ids_train[valid_idxs]
    #         latlons = arrs['latlons_train'][valid_idxs]

    #         print(f"Loaded {len(ids)} valid points.")

    #         points = [Point(lon, lat) for lat, lon in latlons]
    #         gdf_points = gpd.GeoDataFrame({'id': ids}, geometry=points, crs="EPSG:4326")

    #         # Load urban area polygons
    #         print("Loading urban areas...")
    #         urban_areas_path = "/home/libe2152/optimizedsampling/0_data/boundaries/us/us_urban_area_census_2020/tl_2020_us_uac20_with_pop.shp"
    #         gdf_urban = gpd.read_file(urban_areas_path).to_crs("EPSG:4326")
    #         gdf_urban_top = gdf_urban.nlargest(N_urban, 'POP').copy()
    #         print(f"Selected top {N_urban} urban areas by population.")

    #         # Compute distance-based cost
    #         dist_dict =compute_or_load_distances_to_urban(gdf_points, gdf_urban_top, f"/home/libe2152/optimizedsampling/0_data/distances/usavars/{label}/distance_to_top{N_urban}_urban.pkl")
    #         dists = np.array([dist_dict[i] for i in ids])

    #         print(f"Min/Max/Mean distance (m): {dists.min():.2f} / {dists.max():.2f} / {dists.mean():.2f}")

    #         costs = dist_to_cost(dists, scale='sqrt', alpha=0.01)
    #         print(f"Cost stats -> Min: {costs.min():.4f}, Max: {costs.max():.4f}, Mean: {costs.mean():.4f}")

    #         # Save
    #         out_path = f"/home/libe2152/optimizedsampling/0_data/costs/usavars/{label}/convenience_costs/distance_based_costs_top{N_urban}_urban.pkl"
    #         save_cost_array(ids, costs, out_path)

    # for N_urban in [10, 20, 50]:

    #     for label in ["population", "treecover"]:
    #         gdf_points = gpd.read_file(f"/home/libe2152/optimizedsampling/0_data/admin_gdfs/usavars/{label}/gdf_counties_2015.geojson")
    #         gdf_counties = gpd.read_file("/home/libe2152/optimizedsampling/0_data/boundaries/us/us_county_2015/tl_2015_us_county.shp")
    #         gdf_counties['COUNTY_NAME'] = gdf_counties['NAME']

    #         # Load urban area polygons
    #         print("Loading urban areas...")
    #         urban_areas_path = "/home/libe2152/optimizedsampling/0_data/boundaries/us/us_urban_area_census_2020/tl_2020_us_uac20_with_pop.shp"
    #         gdf_urban = gpd.read_file(urban_areas_path).to_crs("EPSG:4326")
    #         gdf_urban_top = gdf_urban.nlargest(N_urban, 'POP').copy()
    #         print(f"Selected top {N_urban} urban areas by population.")

    #         cluster_cols = ['COUNTY_NAME', 'COUNTYFP']
    #         cluster_name = 'county'

    #         # Step 1: Create combined cluster IDs in both DataFrames
    #         gdf_counties['combined_cluster_id'] = gdf_counties[cluster_cols].astype(str).agg('_'.join, axis=1)
    #         gdf_points['combined_cluster_id'] = gdf_points[cluster_cols].astype(str).agg('_'.join, axis=1)

    #         cluster_col = 'combined_cluster_id'

    #         # Step 2: Create gdf_clusters with one geometry per unique cluster ID
    #         # This assumes gdf_counties contains the desired geometries
    #         gdf_clusters = (
    #             gdf_counties[[cluster_col, 'geometry']]
    #             .drop_duplicates(subset=cluster_col)
    #             .set_index(cluster_col)
    #             .to_crs("EPSG:4326")
    #             .copy()
    #         )

    #         cluster_ids = gdf_clusters.index.tolist()

    #         # Compute distance-based cost
    #         dist_dict = compute_or_load_cluster_centroid_distances_to_urban(gdf_clusters, gdf_urban_top, f"/home/libe2152/optimizedsampling/0_data/distances/usavars/{label}/{cluster_name}_distance_to_top{N_urban}_urban.pkl")
    #         dists = np.array([dist_dict[i] for i in cluster_ids])
    #         print(f"Min/Max/Mean distance (m): {dists.min():.2f} / {dists.max():.2f} / {dists.mean():.2f}")

    #         costs = dist_to_cost(dists, scale='sqrt', alpha=0.01)
    #         print(f"Cost stats -> Min: {costs.min():.4f}, Max: {costs.max():.4f}, Mean: {costs.mean():.4f}")

    #         # Save
    #         out_path = f"/home/libe2152/optimizedsampling/0_data/costs/usavars/{label}/convenience_costs/{cluster_name}_distance_based_costs_top{N_urban}_urban.pkl"
    #         save_cost_array(cluster_ids, costs, out_path)

    for N_urban in [10, 20, 50]:
        # Load training data
        print("Loading training data...")
        with open(f"/home/libe2152/optimizedsampling/0_data/features/india_secc/India_SECC_with_splits_4000.pkl", "rb") as f:
            arrs = dill.load(f)

        ids = arrs['ids_train']
        geometries = arrs['geometries_train']

        gdf_points = gpd.GeoDataFrame({'id': ids}, geometry=geometries, crs="EPSG:4326")

        # Load urban area polygons
        print("Loading urban areas...")
        gdf_path = '/share/india_secc/MOSAIKS/train_shrugs_with_admins.geojson'
        gdf_urban = gpd.read_file(gdf_path)

        pop_col = 'pc11_pca_tot_p_combined'
        gdf_urban_top = gdf_urban.nlargest(N_urban, pop_col).copy()
        print(f"Selected top {N_urban} urban areas by population.")
        # Compute distance-based cost
        dist_dict = compute_or_load_distances_to_urban(gdf_points, gdf_urban_top, f"/home/libe2152/optimizedsampling/0_data/distances/india_secc/distance_to_top{N_urban}_urban.pkl",
                                                       id_col='id')
        dists = np.array([dist_dict[str(i)] for i in ids])
        print(f"Min/Max/Mean distance (m): {dists.min():.2f} / {dists.max():.2f} / {dists.mean():.2f}")

        costs = dist_to_cost(dists, scale='sqrt', alpha=0.01)
        print(f"Cost stats -> Min: {costs.min():.4f}, Max: {costs.max():.4f}, Mean: {costs.mean():.4f}")

        # Save
        out_path = f"/home/libe2152/optimizedsampling/0_data/costs/india_secc/convenience_costs/distance_based_costs_top{N_urban}_urban.pkl"
        save_cost_array(ids, costs, out_path)

    for N_urban in [10, 20, 50]:

        data_path = "/share/india_secc/MOSAIKS/train_shrugs_with_admins.geojson"
        gdf_points = gpd.read_file(data_path)

        country_shape_file = '/home/libe2152/optimizedsampling/0_data/boundaries/world/ne_10m_admin_0_countries.shp'
        country_name = 'India'

        strata_col = 'pc11_s_id'
        cluster_col = 'pc11_d_id'
        cluster_name = 'district'

        # Load urban area polygons
        print("Loading urban areas...")
        gdf_path = '/share/india_secc/MOSAIKS/train_shrugs_with_admins.geojson'
        gdf_urban = gpd.read_file(gdf_path)

        pop_col = 'pc11_pca_tot_p_combined'
        gdf_urban_top = gdf_urban.nlargest(N_urban, pop_col).copy()
        print(f"Selected top {N_urban} urban areas by population.")

        if isinstance(cluster_col, list):
            # Create combined cluster ID by joining string values of the columns
            combined_ids = gdf_points[cluster_col].astype(str).agg('_'.join, axis=1)
            gdf_points['combined_cluster_id'] = combined_ids
            cluster_col = 'combined_cluster_id'
        else:
            cluster_col = cluster_col

        gdf_clusters = gdf_points[[cluster_col, 'geometry']].drop_duplicates(cluster_col).set_index(cluster_col).copy()
        gdf_clusters = gdf_clusters.to_crs("EPSG:4326")
        cluster_ids = gdf_clusters.index.unique().tolist()

        # Compute distance-based cost
        dist_dict = compute_or_load_cluster_centroid_distances_to_urban(gdf_clusters, gdf_urban_top, 
                                                                        f"/home/libe2152/optimizedsampling/0_data/distances/india_secc/{cluster_name}_distance_to_top{N_urban}_urban.pkl")
        dists = np.array([dist_dict[i] for i in cluster_ids])
        print(f"Min/Max/Mean distance (m): {dists.min():.2f} / {dists.max():.2f} / {dists.mean():.2f}")

        costs = dist_to_cost(dists, scale='sqrt', alpha=0.01)
        print(f"Cost stats -> Min: {costs.min():.4f}, Max: {costs.max():.4f}, Mean: {costs.mean():.4f}")

        # Save
        out_path = f"/home/libe2152/optimizedsampling/0_data/costs/india_secc/convenience_costs/{cluster_name}_distance_based_costs_top{N_urban}_urban.pkl"
        save_cost_array(cluster_ids, costs, out_path)
