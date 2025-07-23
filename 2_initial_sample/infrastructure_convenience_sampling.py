import os
import dill
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

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

    points_geom_clean = ensure_points(points_geom)
    urban_geom_clean = ensure_points(urban_geom)

    points_coords = np.array([[pt.y, pt.x] for pt in points_geom_clean])
    urban_coords = np.array([[pt.y, pt.x] for pt in urban_geom_clean])

    points_rad = np.radians(points_coords)
    urban_rad = np.radians(urban_coords)

    tree = BallTree(urban_rad, metric='haversine')
    dist_rad, _ = tree.query(points_rad, k=1)

    dist_m = dist_rad.flatten() * 6371000 

    return dist_m

class UrbanConvenienceSampler:
    def __init__(self, 
                id_col,
                gdf_points, 
                gdf_urban=None,
                city_coord=None,
                n_urban=1, 
                pop_col=None,
                cluster_col=None,
                points_per_cluster=None,
                crs="EPSG:4326",
                distances_dir='distances/distances_to_top{n_urban}_urban.pkl',
                cluster_distances_dir='distances/{cluster_col}_cluster_distances_to_top{n_urban}_urban.pkl',
                admin_ids=None):
        self.id_col = id_col

        self.gdf_points = gdf_points.to_crs(crs).copy()

        if gdf_urban is not None:
            self.gdf_urban_top = gdf_urban.to_crs(crs).nlargest(n_urban, pop_col)
        elif city_coord is not None:
            print("Using single city coordinate as urban center.")
            lon, lat = city_coord
            city_point = Point(lon, lat)
            self.gdf_urban_top = gpd.GeoDataFrame({'geometry': [city_point]}, crs=crs)
        self.crs = crs
        self.n_urban = n_urban
        if distances_dir is not None:
            self.distances_dir = distances_dir.format(n_urban=n_urban)

        cluster_col_str = '_'.join(cluster_col) if isinstance(cluster_col, list) else cluster_col
        self.cluster_distance_dir = cluster_distances_dir.format(n_urban=n_urban, cluster_col=cluster_col_str)

        self.points_per_cluster = points_per_cluster

        self.admin_name = admin_ids[cluster_col[0]] if isinstance(cluster_col, list) else cluster_col
        self.admin_ids = admin_ids

        if cluster_col is not None:
            if isinstance(cluster_col, list):
                # Create combined cluster ID by joining string values of the columns
                combined_ids = self.gdf_points[cluster_col].astype(str).agg('_'.join, axis=1)
                self.gdf_points['combined_cluster_id'] = combined_ids
                self.cluster_col = 'combined_cluster_id'
            else:
                self.cluster_col = cluster_col

            self.gdf_clusters = self.gdf_points[[self.cluster_col, 'geometry']].drop_duplicates(self.cluster_col).set_index(self.cluster_col).copy()
            self.gdf_clusters = self.gdf_clusters.to_crs(crs)
        else:
            self.cluster_col = None

    def _load_or_compute_distances(self):
        save_path = self.distances_dir

        if os.path.exists(save_path):
            print(f"Loading precomputed distances from {save_path}...")
            with open(save_path, "rb") as f:
                dist_dict = dill.load(f)
            distance_map = dist_dict["distances_to_urban_area"]
            self.distances = distance_map  # Keep as dict
            # Map distances back to GeoDataFrame column as array for convenience
            self.gdf_points["distance"] = self.gdf_points[self.id_col].astype(str).map(distance_map).to_numpy()
        else:
            print("Computing distances using BallTree haversine nearest neighbor...")
            cluster_centroids = self.gdf_points.geometry
            urban_centroids = self.gdf_urban_top.to_crs("+proj=cea").geometry.centroid.to_crs("EPSG:4326")
            
            distances_array = compute_exact_geodesic_nn_balltree(cluster_centroids, urban_centroids)

            ids = self.gdf_points[self.id_col].astype(str).to_numpy()
            distance_map = dict(zip(ids, distances_array))
            self.distances = distance_map
            self.gdf_points["distance"] = distances_array

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                dill.dump({"distances_to_urban_area": distance_map}, f)

            print(f"Saved distances for {len(distance_map)} points to {save_path}")


    def _load_or_compute_cluster_distances(self):
        save_path = self.cluster_distance_dir

        if os.path.exists(save_path):
            print(f"Loading precomputed cluster distances from {save_path}...")
            with open(save_path, "rb") as f:
                dist_dict = dill.load(f)
            distance_map = dist_dict["distances_to_urban_area"]
            self.cluster_distances = distance_map  # Keep as dict
            self.gdf_clusters["distance"] = self.gdf_clusters.index.astype(str).map(distance_map).to_numpy()
        else:
            print("Computing cluster distances using BallTree haversine nearest neighbor...")
            cluster_centroids = self.gdf_clusters.to_crs("+proj=cea").geometry.centroid.to_crs("EPSG:4326")
            urban_centroids = self.gdf_urban_top.to_crs("+proj=cea").geometry.centroid.to_crs("EPSG:4326")

            distances_array = compute_exact_geodesic_nn_balltree(cluster_centroids, urban_centroids)

            cluster_ids = self.gdf_clusters.index.astype(str).to_numpy()
            distance_map = dict(zip(cluster_ids, distances_array))
            self.cluster_distances = distance_map
            self.gdf_clusters["distance"] = distances_array

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                dill.dump({"distances_to_urban_area": distance_map}, f)

            print(f"Saved cluster distances for {len(distance_map)} clusters to {save_path}")


    def sample(self, n_samples, method="deterministic", temp=0.025, seed=42):
        assert method in ["deterministic", "probabilistic"]
        self._load_or_compute_distances()

        self.method = method
        self.sample_size = n_samples

        if method == "deterministic":
            print("Running deterministic sampling...")
            gdf_shuffled = self.gdf_points.sample(frac=1, random_state=seed)
            sampled_gdf = gdf_shuffled.nsmallest(n_samples, "distance").reset_index(drop=True)
            self.num_samples = len(sampled_gdf)
            self.sampled_ids = sampled_gdf[self.id_col].tolist()
            self.sampled_gdf = sampled_gdf

        elif method == "probabilistic":
            print("Running probabilistic sampling...")
            distances = self.gdf_points["distance"].values
            distances_scaled = (distances - distances.min()) / (distances.max() - distances.min())
            scores = -distances_scaled / temp
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            sampled_gdf = self.gdf_points.sample(n=n_samples, weights=probs, random_state=seed).reset_index(drop=True)
            self.num_samples = len(sampled_gdf)
            self.sampled_ids = sampled_gdf[self.id_col].tolist()
            self.sampled_gdf = sampled_gdf
            self.seed = seed

    def sample_by_clusters(self, total_sample_size, method="probabilistic", temp=0.025, seed=42):
        assert self.cluster_col is not None and self.points_per_cluster is not None, \
            "Both cluster_col and points_per_cluster must be defined."

        if total_sample_size % self.points_per_cluster != 0:
            raise ValueError("Total sample size must be divisible by points_per_cluster.")

        self._load_or_compute_cluster_distances()
        self.method = method
        self.seed = seed
        np.random.seed(seed)

        # Step 1: Filter clusters with at least 1 point
        eligible_clusters = [
            c for c in self.cluster_distances.keys()
            if len(self.gdf_points[self.gdf_points[self.cluster_col] == c]) > 0
        ]

        cluster_distances = pd.Series(self.cluster_distances)
        cluster_distances = cluster_distances.loc[eligible_clusters]

        # Step 2: Rank or sample clusters
        if method == "deterministic":
            sorted_clusters = cluster_distances.nsmallest(len(cluster_distances)).index.tolist()
        else:
            norm_dists = (cluster_distances - cluster_distances.min()) / (cluster_distances.max() - cluster_distances.min())
            scores = -norm_dists / temp
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            sorted_clusters = np.random.default_rng(seed).choice(
                cluster_distances.index, size=len(cluster_distances), replace=False, p=probs
            )

        sampled_points = []
        selected_clusters = []
        points_collected = 0

        for c in sorted_clusters:
            cluster_df = self.gdf_points[self.gdf_points[self.cluster_col] == c]

            # shuffle and sample up to 10 points (or fewer if less available)
            cluster_sample = cluster_df.sample(
                n=min(self.points_per_cluster, len(cluster_df)), 
                random_state=seed
            )

            sampled_points.append(cluster_sample)
            selected_clusters.append(c)
            points_collected += len(cluster_sample)

            if points_collected >= total_sample_size:
                break

        combined_df = pd.concat(sampled_points, axis=0).reset_index(drop=True)

        if len(combined_df) < total_sample_size:
            raise ValueError(f"Insufficient total points: needed {total_sample_size}, got {len(combined_df)}")

        sampled_gdf = combined_df.sample(n=total_sample_size, random_state=seed)

        # Finalize
        self.sampled_gdf = sampled_gdf
        self.sampled_ids = sampled_gdf[self.id_col].tolist()
        self.sample_size = len(self.sampled_ids)
        self.sampled_clusters = selected_clusters


    def save_sampled_ids(self, out_path):
        if hasattr(self, "cluster_col") and self.cluster_col is not None:
            subfolder = 'cluster_based'
            file_name = f"IDS_top{self.n_urban}_urban_cluster_{self.admin_name}_{self.points_per_cluster}_ppc_{len(self.sampled_clusters)}_clusters_{self.sample_size}_size_{self.method}_seed_{self.seed}.pkl"
        else:
            # URBAN-BASED SAMPLING
            subfolder = "urban_based"
            file_name = f"IDS_top{self.n_urban}_urban_{self.num_samples}_points_{self.method}_{self.sample_size}_size_seed_{self.seed}.pkl"

        self.out_dir = os.path.join(out_path, subfolder)
        self.out_path = os.path.join(self.out_dir, file_name)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        with open(self.out_path, "wb") as f:
            dill.dump(self.sampled_ids, f)

        print(f"Saved {len(self.sampled_ids)} sampled IDs to {self.out_path}")


    def plot(self, country_shape_file, country_name=None, exclude_names=None):
        """Plot and save the map of sampled points."""
        fig, ax = plt.subplots(figsize=(12, 10))
        country = gpd.read_file(country_shape_file, engine="pyogrio")

        if country_name is not None:
            country = country[country['NAME'] == country_name]

        if exclude_names:
            country = country[~country['name'].isin(exclude_names)]

        country = country.to_crs('EPSG:4326')

        country.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=4, alpha=0.8)
        self.gdf_points.plot(ax=ax, color='#cccccc', markersize=5, label='All Points', zorder=1, alpha=0.6)
        self.sampled_gdf.plot(ax=ax, color='#d62728', markersize=5, label=f'Sampled ({self.sample_size})', zorder=3, alpha=0.8)

        if hasattr(self, "cluster_col") and self.cluster_col is not None:
            # CLUSTER-BASED SAMPLING
            title = f'Convenience Sampling Cluster\n({len(self.sampled_clusters)} clusters × {self.points_per_cluster} points)'
            filename = f'top{self.n_urban}_urban_areas_cluster_{self.admin_name}_{self.points_per_cluster}_ppc_{len(self.sampled_clusters)}_clusters_{self.method}_{self.sample_size}_size_seed_{self.seed}.png'
        else:
            title = f'Convenience Sampling Urban\n({self.n_urban} urban areas; {self.num_samples} points)'
            filename = f'top{self.n_urban}_urban_areas_{self.num_samples}_points_{self.method}_{self.sample_size}_size_seed_{self.seed}.png'

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10, title='Legend', title_fontsize=11, frameon=True)
        ax.set_aspect('equal')
        ax.axis('off')

        plot_dir = os.path.join(self.out_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    id_col = 'id'

    data_path = f"/home/libe2152/optimizedsampling/0_data/admin_gdfs/togo/gdf_adm3.geojson"
    gdf = gpd.read_file(data_path)
    train_ids = pd.read_csv('/share/togo/splits/train_ids.csv').astype(str)
    gdf = gdf[gdf['id'].astype(str).isin(train_ids.iloc[:, 0])].copy()
    
    gdf['EA'] = gdf['combined_adm_id']
    gdf['prefecture'] = gdf['admin_2']
    gdf['region'] = gdf['admin_1']

    lome_lat = 6.12874
    lome_lon = 1.22154

    distances_dir = f'/home/libe2152/optimizedsampling/0_data/distances/togo/distance_to_lome_urban.pkl'
    cluster_distances_dir = f'/home/libe2152/optimizedsampling/0_data/distances/togo/EA_distance_to_lome_urban.pkl'

    country_shape_file = '/home/libe2152/togo/data/shapefiles/openAfrica/Shapefiles/tgo_admbnda_adm0_inseed_itos_20210107.shp'

    out_path = f'/home/libe2152/optimizedsampling/0_data/initial_samples/togo/convenience_sampling'

    method = 'probabilistic'
    temp = 0.025

    for desired_sample_size in range(100, 1100, 100):
        for seed in [1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415]:
            # Cluster Convenience Sampler (NEW)
            print("Initializing ClusterConvenienceSampler...")
            sampler = UrbanConvenienceSampler(  # make sure you have this class
                gdf_points=gdf,
                id_col=id_col,
                city_coord=(lome_lon, lome_lat),
                n_urban=1,
                cluster_col='EA',
                points_per_cluster=10,
                cluster_distances_dir=cluster_distances_dir
            )

            sampler.sample_by_clusters(
                total_sample_size=desired_sample_size,
                method=method,
                temp=temp,
                seed=seed
            )
            sampler.save_sampled_ids(out_path)
            sampler.plot(country_shape_file=country_shape_file)