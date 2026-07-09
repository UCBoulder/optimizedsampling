import os
import argparse
import dill
import numpy as np
import geopandas as gpd
import pandas as pd

from sklearn.neighbors import BallTree
from shapely.geometry import Point
from plotting_utils import plot_sampled_points_on_map

def compute_exact_geodesic_nn_balltree(points_geom, urban_geom):
    def ensure_points(geoms):
        return [geom.centroid if not geom.geom_type == "Point" else geom for geom in geoms]

    points_coords = np.array([[pt.y, pt.x] for pt in ensure_points(points_geom)])
    urban_coords = np.array([[pt.y, pt.x] for pt in ensure_points(urban_geom)])

    tree = BallTree(np.radians(urban_coords), metric='haversine')
    dist_rad, _ = tree.query(np.radians(points_coords), k=1)

    return dist_rad.flatten() * 6371000

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
            lon, lat = city_coord
            self.gdf_urban_top = gpd.GeoDataFrame({'geometry': [Point(lon, lat)]}, crs=crs)

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
                self.gdf_points['combined_cluster_id'] = self.gdf_points[cluster_col].astype(str).agg('_'.join, axis=1)
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
            with open(save_path, "rb") as f:
                distance_map = dill.load(f)["distances_to_urban_area"]
            self.distances = distance_map
            self.gdf_points["distance"] = self.gdf_points[self.id_col].astype(str).map(distance_map).to_numpy()
        else:
            urban_centroids = self.gdf_urban_top.to_crs("+proj=cea").geometry.centroid.to_crs("EPSG:4326")
            distances_array = compute_exact_geodesic_nn_balltree(self.gdf_points.geometry, urban_centroids)

            ids = self.gdf_points[self.id_col].astype(str).to_numpy()
            distance_map = dict(zip(ids, distances_array))
            self.distances = distance_map
            self.gdf_points["distance"] = distances_array

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                dill.dump({"distances_to_urban_area": distance_map}, f)

    def _load_or_compute_cluster_distances(self):
        save_path = self.cluster_distance_dir

        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                distance_map = dill.load(f)["distances_to_urban_area"]
            self.cluster_distances = distance_map
            self.gdf_clusters["distance"] = self.gdf_clusters.index.astype(str).map(distance_map).to_numpy()
        else:
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

    def sample(self, n_samples, method="deterministic", temp=0.025, seed=42):
        self._load_or_compute_distances()
        self.method = method
        self.sample_size = n_samples

        if method == "deterministic":
            gdf_shuffled = self.gdf_points.sample(frac=1, random_state=seed)
            sampled_gdf = gdf_shuffled.nsmallest(n_samples, "distance").reset_index(drop=True)
        else:
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
        self._load_or_compute_cluster_distances()
        self.method = method
        self.seed = seed
        np.random.seed(seed)

        eligible_clusters = [
            c for c in self.cluster_distances.keys()
            if len(self.gdf_points[self.gdf_points[self.cluster_col] == c]) > 0
        ]
        cluster_distances = pd.Series(self.cluster_distances).loc[eligible_clusters]

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
            cluster_sample = cluster_df.sample(n=min(self.points_per_cluster, len(cluster_df)), random_state=seed)

            sampled_points.append(cluster_sample)
            selected_clusters.append(c)
            points_collected += len(cluster_sample)

            if points_collected >= total_sample_size:
                break

        combined_df = pd.concat(sampled_points, axis=0).reset_index(drop=True)
        sampled_gdf = combined_df.sample(n=total_sample_size, random_state=seed)

        self.sampled_gdf = sampled_gdf
        self.sampled_ids = sampled_gdf[self.id_col].tolist()
        self.sample_size = len(self.sampled_ids)
        self.sampled_clusters = selected_clusters

    def save_sampled_ids(self, out_path):
        if hasattr(self, "cluster_col") and self.cluster_col is not None:
            subfolder = 'cluster_based'
            file_name = f"IDS_top{self.n_urban}_urban_cluster_{self.admin_name}_{self.points_per_cluster}_ppc_{len(self.sampled_clusters)}_clusters_{self.sample_size}_size_{self.method}_seed_{self.seed}.pkl"
        else:
            subfolder = "urban_based"
            file_name = f"IDS_top{self.n_urban}_urban_{self.num_samples}_points_{self.method}_{self.sample_size}_size_seed_{self.seed}.pkl"

        self.out_dir = os.path.join(out_path, subfolder)
        self.out_path = os.path.join(self.out_dir, file_name)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        with open(self.out_path, "wb") as f:
            dill.dump(self.sampled_ids, f)

    def plot(self, country_shape_file, country_name=None, exclude_names=None):
        if hasattr(self, "cluster_col") and self.cluster_col is not None:
            title = f'Convenience Sampling Cluster\n({len(self.sampled_clusters)} clusters × {self.points_per_cluster} points)'
            filename = f'top{self.n_urban}_urban_areas_cluster_{self.admin_name}_{self.points_per_cluster}_ppc_{len(self.sampled_clusters)}_clusters_{self.method}_{self.sample_size}_size_seed_{self.seed}.png'
        else:
            title = f'Convenience Sampling Urban\n({self.n_urban} urban areas; {self.num_samples} points)'
            filename = f'top{self.n_urban}_urban_areas_{self.num_samples}_points_{self.method}_{self.sample_size}_size_seed_{self.seed}.png'

        plot_dir = os.path.join(self.out_dir, "plots")
        save_path = os.path.join(plot_dir, filename)

        plot_sampled_points_on_map(
            self.gdf_points, self.sampled_gdf, self.sample_size, country_shape_file,
            title=title, save_path=save_path, country_name=country_name, exclude_names=exclude_names,
            legend_kwargs={'loc': 'lower left', 'fontsize': 10, 'title': 'Legend', 'title_fontsize': 11, 'frameon': True},
            equal_aspect=True, title_pad=None, use_tight_layout=False,
        )

def main():
    parser = argparse.ArgumentParser(description="Run convenience sampling over USAVars labels.")
    parser.add_argument("--labels", type=str, nargs="+", default=["population", "treecover"])
    parser.add_argument("--data_dir", type=str, default="../../0_data")
    parser.add_argument("--id_col", type=str, default="id")
    parser.add_argument("--urban_shp", type=str, default=None, help="Shapefile of candidate urban areas")
    parser.add_argument("--pop_col", type=str, default="population")
    parser.add_argument("--city_lon", type=float, default=None, help="Use a single city point instead of --urban_shp")
    parser.add_argument("--city_lat", type=float, default=None)
    parser.add_argument("--n_urban", type=int, default=1)
    parser.add_argument("--cluster_col", type=str, nargs="+", default=None,
                        help="Sample whole clusters (e.g. counties) instead of individual points")
    parser.add_argument("--points_per_cluster", type=int, default=5)
    parser.add_argument("--sample_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--method", type=str, default="deterministic", choices=["deterministic", "probabilistic"])
    parser.add_argument("--temp", type=float, default=0.025)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415])
    parser.add_argument("--country_shape_file", type=str,
                        default="../../0_data/boundaries/us/us_states_provinces/ne_110m_admin_1_states_provinces.shp")
    parser.add_argument("--exclude_names", type=str, nargs="+", default=["Alaska", "Hawaii", "Puerto Rico"])
    args = parser.parse_args()

    gdf_urban = gpd.read_file(args.urban_shp) if args.urban_shp else None
    city_coord = (args.city_lon, args.city_lat) if args.city_lon is not None else None
    cluster_col = args.cluster_col
    if cluster_col is not None:
        cluster_col = cluster_col[0] if len(cluster_col) == 1 else cluster_col

    for label in args.labels:
        gdf_points = gpd.read_file(f"{args.data_dir}/admin_gdfs/usavars/{label}/gdf_counties_2015.geojson")
        out_path = f"{args.data_dir}/initial_samples/usavars/{label}/convenience_sampling"

        sampler = UrbanConvenienceSampler(
            id_col=args.id_col, gdf_points=gdf_points, gdf_urban=gdf_urban, city_coord=city_coord,
            n_urban=args.n_urban, pop_col=args.pop_col, cluster_col=cluster_col,
            points_per_cluster=args.points_per_cluster, admin_ids={},
        )
        for total_sample_size in args.sample_sizes:
            for seed in args.seeds:
                if cluster_col is not None:
                    sampler.sample_by_clusters(total_sample_size, method=args.method, temp=args.temp, seed=seed)
                else:
                    sampler.sample(total_sample_size, method=args.method, temp=args.temp, seed=seed)
                sampler.save_sampled_ids(out_path)
                sampler.plot(args.country_shape_file, exclude_names=args.exclude_names)

if __name__ == '__main__':
    main()
