import os
import dill
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points
from shapely.strtree import STRtree
from geopy.distance import geodesic
import matplotlib.pyplot as plt

class UrbanConvenienceSampler:
    def __init__(self, 
                id_col,
                gdf_points, 
                gdf_urban, 
                n_urban, 
                pop_col,
                cluster_col=None,
                points_per_cluster=None,
                crs="EPSG:4326",
                distances_dir='distances/distances_to_top{n_urban}_urban.pkl',
                cluster_distances_dir='distances/{cluster_col}_cluster_distances_to_top{n_urban}_urban.pkl',
                admin_ids=None):
        self.id_col = id_col
        self.gdf_points = gdf_points.to_crs(crs).copy()
        self.gdf_urban_top = gdf_urban.to_crs(crs).nlargest(n_urban, pop_col)  # assuming urban areas have a POP column
        self.crs = crs
        self.n_urban = n_urban
        self.distances_dir = distances_dir.format(n_urban=n_urban)

        cluster_col_str = '_'.join(cluster_col) if isinstance(cluster_col, list) else cluster_col
        self.cluster_distance_dir = cluster_distances_dir.format(n_urban=n_urban, cluster_col=cluster_col_str)

        self.points_per_cluster = points_per_cluster

        if isinstance(cluster_col, list):
            self.admin_ids_keys = cluster_col
        else:
            self.admin_ids_keys = [cluster_col]


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
            self.distances = self.gdf_points[self.id_col].astype(str).map(distance_map)
        else:
            print("Computing distances using spatial index...")
            tree = STRtree(self.gdf_urban_top.geometry.values)
            urban_geoms = self.gdf_urban_top.geometry.values
            def arc_distance_to_urban(polygon):
                centroid = polygon.centroid  #use the centroid of the polygon
                nearest_geom = tree.nearest(centroid)
                nearest_pt = nearest_points(centroid, tree.geometries.take(nearest_geom))[1]
                return geodesic((centroid.y, centroid.x), (nearest_pt.y, nearest_pt.x)).meters
            self.distances = self.gdf_points.geometry.apply(arc_distance_to_urban).to_numpy()

            id_array = self.gdf_points[self.id_col].astype(str).to_numpy()
            distance_dict = dict(zip(id_array, self.distances))

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                dill.dump({"distances_to_urban_area": distance_dict}, f)

            print(f"Saved distances for {len(distance_dict)} points to {save_path}")
        self.gdf_points["distance"] = self.distances

    def _load_or_compute_cluster_distances(self):
        save_path = self.cluster_distance_dir

        if os.path.exists(save_path):
            print(f"Loading precomputed cluster distances from {save_path}...")
            with open(save_path, "rb") as f:
                dist_dict = dill.load(f)
            distance_map = dist_dict["distances_to_urban_area"]
            self.gdf_clusters["distance"] = self.gdf_clusters.index.astype(str).map(distance_map)
            self.cluster_distances = self.gdf_clusters["distance"]
        else:
            print("Computing cluster distances using spatial index...")
            tree = STRtree(self.gdf_urban_top.geometry.values)

            def arc_distance_to_urban(geom):
                centroid = geom.centroid
                nearest_geom = tree.nearest(centroid)
                nearest_pt = nearest_points(centroid, tree.geometries.take(nearest_geom))[1]
                return geodesic((centroid.y, centroid.x), (nearest_pt.y, nearest_pt.x)).meters

            self.gdf_clusters["distance"] = self.gdf_clusters.geometry.apply(arc_distance_to_urban)
            self.cluster_distances = self.gdf_clusters["distance"]

            distance_dict = dict(zip(self.gdf_clusters.index.astype(str), self.cluster_distances))

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                dill.dump({"distances_to_urban_area": distance_dict}, f)

            print(f"Saved cluster distances for {len(distance_dict)} clusters to {save_path}")


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

        n_clusters = total_sample_size // self.points_per_cluster
        if total_sample_size % self.points_per_cluster != 0:
            raise ValueError("Total sample size must be divisible by points_per_cluster.")

        self._load_or_compute_cluster_distances()
        self.method = method
        self.seed = seed
        np.random.seed(seed)

        # Filter out clusters that don't have enough points
        valid_clusters = [
            c for c in self.cluster_distances.index
            if len(self.gdf_points[self.gdf_points[self.cluster_col] == c]) >= self.points_per_cluster
        ]

        if len(valid_clusters) < n_clusters:
            raise ValueError(f"Not enough valid clusters with at least {self.points_per_cluster} points. "
                            f"Needed: {n_clusters}, Available: {len(valid_clusters)}")

        cluster_distances = self.cluster_distances.loc[valid_clusters]

        if method == "deterministic":
            sampled_clusters = cluster_distances.nsmallest(n_clusters).index.tolist()
        else:
            # Min-max scale to [0, 1]
            norm_dists = (cluster_distances - cluster_distances.min()) / (cluster_distances.max() - cluster_distances.min())
            scores = -norm_dists / temp
            exp_scores = np.exp(scores - np.max(scores))  #softmax numerically stable
            probs = exp_scores / exp_scores.sum()
            sampled_clusters = np.random.default_rng(seed).choice(
                cluster_distances.index, size=n_clusters, replace=False, p=probs
            )

        sampled_df = self.gdf_points[self.gdf_points[self.cluster_col].isin(sampled_clusters)].copy()
        print(sampled_df.groupby(self.cluster_col).size())

        # Sample fixed number of points per cluster
        sampled_gdf = sampled_df.groupby(self.cluster_col).apply(
            lambda g: g.sample(n=self.points_per_cluster, random_state=seed)
        ).reset_index(drop=True)
        self.sampled_gdf = sampled_gdf
        self.sampled_ids = sampled_gdf[self.id_col].tolist()
        self.sample_size = len(self.sampled_ids)
        self.sampled_clusters = sampled_clusters


    def save_sampled_ids(self, out_path):
        if hasattr(self, "cluster_col"):
            subfolder = 'cluster_based'
            # CLUSTER-BASED SAMPLING
            if isinstance(self.cluster_col, str) and self.cluster_col == 'combined_cluster_id':
                # Use joined admin ids for the cluster columns
                cluster_admin_name = '_'.join([ADMIN_IDS[c] for c in self.admin_ids_keys])
            else:
                cluster_admin_name = ADMIN_IDS.get(self.cluster_col, self.cluster_col)
            file_name = f"IDS_cluster_{cluster_admin_name}_{self.points_per_cluster}_ppc_{len(self.sampled_clusters)}_clusters_{self.sample_size}_size_{self.method}_seed_{self.seed}.pkl"
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

        if hasattr(self, "cluster_col"):
            # CLUSTER-BASED SAMPLING
            if isinstance(self.cluster_col, str) and self.cluster_col == 'combined_cluster_id':
                # Use joined admin ids for the cluster columns
                cluster_admin_name = '_'.join([ADMIN_IDS[c] for c in self.admin_ids_keys])
            else:
                cluster_admin_name = ADMIN_IDS.get(self.cluster_col, self.cluster_col)

            title = f'Convenience Sampling Cluster\n({len(self.sampled_clusters)} clusters × {self.points_per_cluster} points)'
            filename = f'cluster_{cluster_admin_name}_{self.points_per_cluster}_ppc_{len(self.sampled_clusters)}_clusters_{self.method}_{self.sample_size}_size_seed_{self.seed}.png'
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
    ADMIN_IDS = {
        'STATEFP': 'state',
        'STATE_NAME': 'state',
        'COUNTYFP': 'county',
        'COUNTY_NAME': 'county'
    }

    cluster_col = ['COUNTY_NAME', 'COUNTYFP']
  # or another appropriate cluster ID
    points_per_cluster = 5   # example value, adjust as needed
    id_col="id"

    for label in ['population', 'treecover']:
        gdf_path = f"/home/libe2152/optimizedsampling/0_data/admin_gdfs/usavars/{label}/gdf_counties_2015.geojson"
        gdf = gpd.read_file(gdf_path)

        n_urban = 10
        pop_col = 'POP'
        gdf_urban_path = '/home/libe2152/optimizedsampling/0_data/boundaries/us/us_urban_area_census_2020/tl_2020_us_uac20_with_pop.shp'
        gdf_urban = gpd.read_file(gdf_urban_path)

        distances_dir = f'/home/libe2152/optimizedsampling/0_data/distances/usavars/{label}/distance_to_top{n_urban}_urban.pkl'
        cluster_distances_dir = f'/home/libe2152/optimizedsampling/0_data/distances/usavars/{label}/cluster_distance_to_top{n_urban}_urban.pkl'

        country_shape_file = '/home/libe2152/optimizedsampling/0_data/boundaries/us/us_states_provinces/ne_110m_admin_1_states_provinces.shp'
        exclude_names = ['Alaska', 'Hawaii', 'Puerto Rico']

        out_path = f'/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/convenience_sampling'

        method = 'probabilistic'
        temp = 0.025 if label == "population" else 0.001

        for desired_sample_size in range(100, 1100, 100):
            for seed in [1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415]:
                # Cluster Convenience Sampler (NEW)
                print("Initializing ClusterConvenienceSampler...")
                sampler = UrbanConvenienceSampler(  # make sure you have this class
                    gdf_points=gdf,
                    id_col=id_col,
                    pop_col=pop_col,
                    cluster_col=cluster_col,
                    points_per_cluster=points_per_cluster,
                    gdf_urban=gdf_urban,
                    n_urban=n_urban,
                    distances_dir=distances_dir,
                    cluster_distances_dir=cluster_distances_dir,
                    admin_ids=ADMIN_IDS
                )

                sampler.sample_by_clusters(
                    total_sample_size=desired_sample_size,
                    method=method,
                    temp=temp,
                    seed=seed
                )
                sampler.save_sampled_ids(out_path)
                sampler.plot(country_shape_file=country_shape_file, exclude_names=exclude_names)
