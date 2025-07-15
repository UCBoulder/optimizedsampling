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
                crs="EPSG:4326",
                distances_dir='distances/distances_to_top{n_urban}_urban.pkl'):
        self.id_col = id_col
        self.gdf_points = gdf_points.to_crs(crs).copy()
        self.gdf_urban_top = gdf_urban.to_crs(crs).nlargest(n_urban, pop_col)  # assuming urban areas have a POP column
        self.crs = crs
        self.n_urban = n_urban
        self.distances_dir = distances_dir.format(n_urban=n_urban)

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

    def save_sampled_ids(self, out_path):
        self.out_dir = os.path.join(out_path, "urban_based")

        file_name = f"IDS_top{self.n_urban}_urban_{self.num_samples}_points_{self.method}_seed_{self.seed}.pkl" if self.method == 'probabilistic' else f"deterministic/IDS_top{self.n_urban}_urban_{self.num_samples}_points_{self.method}.pkl"
        self.out_path = os.path.join(out_path, file_name)
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

        ax.set_title(f'Convenience Sampling Urban\n({self.n_urban} urban areas; {self.num_samples} points)', fontsize=16, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10, title='Legend', title_fontsize=11, frameon=True)
        ax.set_aspect('equal')
        ax.axis('off')

        plot_dir = os.path.join(self.out_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f'top{self.n_urban}_urban_areas_{self.num_samples}_points_{self.method}_seed_{self.seed}.png' if self.method == 'probabilistic' else f'top{self.n_urban}_urban_areas_{self.num_samples}_points_{self.method}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {save_path}")

class RegionConvenienceSampler:
    def __init__(self, 
                gdf_points,
                id_col,
                region_col,
                ADMIN_IDS):
        """
        Initialize with a GeoDataFrame and the relevant columns.
        """
        print("[Init] Initializing ClusterSampler...")
        self.gdf_points = gdf_points.copy()

        if self.gdf_points.crs != "EPSG:4326":
            print("Reprojecting to EPSG:4326 for geodesic calculations...")
            self.gdf_points = self.gdf_points.to_crs("EPSG:4326")
            
        self.id_col = id_col
        self.region_col = region_col
        self.ADMIN_IDS = ADMIN_IDS
        self.region_dict = self._determine_regions()
        print(f"[Init] Found {len(self.region_dict)} regions.")

    def _determine_regions(self):
        """Split points by regions."""
        print("[Determine Regions] Stratifying points by column:", self.region_col)
        region_values = self.gdf_points[self.region_col].unique()
        print(f"[Determine Region] Unique regions found: {region_values}")
        return {val: self.gdf_points[self.gdf_points[self.region_col] == val] for val in region_values}

    def sample(self, total_sample_size, region_val, seed):
        """Run the full sampling process."""
        print("[Sample] Starting sampling process...")

        gdf_region = self.region_dict[region_val]
        all_sampled = gdf_region.sample(n=total_sample_size, random_state=seed)

        assert len(all_sampled) == total_sample_size, f'Not enough points to sample in region {region_val}'

        self.sampled_gdf = all_sampled
        self.sampled_ids = all_sampled[self.id_col].tolist()
        self.num_samples = len(self.sampled_ids)
        self.seed = seed
        self.region_val = region_val
        print(f"[Sample] Sampling complete. Total points sampled: {len(self.sampled_ids)}")
        return self.sampled_ids

    def reset_sample(self):
        """Reset all sampling-related attributes to prepare for a fresh sample."""
        print("[Reset Sample] Clearing previous sample state...")
        self.sampled_gdf = None
        self.sampled_ids = None
        self.num_samples = None
        self.seed = None
        self.out_dir = None
        self.region_val = None

    def save_sampled_ids(self, out_path):
        self.out_dir = os.path.join(out_path, "region_based")

        file_name = f"IDS_region_{self.region_val}_{self.num_samples}_points_seed_{self.seed}.pkl"
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
        self.sampled_gdf.plot(ax=ax, color='#d62728', markersize=5, label=f'Sampled ({self.num_samples})', zorder=3, alpha=0.8)

        ax.set_title(f'Convenience Sampling by Region\n(Region {self.region_val}; {self.num_samples} points)', fontsize=16, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10, title='Legend', title_fontsize=11, frameon=True)
        ax.set_aspect('equal')
        ax.axis('off')

        plot_dir = os.path.join(self.out_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f'region_{self.region_val}_{self.num_samples}_points_seed_{self.seed}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {save_path}")

if __name__ == '__main__':
    id_col = 'condensed_shrug_id'
    gdf_path = '/share/india_secc/MOSAIKS/shrugs_with_all_admins.geojson'
    n_urban = 50
    pop_col = 'pc11_pca_tot_p_combined'
    country_shape_file = '/home/libe2152/optimizedsampling/0_data/boundaries/world/ne_10m_admin_0_countries.shp'
    country_name = 'India'
    out_path = f'/home/libe2152/optimizedsampling/0_data/initial_samples/india_secc/convenience_sampling'
    distances_dir = f'/home/libe2152/optimizedsampling/0_data/distances/india_secc/distance_to_top50_urban.pkl'

    print("Reading GeoDataFrame...")
    gdf = gpd.read_file(gdf_path)

    method = 'deterministic'
    # for desired_sample_size in range(1000, 1100, 100):
    #     print("Initializing ConvenienceSampler...")
    #     sampler = ConvenienceSampler(
    #         id_col=id_col,
    #         gdf_points=gdf,
    #         gdf_urban=gdf,
    #         n_urban=n_urban,
    #         pop_col=pop_col,
    #         distances_dir=distances_dir
    #     )

    #     sampler.sample(n_samples=desired_sample_size, method=method, seed=1) #seed needed to break ties
    #     sampler.save_sampled_ids(out_path)
    #     sampler.plot(country_shape_file=country_shape_file, country_name=country_name)

    method = 'probabilistic'
    for desired_sample_size in range(100, 1100, 100):
        for seed in [1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415]:

            print("Initializing ConvenienceSampler...")
            sampler = UrbanConvenienceSampler(
                id_col=id_col,
                gdf_points=gdf,
                gdf_urban=gdf,
                n_urban=n_urban,
                pop_col=pop_col,
                distances_dir=distances_dir
            )

            sampler.sample(n_samples=desired_sample_size, method=method, seed=seed)
            sampler.save_sampled_ids(out_path)
            sampler.plot(country_shape_file=country_shape_file, country_name=country_name)