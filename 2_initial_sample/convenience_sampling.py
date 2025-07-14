import os
import dill
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points
from shapely.strtree import STRtree
from geopy.distance import geodesic
import matplotlib.pyplot as plt

class ConvenienceSampler:
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
            self.distances = self.gdf_points[self.id_col].map(distance_map)
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
        self.out_dir = out_path

        file_name = f"IDS_top{self.n_urban}_urban_{self.num_samples}_points_{self.method}_seed_{self.seed}.pkl" if self.method == 'probabilistic' else f"IDS_top{self.n_urban}_urban_{self.num_samples}_points_{self.method}.pkl"
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

        ax.set_title(f'Convenience Sampling\n({self.n_urban} urban areas; {self.num_samples} points)', fontsize=16, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10, title='Legend', title_fontsize=11, frameon=True)
        ax.set_aspect('equal')
        ax.axis('off')

        os.makedirs(f'convenience_sampling/plots', exist_ok=True)
        save_path = f'convenience_sampling/plots/top{self.n_urban}_urban_areas_{self.num_samples}_points_{self.method}_seed_{self.seed}.png' if self.method == 'probabilistic' else f'convenience_sampling/plots/top{self.n_urban}_urban_areas_{self.num_samples}_points_{self.method}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {save_path}")

if __name__ == '__main__':
    id_col = 'id'
    label = 'treecover'
    gdf_path = f"/home/libe2152/optimizedsampling/0_data/admin_gdfs/usavars/{label}/gdf_counties_2015.geojson"
    gdf = gpd.read_file(gdf_path)

    n_urban = 50
    pop_col = 'POP'
    gdf_urban_path = '/home/libe2152/optimizedsampling/0_data/boundaries/us/us_urban_area_census_2020/tl_2020_us_uac20_with_pop.shp'
    gdf_urban = gpd.read_file(gdf_urban_path)

    distances_dir = f'/home/libe2152/optimizedsampling/0_data/distances/usavars/{label}/distance_to_top50_urban.pkl'

    country_shape_file = '/home/libe2152/optimizedsampling/0_data/boundaries/us/us_states_provinces/ne_110m_admin_1_states_provinces.shp'
    exclude_names = ['Alaska', 'Hawaii', 'Puerto Rico']

    out_path = f'/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/convenience_sampling'

    print("Reading GeoDataFrame...")
    gdf = gpd.read_file(gdf_path)

    method = 'deterministic'
    for desired_sample_size in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        print("Initializing ConvenienceSampler...")
        sampler = ConvenienceSampler(
            id_col=id_col,
            gdf_points=gdf,
            gdf_urban=gdf_urban,
            n_urban=n_urban,
            pop_col=pop_col,
            distances_dir=distances_dir
        )

        sampler.sample(n_samples=desired_sample_size, method=method, seed=1) #seed needed to break ties
        sampler.save_sampled_ids(out_path)
        sampler.plot(country_shape_file=country_shape_file, exclude_names=exclude_names)

    for method in ['probabilistic']:
        for desired_sample_size in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            for seed in [1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415]:

                print("Initializing ConvenienceSampler...")
                sampler = ConvenienceSampler(
                    id_col=id_col,
                    gdf_points=gdf,
                    gdf_urban=gdf_urban,
                    n_urban=n_urban,
                    pop_col=pop_col,
                    distances_dir=distances_dir
                )

                sampler.sample(n_samples=desired_sample_size, method=method, seed=seed)
                sampler.save_sampled_ids(out_path)
                sampler.plot(country_shape_file=country_shape_file, exclude_names=exclude_names)