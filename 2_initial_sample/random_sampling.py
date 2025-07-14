import os
import dill
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

class RandomSampler:
    def __init__(self, gdf_points, id_col):
        self.gdf_points = gdf_points.copy()
        if self.gdf_points.crs != "EPSG:4326":
            self.gdf_points = self.gdf_points.to_crs("EPSG:4326")
        self.id_col = id_col
        self.sampled_gdf = None
        self.sampled_ids = None
        self.sample_size = None
        self.seed = None
        self.out_dir = None

    def sample(self, total_sample_size, seed=None):
        """Randomly sample points from the GeoDataFrame."""
        self.sampled_gdf = self.gdf_points.sample(n=total_sample_size, random_state=seed)
        self.sampled_ids = self.sampled_gdf[self.id_col].tolist()
        self.sample_size = len(self.sampled_ids)
        self.seed = seed
        print(f"[Random Sample] Sampled {self.sample_size} random points.")
        return self.sampled_ids

    def reset_sample(self):
        self.sampled_gdf = None
        self.sampled_ids = None
        self.sample_size = None
        self.seed = None
        self.out_dir = None

    def save_sampled_ids(self, out_path):
        self.out_dir = out_path
        os.makedirs(out_path, exist_ok=True)
        file_name = f"random_sample_{self.sample_size}_points_seed_{self.seed}.pkl"
        self.out_path = os.path.join(out_path, file_name)
        with open(self.out_path, "wb") as f:
            dill.dump(self.sampled_ids, f)
        print(f"[Random Sample] Saved IDs to {self.out_path}")

    def plot(self, country_shape_file, country_name=None, exclude_names=None):
        fig, ax = plt.subplots(figsize=(12, 10))
        country = gpd.read_file(country_shape_file)
        if country_name:
            country = country[country['NAME'] == country_name]
        if exclude_names:
            country = country[~country['name'].isin(exclude_names)]
        country = country.to_crs("EPSG:4326")
        country.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=4, alpha=0.8)
        self.gdf_points.plot(ax=ax, color='#cccccc', markersize=5, label='All Points', zorder=1, alpha=0.6)
        self.sampled_gdf.plot(ax=ax, color='#2ca02c', markersize=5, label=f'Random Sample ({self.sample_size})', zorder=3, alpha=0.9)
        ax.set_title(f'Random Sampling ({self.sample_size} points)', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower left', fontsize=10)
        ax.axis('off')
        plt.tight_layout()

        plot_dir = os.path.join(self.out_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"random_sample_{self.sample_size}_seed_{self.seed}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    label = 'treecover'

    data_path = f"/home/libe2152/optimizedsampling/0_data/admin_gdfs/usavars/{label}/gdf_counties_2015.geojson"
    gdf = gpd.read_file(data_path)

    out_path = f'/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/random_sampling'

    country_shape_file = '/home/libe2152/optimizedsampling/0_data/boundaries/us/us_states_provinces/ne_110m_admin_1_states_provinces.shp'
    exclude_names = ['Alaska', 'Hawaii', 'Puerto Rico']



    sampler = RandomSampler(gdf, id_col="id")

    for total_sample_size in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        for seed in [1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415]:
            sampler.sample(total_sample_size=total_sample_size, seed=seed)
            sampler.save_sampled_ids(out_path)
            sampler.plot(country_shape_file, exclude_names=["Alaska", "Hawaii", "Puerto Rico"])
            sampler.reset_sample()