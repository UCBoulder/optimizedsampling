import os
import dill
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points
from shapely.strtree import STRtree
from geopy.distance import geodesic
import matplotlib.pyplot as plt

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
