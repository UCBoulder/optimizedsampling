import os
import geopandas as gpd
import pandas as pd
import numpy as np
import dill
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import warnings

from pyproj import Geod
from shapely.ops import nearest_points

# Define WGS84 ellipsoid
GEOD = Geod(ellps="WGS84")

def geodesic_distance(geom1, geom2):
    """
    Compute geodesic distance between two geometries using nearest points.
    Assumes geometries are in EPSG:4326.
    """
    point1, point2 = nearest_points(geom1, geom2)
    lon1, lat1 = point1.x, point1.y
    lon2, lat2 = point2.x, point2.y
    _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return dist_m

class ClusterSampler:
    def __init__(self, 
                gdf_points,
                id_col,
                strata_col, 
                cluster_col,   # can be str or list
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
        self.strata_col = strata_col
        
        # Store original cluster_col for metadata/naming
        self.cluster_col_name = cluster_col

        # Handle cluster_col as list or string
        if isinstance(cluster_col, list):
            # Save keys for naming/metadata
            self.cluster_col_keys = cluster_col
            # Create combined cluster id col (joined string)
            combined_ids = self.gdf_points[cluster_col].astype(str).agg('_'.join, axis=1)
            self.gdf_points['combined_cluster_id'] = combined_ids
            self.cluster_col = 'combined_cluster_id'
        else:
            self.cluster_col_keys = [cluster_col]
            self.cluster_col = cluster_col

        self.ADMIN_IDS = ADMIN_IDS
        self.strata_dict = self._stratify_points()
        print(f"[Init] Found {len(self.strata_dict)} strata.")

    def _stratify_points(self):
        """Split points by strata."""
        print("[Stratify] Stratifying points by column:", self.strata_col)
        strata_values = self.gdf_points[self.strata_col].unique()
        print(f"[Stratify] Unique strata found: {strata_values}")
        return {val: self.gdf_points[self.gdf_points[self.strata_col] == val] for val in strata_values}

    def determine_sample_sizes_per_stratum(self, total_sample_size, seed, 
                                        n_strata=None, fixed_strata=None, 
                                        ignore_small_strata=True):
        """
        Return dict mapping strata -> desired sample size (number of points), proportional allocation.
        """
        print(f"[Determine Sample Sizes] Total desired sample size: {total_sample_size}")
        np.random.seed(seed)

        # Select strata
        if fixed_strata is not None:
            selected_strata = fixed_strata
        else:
            all_strata = list(self.strata_dict.keys())
            if ignore_small_strata:
                all_strata = [s for s in all_strata if len(self.strata_dict[s]) > 0]
            if n_strata is not None:
                if n_strata > len(all_strata):
                    raise ValueError("Requested more strata than available.")
                selected_strata = sorted(np.random.choice(all_strata, n_strata, replace=False))
            else:
                selected_strata = sorted(all_strata)

        print(f"[Determine Sample Sizes] Selected strata: {selected_strata}")

        # Compute total points in strata
        strata_sizes = {s: len(self.strata_dict[s]) for s in selected_strata}
        total_points = sum(strata_sizes.values())

        # Raw proportional allocations (floats)
        raw_allocations = {s: total_sample_size * (strata_sizes[s] / total_points) for s in selected_strata}

        # Floor to ints
        sample_sizes = {s: int(np.floor(raw_allocations[s])) for s in selected_strata}

        # Distribute remainder to strata with largest fractional remainders
        remainder = total_sample_size - sum(sample_sizes.values())
        remainders = {s: raw_allocations[s] - sample_sizes[s] for s in selected_strata}
        strata_by_remainder = sorted(remainders.items(), key=lambda x: -x[1])
        for i in range(remainder):
            s = strata_by_remainder[i % len(strata_by_remainder)][0]
            sample_sizes[s] += 1

        print(f"[Determine Sample Sizes] Final sample size per stratum:\n{sample_sizes}")
        return sample_sizes


    def sample_clusters_pps_until_target(self, gdf, points_per_cluster, num_target_points, seed):
        """
        Sample clusters one by one using PPS until total sampled points >= target_points.
        Returns sampled points and list of clusters sampled.
        """
        assert isinstance(self.cluster_col, str), "Something went wrong"
        all_counts = gdf[self.cluster_col].value_counts()

        if len(all_counts) == 0:
            print("[Sample Clusters] No eligible clusters to sample from.")
            return pd.DataFrame(columns=gdf.columns), []

        sampled_clusters = []
        sampled_points_list = []
        num_points_sampled = 0
        remaining = all_counts.copy()
        rng = np.random.default_rng(seed)
        i = 0

        while num_points_sampled < num_target_points and len(remaining) > 0:
            probs = remaining / remaining.sum()
            next_cluster = probs.sample(n=1, weights=probs, random_state=rng).index[0]

            sampled_clusters.append(next_cluster)
            remaining = remaining.drop(next_cluster)

            cluster_pts = gdf[gdf[self.cluster_col] == next_cluster]
            n_pts_to_sample = min(points_per_cluster, len(cluster_pts))

            sampled = cluster_pts.sample(n=n_pts_to_sample, random_state=seed + i)
            sampled_points_list.append(sampled)
            num_points_sampled += n_pts_to_sample
            i += 1

        if num_points_sampled > num_target_points:
            all_points = pd.concat(sampled_points_list).reset_index(drop=True)
            all_points = all_points.sample(n=num_target_points, random_state=seed).reset_index(drop=True)
            num_points_sampled = len(all_points)
        else:
            all_points = pd.concat(sampled_points_list).reset_index(drop=True)

        if num_points_sampled < num_target_points:
            print(f"Increasing points per cluster for strata: {gdf[self.strata_col].iloc[0]}")
            return self.sample_clusters_pps_until_target(gdf, points_per_cluster+5, num_target_points, seed)

        return all_points, sampled_clusters


    def sample(self, total_sample_size, points_per_cluster, seed, 
            n_strata=None, fixed_strata=None, ignore_small_strata=True):
        
        if fixed_strata is not None:
            strata_selection_mode = "fixed"
        elif n_strata is not None:
            strata_selection_mode = "random"
        else:
            raise ValueError("Either `fixed_strata` or `n_strata` must be provided.")
        self.strata_selection_mode = strata_selection_mode

        """Run the full sampling process."""
        print("[Sample] Starting sampling process...")

        sample_sizes_per_stratum = self.determine_sample_sizes_per_stratum(
            total_sample_size=total_sample_size,
            seed=seed,
            n_strata=n_strata,
            fixed_strata=fixed_strata,
            ignore_small_strata=ignore_small_strata
        )

        
        self.sampled_strata = list(sample_sizes_per_stratum.keys())
        self.sampled_clusters = {}
        all_sampled = []
        self.points_per_cluster = points_per_cluster

        for stratum in self.sampled_strata:
            print(f"[Sample] Processing stratum: {stratum}")
            gdf_stratum = self.strata_dict[stratum]
            target_points = sample_sizes_per_stratum[stratum]

            if target_points == 0: #some strata might have 0 allocation if they're very small
                continue

            sampled_points, sampled_clusters = self.sample_clusters_pps_until_target(
                gdf_stratum,
                points_per_cluster=self.points_per_cluster,
                num_target_points=target_points,
                seed=seed
            )

            self.sampled_clusters[stratum] = sampled_clusters
            all_sampled.append(sampled_points)

        all_sampled = pd.concat(all_sampled).reset_index(drop=True)

        self.sampled_gdf = all_sampled
        self.sampled_ids = all_sampled[self.id_col].tolist()
        self.sample_size = len(self.sampled_ids)
        self.n_strata = n_strata if n_strata is not None else None
        
        assert self.sample_size == total_sample_size, 'Something went wrong, incorrect sample size'

        self.seed = seed
        print(f"[Sample] Sampling complete. Total points sampled: {len(self.sampled_ids)}")
        return self.sampled_ids

    def reset_sample(self):
        """Reset all sampling-related attributes to prepare for a fresh sample."""
        print("[Reset Sample] Clearing previous sample state...")
        self.sampled_gdf = None
        self.sampled_ids = None
        self.sample_size = None
        self.seed = None
        self.out_dir = None
        self.sampled_strata = None
        self.sampled_clusters = None
        self.n_strata=None

    def save_sampled_ids(self, out_path):
        if self.strata_selection_mode == "fixed":
            strata_str = "-".join(sorted(self.sampled_strata))  # e.g., '06-12-36'
            subfolder = f"fixedstrata_{strata_str}"
        else:
            subfolder = "randomstrata"

        full_out_dir = os.path.join(out_path, subfolder)
        self.out_dir = full_out_dir
        os.makedirs(full_out_dir, exist_ok=True)

        # Create cluster name string for metadata / filenames
        if len(self.cluster_col_keys) > 1:
            county_keys = {'COUNTYFP', 'COUNTY_NAME'}

            if any(k in county_keys for k in self.cluster_col_keys):
                cluster_name_str = "county"
            else:
                cluster_name_str = "_".join([self.ADMIN_IDS.get(k, k) for k in self.cluster_col_keys])

            print(f"Clusters: {cluster_name_str}")
        else:
            cluster_name_str = self.ADMIN_IDS.get(self.cluster_col_name, self.cluster_col_name)
            print(f"Clusters: {cluster_name_str}")
        
        if hasattr(self, 'n_strata') and self.n_strata is not None:
            strata_str = f"{self.n_strata}_{self.ADMIN_IDS.get(self.strata_col, self.strata_col)}"
        else:
            strata_str = f"{self.ADMIN_IDS.get(self.strata_col, self.strata_col)}"

        file_name = f'sample_{strata_str}_{cluster_name_str}_desired_{self.points_per_cluster}ppc_{self.sample_size}_size_seed_{self.seed}.pkl'

        self.out_path = os.path.join(full_out_dir, file_name)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        assert not any(pd.isna(x) for x in self.sampled_ids), "Something went wrong, NaN values found in sampled_ids"

        sampling_metadata = {
            "sampled_ids": self.sampled_ids,
            "sampled_strata": self.sampled_strata,
            "sampled_clusters": self.sampled_clusters,
            "points_per_cluster": self.points_per_cluster,
            "total_sample_size": self.sample_size,
            "seed": self.seed,
            "strata_col": self.strata_col,
            "cluster_col": self.cluster_col_name  # original cluster col name or list
        }

        with open(self.out_path, "wb") as f:
            dill.dump(sampling_metadata, f)

        print(f"Saved {len(self.sampled_ids)} sampled IDs and metadata to {self.out_path}")

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

        if len(self.cluster_col_keys) > 1:
            cluster_name_str = "_".join([self.ADMIN_IDS.get(k, k) for k in self.cluster_col_keys])
        else:
            cluster_name_str = self.ADMIN_IDS.get(self.cluster_col_name, self.cluster_col_name)

        ax.set_title(f'Cluster Sampling\n(Stratified by {self.ADMIN_IDS.get(self.strata_col, self.strata_col)}, clustered by {cluster_name_str})\n'
                    f'{self.points_per_cluster} points per cluster; {self.sample_size} total points',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower left', fontsize=10)
        ax.axis('off')
        plt.tight_layout()

        plot_dir = os.path.join(self.out_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        if hasattr(self, 'n_strata') and self.n_strata is not None:
            strata_str = f"{self.n_strata}_{self.ADMIN_IDS.get(self.strata_col, self.strata_col)}"
        else:
            strata_str = f"{self.ADMIN_IDS.get(self.strata_col, self.strata_col)}"

        save_file_name = f'sample_{strata_str}_{cluster_name_str}_desired_{self.points_per_cluster}ppc_{self.sample_size}_size_seed_{self.seed}.png'
        save_path = os.path.join(plot_dir, save_file_name)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved plot to {save_path}")

if __name__=="__main__":
    ADMIN_IDS = {
        'STATEFP': 'state',
        'STATE_NAME': 'state',
        'COUNTYFP': 'county',
        'COUNTY_NAME': 'county'
    }

    data_path = f"/home/libe2152/optimizedsampling/0_data/admin_gdfs/togo/gdf_adm3.geojson"
    gdf = gpd.read_file(data_path)
    gdf['EA'] = gdf['combined_adm_id']
    gdf['prefecture'] = gdf['admin_2']
    gdf['region'] = gdf['admin_1']

    out_path = f'/home/libe2152/optimizedsampling/0_data/initial_samples/togo/cluster_sampling'

    country_shape_file = '/home/libe2152/togo/data/shapefiles/openAfrica/Shapefiles/tgo_admbnda_adm0_inseed_itos_20210107.shp'

    strata_col = 'prefecture'
    cluster_col = 'EA'

    n_strata = 5
    for points_per_cluster in [5, 10, 25]:
        sampler = ClusterSampler(gdf, id_col='id', strata_col=strata_col, cluster_col=cluster_col, ADMIN_IDS=ADMIN_IDS)
        for total_sample_size in range(100, 1100, 100):
            
            for seed in [1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415]:
                try:
                    sampler.sample(total_sample_size, points_per_cluster, seed, n_strata=n_strata)
                    sampler.save_sampled_ids(out_path)
                    sampler.plot(country_shape_file=country_shape_file)
                except Exception as e:
                    print(e)
                sampler.reset_sample()