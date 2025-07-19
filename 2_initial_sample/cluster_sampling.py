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

    def determine_clusters(self, total_sample_size, points_per_cluster, seed, 
                        n_strata=None, fixed_strata=None, 
                        ignore_small_clusters=True, ignore_small_strata=True):
        """Determine how many clusters to sample per stratum such that 
        points_per_cluster * clusters == total_sample_size.

        Strategy:
        - If total_clusters <= num_strata:
            Assign 1 cluster to total_clusters many randomly chosen strata.
        - If total_clusters > num_strata:
            Assign floor(total_clusters / num_strata) clusters to each stratum.
            Then distribute the remaining clusters randomly (without exceeding availability).
        """
        print(f"[Determine Clusters] Total desired sample size: {total_sample_size}")
        np.random.seed(seed)
        if total_sample_size % points_per_cluster != 0:
            raise ValueError("Total sample size must be divisible by points_per_cluster.")
        
        total_clusters = total_sample_size // points_per_cluster

        strata_grouped = self.gdf_points.groupby(self.strata_col)

        eligible_strata = []
        for stratum, gdf in strata_grouped:
            if ignore_small_strata and len(gdf) < points_per_cluster:
                continue
            if ignore_small_clusters:
                cluster_sizes = gdf[self.cluster_col].value_counts()
                cluster_sizes = cluster_sizes[cluster_sizes >= points_per_cluster]
                if len(cluster_sizes) == 0:
                    continue
            eligible_strata.append(stratum)

        if fixed_strata is not None:
            # Validate user-specified fixed strata
            for s in fixed_strata:
                if s not in eligible_strata:
                    raise ValueError(f"Fixed stratum {s} is not eligible.")
            strata_vals = fixed_strata
        elif n_strata is not None:
            if len(eligible_strata) < n_strata:
                raise ValueError(f"Requested {n_strata} strata but only {len(eligible_strata)} eligible.")
            np.random.seed(seed)
            strata_vals = list(np.random.choice(eligible_strata, size=n_strata, replace=False))
        else:
            raise ValueError("Either `fixed_strata` or `n_strata` must be provided.")

        assert n_strata is not None, 'Need to specify number of strata to sample from.'
        if len(eligible_strata) < n_strata:
            raise ValueError(f"Requested {n_strata} strata but only {len(eligible_strata)} eligible.")
        np.random.seed(seed)
        strata_vals = list(np.random.choice(eligible_strata, size=n_strata, replace=False))

        num_strata = len(strata_vals)
        if num_strata == 0:
            raise ValueError("No eligible strata to assign clusters to.")

        cluster_counts = {stratum: 0 for stratum in strata_vals}

        if total_clusters <= num_strata:
            chosen_strata = np.random.choice(strata_vals, size=total_clusters, replace=False)
            for stratum in chosen_strata:
                cluster_counts[stratum] = 1
        else:
            #step 1: assign floor(total_clusters / num_strata) to all strata
            min_clusters_per_stratum = total_clusters // num_strata
            for stratum in strata_vals:
                gdf_stratum = self.gdf_points[self.gdf_points[self.strata_col] == stratum]
                cluster_sizes = gdf_stratum[self.cluster_col].value_counts()
                if ignore_small_clusters:
                    cluster_sizes = cluster_sizes[cluster_sizes >= points_per_cluster]
                max_available = len(cluster_sizes)
                cluster_counts[stratum] = min(min_clusters_per_stratum, max_available)

            #step 2: distribute remaining clusters
            assigned_clusters = sum(cluster_counts.values())
            remaining = total_clusters - assigned_clusters

            while remaining > 0:
                candidates = [
                    stratum for stratum in cluster_counts.keys()
                    if cluster_counts[stratum] < (
                        self.gdf_points[self.gdf_points[self.strata_col] == stratum][self.cluster_col]
                        .value_counts()
                        .loc[lambda x: x >= points_per_cluster if ignore_small_clusters else slice(None)]
                        .shape[0]
                    )
                ]
                if not candidates:
                    print("[Determine Clusters] No more eligible strata with available clusters.")
                    break

                np.random.shuffle(candidates)
                for stratum in candidates:
                    cluster_counts[stratum] += 1
                    remaining -= 1
                    if remaining == 0:
                        break

        print(f"[Determine Clusters] Cluster allocation per stratum:\n{cluster_counts}")
        return cluster_counts

    def sample_clusters(self, gdf, n_clusters, seed, points_per_cluster, ignore_small_clusters=True):
        print(f"[Sample Clusters] Sampling {n_clusters} clusters")

        counts = gdf[self.cluster_col].value_counts()

        if ignore_small_clusters:
            counts = counts[counts >= points_per_cluster]
            print(f"[Sample Clusters] Ignoring clusters with < {points_per_cluster} points. Remaining: {len(counts)}")

        if len(counts) == 0:
            print("[Sample Clusters] No eligible clusters to sample from.")
            return []

        probs = counts / counts.sum()

        try:
            sampled = probs.sample(n=n_clusters, weights=probs, replace=False, random_state=seed)
        except Exception as e:
            print(f"[Sample Clusters] Error during sampling: {e}")
            from IPython import embed; embed()

        print(f"[Sample Clusters] Sampled cluster IDs: {sampled.index.tolist()}")
        return sampled.index.tolist()


    def sample_points_within(self, gdf, clusters, points_per_cluster, seed):
        """Sample points within each selected cluster."""

        if not clusters:
            print("[Sample Points] No clusters provided — returning empty DataFrame.")
            return pd.DataFrame(columns=gdf.columns)

        print(f"[Sample Points] Sampling up to {points_per_cluster} points from each of {len(clusters)} clusters...")
        points = []
        for cluster in clusters:
            cluster_pts = gdf[gdf[self.cluster_col] == cluster]
            if len(cluster_pts) <= points_per_cluster:
                print(f"  [Sample Points] Using all {len(cluster_pts)} points from cluster {cluster}")
                sampled = cluster_pts
            else:
                print(f"  [Sample Points] Sampling {points_per_cluster} points from cluster {cluster}")
                sampled = cluster_pts.sample(n=points_per_cluster, random_state=seed)
            points.append(sampled)
        return pd.concat(points)

    def sample(self, total_sample_size, points_per_cluster, seed, 
            n_strata=None, fixed_strata=None, 
            ignore_small_clusters=True, ignore_small_strata=True):
        
        if fixed_strata is not None:
            strata_selection_mode = "fixed"
        elif n_strata is not None:
            strata_selection_mode = "random"
        else:
            raise ValueError("Either `fixed_strata` or `n_strata` must be provided.")
        self.strata_selection_mode = strata_selection_mode

        """Run the full sampling process."""
        print("[Sample] Starting sampling process...")

        clusters_per_stratum = self.determine_clusters(
            total_sample_size=total_sample_size,
            points_per_cluster=points_per_cluster,
            seed=seed,
            n_strata=n_strata,
            fixed_strata=fixed_strata,
            ignore_small_clusters=ignore_small_clusters,
            ignore_small_strata=ignore_small_strata
        )
        
        self.sampled_strata = list(clusters_per_stratum.keys())
        self.sampled_clusters = {}
        all_sampled = []

        for stratum in self.sampled_strata:
            print(f"[Sample] Processing stratum: {stratum}")
            gdf_stratum = self.strata_dict[stratum]
            n_clusters = clusters_per_stratum[stratum]
            if n_clusters == 0:
                continue

            sampled_clusters = self.sample_clusters(
                gdf_stratum, n_clusters, seed, points_per_cluster, ignore_small_clusters)
            self.sampled_clusters[stratum] = sampled_clusters

            sampled_points = self.sample_points_within(
                gdf_stratum, sampled_clusters, points_per_cluster, seed)
            all_sampled.append(sampled_points)

        all_sampled = pd.concat(all_sampled).reset_index(drop=True)

        self.sampled_gdf = all_sampled
        self.sampled_ids = all_sampled[self.id_col].tolist()
        self.points_per_cluster = points_per_cluster
        self.sample_size = len(self.sampled_ids)
        
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
            cluster_name_str = "_".join([self.ADMIN_IDS.get(k, k) for k in self.cluster_col_keys])
        else:
            cluster_name_str = self.ADMIN_IDS.get(self.cluster_col_name, self.cluster_col_name)

        file_name = f'sample_{self.ADMIN_IDS.get(self.strata_col, self.strata_col)}_{cluster_name_str}_{self.points_per_cluster}ppc_{self.sample_size}_size_seed_{self.seed}.pkl'

        self.out_path = os.path.join(full_out_dir, file_name)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

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

        save_file_name = f'sample_{self.ADMIN_IDS.get(self.strata_col, self.strata_col)}_{cluster_name_str}_{self.points_per_cluster}ppc_{self.sample_size}_size_seed_{self.seed}.png'
        save_path = os.path.join(plot_dir, save_file_name)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved plot to {save_path}")

if __name__=="__main__":
    ADMIN_IDS = {
        'pc11_s_id': 'state',
        'pc11_d_id': 'district',
        'pc11_sd_id': 'subdistrict'
    }

    data_path = "/share/india_secc/MOSAIKS/train_shrugs_with_admins.geojson"
    gdf = gpd.read_file(data_path)

    country_shape_file = '/home/libe2152/optimizedsampling/0_data/boundaries/world/ne_10m_admin_0_countries.shp'
    country_name = 'India'

    strata_col = 'pc11_s_id'
    cluster_col = 'pc11_d_id'

    out_path = f'/home/libe2152/optimizedsampling/0_data/initial_samples/india_secc/cluster_sampling'

    sampler = ClusterSampler(gdf, id_col='condensed_shrug_id', strata_col=strata_col, cluster_col=cluster_col, ADMIN_IDS=ADMIN_IDS)

    n_strata = 5
    for points_per_cluster in [50]:
        sampler.cluster_col = cluster_col
        #sampler.merge_small_strata(points_per_cluster)
        #sampler.merge_small_clusters(points_per_cluster)
        for total_sample_size in range(1000, 6000, 1000):
            
            for seed in [1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415]:
                try:
                    sampler.sample(total_sample_size, points_per_cluster, seed, n_strata=n_strata)
                    sampler.save_sampled_ids(out_path)
                    sampler.plot(country_shape_file=country_shape_file, country_name=country_name)
                except Exception as e:
                    print(e)
                    from IPython import embed; embed()
                sampler.reset_sample()