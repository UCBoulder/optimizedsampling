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
                cluster_col,
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
        self.cluster_col_name = cluster_col
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

    def merge_small_strata(self, points_per_cluster):
        """
        Merge strata that have fewer than points_per_cluster points.
        First try merging with touching (neighbor) strata, otherwise merge with closest stratum.
        Updates self.gdf_points and self.strata_dict accordingly.
        """
        print("[Stratum Merge] Checking for small strata...")
        
        # Get current stratum sizes and geometries
        stratum_sizes = {
            val: len(gdf) for val, gdf in self._stratify_points().items()
        }

        stratum_geoms = {
            val: unary_union(self.gdf_points[self.gdf_points[self.strata_col] == val].geometry)
            for val in stratum_sizes
        }

        # Track changes
        merged_map = {}
        updated = False

        # Work on a mutable copy of strata info
        strata_to_check = sorted(stratum_sizes.items(), key=lambda x: x[1])  # smallest first
        visited = set()

        for s, sz in strata_to_check:
            if sz >= points_per_cluster or s in visited:
                continue

            print(f"  [Stratum Merge] Stratum {s} too small ({sz} points). Trying to merge...")

            geom_s = stratum_geoms[s]

            # Try to merge with touching neighbor first
            neighbors = [
                o for o in stratum_sizes
                if o != s and o not in visited and geom_s.touches(stratum_geoms[o])
            ]

            if neighbors:
                # Pick smallest touching neighbor
                merge_target = min(neighbors, key=lambda o: stratum_sizes[o])
                print(f"    [Stratum Merge] Merging with touching neighbor stratum {merge_target}")
            else:
                # No touching neighbors; merge with closest stratum
                others = [o for o in stratum_sizes if o != s and o not in visited]
                if not others:
                    print(f"    [Stratum Merge] No other strata to merge with for {s}. Skipping.")
                    continue
                merge_target = min(others, key=lambda o: geodesic_distance(geom_s, stratum_geoms[o]))
                print(f"    [Stratum Merge] No neighbors. Merging with closest stratum {merge_target}")

            # Perform merge
            self.gdf_points.loc[self.gdf_points[self.strata_col] == s, self.strata_col] = merge_target
            visited.add(s)
            updated = True

        # Update strata_dict
        if updated:
            self.strata_dict = self._stratify_points()
            print("[Stratum Merge] Completed merging small strata.")
        else:
            print("[Stratum Merge] No strata needed merging.")

    def merge_small_clusters(self, points_per_cluster):
        """
        Iteratively merge clusters so all have at least points_per_cluster points.
        Always pick smallest unassigned cluster, merge with smallest touching neighbor or nearest cluster.
        """
        print("[Merge] Starting iterative merging until all clusters meet minimum size...")
        gdf = self.gdf_points.copy()
        gdf['merged_cluster_id'] = None

        for stratum_value in gdf[self.strata_col].unique():
            print(f"\n[Merge] Processing stratum {stratum_value}")
            gdf_stratum = gdf[gdf[self.strata_col] == stratum_value]

            cluster_counts = gdf_stratum[self.cluster_col].value_counts().to_dict()
            cluster_geoms = gdf_stratum.groupby(self.cluster_col)['geometry'].apply(lambda x: unary_union(x)).to_dict()

            all_clusters = {
                cid: (cluster_counts[cid], cluster_geoms[cid])
                for cid in cluster_counts.keys()
            }

            merged_id_counter = 0
            merged_clusters = {}

            while True:
                sorted_clusters = sorted(all_clusters.items(), key=lambda x: x[1][0])
                if not sorted_clusters:
                    break

                smallest_id, (smallest_size, smallest_geom) = sorted_clusters[0]
                if smallest_size >= points_per_cluster:
                    break

                neighbors = [
                    (cid, data) for cid, data in all_clusters.items()
                    if cid != smallest_id and smallest_geom.touches(data[1])
                ]
                if neighbors:
                    neighbors.sort(key=lambda x: x[1][0])
                    selected_id, (selected_size, selected_geom) = neighbors[0]
                else:
                    other_clusters = [
                        (cid, data) for cid, data in all_clusters.items() if cid != smallest_id
                    ]
                    if not other_clusters:
                        warnings.warn(f"Cluster {smallest_id} in stratum {stratum_value} isolated and cannot be merged further.")
                        break
                    selected_id, (selected_size, selected_geom) = min(
                        other_clusters,
                        key=lambda x: geodesic_distance(smallest_geom, x[1][1])
                    )
                new_id = f"{stratum_value}_merged_{merged_id_counter}"
                merged_id_counter += 1

                new_size = smallest_size + selected_size
                new_geom = unary_union([smallest_geom, selected_geom])

                del all_clusters[smallest_id]
                del all_clusters[selected_id]

                all_clusters[new_id] = (new_size, new_geom)

                merged_clusters[new_id] = {smallest_id, selected_id}
                print(f"  [Merge] Merged clusters {smallest_id} + {selected_id} -> {new_id} (size {new_size})")

            # Now assign merged IDs back to points
            # For clusters never merged, assign their original IDs
            final_mapping = {}

            # Start with identity mapping for unmerged original clusters
            for cid in cluster_counts.keys():
                final_mapping[cid] = cid

            # Recursively update mapping for merged clusters
            def expand_group(group_id):
                if group_id in merged_clusters:
                    expanded = set()
                    for sub_id in merged_clusters[group_id]:
                        expanded.update(expand_group(sub_id))
                    return expanded
                else:
                    return {group_id}

            # Replace merged IDs with expanded original cluster IDs
            for merged_id in merged_clusters.keys():
                original_ids = expand_group(merged_id)
                for oid in original_ids:
                    final_mapping[oid] = merged_id

            cluster_vals = gdf.loc[gdf[self.strata_col] == stratum_value, self.cluster_col].unique()
            missing = [cid for cid in cluster_vals if cid not in final_mapping]
            if missing:
                print(f"[Warning] {len(missing)} clusters in stratum {stratum_value} missing in final_mapping: {missing[:5]}")
                from IPython import embed; embed()

            # Map original cluster IDs in gdf to merged cluster IDs
            gdf.loc[gdf[self.strata_col] == stratum_value, 'merged_cluster_id'] = \
                gdf.loc[gdf[self.strata_col] == stratum_value, self.cluster_col].map(final_mapping)

            print(f"  [Merge] Completed merging for stratum {stratum_value}")

        # Update state
        self.gdf_points = gdf
        self.cluster_col = 'merged_cluster_id'
        self.strata_dict = self._stratify_points()
        print(f"[Merge] All strata merged. Using '{self.cluster_col}' as cluster column.")


    def determine_clusters(self, total_sample_size, points_per_cluster):
        """Determine how many clusters to sample per stratum such that 
        points_per_cluster * clusters == total_sample_size.
        Ensures that no stratum is assigned more clusters than it has available.
        """
        print(f"[Determine Clusters] Total desired sample size: {total_sample_size}")
        total_clusters = total_sample_size // points_per_cluster
        if total_sample_size % points_per_cluster != 0:
            raise ValueError("Total sample size must be divisible by points_per_cluster.")

        strata_sizes = {s: len(gdf) for s, gdf in self.strata_dict.items()}
        total_size = sum(strata_sizes.values())

        raw_allocations = {
            s: (strata_sizes[s] / total_size) * total_clusters
            for s in strata_sizes
        }

        # Start with integer part of allocations
        clusters_per_stratum = {s: int(raw_allocations[s]) for s in raw_allocations}
        remainders = {
            s: raw_allocations[s] - clusters_per_stratum[s] for s in raw_allocations
        }

        # Cap to available clusters
        cluster_capacities = {
            s: self.gdf_points[self.gdf_points[self.strata_col] == s][self.cluster_col].nunique()
            for s in strata_sizes
        }

        for s in clusters_per_stratum:
            if clusters_per_stratum[s] > cluster_capacities[s]:
                print(f"  [Adjust] Reducing {s} from {clusters_per_stratum[s]} to {cluster_capacities[s]} (max available clusters)")
                clusters_per_stratum[s] = cluster_capacities[s]

        # Redistribute leftover clusters
        assigned_clusters = sum(clusters_per_stratum.values())
        remaining = total_clusters - assigned_clusters
        if remaining > 0:
            eligible = {
                s: cluster_capacities[s] - clusters_per_stratum[s]
                for s in strata_sizes
                if clusters_per_stratum[s] < cluster_capacities[s]
            }

            sorted_eligible = sorted(eligible.items(), key=lambda x: -remainders[x[0]])
            i = 0
            while remaining > 0 and sorted_eligible:
                s, capacity_left = sorted_eligible[i % len(sorted_eligible)]
                if capacity_left > 0:
                    clusters_per_stratum[s] += 1
                    remaining -= 1
                    eligible[s] -= 1
                i += 1
                sorted_eligible = [(k, v) for k, v in eligible.items() if v > 0]

        print(f"[Determine Clusters] Final clusters per stratum: {clusters_per_stratum}")
        return clusters_per_stratum


    def sample_clusters(self, gdf, n_clusters, seed):
        """Sample clusters with probability proportional to size."""
        print(f"[Sample Clusters] Sampling {n_clusters} clusters")
        counts = gdf[self.cluster_col].value_counts()
        probs = counts / counts.sum()
        n_clusters = min(n_clusters, len(probs))
        try:
            sampled = probs.sample(n=n_clusters, weights=probs, replace=False, random_state=seed)
        except Exception as e:
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

    def sample(self, total_sample_size, points_per_cluster, seed):
        """Run the full sampling process."""
        print("[Sample] Starting sampling process...")
        clusters_per_stratum = self.determine_clusters(total_sample_size, points_per_cluster)
        all_sampled = []

        for stratum, gdf_stratum in self.strata_dict.items():
            print(f"[Sample] Processing stratum: {stratum}")
            n_clusters = clusters_per_stratum[stratum]
            if n_clusters == 0:
                continue

            sampled_clusters = self.sample_clusters(gdf_stratum, n_clusters, seed)
            
            sampled_points = self.sample_points_within(gdf_stratum, sampled_clusters, points_per_cluster, seed)
            all_sampled.append(sampled_points)

        all_sampled = pd.concat(all_sampled).reset_index(drop=True)

        self.sampled_gdf = all_sampled
        self.sampled_ids = all_sampled[self.id_col].tolist()
        self.points_per_cluster = points_per_cluster
        self.sample_size = len(self.sampled_ids)
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

    def save_sampled_ids(self, out_path):
        self.out_dir = out_path

        file_name = 'IDs_{strata_name}_strata_{cluster_name}_clusters_{points_per_cluster}_points_per_cluster_{sample_size}_size_seed_{seed}.pkl'
        file_name = file_name.format(strata_name=self.ADMIN_IDS[self.strata_col], 
                                    cluster_name=self.ADMIN_IDS[self.cluster_col_name], 
                                    points_per_cluster=self.points_per_cluster, 
                                    sample_size=self.sample_size,
                                    seed=self.seed)
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

        ax.set_title(f'Cluster Sampling\n(Stratified by {self.ADMIN_IDS[self.strata_col]}, clustered by {self.ADMIN_IDS[self.cluster_col_name]})\n'
                    f'{self.points_per_cluster} points per cluster; {self.sample_size} total points',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower left', fontsize=10)
        ax.axis('off')
        plt.tight_layout()

        plot_dir = os.path.join(self.out_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        save_path = '{strata_name}_strata_{cluster_name}_clusters_{points_per_cluster}_points_per_cluster_{sample_size}_size_seed_{seed}.png'
        save_path = save_path.format(strata_name=self.ADMIN_IDS[self.strata_col], 
                                    cluster_name=self.ADMIN_IDS[self.cluster_col_name], 
                                    points_per_cluster=self.points_per_cluster, 
                                    sample_size=self.sample_size,
                                    seed=self.seed)
        save_path = os.path.join(plot_dir, save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    ADMIN_IDS = {
        'STATEFP': 'state',
        'STATE_NAME': 'state',
        'COUNTYFP': 'county',
        'COUNTY_NAME': 'county'
    }

    for label in ['population', 'treecover']:

        data_path = f"/home/libe2152/optimizedsampling/0_data/admin_gdfs/usavars/{label}/gdf_counties_2015.geojson"
        gdf = gpd.read_file(data_path)

        out_path = f'/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/cluster_sampling'

        country_shape_file = '/home/libe2152/optimizedsampling/0_data/boundaries/us/us_states_provinces/ne_110m_admin_1_states_provinces.shp'
        exclude_names = ['Alaska', 'Hawaii', 'Puerto Rico']

        strata_col = 'STATEFP'
        cluster_col = 'COUNTYFP'

        sampler = ClusterSampler(gdf, id_col='id', strata_col=strata_col, cluster_col=cluster_col, ADMIN_IDS=ADMIN_IDS)

        for points_per_cluster in [2, 5, 10, 25]:
            sampler.cluster_col = cluster_col
            sampler.merge_small_strata(points_per_cluster)
            sampler.merge_small_clusters(points_per_cluster)
            for total_sample_size in [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]:
                
                for seed in [1, 42, 123, 456, 789, 1234, 5678, 9101, 1213, 1415]:
                    try:
                        sampler.sample(total_sample_size, points_per_cluster, seed=seed)
                        sampler.save_sampled_ids(out_path)
                        sampler.plot(country_shape_file=country_shape_file, exclude_names=exclude_names)
                    except Exception as e:
                        print(e)
                    sampler.reset_sample()