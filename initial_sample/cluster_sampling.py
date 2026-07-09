import os
import pandas as pd
import numpy as np
import dill

from plotting_utils import plot_sampled_points_on_map

class ClusterSampler:
    def __init__(self, gdf_points, id_col, strata_col, cluster_col, ADMIN_IDS):
        self.gdf_points = gdf_points.copy()
        if self.gdf_points.crs != "EPSG:4326":
            self.gdf_points = self.gdf_points.to_crs("EPSG:4326")

        self.id_col = id_col
        self.strata_col = strata_col
        self.cluster_col_name = cluster_col

        if isinstance(cluster_col, list):
            self.cluster_col_keys = cluster_col
            self.gdf_points['combined_cluster_id'] = self.gdf_points[cluster_col].astype(str).agg('_'.join, axis=1)
            self.cluster_col = 'combined_cluster_id'
        else:
            self.cluster_col_keys = [cluster_col]
            self.cluster_col = cluster_col

        self.ADMIN_IDS = ADMIN_IDS
        self.strata_dict = self._stratify_points()

    def _stratify_points(self):
        strata_values = self.gdf_points[self.strata_col].unique()
        return {val: self.gdf_points[self.gdf_points[self.strata_col] == val] for val in strata_values}

    def determine_sample_sizes_per_stratum(self, total_sample_size, seed,
                                        n_strata=None, fixed_strata=None,
                                        ignore_small_strata=True):
        np.random.seed(seed)

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

        strata_sizes = {s: len(self.strata_dict[s]) for s in selected_strata}
        total_points = sum(strata_sizes.values())

        raw_allocations = {s: total_sample_size * (strata_sizes[s] / total_points) for s in selected_strata}
        sample_sizes = {s: int(np.floor(raw_allocations[s])) for s in selected_strata}

        # give leftover units to strata closest to rounding up
        remainder = total_sample_size - sum(sample_sizes.values())
        remainders = {s: raw_allocations[s] - sample_sizes[s] for s in selected_strata}
        strata_by_remainder = sorted(remainders.items(), key=lambda x: -x[1])
        for i in range(remainder):
            s = strata_by_remainder[i % len(strata_by_remainder)][0]
            sample_sizes[s] += 1

        return sample_sizes

    def sample_clusters_pps_until_target(self, gdf, points_per_cluster, num_target_points, seed):
        all_counts = gdf[self.cluster_col].value_counts()

        if len(all_counts) == 0 or num_target_points == 0:
            return pd.DataFrame(columns=gdf.columns), []

        sampled_clusters = []
        sampled_points_list = []
        num_points_sampled = 0
        remaining = all_counts.copy()
        rng = np.random.default_rng(seed)
        i = 0

        # draw clusters without replacement, weighted by remaining point counts
        while num_points_sampled < num_target_points and len(remaining) > 0:
            probs = remaining / remaining.sum()
            next_cluster = probs.sample(n=1, weights=probs, random_state=rng).index[0]

            sampled_clusters.append(next_cluster)
            remaining = remaining.drop(next_cluster)

            cluster_pts = gdf[gdf[self.cluster_col] == next_cluster]
            n_pts_to_sample = min(points_per_cluster, len(cluster_pts))

            sampled = cluster_pts.sample(n=n_pts_to_sample, random_state=seed + i)
            if num_points_sampled + n_pts_to_sample > num_target_points:
                break

            num_points_sampled += n_pts_to_sample
            sampled_points_list.append(sampled)
            i += 1

        if len(sampled_points_list) == 0:
            return pd.DataFrame(columns=gdf.columns), []

        all_points = pd.concat(sampled_points_list).reset_index(drop=True)
        return all_points, sampled_clusters

    def sample(self, total_sample_size, points_per_cluster, seed,
            n_strata=None, fixed_strata=None, ignore_small_strata=True):

        if fixed_strata is not None:
            self.strata_selection_mode = "fixed"
        elif n_strata is not None:
            self.strata_selection_mode = "random"
        else:
            raise ValueError("Either `fixed_strata` or `n_strata` must be provided.")

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
            gdf_stratum = self.strata_dict[stratum]
            target_points = sample_sizes_per_stratum[stratum]

            if target_points == 0:
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

        self.desired_sample_size = total_sample_size
        self.n_strata = n_strata if n_strata is not None else None

        self.seed = seed
        return self.sampled_ids

    def reset_sample(self):
        self.sampled_gdf = None
        self.sampled_ids = None
        self.sample_size = None
        self.desired_sample_size = None
        self.seed = None
        self.out_dir = None
        self.sampled_strata = None
        self.sampled_clusters = None
        self.n_strata = None

    def save_sampled_ids(self, out_path):
        if self.strata_selection_mode == "fixed":
            strata_str = "-".join(sorted(self.sampled_strata))
            subfolder = f"fixedstrata_{strata_str}"
        else:
            subfolder = "randomstrata"

        full_out_dir = os.path.join(out_path, subfolder)
        self.out_dir = full_out_dir
        os.makedirs(full_out_dir, exist_ok=True)

        if len(self.cluster_col_keys) > 1:
            county_keys = {'COUNTYFP', 'COUNTY_NAME'}
            if any(k in county_keys for k in self.cluster_col_keys):
                cluster_name_str = "county"
            else:
                cluster_name_str = "_".join([self.ADMIN_IDS.get(k, k) for k in self.cluster_col_keys])
        else:
            cluster_name_str = self.ADMIN_IDS.get(self.cluster_col_name, self.cluster_col_name)

        if hasattr(self, 'n_strata') and self.n_strata is not None:
            strata_str = f"{self.n_strata}_{self.ADMIN_IDS.get(self.strata_col, self.strata_col)}"
        else:
            strata_str = f"{self.ADMIN_IDS.get(self.strata_col, self.strata_col)}"

        file_name = f'sample_{strata_str}_{cluster_name_str}_{self.points_per_cluster}ppc_{self.desired_sample_size}_size_seed_{self.seed}.pkl'

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
            "cluster_col": self.cluster_col_name
        }

        with open(self.out_path, "wb") as f:
            dill.dump(sampling_metadata, f)

    def plot(self, country_shape_file, country_name=None, exclude_names=None):
        if len(self.cluster_col_keys) > 1:
            cluster_name_str = "_".join([self.ADMIN_IDS.get(k, k) for k in self.cluster_col_keys])
        else:
            cluster_name_str = self.ADMIN_IDS.get(self.cluster_col_name, self.cluster_col_name)

        if hasattr(self, 'n_strata') and self.n_strata is not None:
            strata_str = f"{self.n_strata}_{self.ADMIN_IDS.get(self.strata_col, self.strata_col)}"
        else:
            strata_str = f"{self.ADMIN_IDS.get(self.strata_col, self.strata_col)}"

        plot_dir = os.path.join(self.out_dir, 'plots')
        save_file_name = f'sample_{strata_str}_{cluster_name_str}_{self.points_per_cluster}ppc_{self.sample_size}_size_seed_{self.seed}.png'
        save_path = os.path.join(plot_dir, save_file_name)

        title = (f'Cluster Sampling\n(Stratified by {self.ADMIN_IDS.get(self.strata_col, self.strata_col)}, clustered by {cluster_name_str})\n'
                  f'{self.points_per_cluster} points per cluster; {self.sample_size} total points')

        plot_sampled_points_on_map(
            self.gdf_points, self.sampled_gdf, self.sample_size, country_shape_file,
            title=title, save_path=save_path, country_name=country_name, exclude_names=exclude_names,
        )
