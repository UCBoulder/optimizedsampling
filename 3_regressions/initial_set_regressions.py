import os
import dill
import numpy as np
import pandas as pd
from tqdm import tqdm

import csv

def load_data_from_pkl(features_path, dataset=None, label=None):
    if dataset != "togo":
        print(f"Loading features from {features_path} ...")
        try:
            with open(features_path, "rb") as f:
                arrs = dill.load(f)
            full_ids = arrs['ids_train']
            X_train_full = arrs['X_train']
            y_train_full = arrs['y_train']
            X_test = arrs['X_test']
            y_test = arrs['y_test']
            return full_ids, X_train_full, y_train_full, X_test, y_test
        except Exception as e:
            print(f"[ERROR] Could not load features: {e}")
            return
    else:
        print("Loading togo features...")
        try:
            with open(features_path, "rb") as f:
                arrs = dill.load(f)
            full_ids = arrs['ids_train']
            X_train_full = arrs['X_train']
            y_train_full = arrs[f'{label}_train']

            X_test = arrs['X_test']
            y_test = arrs[f'{label}_test']
            invalid_idxs = np.where(np.isnan(y_test))[0]
            valid_idxs = np.where(~np.isnan(y_test))[0]

            print(f"Number of invalid indices: {len(invalid_idxs)}")

            X_test = X_test[valid_idxs]
            y_test = y_test[valid_idxs]
            return full_ids, X_train_full, y_train_full, X_test, y_test
        except Exception as e:
            print(f"[ERROR] Could not load features: {e}")
            return


def sampling_r2_scores(
    features_path,
    sampling_dir,
    results_dir,
    ridge_regression_fn,
    metadata_parser,
    results_filename_suffix,
    min_samples=100,
    verbose=True,
    **kwargs
):
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{results_filename_suffix}.csv")
    file_exists = os.path.isfile(csv_path)

    full_ids, X_train_full, y_train_full, X_test, y_test = load_data_from_pkl(features_path, **kwargs)

    id_to_index = {str(id_): i for i, id_ in enumerate(full_ids)}

    if verbose:
        print(f"Processing samples from {sampling_dir} ...")

    # Open CSV file once for appending
    with open(csv_path, 'a', newline='') as csvfile:
        writer = None

        for fname in tqdm(sorted(os.listdir(sampling_dir)), desc=f"Processing samples"):
            if not fname.endswith(".pkl"):
                continue

            full_path = os.path.join(sampling_dir, fname)
            try:
                with open(full_path, "rb") as f:
                    loaded = dill.load(f)
                    
                    if isinstance(loaded, dict) and 'sampled_ids' in loaded:
                        sampled_ids = loaded['sampled_ids']
                    else:
                        sampled_ids = loaded  # assume it's already a list or compatible

                    sampled_ids = [x if isinstance(x, str) else str(x) for x in sampled_ids]
            except Exception as e:
                if verbose:
                    from IPython import embed; embed()
                    print(f"[WARNING] Failed to load {fname}: {e}")
                continue

            sampled_indices = [id_to_index[i] for i in sampled_ids if i in id_to_index]

            if len(sampled_indices) < min_samples:
                if verbose:
                    print(f"[SKIP] Not enough samples ({len(sampled_indices)}) in {fname}")
                continue

            X_subset = X_train_full[sampled_indices]
            y_subset = y_train_full[sampled_indices]

            try:
                r2 = ridge_regression_fn(X_subset, y_subset, X_test, y_test)
            except Exception as e:
                if verbose:
                    print(f"[ERROR] Ridge regression failed on {fname}: {e}")
                    from IPython import embed; embed()
                continue

            metadata = metadata_parser(fname, sampled_indices, r2)

            # Initialize writer with header once
            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=metadata.keys())
                if not file_exists:
                    writer.writeheader()
                    file_exists = True  # Only write header once

            writer.writerow(metadata)
            if verbose:
                print(f"[WRITE] Saved result from {fname} to {csv_path}")

    if verbose:
        print(f"Results written incrementally to {csv_path}")


# --- Metadata parsers for each sampling type ---

import re
import os

def parse_cluster_metadata(fname, sampled_indices, r2):
    base = os.path.basename(fname).replace(".pkl", "")
    parts = base.split("_")
    return {
        "filename": fname,
        "seed": parts[-1],                      # 1
        "sample_size": len(sampled_indices),
        "points_per_cluster": parts[-4].replace("ppc", ""),  # "2ppc" → "2"
        "strata": parts[3],                     # "county"
        "cluster": parts[4],                    # "county"
        "r2": r2,
    }

def parse_convenience_metadata(fname, sampled_indices, r2):
    base = os.path.basename(fname).replace(".pkl", "")
    parts = base.split("_")

    # Get source from the directory
    source = "unknown"
    if "urban_based" in fname:
        source = "urban_based"
    elif "region_based" in fname:
        source = "region_based"
    elif "cluster_based" in fname:
        source = "cluster_based"

    # Extract common metadata
    seed = parts[-1] if "seed" in parts else None
    method = parts[-2] if "seed" in parts else None

    # Initialize optional fields
    cluster_type = None
    points_per_cluster = None
    num_clusters = None

    num_urban = None
    for part in parts:
        match = re.match(r"top(\d+)", part)
        if match:
            num_urban = int(match.group(1))
            break

    if source == "cluster_based":
        # Look for "cluster" in parts and get the type following it
        if "cluster" in parts:
            cluster_idx = parts.index("cluster")
            cluster_type = parts[cluster_idx + 1]
        
        if "ppc" in parts:
            ppc_idx = parts.index("ppc")
            try:
                points_per_cluster = int(parts[ppc_idx - 1])
            except ValueError:
                points_per_cluster = None
        
        if "clusters" in parts:
            cluster_count_idx = parts.index("clusters")
            try:
                num_clusters = int(parts[cluster_count_idx - 1])
            except ValueError:
                num_clusters = None

        return {
            "filename": os.path.basename(fname),
            "sample_size": len(sampled_indices),
            "seed": seed,
            "source": source,
            "method": method,
            "num_urban": num_urban,
            "cluster_type": cluster_type,
            "points_per_cluster": points_per_cluster,
            "num_clusters": num_clusters,
            "r2": r2,
        }
    else:
        return {
            "filename": os.path.basename(fname),
            "sample_size": len(sampled_indices),
            "seed": seed,
            "source": source,
            "method": method,
            "num_urban": num_urban,
            "r2": r2,
        }



def parse_random_metadata(fname, sampled_indices, r2):
    m = re.search(r"random_sample_(\d+)_points_seed_(\d+)\.pkl$", fname)
    sample_size = int(m.group(1)) if m else None
    seed = int(m.group(2)) if m else None
    return {
        "filename": fname,
        "sample_size": sample_size or len(sampled_indices),
        "seed": seed,
        "r2": r2,
    }


# --- Specific functions for each sampling type ---

def cluster_sampling_r2_scores(*args, **kwargs):
    return sampling_r2_scores(
        *args,
        metadata_parser=parse_cluster_metadata,
        results_filename_suffix="cluster_sampling_r2_scores",
        **kwargs,
    )

def convenience_sampling_r2_scores(*args, **kwargs):
    return sampling_r2_scores(
        *args,
        metadata_parser=parse_convenience_metadata,
        results_filename_suffix="convenience_sampling_r2_scores",
        **kwargs,
    )

def random_sampling_r2_scores(*args, **kwargs):
    return sampling_r2_scores(
        *args,
        metadata_parser=parse_random_metadata,
        results_filename_suffix="random_sampling_r2_scores",
        **kwargs,
    )

if __name__ == "__main__":
    from regressions import ridge_regression

    CLUSTER_SAMPLING_DIR = {
        'population': "/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/population/cluster_sampling/fixedstrata_Idaho_16-Louisiana_22-Mississippi_28-New Mexico_35-Pennsylvania_42",
        "treecover": "/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/treecover/cluster_sampling/fixedstrata_Alabama_01-Colorado_08-Montana_30-New York_36-Ohio_39"
    }

    for label in ['population', 'treecover']:

        features_path = f"/home/libe2152/optimizedsampling/0_data/features/usavars/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl"
        cluster_sampling_dir= CLUSTER_SAMPLING_DIR[label]
        convenience_sampling_urban_dir = f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/convenience_sampling/urban_based"
        convenience_sampling_region_dir = f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/convenience_sampling/region_based"
        random_sampling_dir = f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/random_sampling"

        results_dir = f"/home/libe2152/optimizedsampling/0_results/usavars/{label}"

        # # Run cluster sampling R2 scores
        # cluster_sampling_r2_scores(
        #     features_path=features_path,
        #     sampling_dir=cluster_sampling_dir,
        #     results_dir=results_dir,
        #     ridge_regression_fn=ridge_regression,
        #     verbose=True,
        #     min_samples=100
        # )

        # Run convenience sampling R2 scores
        convenience_sampling_r2_scores(
            features_path=features_path,
            sampling_dir=convenience_sampling_urban_dir,
            results_dir=results_dir,
            ridge_regression_fn=ridge_regression,
            verbose=True,
            min_samples=100
        )

        # convenience_sampling_r2_scores(
        #     features_path=features_path,
        #     sampling_dir=convenience_sampling_region_dir,
        #     results_dir=results_dir,
        #     ridge_regression_fn=ridge_regression,
        #     verbose=True,
        #     min_samples=100
        # )

        # # Run random sampling R2 scores
        # random_sampling_r2_scores(
        #     features_path=features_path,
        #     sampling_dir=random_sampling_dir,
        #     results_dir=results_dir,
        #     ridge_regression_fn=ridge_regression,
        #     verbose=True,
        # )