import os
import dill
import pandas as pd
from tqdm import tqdm

import csv

def sampling_r2_scores(
    features_path,
    sampling_dir,
    results_dir,
    ridge_regression_fn,
    metadata_parser,
    results_filename_suffix,
    min_samples=10,
    verbose=True,
    **kwargs
):
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{results_filename_suffix}.csv")
    file_exists = os.path.isfile(csv_path)

    if verbose:
        print(f"Loading features from {features_path} ...")
    try:
        with open(features_path, "rb") as f:
            arrs = dill.load(f)
        full_ids = arrs['ids_train']
        X_train_full = arrs['X_train']
        y_train_full = arrs['y_train']
        X_test = arrs['X_test']
        y_test = arrs['y_test']
    except Exception as e:
        print(f"[ERROR] Could not load features: {e}")
        return

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
                    sampled_ids = dill.load(f)
                sampled_ids = [str(x) for x in sampled_ids]
            except Exception as e:
                if verbose:
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

def parse_cluster_metadata(fname, sampled_indices, r2):
    parts = fname.replace(".pkl", "").split("_")
    return {
        "filename": fname,
        "seed": parts[-1],
        "sample_size": len(sampled_indices),
        "points_per_cluster": parts[-8],
        "strata": parts[1],
        "cluster": parts[3],
        "r2": r2,
    }

def parse_convenience_metadata(fname, sampled_indices, r2):
    seed = None
    # Determine source type
    if "urban" in fname:
        source = "urban_based"
    elif "region" in fname:
        source = "region_based"
    else:
        source = "unknown"

    return {
        "filename": os.path.basename(fname),
        "sample_size": len(sampled_indices),
        "seed": seed,
        "source": source,
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

    for label in ['population', 'treecover']:

        features_path = f"/home/libe2152/optimizedsampling/0_data/features/usavars/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl"
        cluster_sampling_dir= f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/cluster_sampling"
        convenience_sampling_dir = f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/convenience_sampling"
        random_sampling_dir = f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/random_sampling"

        results_dir = f"/home/libe2152/optimizedsampling/0_results/usavars/{label}"

        # Run cluster sampling R2 scores
        cluster_sampling_r2_scores(
            features_path=features_path,
            sampling_dir=cluster_sampling_dir,
            results_dir=results_dir,
            ridge_regression_fn=ridge_regression,
            verbose=True,
        )

        # Run convenience sampling R2 scores
        convenience_sampling_r2_scores(
            features_path=features_path,
            sampling_dir=convenience_sampling_dir,
            results_dir=results_dir,
            ridge_regression_fn=ridge_regression,
            verbose=True,
        )

        # Run random sampling R2 scores
        random_sampling_r2_scores(
            features_path=features_path,
            sampling_dir=random_sampling_dir,
            results_dir=results_dir,
            ridge_regression_fn=ridge_regression,
            verbose=True,
        )