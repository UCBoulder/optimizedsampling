import os
import dill
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import re


def load_data_from_pkl(features_path, dataset=None, label=None):
    """Load training and test data from pickle file."""
    print(f"Loading features from {features_path} ...")
    
    try:
        with open(features_path, "rb") as f:
            arrs = dill.load(f)
        print("✓ Successfully loaded pickle file")
    except Exception as e:
        print(f"[ERROR] Could not load features: {e}")
        return None
    
    if dataset != "togo":
        return _extract_standard_data(arrs)
    else:
        return _extract_togo_data(arrs, label)


def _extract_standard_data(arrs):
    """Extract data for standard datasets."""
    print("Extracting standard dataset format...")
    full_ids = arrs['ids_train']
    X_train_full = arrs['X_train']
    y_train_full = arrs['y_train']
    X_test = arrs['X_test']
    y_test = arrs['y_test']
    
    print(f"Training samples: {len(X_train_full)}, Test samples: {len(X_test)}")
    return full_ids, X_train_full, y_train_full, X_test, y_test


def _extract_togo_data(arrs, label):
    """Extract data for Togo dataset with NaN filtering."""
    print(f"Extracting Togo dataset format for label: {label}")
    full_ids = arrs['ids_train']
    X_train_full = arrs['X_train']
    y_train_full = arrs[f'{label}_train']
    X_test = arrs['X_test']
    y_test = arrs[f'{label}_test']
    
    # Filter out NaN values
    invalid_idxs = np.where(np.isnan(y_test))[0]
    valid_idxs = np.where(~np.isnan(y_test))[0]
    
    print(f"Invalid indices (NaN): {len(invalid_idxs)}")
    print(f"Valid indices: {len(valid_idxs)}")
    
    X_test = X_test[valid_idxs]
    y_test = y_test[valid_idxs]
    
    print(f"Training samples: {len(X_train_full)}, Test samples: {len(X_test)} (after filtering)")
    return full_ids, X_train_full, y_train_full, X_test, y_test


def create_id_mapping(full_ids):
    """Create mapping from ID to index."""
    print(f"Creating ID mapping for {len(full_ids)} samples...")
    id_to_index = {str(id_): i for i, id_ in enumerate(full_ids)}
    print(f"✓ Created mapping for {len(id_to_index)} unique IDs")
    return id_to_index


def load_sample_file(filepath, verbose=True):
    """Load a single sample file and extract sampled IDs."""
    try:
        with open(filepath, "rb") as f:
            loaded = dill.load(f)
            
        if isinstance(loaded, dict) and 'sampled_ids' in loaded:
            sampled_ids = loaded['sampled_ids']
        else:
            sampled_ids = loaded  # assume it's already a list or compatible

        # Ensure all IDs are strings
        sampled_ids = [x if isinstance(x, str) else str(x) for x in sampled_ids]
        
        if verbose:
            print(f"✓ Loaded {len(sampled_ids)} IDs from {os.path.basename(filepath)}")
        
        return sampled_ids
        
    except Exception as e:
        if verbose:
            print(f"[WARNING] Failed to load {os.path.basename(filepath)}: {e}")
        return None


def get_valid_sample_indices(sampled_ids, id_to_index, min_samples, filepath, verbose=True):
    """Convert sampled IDs to indices and validate sample size."""
    sampled_indices = [id_to_index[i] for i in sampled_ids if i in id_to_index]
    
    if len(sampled_indices) < min_samples:
        if verbose:
            print(f"[SKIP] Not enough samples ({len(sampled_indices)}) in {os.path.basename(filepath)}")
        return None
    
    if verbose:
        print(f"✓ {len(sampled_indices)} valid indices found")
    
    return sampled_indices


def run_regression_analysis(X_subset, y_subset, X_test, y_test, ridge_regression_fn, filepath, verbose=True):
    """Run ridge regression analysis on subset data."""
    try:
        if verbose:
            print(f"Running regression on {len(X_subset)} training samples...")
        
        r2 = ridge_regression_fn(X_subset, y_subset, X_test, y_test)
        
        if verbose:
            print(f"✓ R² score: {r2:.4f}")
        
        return r2
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Ridge regression failed on {os.path.basename(filepath)}: {e}")
        return None


def write_result_to_csv(writer, metadata, csv_path, verbose=True):
    """Write a single result row to CSV."""
    writer.writerow(metadata)
    if verbose:
        print(f"✓ Saved result to {csv_path}")


def process_sampling_files(sampling_dir, id_to_index, X_train_full, y_train_full, X_test, y_test,
                         ridge_regression_fn, metadata_parser, min_samples, verbose=True):
    """Process all sampling files in a directory."""
    print(f"\nProcessing samples from {sampling_dir} ...")
    
    results = []
    sample_files = [f for f in sorted(os.listdir(sampling_dir)) if f.endswith(".pkl")]
    
    print(f"Found {len(sample_files)} sample files to process")
    
    for fname in tqdm(sample_files, desc="Processing samples"):
        full_path = os.path.join(sampling_dir, fname)
        
        if verbose:
            print(f"\n--- Processing {fname} ---")
        
        # Load sample file
        sampled_ids = load_sample_file(full_path, verbose)
        if sampled_ids is None:
            continue
        
        # Get valid indices
        sampled_indices = get_valid_sample_indices(sampled_ids, id_to_index, min_samples, full_path, verbose)
        if sampled_indices is None:
            continue
        
        # Extract subset data
        X_subset = X_train_full[sampled_indices]
        y_subset = y_train_full[sampled_indices]
        
        # Run regression
        r2 = run_regression_analysis(X_subset, y_subset, X_test, y_test, ridge_regression_fn, full_path, verbose)
        if r2 is None:
            continue
        
        # Parse metadata
        metadata = metadata_parser(fname, sampled_indices, r2)
        results.append(metadata)
        
        if verbose:
            print(f"✓ Successfully processed {fname}")
    
    print(f"\nCompleted processing {len(results)} files successfully")
    return results


def save_results_to_csv(results, csv_path, verbose=True):
    """Save all results to CSV file."""
    if not results:
        print("No results to save")
        return
    
    print(f"\nSaving {len(results)} results to {csv_path}...")
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
        
        if not file_exists:
            writer.writeheader()
            print("✓ Created new CSV file with headers")
        
        for result in results:
            writer.writerow(result)
    
    print(f"✓ Successfully saved all results to {csv_path}")


def sampling_r2_scores(features_path, sampling_dir, results_dir, ridge_regression_fn,
                      metadata_parser, results_filename_suffix, min_samples=100, verbose=True, **kwargs):
    """Main function to compute R² scores for sampling methods."""
    print(f"\n{'='*60}")
    print(f"STARTING SAMPLING R² ANALYSIS")
    print(f"{'='*60}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Load data
    data = load_data_from_pkl(features_path, **kwargs)
    if data is None:
        return
    
    full_ids, X_train_full, y_train_full, X_test, y_test = data
    
    # Create ID mapping
    id_to_index = create_id_mapping(full_ids)
    
    # Process all sampling files
    results = process_sampling_files(
        sampling_dir, id_to_index, X_train_full, y_train_full, X_test, y_test,
        ridge_regression_fn, metadata_parser, min_samples, verbose
    )
    
    # Save results
    csv_path = os.path.join(results_dir, f"{results_filename_suffix}.csv")
    save_results_to_csv(results, csv_path, verbose)
    
    print(f"\n{'='*60}")
    print(f"COMPLETED SAMPLING R² ANALYSIS")
    print(f"{'='*60}")


# --- Metadata parsers for each sampling type ---

def parse_cluster_metadata(fname, sampled_indices, r2):
    """Parse metadata for cluster sampling files."""
    base = os.path.basename(fname).replace(".pkl", "")
    parts = base.split("_")
    return {
        "filename": fname,
        "seed": parts[-1],
        "sample_size": len(sampled_indices),
        "points_per_cluster": parts[-4].replace("ppc", ""),
        "strata": parts[3],
        "cluster": parts[4],
        "r2": r2,
    }


def parse_convenience_metadata(fname, sampled_indices, r2):
    """Parse metadata for convenience sampling files."""
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

    # Extract num_urban
    for part in parts:
        match = re.match(r"top(\d+)", part)
        if match:
            num_urban = int(match.group(1))
            break

    if source == "cluster_based":
        # Extract cluster-specific metadata
        if "cluster" in parts:
            cluster_idx = parts.index("cluster")
            if cluster_idx + 1 < len(parts):
                cluster_type = parts[cluster_idx + 1]
        
        if "ppc" in parts:
            ppc_idx = parts.index("ppc")
            if ppc_idx > 0:
                try:
                    points_per_cluster = int(parts[ppc_idx - 1])
                except ValueError:
                    points_per_cluster = None
        
        if "clusters" in parts:
            cluster_count_idx = parts.index("clusters")
            if cluster_count_idx > 0:
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
    """Parse metadata for random sampling files."""
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
    """Run R² analysis for cluster sampling."""
    print("Running CLUSTER SAMPLING analysis...")
    return sampling_r2_scores(
        *args,
        metadata_parser=parse_cluster_metadata,
        results_filename_suffix="cluster_sampling_r2_scores",
        **kwargs,
    )


def convenience_sampling_r2_scores(*args, **kwargs):
    """Run R² analysis for convenience sampling."""
    print("Running CONVENIENCE SAMPLING analysis...")
    return sampling_r2_scores(
        *args,
        metadata_parser=parse_convenience_metadata,
        results_filename_suffix="convenience_sampling_r2_scores",
        **kwargs,
    )


def random_sampling_r2_scores(*args, **kwargs):
    """Run R² analysis for random sampling."""
    print("Running RANDOM SAMPLING analysis...")
    return sampling_r2_scores(
        *args,
        metadata_parser=parse_random_metadata,
        results_filename_suffix="random_sampling_r2_scores",
        **kwargs,
    )


def get_sampling_directories(dataset, label=None):
    """Get sampling directories for a given dataset and label."""
    if dataset == "usavars":
        cluster_dirs = {
            'population': "/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/population/cluster_sampling/fixedstrata_Idaho_16-Louisiana_22-Mississippi_28-New Mexico_35-Pennsylvania_42",
            "treecover": "/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/treecover/cluster_sampling/fixedstrata_Alabama_01-Colorado_08-Montana_30-New York_36-Ohio_39"
        }
        
        return {
            'features_path': f"/home/libe2152/optimizedsampling/0_data/features/usavars/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl",
            'cluster_sampling_dir': cluster_dirs[label],
            'convenience_sampling_urban_dir': f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/convenience_sampling/urban_based",
            'convenience_sampling_region_dir': f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/convenience_sampling/region_based",
            'random_sampling_dir': f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/random_sampling",
            'results_dir': f"/home/libe2152/optimizedsampling/0_results/usavars/{label}",
            'dataset': dataset
        }
    
    elif dataset == "togo":
        return {
            'features_path': "/home/libe2152/optimizedsampling/0_data/features/togo/togo_fertility_data_all_2022_Jul_Dec_P20.pkl",
            'cluster_sampling_dir': "/home/libe2152/optimizedsampling/0_data/initial_samples/togo/cluster_sampling/fixedstrata_plateaux",
            'convenience_sampling_urban_dir': None,  # Not available for Togo
            'convenience_sampling_region_dir': None,  # Not available for Togo
            'random_sampling_dir': None,  # Not available for Togo
            'results_dir': "/home/libe2152/optimizedsampling/0_results/togo",
            'dataset': dataset,
            'label': 'ph_h2o'  # Default label for Togo
        }
    
    elif dataset == "india_secc":
        return {
            'features_path': "/home/libe2152/optimizedsampling/0_data/features/india_secc/India_SECC_with_splits_4000.pkl",
            'cluster_sampling_dir': "/home/libe2152/optimizedsampling/0_data/initial_samples/india_secc/cluster_sampling/randomstrata",
            'convenience_sampling_urban_dir': "/home/libe2152/optimizedsampling/0_data/initial_samples/india_secc/convenience_sampling/urban_based",
            'convenience_sampling_region_dir': "/home/libe2152/optimizedsampling/0_data/initial_samples/india_secc/convenience_sampling/cluster_based",
            'random_sampling_dir': "/home/libe2152/optimizedsampling/0_data/initial_samples/india_secc/random_sampling",
            'results_dir': "/home/libe2152/optimizedsampling/0_results/india_secc",
            'dataset': dataset
        }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported datasets: 'usavars', 'togo', 'india_secc'")


def process_label(dataset, label=None, ridge_regression_fn=None, run_cluster=False, 
                 run_convenience_urban=True, run_convenience_region=False, run_random=False):
    """Process a single dataset/label with specified sampling methods."""
    print(f"\n{'#'*80}")
    if label:
        print(f"PROCESSING DATASET: {dataset.upper()} - LABEL: {label.upper()}")
    else:
        print(f"PROCESSING DATASET: {dataset.upper()}")
    print(f"{'#'*80}")
    
    dirs = get_sampling_directories(dataset, label)
    
    # Build common arguments
    common_args = {
        'features_path': dirs['features_path'],
        'results_dir': dirs['results_dir'],
        'ridge_regression_fn': ridge_regression_fn,
        'verbose': True,
        'min_samples': 100
    }
    
    # Add dataset-specific arguments
    if dirs.get('dataset'):
        common_args['dataset'] = dirs['dataset']
    if dirs.get('label'):
        common_args['label'] = dirs['label']
    
    # Run sampling methods based on availability and flags
    if run_cluster and dirs['cluster_sampling_dir']:
        print(f"\n--- Running cluster sampling for {dataset}" + (f"/{label}" if label else "") + " ---")
        cluster_sampling_r2_scores(sampling_dir=dirs['cluster_sampling_dir'], **common_args)
    elif run_cluster:
        print(f"[SKIP] Cluster sampling not available for {dataset}")
    
    if run_convenience_urban and dirs['convenience_sampling_urban_dir']:
        print(f"\n--- Running urban-based convenience sampling for {dataset}" + (f"/{label}" if label else "") + " ---")
        convenience_sampling_r2_scores(sampling_dir=dirs['convenience_sampling_urban_dir'], **common_args)
    elif run_convenience_urban:
        print(f"[SKIP] Urban-based convenience sampling not available for {dataset}")
    
    if run_convenience_region and dirs['convenience_sampling_region_dir']:
        print(f"\n--- Running region-based convenience sampling for {dataset}" + (f"/{label}" if label else "") + " ---")
        convenience_sampling_r2_scores(sampling_dir=dirs['convenience_sampling_region_dir'], **common_args)
    elif run_convenience_region:
        print(f"[SKIP] Region-based convenience sampling not available for {dataset}")
    
    if run_random and dirs['random_sampling_dir']:
        print(f"\n--- Running random sampling for {dataset}" + (f"/{label}" if label else "") + " ---")
        random_sampling_r2_scores(sampling_dir=dirs['random_sampling_dir'], **common_args)
    elif run_random:
        print(f"[SKIP] Random sampling not available for {dataset}")


def process_usavars_datasets(ridge_regression_fn):
    """Process US agricultural variables datasets."""
    print("\n" + "="*80)
    print("PROCESSING USAVARS DATASETS")
    print("="*80)
    
    labels = ['population', 'treecover']
    
    for label in labels:
        process_label(
            dataset="usavars",
            label=label,
            ridge_regression_fn=ridge_regression_fn,
            run_cluster=False,
            run_convenience_urban=True,
            run_convenience_region=False,
            run_random=False
        )


def process_togo_dataset(ridge_regression_fn):
    """Process Togo dataset."""
    print("\n" + "="*80)
    print("PROCESSING TOGO DATASET")
    print("="*80)
    
    process_label(
        dataset="togo",
        ridge_regression_fn=ridge_regression_fn,
        run_cluster=True,
        run_convenience_urban=False,
        run_convenience_region=False,
        run_random=False
    )


def process_india_dataset(ridge_regression_fn):
    """Process India SECC dataset."""
    print("\n" + "="*80)
    print("PROCESSING INDIA SECC DATASET")
    print("="*80)
    
    process_label(
        dataset="india_secc",
        ridge_regression_fn=ridge_regression_fn,
        run_cluster=False,
        run_convenience_urban=True,
        run_convenience_region=True,
        run_random=False
    )


def main():
    """Main function to run the analysis."""
    from regressions import ridge_regression
    
    print("STARTING SAMPLING R² SCORE ANALYSIS")
    print("="*80)
    
    # Process different datasets
    # process_usavars_datasets(ridge_regression)
    process_togo_dataset(ridge_regression)
    process_india_dataset(ridge_regression)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()