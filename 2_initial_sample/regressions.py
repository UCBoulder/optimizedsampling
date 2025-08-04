import os
import dill
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

'''
Run ridge regression and return R2 score
'''
def ridge_regression(X_train, 
                     y_train, 
                     X_test, 
                     y_test, 
                     n_folds=5, 
                     alphas=np.logspace(-5, 5, 10)):
    
    n_samples = X_train.shape[0]

    if n_samples < 2*n_folds:
        print("Not enough samples for cross-validation.")
        return
     
    print("Fitting regression...")

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    param_grid = {
        'ridge__alpha': alphas
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    ridge_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',        
        cv=cv,
        n_jobs=-1                   #parallelize across folds
    )


    def evaluate_r2(model, X_test, y_test):
        return model.score(X_test, y_test)
    
    ridge_search.fit(X_train, y_train)
    r2 = evaluate_r2(ridge_search, X_test, y_test)

    if abs(r2) > 1:
        print("Warning: Severe overfitting. Add more samples.")
    return r2


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


def parse_seeded_filename(filename):
    """Parse filename to extract metadata including seed."""
    base = os.path.basename(filename).replace(".pkl", "")
    
    # Extract seed (should be at the end)
    seed_match = re.search(r'seed_(\d+)$', base)
    if not seed_match:
        return None
    
    seed = int(seed_match.group(1))
    base_without_seed = base.replace(f'_seed_{seed}', '')
    
    # Parse different sampling types
    if 'cluster_sampling' in filename or 'cluster' in base:
        return parse_cluster_seeded_filename(base_without_seed, seed, filename)
    elif 'convenience_sampling' in filename or any(x in filename for x in ['urban_based', 'region_based', 'cluster_based']):
        return parse_convenience_seeded_filename(base_without_seed, seed, filename)
    elif 'random_sampling' in filename or 'random_sample' in base:
        return parse_random_seeded_filename(base_without_seed, seed, filename)
    else:
        print(f"[WARNING] Unknown sampling type for {filename}")
        return None


def parse_cluster_seeded_filename(base, seed, filename):
    """Parse cluster sampling seeded filename using ppc and size patterns."""
    parts = base.split("_")
    
    sample_size = None
    points_per_cluster = None
    
    for i, part in enumerate(parts):
        if part.endswith("ppc"):
            try:
                points_per_cluster = int(part.replace("ppc", ""))
            except ValueError:
                pass
        
        if part == "size" and i > 0:
            try:
                sample_size = int(parts[i - 1])
            except ValueError:
                pass

    return {
        'sampling_type': 'cluster',
        'seed': seed,
        'base_name': base,
        'intended_size': sample_size,
        'points_per_cluster': points_per_cluster,
        'filename': filename
    }


def parse_convenience_seeded_filename(base, seed, filename):
    """Parse convenience sampling seeded filename."""
    parts = base.split("_")
    
    # Get source from the directory or filename
    source = "unknown"
    if "urban_based" in filename:
        source = "urban_based"
    elif "region_based" in filename:
        source = "region_based"
    elif "cluster_based" in filename:
        source = "cluster_based"
    
    # Extract intended sample size - look for patterns like "top5", numbers in filename
    intended_size = None
    num_urban = None
    
    for part in parts:
        # Look for "top{number}" pattern
        match = re.match(r"(\d+)_points", part)
        match2 = re.match(r"top(\d+)", part)
        if match:
            num_urban = int(match2.group(1))
            # For convenience sampling, the intended size might be encoded differently
            # This might need adjustment based on actual filename patterns
            intended_size = int(match.group(1))  # Assuming top{N} means N samples initially
            break
        
        # Look for explicit size indicators
        if part.isdigit():
            intended_size = int(part)
    
    return {
        'sampling_type': 'convenience',
        'seed': seed,
        'base_name': base,
        'intended_size': intended_size,
        'source': source,
        'num_urban': num_urban,
        'filename': filename
    }


def parse_random_seeded_filename(base, seed, filename):
    """Parse random sampling seeded filename."""
    # Extract sample size from filename
    size_match = re.search(r'random_sample_(\d+)_points', base)
    intended_size = int(size_match.group(1)) if size_match else None
    
    return {
        'sampling_type': 'random',
        'seed': seed,
        'base_name': base,
        'intended_size': intended_size,
        'filename': filename
    }


def group_files_by_type_and_size(sampling_dir, verbose=True):
    """Group sampling files by type and intended size, maintaining seed information."""
    print(f"\nGrouping files by type and intended size from {sampling_dir}...")
    
    groups = defaultdict(list)  # key: (sampling_type, intended_size), value: list of (metadata, filepath)
    
    sample_files = [os.path.join(sampling_dir, f) for f in sorted(os.listdir(sampling_dir)) if f.endswith(".pkl")]
    print(f"Found {len(sample_files)} sample files")
    
    for fname in sample_files:
        full_path = os.path.join(sampling_dir, fname)
        metadata = parse_seeded_filename(fname)
        
        if metadata is None:
            if verbose:
                print(f"[SKIP] Could not parse {fname}")
            continue
        
        intended_size = metadata.get('intended_size')
        if intended_size is None:
            if verbose:
                print(f"[SKIP] Could not determine intended size for {fname}")
            continue
        
        # Load file to get sampled IDs (but group by intended size)
        sampled_ids = load_sample_file(full_path, verbose=False)
        if sampled_ids is None:
            continue
        
        actual_size = len(sampled_ids)
        metadata['actual_size'] = actual_size
        
        # Create grouping key using intended size
        key = (metadata['sampling_type'], intended_size, metadata['base_name'])
        groups[key].append((metadata, full_path, sampled_ids))
        
        if verbose:
            print(f"✓ Grouped {fname} -> type: {metadata['sampling_type']}, intended_size: {intended_size}, actual_size: {actual_size}")
    
    print(f"Created {len(groups)} groups")
    return groups


def verify_subset_relationship(groups, verbose=True):
    """Verify that smaller sample sets are subsets of larger ones within the same type.
    
    Raises:
        ValueError: If subset relationships are violated, stopping the entire process.
    """
    print("\nVerifying subset relationships...")
    
    # Group by sampling type and base name
    type_groups = defaultdict(lambda: defaultdict(list))
    for (sampling_type, intended_size, base_name), files in groups.items():
        type_groups[sampling_type][base_name].append((intended_size, files))
    
    verification_results = {}
    all_violations = []  # Track all violations before stopping
    
    for sampling_type, base_groups in type_groups.items():
        print(f"\n--- Verifying {sampling_type} sampling ---")
        
        for base_name, size_files in base_groups.items():
            # Sort by intended size
            size_files.sort(key=lambda x: x[0])
            sizes = [sf[0] for sf in size_files]
            
            if verbose:
                print(f"Base: {base_name}, Intended sizes: {sizes}")
            
            # Check subset relationships for each seed
            seeds_verified = []
            violations_for_base = []
            
            if len(size_files) > 1:
                # Get all seeds from smallest size group
                smallest_size, smallest_files = size_files[0]
                seeds = set(metadata['seed'] for metadata, _, _ in smallest_files)
                
                for seed in seeds:
                    seed_samples = {}
                    
                    # Collect samples for this seed across all intended sizes
                    for intended_size, files in size_files:
                        for metadata, filepath, sampled_ids in files:
                            if metadata['seed'] == seed:
                                seed_samples[intended_size] = set(sampled_ids)
                                break
                    
                    # Verify subset relationships
                    subset_verified = True
                    for i in range(len(sizes) - 1):
                        smaller_size = sizes[i]
                        larger_size = sizes[i + 1]
                        
                        if smaller_size in seed_samples and larger_size in seed_samples:
                            smaller_set = seed_samples[smaller_size]
                            larger_set = seed_samples[larger_size]
                            
                            if not smaller_set.issubset(larger_set):
                                subset_verified = False
                                violation_msg = (
                                    f"SUBSET VIOLATION: {sampling_type} sampling, {base_name}, "
                                    f"seed {seed}: size {smaller_size} ({len(smaller_set)} samples) "
                                    f"is NOT a subset of size {larger_size} ({len(larger_set)} samples)"
                                )
                                violations_for_base.append(violation_msg)
                                all_violations.append(violation_msg)
                                
                                # Show some details about the violation
                                missing_samples = smaller_set - larger_set
                                extra_samples = larger_set - smaller_set
                                print(f"[ERROR] {violation_msg}")
                                print(f"        Missing {len(missing_samples)} samples in larger set: {list(missing_samples)[:5]}{'...' if len(missing_samples) > 5 else ''}")
                                print(f"        Extra {len(extra_samples)} samples in larger set: {list(extra_samples)[:5]}{'...' if len(extra_samples) > 5 else ''}")
                                break
                    
                    if subset_verified:
                        seeds_verified.append(seed)
                        if verbose:
                            print(f"✓ Seed {seed}: Subset relationship verified for intended sizes")
            
            verification_results[(sampling_type, base_name)] = {
                'intended_sizes': sizes,
                'verified_seeds': seeds_verified,
                'total_seeds': len(seeds) if 'seeds' in locals() else 0,
                'violations': violations_for_base
            }
    
    # If any violations were found, stop the entire process
    if all_violations:
        print(f"\n{'='*80}")
        print("CRITICAL ERROR: SUBSET RELATIONSHIP VIOLATIONS DETECTED")
        print(f"{'='*80}")
        print(f"Found {len(all_violations)} violations:")
        for i, violation in enumerate(all_violations, 1):
            print(f"{i}. {violation}")
        
        print(f"\n{'='*80}")
        print("STOPPING ANALYSIS - SUBSET RELATIONSHIPS MUST BE MAINTAINED")
        print("Please fix the sampling data before proceeding.")
        print(f"{'='*80}")
        
        raise ValueError(
            f"Subset relationship verification failed with {len(all_violations)} violations. "
            "Smaller sample sets must be subsets of larger ones within the same sampling configuration."
        )
    
    print(f"\n✓ SUCCESS: All subset relationships verified across all sampling types and configurations")
    return verification_results


def compute_r2_statistics(groups, id_to_index, X_train_full, y_train_full, X_test, y_test, 
                         ridge_regression_fn, min_samples=1100, max_samples=1100, verbose=True):
    """Compute R² statistics for each group of samples."""
    print("\nComputing R² statistics...")
    
    results = {}
    
    for (sampling_type, intended_size, base_name), files in tqdm(groups.items(), desc="Processing groups"):
        if intended_size < min_samples:
            if verbose:
                print(f"[SKIP] Intended size {intended_size} below minimum {min_samples}")
            continue
        if intended_size > max_samples:
            if verbose:
                print(f"[SKIP] Intended size {intended_size} above maximum {max_samples}")
            continue
        
        print(f"\n--- Processing {sampling_type} sampling, intended size {intended_size} ---")
        print(f"Base: {base_name}")
        
        r2_scores = []
        valid_files = 0
        actual_sizes = []
        
        for metadata, filepath, sampled_ids in files:
            # Convert IDs to indices
            sampled_indices = [id_to_index[i] for i in sampled_ids if i in id_to_index]
            actual_size = len(sampled_indices)
            actual_sizes.append(actual_size)
            
            if actual_size < min_samples:
                if verbose:
                    print(f"[SKIP] Not enough valid indices for seed {metadata['seed']} (actual: {actual_size})")
                continue
            
            # Extract subset data
            X_subset = X_train_full[sampled_indices]
            y_subset = y_train_full[sampled_indices]
            
            # Run regression
            try:
                r2 = ridge_regression_fn(X_subset, y_subset, X_test, y_test)
                r2_scores.append(r2)
                valid_files += 1
                
                if verbose:
                    print(f"✓ Seed {metadata['seed']}: R² = {r2:.4f} (actual size: {actual_size})")
                    
            except Exception as e:
                if verbose:
                    print(f"[ERROR] Regression failed for seed {metadata['seed']}: {e}")
                continue
        
        if r2_scores:
            stats = {
                'sampling_type': sampling_type,
                'base_name': base_name,
                'intended_sample_size': intended_size,
                'actual_sample_size_mean': float(np.mean(actual_sizes)) if actual_sizes else None,
                'actual_sample_size_std': float(np.std(actual_sizes)) if actual_sizes else None,
                'num_seeds': valid_files,
                'r2_mean': float(np.mean(r2_scores)),
                'r2_std': float(np.std(r2_scores)),
                'r2_min': float(np.min(r2_scores)),
                'r2_max': float(np.max(r2_scores)),
                'r2_scores': [float(r2) for r2 in r2_scores]  # Keep individual scores for debugging
            }
            
            results[(sampling_type, intended_size, base_name)] = stats
            
            print(f"✓ {sampling_type} intended size {intended_size}: μ={stats['r2_mean']:.4f}, σ={stats['r2_std']:.4f} (n={valid_files})")
            if stats['actual_sample_size_mean']:
                print(f"  Actual sizes: μ={stats['actual_sample_size_mean']:.1f}, σ={stats['actual_sample_size_std']:.1f}")
        else:
            print(f"[WARNING] No valid R² scores for {sampling_type} intended size {intended_size}")
    
    return results


def save_results_to_json(results, output_dir, verbose=True):
    """Save each result to its own JSON file based on its 'base_name'."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving individual results to {output_dir}...")

    count = 0
    for (_, _, _), stats in results.items():
        base_name = stats.get("base_name", f"unnamed_{count}")
        file_name = f"{base_name}.json"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, 'w') as f:
            json.dump(stats, f, indent=2)

        count += 1
        if verbose:
            print(f"✓ Saved {file_path}")

    print(f"\nSaved {count} JSON files.")

def seeded_sampling_r2_analysis(dataset_name, features_path, sampling_dir, results_dir, ridge_regression_fn,
                               min_samples=1100, verify_subsets=True, verbose=True, **kwargs):
    """Main function to perform seeded sampling R² analysis."""
    print(f"\n{'='*80}")
    print(f"STARTING SEEDED SAMPLING R² ANALYSIS")
    print(f"{'='*80}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory: {results_dir}")
    print(f"Sampling directory: {sampling_dir}")
    
    # Load data
    data = load_data_from_pkl(features_path, **kwargs)
    if data is None:
        return
    
    full_ids, X_train_full, y_train_full, X_test, y_test = data
    
    # Create ID mapping
    id_to_index = create_id_mapping(full_ids)
    
    # Group files by type and size
    groups = group_files_by_type_and_size(sampling_dir, verbose)
    
    if not groups:
        print("[ERROR] No valid groups found")
        return
    
    # Verify subset relationships
    if verify_subsets:
        try:
            verification_results = verify_subset_relationship(groups, verbose)
        except ValueError as e:
            print(f"\n[CRITICAL ERROR] {e}")
            return None  # Stop processing and return None
    
    # Compute R² statistics
    results = compute_r2_statistics(
        groups, id_to_index, X_train_full, y_train_full, X_test, y_test,
        ridge_regression_fn, min_samples, verbose=True
    )
    from IPython import embed; embed()
    # Save results
    output_dir = results_dir  # or wherever you want the per-base_name files saved
    save_results_to_json(results, output_dir, verbose=verbose)
    
    print(f"\n{'='*80}")
    print(f"COMPLETED SEEDED SAMPLING R² ANALYSIS")
    print(f"{'='*80}")
    
    return results


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
            'random_sampling_dir': None,
            'results_dir': f"{dataset}_{label}_sampling_stats_json",
            'dataset': dataset
        }
    
    elif dataset == "togo":
        return {
            'features_path': "/home/libe2152/optimizedsampling/0_data/features/togo/togo_fertility_data_all_2021_Jan_Jun_P20.pkl",
            'cluster_sampling_dir': "/home/libe2152/optimizedsampling/0_data/initial_samples/togo/cluster_sampling/fixedstrata_kara-plateaux",
            'convenience_sampling_urban_dir': None,
            'convenience_sampling_region_dir': None,
            'random_sampling_dir': None,
            'results_dir': f"{dataset}_sampling_stats_json",
            'dataset': dataset,
            'label': 'ph_h2o'
        }
    
    elif dataset == "india_secc":
        return {
            'features_path': "/home/libe2152/optimizedsampling/0_data/features/india_secc/India_SECC_with_splits_4000.pkl",
            'cluster_sampling_dir': "/home/libe2152/optimizedsampling/0_data/initial_samples/india_secc/cluster_sampling/fixedstrata_01-04-06-08-09-11-21-23-25-33",
            'convenience_sampling_urban_dir': "/home/libe2152/optimizedsampling/0_data/initial_samples/india_secc/convenience_sampling/urban_based",
            'convenience_sampling_region_dir': "/home/libe2152/optimizedsampling/0_data/initial_samples/india_secc/convenience_sampling/cluster_based",
            'random_sampling_dir': None,
            'results_dir': f"{dataset}_sampling_stats_json",
            'dataset': dataset
        }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported datasets: 'usavars', 'togo', 'india_secc'")


def process_dataset_seeded_analysis(dataset, label=None, ridge_regression_fn=None,
                                   run_cluster=True, run_convenience_urban=False,
                                   run_convenience_region=False, run_random=False):
    """Process seeded analysis for a dataset."""
    print(f"\n{'#'*80}")
    if label:
        print(f"SEEDED ANALYSIS - DATASET: {dataset.upper()} - LABEL: {label.upper()}")
    else:
        print(f"SEEDED ANALYSIS - DATASET: {dataset.upper()}")
    print(f"{'#'*80}")
    
    dirs = get_sampling_directories(dataset, label)
    
    # Build common arguments
    common_args = {
        'features_path': dirs['features_path'],
        'results_dir': dirs['results_dir'],
        'ridge_regression_fn': ridge_regression_fn,
        'verbose': True,
        'min_samples': 1100,
        'verify_subsets': True
    }
    
    # Add dataset-specific arguments
    if dirs.get('dataset'):
        common_args['dataset'] = dirs['dataset']
    if dirs.get('label'):
        common_args['label'] = dirs['label']
    
    # Run seeded analysis for available sampling methods
    if run_cluster and dirs['cluster_sampling_dir']:
        print(f"\n--- Running seeded cluster sampling analysis ---")
        seeded_sampling_r2_analysis(dataset_name=dataset, sampling_dir=dirs['cluster_sampling_dir'], **common_args)
    
    if run_convenience_urban and dirs['convenience_sampling_urban_dir']:
        print(f"\n--- Running seeded urban convenience sampling analysis ---")
        seeded_sampling_r2_analysis(dataset_name=dataset, sampling_dir=dirs['convenience_sampling_urban_dir'], **common_args)
    
    if run_convenience_region and dirs['convenience_sampling_region_dir']:
        print(f"\n--- Running seeded region convenience sampling analysis ---")
        seeded_sampling_r2_analysis(dataset_name=dataset, sampling_dir=dirs['convenience_sampling_region_dir'], **common_args)
    
    if run_random and dirs['random_sampling_dir']:
        print(f"\n--- Running seeded random sampling analysis ---")
        seeded_sampling_r2_analysis(dataset_name=dataset, sampling_dir=dirs['random_sampling_dir'], **common_args)


def main():
    """Main function to run seeded sampling analysis."""
    
    print("STARTING SEEDED SAMPLING R² ANALYSIS")
    print("="*80)
    
    # Process Togo dataset
    # process_dataset_seeded_analysis(
    #     dataset="togo",
    #     ridge_regression_fn=ridge_regression,
    #     run_cluster=True,
    #     run_convenience_urban=False,
    #     run_convenience_region=False,
    #     run_random=False
    # )
    
    # Process India dataset
    # process_dataset_seeded_analysis(
    #     dataset="india_secc",
    #     ridge_regression_fn=ridge_regression,
    #     run_cluster=True,
    #     run_convenience_urban=False,
    #     run_convenience_region=False,
    #     run_random=False
    # )
    
    # Process USAVars datasets (commented out)
    for label in ['population', 'treecover']:
        process_dataset_seeded_analysis(
            dataset="usavars",
            label=label,
            ridge_regression_fn=ridge_regression,
            run_cluster=True,
            run_convenience_urban=False,
            run_convenience_region=False,
            run_random=False
        )
    
    print("\n" + "="*80)
    print("SEEDED ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()