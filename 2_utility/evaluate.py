import pandas as pd
import dill
import os
import numpy as np
from utility_functions import *
from glob import glob
from tqdm import tqdm

def compute_utilities_from_ids(
    sampled_ids,
    all_ids,
    utility_fns,
    group_dict,
    **kwargs
):
    """
    Compute a dictionary of utilities from a list of sampled IDs.

    Args:
        sampled_ids (list): IDs selected in the sample.
        all_ids (list): Full list of all possible IDs (defines vector length).
        utility_fns (dict): Dict of utility_name -> function(s), which take a binary vector s.
        group_dict (dict): Mapping of ID to group assignment.
        **kwargs: Additional arguments like similarity_matrix, distance_matrix.

    Returns:
        dict: utility_name -> computed utility value
    """
    print(f"Computing utilities for {len(sampled_ids)} sampled IDs out of {len(all_ids)} total IDs")
    
    id_to_idx = {id_: i for i, id_ in enumerate(all_ids)}
    s = np.zeros(len(all_ids), dtype=int)
    for sid in sampled_ids:
        if sid in id_to_idx:
            s[id_to_idx[sid]] = 1
    
    group_assignments_per_id = [[(group_dict[id_], 1.0)] for id_ in all_ids]

    results = {}
    for name, fn in utility_fns.items():
        print(f"  Computing utility: {name}")
        if name.startswith('pop_risk'):
            results[name] = fn(s, group_assignments_per_id)
        elif name in ['similarity']:
            similarity_matrix = kwargs.get('similarity_matrix', None)
            results[name] = fn(s, similarity_matrix)
        elif name in ['diversity']:
            similarity_matrix = kwargs.get('similarity_matrix', None)
            results[name] = fn(s, similarity_matrix)
        else:
            results[name] = fn(s)

    return results

def save_utilities_results_to_csv(results_list, csv_path):
    """
    Save or append utility results from multiple samples to a CSV file.

    Args:
        results_list (list of tuples): Each tuple is (sampled_path, sampling_type, results_dict).
        csv_path (str): Path to output CSV file.
    """
    print(f"Saving {len(results_list)} results to {csv_path}")
    
    rows = []
    for sampled_path, sampling_type, results in results_list:
        row = {'sampled_path': sampled_path, 'sampling_type': sampling_type}
        row.update(results)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Results saved successfully")

def load_data_for_label(label, group_type='nlcd'):
    """
    Load all necessary data for a given label and group type.
    
    Args:
        label (str): Either 'population' or 'treecover'
        group_type (str): Either 'nlcd' or 'image_clusters_8'
    
    Returns:
        tuple: (all_ids, group_dict, similarity_matrix)
    """
    print(f"Loading data for label: {label}, group_type: {group_type}")
    
    invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
    
    # Load all_ids
    features_path = f"/home/libe2152/optimizedsampling/0_data/features/usavars/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl"
    print(f"Loading features from: {features_path}")
    with open(features_path, "rb") as f:
        arrs = dill.load(f)
    all_ids = arrs['ids_train']
    
    # Filter out invalid ids
    all_ids = [id_ for id_ in all_ids if id_ not in invalid_ids]
    print(f"Loaded {len(all_ids)} valid IDs (filtered out {len(invalid_ids)} invalid IDs)")
    
    # Load groups info
    if group_type == 'nlcd':
        group_path = f"/home/libe2152/optimizedsampling/0_data/groups/usavars/{label}/NLCD_cluster_assignments_8_dict.pkl"
    elif group_type == 'image_clusters_8':
        group_path = f"/home/libe2152/optimizedsampling/0_data/groups/usavars/{label}/image_8_cluster_assignments.pkl"
    else:
        raise ValueError(f"Unknown group_type: {group_type}")
    
    print(f"Loading groups from: {group_path}")
    with open(group_path, "rb") as f:
        arrs = dill.load(f)

    # arrs is a dict of id -> assignment
    ids, groups = zip(*[(i, g) for i, g in arrs.items() if i not in invalid_ids])
    group_dict = dict(zip(ids, groups))
    print(f"Loaded group assignments for {len(group_dict)} IDs")
    
    # Load similarity matrix
    similarity_path = f"/home/libe2152/optimizedsampling/0_data/cosine_similarity/usavars/{label}/cosine_similarity_train_test.npy"
    print(f"Loading similarity matrix from: {similarity_path}")
    similarity_matrix = np.load(similarity_path)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    return all_ids, group_dict, similarity_matrix

def get_utility_functions(group_type='nlcd'):
    """
    Get utility functions dictionary based on group type.
    
    Args:
        group_type (str): Either 'nlcd' or 'image_clusters_8'
    
    Returns:
        dict: utility function mappings
    """
    print(f"Setting up utility functions for group_type: {group_type}")
    
    utility_fns = {
        "size": greedy,
        'diversity': lambda x, sim_matrix: diversity(x, sim_matrix),
        f"pop_risk_{group_type}_0.5": lambda x, groups: pop_risk(x, groups, l=0.5),
        f"pop_risk_{group_type}_0": lambda x, groups: pop_risk(x, groups, l=0),
        f"pop_risk_{group_type}_0.01": lambda x, groups: pop_risk(x, groups, l=0.01),
        f"pop_risk_{group_type}_0.1": lambda x, groups: pop_risk(x, groups, l=0.1),
        f"pop_risk_{group_type}_0.9": lambda x, groups: pop_risk(x, groups, l=0.9),
        f"pop_risk_{group_type}_0.99": lambda x, groups: pop_risk(x, groups, l=0.99),
        f"pop_risk_{group_type}_1": lambda x, groups: pop_risk(x, groups, l=1),
    }
    
    print(f"Created {len(utility_fns)} utility functions:")
    for func_name in utility_fns.keys():
        print(f"  - {func_name}")
    return utility_fns

def process_sampling_files(base_dir, sampling_type, all_data_dict, label):
    """
    Process all sampling files for a given sampling type.
    
    Args:
        base_dir (str): Base directory path
        sampling_type (str): Type of sampling
        all_data_dict (dict): Dictionary containing data for all group types
        label (str): Current label being processed ('population' or 'treecover')
    
    Returns:
        list: Results list with tuples of (filename, sampling_label, results)
    """
    print(f"\nProcessing sampling type: {sampling_type}")
    results_list = []
    invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
    
    if sampling_type == "convenience_sampling":
        subfolders = ["urban_based"]
        for subfolder in subfolders:
            print(f"  Processing subfolder: {subfolder}")
            dir_path = os.path.join(base_dir, sampling_type, subfolder)
            results_list.extend(
                process_directory(dir_path, f"{sampling_type}_{subfolder}", all_data_dict, invalid_ids)
            )
    
    elif sampling_type == "cluster_sampling":
        # Map subfolders to their corresponding labels
        subfolder_map = {
            "population": "fixedstrata_Idaho_16-Louisiana_22-Mississippi_28-New Mexico_35-Pennsylvania_42",
            "treecover": "fixedstrata_Alabama_01-Colorado_08-Montana_30-New York_36-Ohio_39"
        }
        
        # Only process the subfolder that matches the current label
        if label in subfolder_map:
            subfolder = subfolder_map[label]
            print(f"  Processing subfolder for {label}: {subfolder}")
            dir_path = os.path.join(base_dir, sampling_type, subfolder)
            results_list.extend(
                process_directory(dir_path, f"{sampling_type}_{subfolder}", all_data_dict, invalid_ids)
            )
        else:
            print(f"  No matching subfolder found for label: {label}")
    
    else:  # random_sampling
        print(f"  Processing directory directly")
        dir_path = os.path.join(base_dir, sampling_type)
        results_list.extend(
            process_directory(dir_path, sampling_type, all_data_dict, invalid_ids)
        )
    
    return results_list

def process_directory(dir_path, sampling_label, all_data_dict, invalid_ids):
    """
    Process all pickle files in a directory, computing utilities for all group types per file.
    
    Args:
        dir_path (str): Directory path to process
        sampling_label (str): Label for this sampling method
        all_data_dict (dict): Dictionary containing data for all group types
        invalid_ids (np.ndarray): IDs to filter out
    
    Returns:
        list: Results for this directory
    """
    results_list = []
    
    if not os.path.isdir(dir_path):
        print(f"    [Warning] Directory does not exist: {dir_path}")
        return results_list
    
    pkl_files = glob(os.path.join(dir_path, "*.pkl"))
    print(f"    Found {len(pkl_files)} pickle files")
    
    processed_count = 0
    for sampled_path in tqdm(pkl_files, desc=f"Processing {sampling_label}", leave=False):
        if not sampled_path.endswith("seed_1.pkl"):
            continue
        
        try:
            print(f"    Processing: {os.path.basename(sampled_path)}")
            with open(sampled_path, "rb") as f:
                sampled_ids = dill.load(f)
            
            # Handle different data structures
            if isinstance(sampled_ids, dict) and "sampled_ids" in sampled_ids:
                sampled_ids = sampled_ids["sampled_ids"]
            
            # Filter out invalid ids
            original_count = len(sampled_ids)
            sampled_ids = [id_ for id_ in sampled_ids if id_ not in invalid_ids]
            print(f"      Filtered sampled IDs: {original_count} -> {len(sampled_ids)}")
            
            # Compute utilities for ALL group types for this file
            combined_results = {}
            filename = os.path.basename(sampled_path)
            
            for group_type in all_data_dict.keys():
                print(f"      Computing utilities for group_type: {group_type}")
                data = all_data_dict[group_type]
                
                results = compute_utilities_from_ids(
                    sampled_ids,
                    data['all_ids'],
                    data['utility_fns'],
                    data['group_dict'],
                    similarity_matrix=data['similarity_matrix']
                )
                
                # Add results to combined dictionary
                combined_results.update(results)
                print(f"        {group_type} results: {results}")
            
            results_list.append((filename, sampling_label, combined_results))
            processed_count += 1
            
        except Exception as e:
            print(f"    [Error] Could not process {sampled_path}: {e}")
    
    print(f"    Successfully processed {processed_count} files")
    return results_list

def main():
    """
    Main function to process all labels and group types.
    """
    print("Starting utility computation script")
    
    labels = ['population', 'treecover']
    group_types = ['nlcd', 'image_clusters_8']
    sampling_types = ["cluster_sampling", "convenience_sampling", "random_sampling"]
    
    for label in labels:
        print(f"\n{'='*50}")
        print(f"Processing label: {label}")
        print(f"{'='*50}")
        
        # Load data for ALL group types first
        all_data_dict = {}
        for group_type in group_types:
            print(f"\nLoading data for group type: {group_type}")
            try:
                all_ids, group_dict, similarity_matrix = load_data_for_label(label, group_type)
                utility_fns = get_utility_functions(group_type)
                
                all_data_dict[group_type] = {
                    'all_ids': all_ids,
                    'group_dict': group_dict,
                    'similarity_matrix': similarity_matrix,
                    'utility_fns': utility_fns
                }
                print(f"Successfully loaded data for {group_type}")
                
            except Exception as e:
                print(f"[Error] Failed to load data for {label} - {group_type}: {e}")
                continue
        
        if not all_data_dict:
            print(f"No data loaded for {label}, skipping...")
            continue
        
        print(f"\nLoaded data for {len(all_data_dict)} group types: {list(all_data_dict.keys())}")
        
        # Process each sampling type with all group types
        all_combined_results = []
        base_dir = f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}"
        
        for sampling_type in sampling_types:
            results = process_sampling_files(base_dir, sampling_type, all_data_dict, label)
            all_combined_results.extend(results)
        
        # Save all results for this label in one CSV file
        if all_combined_results:
            output_path = f"/home/libe2152/optimizedsampling/0_results/usavars/{label}/utilities.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_utilities_results_to_csv(all_combined_results, output_path)
            
            print(f"\nCompleted processing for {label}")
            print(f"Total combined results saved: {len(all_combined_results)}")
        else:
            print(f"\nNo results to save for {label}")
    
    print(f"\n{'='*50}")
    print("Script completed successfully!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()