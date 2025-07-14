import pandas as pd
import dill
import os
import numpy as np
from utility_functions import *

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
        units (array-like): Optional mapping of each ID to a unit.
        groups (array-like): Optional mapping of each ID to a group.
        extra_args (dict): Optional dict mapping utility_name -> dict of extra kwargs.

    Returns:
        dict: utility_name -> computed utility value
    """
    id_to_idx = {id_: i for i, id_ in enumerate(all_ids)}
    s = np.zeros(len(all_ids), dtype=int)
    for sid in sampled_ids:
        if sid in id_to_idx:
            s[id_to_idx[sid]] = 1
    
    if group_dict is not None:
        id_to_group = {id_: group_dict[id_] for id_ in all_ids}
        groups = np.zeros(len(all_ids), dtype=int)
        for sid in sampled_ids:
            if sid in id_to_idx:
                groups[id_to_idx[sid]] = id_to_group[sid]

    results = {}
    for name, fn in utility_fns.items():
        if name.startswith('pop_risk'):
            results[name] = fn(s, groups)
        elif name in ['similarity']:
            similarity_matrix = kwargs.get('similarity_matrix', None)
            results[name] = fn(s, similarity_matrix)
        elif name in ['diversity']:
            distance_matrix = kwargs.get('distance_matrix', None)
            results[name] = fn(s, distance_matrix)
        else:
            results[name] = fn(s)

    return results

def save_utilities_results_to_csv(results_list, csv_path):
    """
    Save or append utility results from multiple samples to a CSV file.

    Args:
        results_list (list of tuples): Each tuple is (sampled_path, results_dict),
            where results_dict maps utility names to values.
        csv_path (str): Path to output CSV file.

    CSV columns will be:
        'sampled_path', <utility_fn_1>, <utility_fn_2>, ...
    """
    rows = []
    for sampled_path, sampling_type, results in results_list:
        row = {'sampled_path': sampled_path, 'sampling_type': sampling_type}
        row.update(results)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Append if file exists, else write new with header
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    import numpy as np
    import os
    import dill
    from glob import glob

    utility_fns = {
        "size": greedy,
        "pop_risk_0.5": lambda x, groups: pop_risk(x, groups, l=0.5),
        "pop_risk_0": lambda x, groups: pop_risk(x, groups, l=0),
        "pop_risk_0.01": lambda x, groups: pop_risk(x, groups, l=0.01),
        "pop_risk_0.1": lambda x, groups: pop_risk(x, groups, l=0.1),
        "pop_risk_0.9": lambda x, groups: pop_risk(x, groups, l=0.9),
        "pop_risk_0.99": lambda x, groups: pop_risk(x, groups, l=0.99),
        "pop_risk_1": lambda x, groups: pop_risk(x, groups, l=1),
        #"similarity": similarity,
        #"diversity": diversity
    }

    invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])

    for label in ['population', 'treecover']:

        # Load all_ids once
        features_path = f"/home/libe2152/optimizedsampling/0_data/features/usavars/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl"
        with open(features_path, "rb") as f:
            arrs = dill.load(f)
        all_ids = arrs['ids_train']

        # Filter out invalid ids from all_ids
        all_ids = [id_ for id_ in all_ids if id_ not in invalid_ids]

        # Load groups info once
        group_path = f"/home/libe2152/optimizedsampling/0_data/groups/usavars/{label}/NLCD_cluster_assignments_8.pkl"
        with open(group_path, "rb") as f:
            arrs = dill.load(f)
        ids = arrs['ids']
        groups = arrs['assignments']

        similarity_path = f"/home/libe2152/optimizedsampling/0_data/cosine_similarity/usavars/{label}/cosine_similarity_train_test.npz"
        similarity_matrix = np.load(similarity_path)['arr_0']

        distance_path = f"/home/libe2152/optimizedsampling/0_data/cosine_distance/usavars/{label}/cosine_distance.npz"
        distance_matrix = np.load(distance_path)['arr_0']

        # Filter out invalid ids from groups and ids
        filtered_pairs = [(i, g) for i, g in zip(ids, groups) if i not in invalid_ids]
        if filtered_pairs:
            ids, groups = zip(*filtered_pairs)
        else:
            ids, groups = [], []

        group_dict = dict(zip(ids, groups))

        base_dir = f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}"
        sampling_types = [
            "cluster_sampling",
            "convenience_sampling",  # parent convenience folder
            "random_sampling"
        ]

        results_list = []

        for sampling_type in sampling_types:
            if sampling_type == "convenience_sampling":
                # iterate over the two subfolders inside convenience_sampling
                for subfolder in ["probabilistic", "deterministic"]:
                    dir_path = os.path.join(base_dir, sampling_type, subfolder)
                    if not os.path.isdir(dir_path):
                        print(f"[Warning] Directory does not exist: {dir_path}")
                        continue

                    pkl_files = glob(os.path.join(dir_path, "*.pkl"))

                    for sampled_path in pkl_files:
                        try:
                            with open(sampled_path, "rb") as f:
                                sampled_ids = dill.load(f)

                            # Filter out invalid ids from sampled_ids
                            sampled_ids = [id_ for id_ in sampled_ids if id_ not in invalid_ids]

                            results = compute_utilities_from_ids(
                                sampled_ids,
                                all_ids,
                                utility_fns,
                                group_dict,
                                similarity_matrix = similarity_matrix,
                                distance_matrix = distance_matrix
                            )
                            filename = os.path.basename(sampled_path)
                            sampling_label = f"{sampling_type}_{subfolder}"
                            results_list.append((filename, sampling_label, results))

                        except Exception as e:
                            print(f"[Error] Could not process {sampled_path}: {e}")
                            from IPython import embed; embed()

            else:
                # For cluster_sampling and random_sampling (no subfolders)
                dir_path = os.path.join(base_dir, sampling_type)
                if not os.path.isdir(dir_path):
                    print(f"[Warning] Directory does not exist: {dir_path}")
                    continue

                pkl_files = glob(os.path.join(dir_path, "*.pkl"))

                for sampled_path in pkl_files:
                    try:
                        with open(sampled_path, "rb") as f:
                            sampled_ids = dill.load(f)

                        # Filter out invalid ids from sampled_ids
                        sampled_ids = [id_ for id_ in sampled_ids if id_ not in invalid_ids]

                        results = compute_utilities_from_ids(
                            sampled_ids,
                            all_ids,
                            utility_fns,
                            group_dict,
                            similarity_matrix = similarity_matrix,
                            distance_matrix = distance_matrix
                        )
                        filename = os.path.basename(sampled_path)
                        results_list.append((filename, sampling_type, results))

                    except Exception as e:
                        print(f"[Error] Could not process {sampled_path}: {e}")
                        from IPython import embed; embed()

        save_utilities_results_to_csv(
            results_list,
            f"/home/libe2152/optimizedsampling/0_results/usavars/{label}/utilities.csv"
        )
