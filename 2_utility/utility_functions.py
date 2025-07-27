import numpy as np

def random(s: np.ndarray) -> float:
    return 0.0

def greedy(s: np.ndarray) -> float:
    return np.sum(s)

def pop_risk(s: np.ndarray, groups_per_unit, l=0.5, ignored_groups=None) -> float:
    if ignored_groups is None:
        ignored_groups = set()
    else:
        ignored_groups = set(ignored_groups)

    num_units = len(groups_per_unit)
    all_groups = sorted({g for unit_pair in groups_per_unit for (g, _) in unit_pair if g not in ignored_groups})
    group_to_idx = {g: i for i, g in enumerate(all_groups)}
    num_groups = len(all_groups)

    group_counts = np.zeros(num_groups, dtype=int)
    A_np = np.zeros((num_groups, num_units), dtype=int)

    for unit_idx, unit_pair in enumerate(groups_per_unit):
        for (group, count) in unit_pair:
            if group in ignored_groups:
                continue
            group_idx = group_to_idx[group]
            group_counts[group_idx] += count
            A_np[group_idx, unit_idx] = count

    group_weights_np = group_counts / group_counts.sum()
    group_sizes = A_np @ s
    total_size = group_sizes.sum()

    # Avoid divide-by-zero with eps
    eps = 1e-8
    group_sizes_safe = np.sqrt(np.maximum(group_sizes, eps))
    total_size_safe = np.sqrt(max(total_size, eps))

    group_risks = l * (1.0 / group_sizes_safe) + (1 - l) * (1.0 / total_size_safe)
    weighted_risks = group_weights_np * group_risks

    return -np.sum(weighted_risks)  # negative to match maximization convention

def similarity(s: np.ndarray, similarity_per_unit: np.ndarray) -> float:
    """
    similarity_per_unit: shape (n_units, n_test)
    s: shape (n_units,)
    """
    test_similarity = similarity_per_unit.mean(axis=1)
    return float(np.dot(s, test_similarity))

def diversity(s: np.ndarray, distance_per_unit: np.ndarray) -> float:
    """
    distance_per_unit: shape (n_units, n_units)
    s: shape (n_units,)
    """
    n_units = len(s)
    s_i = s.reshape((n_units, 1))
    s_j = s.reshape((1, n_units))
    pairwise_min = np.minimum(s_i, s_j)

    diag_mask = np.ones((n_units, n_units)) - np.eye(n_units)
    valid_distances = distance_per_unit * pairwise_min * diag_mask

    return float(valid_distances[valid_distances > 0].min()) if np.any(valid_distances > 0) else 0.0
