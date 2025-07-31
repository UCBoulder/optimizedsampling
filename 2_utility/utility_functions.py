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

import numpy as np

def diversity_(s, similarity_per_unit, l=100, epsilon=1e-4):
    S_eps = similarity_per_unit + epsilon * np.eye(similarity_per_unit.shape[0])
    s = s.astype(float)  # ensure dot products work
    return l * np.sum(s) - s.T @ S_eps @ s

