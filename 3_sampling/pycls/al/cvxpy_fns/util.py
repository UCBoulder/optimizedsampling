import cvxpy as cp
import numpy as np

def random(s):
    return 0

def greedy(s):
    return cp.sum(s)

#Workshop paper objective (from rep. matters paper)
def pop_risk(s, groups_per_unit, l=0.5, ignored_groups=None):
    """
    Compute weighted risk over groups from input values and group labels,
    using group weights proportional to group frequency in `groups`.

    Args:
        s: cp.Variable over units (e.g., binary indicators or weights)
        groups_per_unit: list of dicts; each dict maps group → count
        l: weighting parameter in [0,1]

    Returns:
        scalar risk value (cp scalar)
    """
    print(f"Population risk utility function with lambda={l}")

    if ignored_groups is None:
        ignored_groups = set()
    else:
        print(f"Ignore groups: {ignored_groups}")
        ignored_groups = set(ignored_groups)

    num_units = len(groups_per_unit)
    # Collect all groups, excluding ignored
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
    group_weights = cp.Constant(group_weights_np)

    A = cp.Constant(A_np)

    group_sizes = A @ s
    total_size = cp.sum(group_sizes)

    group_risks = l * cp.inv_pos(cp.sqrt(group_sizes)) + (1 - l) * cp.inv_pos(cp.sqrt(total_size))
    weighted_risks = cp.multiply(group_weights, group_risks)
    
    return -cp.sum(weighted_risks)  #negative so optimization will maximize this

def pop_risk_mod(s, groups_per_unit, l=0.5, ignored_groups=None):
    """
    Compute weighted risk over groups from input values and group labels,
    using group weights proportional to group frequency in `groups`.

    Args:
        s: cp.Variable over units (e.g., binary indicators or weights)
        groups_per_unit: list of dicts; each dict maps group → count
        l: weighting parameter in [0,1]

    Returns:
        scalar risk value (cp scalar)
    """
    print(f"Population risk utility function with lambda={l}")

    if ignored_groups is None:
        ignored_groups = set()
    else:
        print(f"Ignore groups: {ignored_groups}")
        ignored_groups = set(ignored_groups)

    num_units = len(groups_per_unit)
    # Collect all groups, excluding ignored
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
    group_weights = cp.Constant(group_weights_np)

    A = cp.Constant(A_np)

    group_sizes = A @ s
    total_size = cp.sum(group_sizes)

    group_risks = l * cp.inv_pos(group_sizes) + (1 - l) * cp.inv_pos(total_size) #NO SQRT HERE
    weighted_risks = cp.multiply(group_weights, group_risks)
    
    return -cp.sum(weighted_risks)  #negative so optimization will maximize this

def similarity(s, similarity_per_unit):
    test_similarity = similarity_per_unit.mean(axis=1) #this sums the similarity for now, might want to change to softmax
    return s @ test_similarity

def diversity(s, distance_per_unit):    
    n = s.shape[0]
    M = distance_per_unit
    M_sym = 0.5 * (M + M.T)

    s_i = cp.reshape(s, (n, 1), order='C')
    s_j = cp.reshape(s, (1, n), order='C')

    pairwise_min = cp.minimum(s_i, s_j)

    return cp.sum(cp.multiply(M_sym, pairwise_min))
