import numpy as np

def random(s):
    return 0

def greedy(s):
    return np.sum(s)

def stratified(s, groups, l=0.5):
    unique_groups = np.unique(groups)
    group_sizes = np.array([np.sum(s[groups == g]) for g in unique_groups])
    total_size = np.sum(group_sizes)
    
    group_risks = l / np.sqrt(group_sizes) + (1 - l) / np.sqrt(total_size)
    return -np.sum(group_risks)

def pop_risk(s, units, groups, l=0.5):
    """
    NumPy version of population risk utility.
    
    Args:
        s: 1D NumPy array of selection indicators or weights (length: n or num units)
        units: 1D array mapping each point to a unit
        groups: 1D array mapping each point to a group
        l: float in [0,1]
        
    Returns:
        Scalar utility value (float)
    """
    print(f"Population risk utility function with lambda={l}")
    
    if s.shape[0] == len(groups):  # unit = point
        unique_groups = np.unique(groups)
        group_sizes = np.array([np.sum(s[groups == g]) for g in unique_groups])
    else:
        unique_groups, group_idx = np.unique(groups, return_inverse=True)
        unique_units, unit_idx = np.unique(units, return_inverse=True)

        A = np.zeros((len(unique_groups), len(unique_units)), dtype=int)
        np.add.at(A, (group_idx, unit_idx), 1)

        group_sizes = A @ s

    total_size = np.sum(group_sizes)
    group_weights = np.bincount(group_idx)[np.arange(len(unique_groups))] / len(groups)

    group_risks = l / np.sqrt(group_sizes) + (1 - l) / np.sqrt(total_size)
    weighted_risks = group_weights * group_risks

    return -np.sum(weighted_risks)

def similarity(s, similarity_matrix):
    """
    Maximize average similarity to test set (or predefined reference).
    """
    test_similarity = similarity_matrix.sum(axis=1)  # total similarity for each training point
    return s @ test_similarity

def diversity(s, distance_matrix):
    """
    Diversity penalty based on distance matrix: discourages picking similar (nearby) points.
    """
    return s @ distance_matrix @ s