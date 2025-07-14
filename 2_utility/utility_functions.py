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

def pop_risk(s, groups, l=0.5):
    """
    NumPy version of population risk utility.
    
    Args:
        s: 1D NumPy array of selection indicators or weights (length: n or num units)
        groups: 1D array mapping each point to a group
        l: float in [0,1]
        
    Returns:
        Scalar utility value (float)
    """
    print(f"Population risk utility function with lambda={l}")
    
    unique_groups, group_idx = np.unique(groups, return_inverse=True)
    group_sizes = np.array([np.sum(s[groups == g]) for g in unique_groups])

    total_size = np.sum(group_sizes)
    group_weights = np.bincount(group_idx)[np.arange(len(unique_groups))] / len(groups)

    group_risks = l / np.sqrt(group_sizes+1) + (1 - l) / np.sqrt(total_size+1)
    weighted_risks = group_weights * group_risks

    return -np.sum(weighted_risks)

def similarity(s, similarity_matrix):
    """
    Maximize average similarity to test set (or predefined reference).
    """
    print('Computing similarity utility...')
    test_similarity = similarity_matrix.sum(axis=1)  # total similarity for each training point
    return s @ test_similarity

def diversity(s, distance_matrix):
    """
    Diversity penalty based on distance matrix: discourages picking similar (nearby) points.
    """
    print("Computing diversity utility...")
    return s @ distance_matrix @ s