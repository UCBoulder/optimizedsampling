import cvxpy as cp
import numpy as np

def random(s):
    return 0

def greedy(s):
    return cp.sum(s)

#Workshop paper objective (from rep. matters paper)
def pop_risk(s, units, groups, l=0.5):
    """
    Compute weighted risk over groups from input values and group labels,
    using group weights proportional to group frequency in `groups`.
    
    Args:
        x: cp.array of values (e.g., weights or indicators)
        groups: numpy array of group labels
        l: weighting parameter in [0,1]
        
    Returns:
        scalar risk value (cp scalar)
    """
    print(f"Population risk utility function with lambda={l}")
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    group_weights = cp.Constant(group_counts / group_counts.sum())  #proportions
    
    if s.shape[0] == len(groups):
        group_sizes = cp.hstack([cp.sum(s[groups == g]) for g in unique_groups])
    else: #make note of how this is done
        unique_groups, group_idx = np.unique(groups, return_inverse=True)
        unique_units, unit_idx = np.unique(units, return_inverse=True)
        
        A_np = np.zeros((len(unique_groups), len(unique_units)), dtype=int)
        np.add.at(A_np, (group_idx, unit_idx), 1) 

        A = cp.Constant(A_np)
        group_sizes = A @ s  #this does not take into account points per cluster, but might not have to
    total_size = cp.sum(group_sizes)
    
    # risk per group: l*(1/sqrt(nj)) + (1-l)*(1/sqrt(n))
    group_risks = l * cp.inv_pos(cp.sqrt(group_sizes)) + (1 - l) * cp.inv_pos(cp.sqrt(total_size))
    weighted_risks = cp.multiply(group_weights, group_risks)
    return -cp.sum(weighted_risks) #negative since opt will maximize this

def similarity(s, units, similarity_matrix):
    unique_units = np.unique(units)
    n_units = len(unique_units)

    n_test = similarity_matrix.shape[1]
    similarity_per_unit = np.zeros((n_units, n_test))

    for i, unit in enumerate(unique_units):
        mask = units == unit
        similarity_per_unit[i] = similarity_matrix[mask].sum(axis=0) 

    test_similarity = similarity_per_unit.sum(axis=1) #this sums the similarity for now, might want to change to softmax
    return s @ test_similarity

def diversity(s, distance_matrix):
    #slow as is, might need sparse distance implementation like knn
    return s @ distance_matrix @ s #penalizes close together points