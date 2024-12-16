import numpy as np
from utils import distance_of_subset

def compute_unif_cost(dist_path, ids, **kwargs):
    return compute_lin_cost(dist_path, ids, alpha=0, beta=1)

def compute_lin_cost(dist_path, ids, **kwargs):
    dist_of_subset = distance_of_subset(dist_path, ids)
    alpha = kwargs.get('alpha', 1)
    beta = kwargs.get('beta', 1)

    cost_str = f"linear wrt distance with alpha={alpha}, beta={beta}"
    print("Cost function is {cost_str}".format(cost_str=cost_str))

    costs = (alpha*(dist_of_subset) + beta)

    return costs

def compute_lin_w_r_cost(dist_path, ids,**kwargs):
    dist_of_subset = distance_of_subset(dist_path, ids)
    costs = np.empty(len(dist_of_subset), dtype=np.float32)

    alpha = kwargs.get('alpha', 1)
    beta = kwargs.get('beta', 1)
    gamma = kwargs.get('gamma', 1)
    r = kwargs.get('r', 0)

    cost_str = f"linear outside or radius {r} km with alpha={alpha}, beta = {beta}, gamma={gamma}"
    print("Cost function is {cost_str}".format(cost_str=cost_str))

    for i in range(len(dist_of_subset)):
        if (dist_of_subset[i] <= r):
            costs[i] = gamma
        if (dist_of_subset[i] > r):
            costs[i] = alpha*(dist_of_subset[i]) + beta

    return costs