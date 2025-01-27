import dill

from cost import *
from clusters import retrieve_all_clusters, retrieve_clusters

import cvxpy as cp
import numpy as np

'''
Calculates expect number of samples in each cluster
'''
def cluster_distribution(x, clusters):
    print("Calculating cluster_distribution...")
    num_clusters = len(np.unique(clusters))

    #If x is a Variable, need to us cvxpy methods
    if isinstance(x, cp.Variable):
        cluster_distribution = [cp.sum(cp.multiply(x, clusters == c)) for c in range(num_clusters)]
    else:
        cluster_distribution = np.zeros((num_clusters,), dtype=float)
        for i in range(x.shape[0]):
            c = clusters[i]
            cluster_distribution[c] += x[i]
    return cluster_distribution

'''
Calculates expected sample size
'''
def expected_sample_size(x):
    print("Calculating cost distribution...")

    #If x is a Variable, need to use cvxpy methods
    if isinstance(x, cp.Variable):
        return cp.sum(x)
    else:
        return np.sum(x)

'''
Calculates negated L2 norm
'''
def distribution_similarity(p, q):
    print("Calculating distribution similarity...")
    q = cp.vstack(q)
    if q.shape != p.shape:
        q = cp.reshape(q, p.shape, order = 'C')
    return -cp.norm(p-q)**2

'''
Combines representativeness and sample size
l: parameter controlling influence of expected sample size
'''
def joint_objective(x, l1, l2, clusters, p_target):
    p_sample = cluster_distribution(x, clusters)
    return cp.multiply(l1, distribution_similarity(p_target, p_sample)) + cp.multiply(l2, expected_sample_size(x))

'''
Sets up cvxpy problem and solves
'''
def solve(ids, costs, budget, l1, l2):
    x = cp.Variable(len(ids), nonneg=True)
    l1 = cp.Parameter(nonneg=True, value=l1)
    l2 = cp.Parameter(nonneg=True, value=l2)

    clusters = retrieve_clusters(ids, "data/clusters/NLCD_percentages_cluster_assignment.pkl")
    
    #Target cluster distribution
    p_target = cluster_distribution(np.ones((len(ids),), dtype=int), clusters)

    #cvxpy Problem setup
    objective = joint_objective(x, l1, l2, clusters, p_target)
    constraints = [0 <= x, x <= 1, x.T@costs <= budget]
    prob = cp.Problem(cp.Maximize(objective), constraints)

    prob.solve(verbose=True, max_iter=100000)

    print("For lambda1=", l1.value, " and lambda2=", l2.value, ":")
    print("Optimal x is: ", x.value)
    return x.value

if __name__ == '__main__':
    l1 = 1
    l2 = 1
    budget = 1000
    with open("data/int/feature_matrices/CONTUS_UAR_population_with_splits.pkl", "rb") as f:
        arrs = dill.load(f)
    ids = arrs['ids_train']

    costs = compute_unif_cost(ids)

    solve(ids, costs, budget, l1, l2)



