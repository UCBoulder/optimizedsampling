from cost import *
from clusters import retrieve_all_clusters

import cvxpy as cp
import numpy as np
from scipy.stats import entropy

clusters, ids = retrieve_all_clusters("data/clusters/NLCD_percentages_cluster_assignment.pkl")
num_clusters = len(np.unique(clusters))
total_num_samples = len(clusters)

#Calculates cluster distribution from sample:
def cluster_distribution(x):
    if isinstance(x,cp.Variable):
        cluster_distribution = [cp.sum(cp.multiply(x, clusters == c)) for c in range(num_clusters)]

        #Normalize
        cluster_distribution = cp.multiply(cluster_distribution, cp.inv_pos(cp.sum(cluster_distribution)))
    else:
        cluster_distribution = np.zeros((num_clusters,), dtype=float)
        for i in range(x.shape[0]):
            print(i)
            c = clusters[i]
            cluster_distribution[c] += x[i]

        #Normalize
        cluster_distribution = cluster_distribution / np.sum(cluster_distribution)
    return cluster_distribution

p_target = cluster_distribution(np.ones((total_num_samples, ), dtype=int))

#Calculates expected sample size
def expected_sample_size(x):
    return np.sum(x)

#Calculates negated KL-divergence
def distribution_similarity(p, q):
    return -entropy(p, q)

#Jointly demonstrate representativeness and samplesize
def joint_objective(x, l):
    p_sample = cluster_distribution(x)
    return distribution_similarity(p_target, p_sample) + l*expected_sample_size(x)

def solve(l, budget, cost_func, ids=None):
    x = cp.Variable(total_num_samples)
    costs = cost_func(ids)

    objective = joint_objective(x, l)
    constraints = [0 <= x, x <= 1, x.T@costs <= budget]
    prob = cp.Problem(cp.Maximize(objective), constraints)

    print("The optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    return x.value

if __name__ == '__main__':
    l = 0.5
    budget = 1000
    cost_func = compute_unif_cost

    solve(l, budget, cost_func, ids=ids)



