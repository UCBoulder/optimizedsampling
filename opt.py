from cost import *
from clusters import retrieve_all_clusters, retrieve_clusters

import cvxpy as cp
import numpy as np
from scipy.stats import entropy

#Calculates cluster distribution from sample:
def cluster_distribution(x, clusters):
    print("Calculating cluster_distribution...")
    num_clusters = len(np.unique(clusters))
    if isinstance(x, cp.Variable):
        cluster_distribution = [cp.sum(cp.multiply(x, clusters == c)) for c in range(num_clusters)]

        #Normalize
        #stacked_distribution = cp.vstack(cluster_distribution)  # Stack the expressions vertically
        #normalized_cluster_distribution = stacked_distribution / cp.sum(stacked_distribution) #Normalizing leads to non dcp *********
    else:
        cluster_distribution = np.zeros((num_clusters,), dtype=float)
        for i in range(x.shape[0]):
            c = clusters[i]
            cluster_distribution[c] += x[i]

        #Normalize
        cluster_distribution = cluster_distribution / np.sum(cluster_distribution)
    return cluster_distribution

#Calculates expected sample size
def expected_sample_size(x):
    print("Calculating cost distribution...")
    if isinstance(x, cp.Variable):
        return cp.sum(x)
    else:
        return np.sum(x)

#Calculates negated KL-divergence
def distribution_similarity(p, q):
    print("Calculating distribution similarity...")
    q = cp.vstack(q)
    if q.shape != p.shape:
        q = cp.reshape(q, p.shape, order = 'C')
    return -cp.sum(cp.kl_div(p, q))

#Jointly demonstrate representativeness and sample size
def joint_objective(x, clusters, p_target):
    p_sample = cluster_distribution(x, clusters)
    return distribution_similarity(p_target, p_sample)
    #+ l * expected sample size

def solve(l, budget, cost_func, ids=None, **kwargs):
    x = cp.Variable(len(ids))
    clusters = retrieve_clusters(ids, "data/clusters/NLCD_percentages_cluster_assignment.pkl")
    p_target = cluster_distribution(np.ones((len(ids),), dtype=int), clusters)

    if cost_func == compute_state_cost:
        states = kwargs.get('states', 1)
        costs = cost_func(states, ids=ids)
    elif cost_func == compute_unif_cost:
        costs = cost_func(ids_train)
    else:
        dist_path = "data/cost/distance_to_closest_city.pkl"
        costs = cost_func(dist_path, ids_train, **kwargs)

    objective = joint_objective(x, clusters, p_target)
    constraints = [0 <= x, x <= 1, x.T@costs <= budget, cp.sum(x) == 1]
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(verbose=True, max_iter=100000)

    print("The optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    return x.value

if __name__ == '__main__':
    l = 0.5
    budget = 1000
    cost_func = compute_state_cost

    #solve(l, budget, cost_func, ids=ids, states=["Colorado"])



