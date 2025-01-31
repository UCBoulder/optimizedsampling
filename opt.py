import dill

from cost import *
from clusters import retrieve_all_clusters, retrieve_clusters

import cvxpy as cp
import numpy as np

def group_size(x, groups, g):
    return cp.sum(x[groups == g])

def pop_size(x):
    return cp.sum(x)

def group_risk(nj, n, sigmaj_sq, tauj_sq, pj, qj, deltaj):
    return deltaj + sigmaj_sq * cp.exp(-pj*cp.log(nj)) + tauj_sq * cp.exp(-qj*cp.log(n))

def pop_risk(njs, n, sigmaj_sqs, tauj_sqs, pjs, qjs, deltajs, gammajs):
    group_risks = [group_risk(njs[i], n, sigmaj_sqs[i], tauj_sqs[i], pjs[i], qjs[i], deltajs[i]) for i in range(len(njs))]
    weighted_group_risks = [gammajs[i]*group_risks[i] for i in range(len(njs))]
    return cp.sum(weighted_group_risks)

def risk_by_prob(x, groups, sigmaj_sqs, tauj_sqs, pjs, qjs, deltajs, gammajs):
    njs = [group_size(x, groups, g) for g in np.unique(groups)]
    n = pop_size(x)
    return pop_risk(njs, n, sigmaj_sqs, tauj_sqs, pjs, qjs, deltajs, gammajs)

'''
Sets up cvxpy problem and solves
'''
def solve(ids, 
          costs, 
          budget, 
          sigmaj_sqs=np.full((8,),2), 
          tauj_sqs=np.ones((8,)), 
          pjs=np.ones((8,)), 
          qjs=np.ones((8,)), 
          deltajs=np.ones((8,))):

    n = len(ids)
    x = cp.Variable(n, nonneg=True)

    clusters = retrieve_clusters(ids, "data/clusters/urban_areas_cluster_assignment.pkl")
    #gammajs = [np.sum(clusters == c)/len(ids) for c in np.unique(clusters)]
    gammajs = [1 for c in np.unique(clusters)]

    #Need to set values to finite, large value instead
    if len(np.where(costs == np.inf)[0])>0:
        costs = np.array([np.min([1e6, c]) for c in costs]).reshape(-1,1)

    #cvxpy Problem setup
    objective = risk_by_prob(x, clusters, sigmaj_sqs, tauj_sqs, pjs, qjs, deltajs, gammajs)
    constraints = [0 <= x, x <= 1, costs.T@x <= budget]
    prob = cp.Problem(cp.Minimize(objective), constraints)

    prob.solve(solver=cp.MOSEK, verbose=True)

    print("Optimal x is: ", x.value)
    return x.value

if __name__ == '__main__':
    budget = 1000
    with open("data/int/feature_matrices/CONTUS_UAR_population_with_splits.pkl", "rb") as f:
        arrs = dill.load(f)
    ids = arrs['ids_train']

    dist_path = "data/cost/distance_to_closest_city.pkl"

    costs = compute_lin_cost(dist_path, ids, alpha=1, beta=1)

    solve(ids, costs, budget)



