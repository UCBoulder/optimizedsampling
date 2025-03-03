import dill
import os

#from cost import *
#from clusters import retrieve_all_clusters, retrieve_clusters

import cvxpy as cp
import numpy as np
import mosek

with mosek.Env() as env:
    with env.Task() as task:
        task.putintparam(mosek.iparam.num_threads, 1) 
        task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.primal_simplex) 
        task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1e-9)  # Tighten numerical tolerances
os.environ["MKL_NUM_THREADS"] = "1"

def group_size(x, groups, g):
    return cp.sum(x[groups == g])

def pop_size(x):
    return cp.sum(x)

def group_risk(nj, n, l):
    return l*cp.exp(-1*cp.log(nj)) + (1-l)*cp.exp(-1*cp.log(n))

def pop_risk(njs, n, gammajs, l):
    group_risks = [group_risk(njs[i], n, l) for i in range(njs.shape[0])]
    weighted_group_risks = [gammajs[i]*group_risks[i] for i in range(njs.shape[0])]
    return cp.sum(weighted_group_risks)

def risk_by_prob(x, groups, gammajs, l):
    njs = [group_size(x, groups, g) for g in np.unique(groups)]
    n = pop_size(x)
    return pop_risk(njs, n, gammajs, l)

def risk_by_group_size(njs, gammajs, l):
    n = sum(njs)
    return pop_risk(njs, n, gammajs, l)

'''
Sets up cvxpy problem and solves
'''
def solve_usavars(ids, 
                  costs, 
                  budget,
                  obj_type='sample', #sample or group size
                  l=0.5):

    n = len(ids)
    x = cp.Variable(n, nonneg=True)

    clusters = retrieve_clusters(ids, f"data/clusters/{group_type}_cluster_assignment.pkl")

    gammajs = [np.sum(clusters == c)/len(ids) for c in np.unique(clusters)]
    #gammajs = [1 for c in np.unique(clusters)]

    #Need to set values to finite, large value instead
    if len(np.where(costs == np.inf)[0])>0:
        costs = np.array([np.min([1e6, c]) for c in costs]).reshape(-1,1)

    #cvxpy Problem setup
    objective = risk_by_prob(x, clusters, gammajs, l)
    constraints = [0 <= x, x <= 1, costs.T@x <= budget]
    prob = cp.Problem(cp.Minimize(objective), constraints)

    prob.solve(solver=cp.MOSEK, mosek_params={
        'MSK_DPAR_MIO_TOL_REL_GAP': 1e-6,
        'MSK_IPAR_NUM_THREADS': 1,
        'MSK_IPAR_LOG': 2
    })

    print("Optimal x is: ", x.value)
    return x.value

def solve_rep_matters(groups,
                      max_group_sizes,
                      costs, 
                      budget,
                      prop=True, #use gammajs
                      l=0.5):

    num_groups = len(groups)
    njs = cp.Variable(num_groups, nonneg=True)
    total_num = sum(max_group_sizes)

    if prop == True:
        gammajs = [max_group_sizes[i]/total_num for i in range(num_groups)]
    else:
        gammajs = [1 for g in range(num_groups)]

    #cvxpy Problem setup
    objective = risk_by_group_size(njs, gammajs, l)
    constraints = [0 <= njs, costs.T@njs <= budget]
    prob = cp.Problem(cp.Minimize(objective), constraints)

    prob.solve(solver=cp.MOSEK, mosek_params={
        'MSK_DPAR_MIO_TOL_REL_GAP': 1e-6,
        'MSK_IPAR_NUM_THREADS': 1,
        'MSK_IPAR_LOG': 2
    })

    print("Optimal group sizes are: ", njs.value)
    return njs.value

if __name__ == '__main__':
    budget = 1000
    with open("data/int/feature_matrices/CONTUS_UAR_population_with_splits.pkl", "rb") as f:
        arrs = dill.load(f)
    ids = arrs['ids_train']

    dist_path = "data/cost/distance_to_closest_city.pkl"

    costs = compute_lin_cost(dist_path, ids, alpha=0.1, beta=1)

    solve(ids, costs, budget)



