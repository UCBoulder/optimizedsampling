import numpy as np
import random

from mosaiks.code.mosaiks.utils import io
from mosaiks.code.mosaiks import config as c
from oed import *
from feasibility import *

'''
Spatial-only baseline
Takes a random subset of training data
'''
def random_subset(X_train, Y_train, latlon, size):
    print("Generating subset using SRS...")
    subset_idxs = np.random.choice(len(X_train), size=size, replace=False)
    return X_train[subset_idxs], Y_train[subset_idxs], latlon[subset_idxs]

'''
Spatial-only baseline
Takes a random subset of training data and records cost
'''
def random_subset_and_cost(X_train, Y_train, latlon_train, ids_train, costs, size):
    print("Generating subset using SRS...")
    subset_idxs = np.random.choice(len(X_train), size=size, replace=False)

    #Get costs of subset
    total_cost = cost_of_subset(costs, subset_idxs)

    return X_train[subset_idxs], Y_train[subset_idxs], latlon_train[subset_idxs], ids_train[subset_idxs], total_cost

'''
Image-only baseline
Takes an OED subset of training data--Image-only baseline
Only works for V-optimality as of 11/4
'''
def image_subset(X_train, Y_train, latlon_train, ids_train, costs, rule, size):
    print("Generating subset using {rule}...".format(rule='V Optimal Design'))
    scores_path = "data/scores/CONTUS_UAR_leverage_scores.pkl"
    subset_idxs = sampling_with_scores(scores_path, ids_train, size, rule)

    #Get costs of subset
    total_cost = cost_of_subset(costs, subset_idxs)
    return X_train[subset_idxs], Y_train[subset_idxs], latlon_train[subset_idxs], ids_train[subset_idxs], total_cost

'''
Image and Spatial based
Takes an OED subset of training data using SatCLIP embeddings
'''
def satclip_subset(X_train, Y_train, latlon_train, loc_emb_train, ids_train, costs, rule, size):
    print("Generating subset using satclip embeddings...")
    scores_path = "data/scores/CONTUS_UAR_satclip_leverage_scores.pkl"
    subset_idxs = sampling_with_scores(scores_path, ids_train, size, rule)

    #Get costs of subset
    total_cost = cost_of_subset(costs, subset_idxs)
    return X_train[subset_idxs], Y_train[subset_idxs], latlon_train[subset_idxs], ids_train[subset_idxs], total_cost

'''
Takes subsets greedily with lowest cost until number of samples is reached
'''
def greedy_by_cost(X_train, Y_train, latlon_train, ids_train, costs, size):
    print("Generating subset using greedy cost algorithm...")
    lowest_cost_idxs = np.argpartition(costs, size)[:size]
    cost_subset = costs[lowest_cost_idxs]

    boundary_cost = max(cost_subset)
    tied_idxs = np.where(costs == boundary_cost)[0]
    idxs_in_cost_subset = lowest_cost_idxs[np.where(cost_subset == boundary_cost)[0]]

    #Break ties randomly
    if (len(tied_idxs)>len(idxs_in_cost_subset)):
        lowest_cost_idxs = lowest_cost_idxs[lowest_cost_idxs != idxs_in_cost_subset]
        lowest_cost_idxs = np.concatenate([lowest_cost_idxs, np.random.choice(tied_idxs, size=len(idxs_in_cost_subset), replace=False)])

    total_cost = cost_of_subset(costs, lowest_cost_idxs)
    return X_train[lowest_cost_idxs], Y_train[lowest_cost_idxs], latlon_train[lowest_cost_idxs], ids_train[lowest_cost_idxs], total_cost

'''
Takes a SRS subset of training data such that total cost does not exceed budget
'''
def constrained_random_subset(X_train, Y_train, latlon_train, cost_train, budget):
    cost = 0
    idxs = [i for i in range(len(X_train))]
    subset_idxs = []
    while cost<=budget:
       sampled_idx = random.sample(idxs, 1)
       subset_idxs.append(sampled_idx)
       idxs = idxs[:sampled_idx] + idxs[sampled_idx + 1:]
       cost += cost_train[sampled_idx]
    return X_train[subset_idxs], Y_train[subset_idxs], latlon_train[subset_idxs]