import numpy as np
import random

from mosaiks.code.mosaiks.utils import io
from mosaiks.code.mosaiks import config as c
from oed import *
from feasibility import *

class Sampler:
    '''
    Class for sampling
    '''
    def __init__(self, *datasets, rule="random", loc_emb=None):
        '''Initialize a new Sampler instance.

        Args:
            data: data to sample from (df or array)
            rule: use rule to sample
        '''
        i = 0
        for dataset in datasets:
            if isinstance(dataset, pd.DataFrame):
                dataset = dataset.to_numpy()
            setattr(self, f"dataset{i+1}", dataset)
            i += 1
        
        self.rule = rule

        if loc_emb is not None:
            self.loc_emb = loc_emb

    '''
    Determine indexes of subset to sample
    '''
    def subset_idxs(self, n=0):
        if self.rule=="random":
            subset_idxs = np.random.choice(len(self.dataset1), size=n, replace=False)

        if self.rule=="image":
            subset_idxs = sampling_with_scores(self.dataset1, n, v_optimal_design)

        if self.rule=="satclip":
            subset_idxs = sampling_with_scores(self.loc_emb, n, v_optimal_design)

        return subset_idxs
    
    def sample(self, n=0):
        subset_idxs = self.subset_idxs(n)

        i = 1
        while True:
            dataset = getattr(self, f"dataset{i}", None)
            if dataset is None:
                return
            yield dataset[subset_idxs]
            i += 1

#-----------------------------------------------------------------------------------------

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
def random_subset_and_cost(X_train, Y_train, latlon, size, costs):
    print("Generating subset using SRS...")
    subset_idxs = np.random.choice(len(X_train), size=size, replace=False)

    #Get costs of subset
    total_cost = total_cost(costs, subset_idxs)

    return X_train[subset_idxs], Y_train[subset_idxs], latlon[subset_idxs], total_cost

'''
Image-only baseline
Takes an OED subset of training data--Image-only baseline
Only works for V-optimality as of 11/4
'''
def image_subset(X_train, Y_train, latlon_train, rule, size):
    print("Generating subset using {rule}...".format(rule='V Optimal Design'))
    subset_idxs = sampling_with_scores(X_train, size, rule)
    return X_train[subset_idxs], Y_train[subset_idxs], latlon_train[subset_idxs]

'''
Image and Spatial based
Takes an OED subset of training data using SatCLIP embeddings
'''
def satclip_subset(X_train, Y_train, latlon_train, loc_emb_train, rule, size):
    print("Generating subset using satclip embeddings...")
    subset_idxs = sampling_with_scores(loc_emb_train, size, rule)
    return X_train[subset_idxs], Y_train[subset_idxs], latlon_train[subset_idxs]

'''
Takes subsets greedily with lowest cost until number of samples is reached
'''
def greedy_by_cost(X_train, Y_train, latlon_train, size):
    costs = get_costs(c, X_train).values
    lowest_cost_idxs = np.argpartition(costs, size)[:size]
    return X_train[lowest_cost_idxs], Y_train[lowest_cost_idxs], latlon_train[lowest_cost_idxs]

'''
Takes a SRS subset of training data such that total cost does not exceed budget
'''
def constrained_random_subset(X_train, Y_train, latlon_train, budget):
    cost = 0
    idxs = [i for i in range(len(X_train))]
    subset_idxs = []
    while cost<=budget:
       sampled_idx = random.sample(idxs, 1)
       subset_idxs.append(sampled_idx)
       idxs = idxs[:sampled_idx] + idxs[sampled_idx + 1:]
       cost += cost[sampled_idx]
    return X_train[subset_idxs], Y_train[subset_idxs], latlon_train[subset_idxs]