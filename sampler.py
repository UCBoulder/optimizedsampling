import pandas as pd
import numpy as np
from oed import *

class Sampler:
    '''
    Class for sampling
    '''
    def __init__(self, ids, *datasets, rule="random", loc_emb=None, costs=None):
        '''Initialize a new Sampler instance.

        Args:
            datasets: data to sample from (df or array)
            rule: use rule to sample ("random", "image", "satclip")
            loc_emb: satclip embeddings if rule is satclip
        '''
        i = 0
        for dataset in datasets:
            if isinstance(dataset, pd.DataFrame):
                dataset = dataset.to_numpy()
            setattr(self, f"dataset{i+1}", dataset)
            i += 1
        
        self.ids = ids
        self.rule = rule

        if loc_emb is not None:
            self.loc_emb = loc_emb
        
        if costs is None:
            self.costs = np.ones(len(datasets[0]))
        else:
            self.costs = costs

        self.scores = None
        self.set_scores()

    '''
    Sets scores according to rule
    '''
    def set_scores(self):
        if self.rule == "random":
            self.scores = np.ones(len(self.dataset1))

        elif self.rule == "image":
            scores_path = "data/scores/CONTUS_UAR_leverage_scores.pkl"
            self.scores = scores_from_path(scores_path, self.ids)

        elif self.rule == "satclip":
            scores_path = "data/scores/CONTUS_UAR_satclip_leverage_scores.pkl"
            self.scores = scores_from_path(scores_path, self.ids)

        elif self.rule == 'greedycost':
            self.scores = -self.costs #Highest score corresponds to lowest cost
    
    '''
    Determine indexes of subset to sample
    Args:
        self:
        budget: cost limit
    '''
    def subset_idxs_with_budget(self, budget=0):
        subset_idxs = []
        total_cost = 0
        scores = self.scores.copy()

        while total_cost < budget:
            max_score = np.max(scores)
            max_idxs = np.where(scores == max_score)[0]

            # Randomly pick one of the indices with the maximum leverage score
            max_idx = np.random.choice(max_idxs)
            # Update cost
            total_cost += self.costs[max_idx]

            if total_cost >= budget:
                break

            # Add to the sampled set
            subset_idxs.append(max_idx)

            #Set this points score to -inf so it's not chosen again
            scores[max_idx] = -np.inf

        return subset_idxs
    
    '''
    Sample subset from each dataset
    '''
    def sample_with_budget(self, budget=0):
        subset_idxs = self.subset_idxs_with_budget(budget)

        i = 1
        while True:
            dataset = getattr(self, f"dataset{i}", None)
            if dataset is None:
                return
            yield dataset[subset_idxs]
            i += 1