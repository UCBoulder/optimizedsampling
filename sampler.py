import pandas as pd
import numpy as np

from oed import *
import config as c
from clusters import retrieve_clusters

class Sampler:
    '''
    Class for sampling
    '''
    def __init__(
            self, 
            ids, 
            *datasets, 
            rule="random",
            prob_dist=None, 
            loc_emb=None, 
            costs=None,
            cluster_type="NLCD_percentages"):
        '''Initialize a new Sampler instance.

        Args:
            datasets: data to sample from (df or array)
            rule: use rule to sample ("random", "image", "satclip", "lowcost")
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

        if prob_dist is not None:
            self.prob_dist = prob_dist

        if loc_emb is not None:
            self.loc_emb = loc_emb
        
        if costs is None:
            self.costs = np.ones(len(datasets[0]))
        else:
            self.costs = costs

        self.scores = None

        if rule == "clusters":
            self.set_clusters(cluster_type)
        else:
            self.set_scores()

        self.finite_idxs = np.where(self.costs != np.inf)[0]
        self.total_valid = len(self.finite_idxs)

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

        elif self.rule == 'prob':
            self.scores = self.prob_dist

    
    '''
    Set clusters from cluster path
        cluster_type: NLCD, NLCD_percentages, 
    '''
    def set_clusters(self, cluster_type):
        cluster_path = f"data/clusters/{cluster_type}_cluster_assignment.pkl"
        self.clusters = retrieve_clusters(self.ids, cluster_path)

    '''
    Determine indexes of subset to sample
    Args:
        self:
        budget: cost limit
    '''
    def subset_idxs_with_scores(self, budget=0, seed=42):
        subset_idxs = []
        total_cost = 0
        scores = self.scores.copy()
        finite_idxs = np.where(self.costs != np.inf)[0]

        while total_cost < budget and len(finite_idxs) > 0:
            max_score = np.max(scores[finite_idxs])
            max_idxs = finite_idxs[scores[finite_idxs] == max_score]

            # Randomly pick one of the indices with the maximum score
            np.random.seed(seed)
            max_idx = np.random.choice(max_idxs)

            # Update cost
            total_cost += self.costs[max_idx]

            if total_cost > budget:
                break

            # Add to the sampled set
            subset_idxs.append(max_idx)

            #Make sure the point is not chosen again
            finite_idxs = finite_idxs[finite_idxs != max_idx]

            if len(subset_idxs) == len(self.dataset1):
                break

        return subset_idxs
    
    '''
    Determine indexes of subset to sample using clustering
    Args:
        self
        budget: cost limit
    '''
    def subset_idxs_with_clusters(self, budget=0, seed=42):
        subset_idxs = []
        total_cost = 0
        clusters = self.clusters.copy()
        unique_clusters  = np.unique(self.clusters)
        finite_idxs = self.finite_idxs.copy()

        stop = False
        while total_cost < budget and len(finite_idxs) > 0:
            for c in unique_clusters:
                cluster_idxs = finite_idxs[clusters[finite_idxs] == c]

                if len(cluster_idxs) == 0:
                    continue

                #Randomly pick index in cluster
                np.random.seed(seed)
                idx = np.random.choice(cluster_idxs)

                # Update cost
                total_cost += self.costs[idx]

                if total_cost > budget:
                    stop = True
                    break
                
                subset_idxs.append(idx)
                
                #Ensure this point is not chosen again
                finite_idxs = finite_idxs[finite_idxs != idx]

            if stop:
                break

        return subset_idxs

    
    '''
    Sample subset from each dataset
    '''
    def sample_with_budget(self, budget=0, seed=42):
        print(f"Sampling with respect to budget {budget}")

        if self.rule == 'clusters':
            subset_idxs = self.subset_idxs_with_clusters(budget, seed)
        else:
            subset_idxs = self.subset_idxs_with_scores(budget, seed)

        i = 1
        while True:
            dataset = getattr(self, f"dataset{i}", None)
            if dataset is None:
                return
            yield dataset[subset_idxs]
            i += 1