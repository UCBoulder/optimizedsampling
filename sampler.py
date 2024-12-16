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
    
    # '''
    # Set clusters from KMeans
    # '''
    # def set_clusters(self, feat_type):
    #     cluster_path = f"data/clusters/KMeans_{feat_type}_cluster_assignment.pkl"
    #     self.clusters = retrieve_clusters(self.ids, cluster_path)

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

        while total_cost < budget:
            max_score = np.max(scores)
            max_idxs = np.where(scores == max_score)[0]

            # Randomly pick one of the indices with the maximum leverage score
            np.random.seed(seed)
            max_idx = np.random.choice(max_idxs)
            # Update cost
            total_cost += self.costs[max_idx]

            if total_cost >= budget:
                break

            # Add to the sampled set
            subset_idxs.append(max_idx)

            #Set this points score to -inf so it's not chosen again
            scores[max_idx] = -np.inf

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

        while total_cost < budget:
            for c in unique_clusters:
                cluster_idxs = np.where(clusters == c)[0]

                if len(cluster_idxs) == 0:
                    continue

                #Randomly pick index in cluster
                np.random.seed(seed)
                cluster_idx = np.random.choice(cluster_idxs)

                # Update cost
                total_cost += self.costs[cluster_idx]

                if total_cost >= budget:
                    break
                
                subset_idxs.append(cluster_idx)
                
                #Ensure this point is not chosen again
                clusters[cluster_idx] = -1

        return subset_idxs
    
    '''
    Sample subset from each dataset
    '''
    def sample_with_budget(self, budget=0, seed=42):
        print(f"Sampling with respect to budget {budget}")
        subset_idxs = self.subset_idxs_with_scores(budget, seed)

        i = 1
        while True:
            dataset = getattr(self, f"dataset{i}", None)
            if dataset is None:
                return
            yield dataset[subset_idxs]
            i += 1