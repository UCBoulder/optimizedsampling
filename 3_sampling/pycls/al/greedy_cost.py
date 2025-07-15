import os
import dill
import numpy as np
import pandas as pd
import numpy as np

from pycls.datasets.usavars import USAVars

labels = {
    'USAVARS_POP': "population",
    'USAVARS_TC': "treecover",
    'USAVARS_EL': "elevation"
}

def get_all_ids(label):
    usavars = USAVars(root='/share/usavars', isTrain=True, label=label)

    def get_usavars_ids(idx):
        _, _, _, id = usavars[idx]
        return id

    ids = [get_usavars_ids(i) for i in range(len(usavars))]

    return ids

def get_all_costs(label, cost_dict_path):
    ids = get_all_ids(label)

    with open(cost_dict_path, 'rb') as f:
        cost_dict = dill.load(f)

    all_costs = np.array([
        np.inf if (id in cost_dict and np.isnan(cost_dict[id])) 
        else cost_dict[id] 
        for id in ids if id in cost_dict
    ])

    return all_costs

class GreedyCost:
    def __init__(self, cfg, lSet, uSet, budgetSize):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.lSet = lSet
        self.uSet = uSet
        self.budget = budgetSize
        self.costs = get_all_costs(labels[self.ds_name], cost_path)

    def select_samples(self):
        # using only labeled+unlabeled indices, without validation set.
        relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        costs = self.costs[relevant_indices]

        all_indices = np.arange(len(relevant_indices))
        existing_indices = np.arange(len(self.lSet))
        new_indices = np.setdiff1d(all_indices, existing_indices)

        #randomly choose order
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(new_indices)
        selected = perm[np.argsort(costs[perm])][:self.budget]

        print(f"Total Sample Cost: {np.sum(costs[selected])}")

        selected = np.array(selected)
        assert len(selected) == self.budget, 'added a different number of samples'
        assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
        activeSet = relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        return activeSet, remainSet
