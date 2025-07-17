import os
import dill
import numpy as np
import pandas as pd
import numpy as np

import cvxpy as cp
import mosek

from .cvxpy_fns import cost, util
from . import cost as np_cost

UTILITY_FNS = {
    "random": util.random,
    "greedycost": util.greedy,
    "poprisk": util.pop_risk,
    "similarity": util.similarity,
    "diversity": util.diversity
}
COST_FNS = {
    "uniform": cost.uniform,
    "pointwise_by_array": cost.pointwise_by_array,
    "unit_aware_pointwise_cost": cost.unit_aware_pointwise_cost
}

NP_COST_FNS = {
    "uniform": np_cost.uniform,
    "pointwise_by_array": np_cost.pointwise_by_array,
    "unit_aware_pointwise_cost": np_cost.unit_aware_pointwise_cost
}

class Opt:
    def __init__(self, cfg, lSet, uSet, budgetSize):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.lSet = lSet.astype(int)
        self.uSet = uSet.astype(int)
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.budget = budgetSize
        self.set_unit_assignment()
        self._resolve_cost_func()

    def set_unit_assignment(self):
        self.unit_assignment = np.array(self.cfg.UNITS.UNIT_ASSIGNMENT) if self.cfg.UNITS.UNIT_ASSIGNMENT is not None else np.arange(len(self.relevant_indices))
        self.unit_assignment = self.unit_assignment[self.relevant_indices]

        self.units = np.unique(self.unit_assignment)
        self.points_per_unit = self.cfg.UNITS.POINTS_PER_UNIT if self.cfg.UNITS.POINTS_PER_UNIT is not None else None #if none, we will select the whole unit

    def set_utility_func(self, utility_func_type):
        if utility_func_type not in UTILITY_FNS:
            raise ValueError(f"Invalid utility function type: {utility_func_type}")
        self.utility_func_type = utility_func_type
        self.utility_func = UTILITY_FNS[utility_func_type]

        if utility_func_type == "poprisk":
            assert self.cfg.GROUPS.GROUP_ASSIGNMENT is not None, "Group assignment must not be none for poprisk utility function"

            self.group_assignment = np.array(self.cfg.GROUPS.GROUP_ASSIGNMENT)

            self.group_assignment = self.group_assignment[self.relevant_indices]

            self.utility_func = lambda s: util.pop_risk(s, self.unit_assignment, self.group_assignment, l=self.cfg.ACTIVE_LEARNING.UTIL_LAMBDA)
        elif utility_func_type == "similarity":
            assert self.cfg.ACTIVE_LEARNING.SIMILARITY_MATRIX_PATH is not None, "Need to specify similarity matrix path"
            similarity_matrix = np.load(self.cfg.ACTIVE_LEARNING.SIMILARITY_MATRIX_PATH)['arr_0']
            similarity_matrix = similarity_matrix[np.ix_(self.relevant_indices, self.relevant_indices)]

            self.utility_func = lambda s: util.similarity(s, similarity_matrix)

        elif utility_func_type == "diversity":
            assert self.cfg.ACTIVE_LEARNING.DISTANCE_MATRIX_PATH is not None, "Need to specify distance matrix path"
            distance_matrix = np.load(self.cfg.ACTIVE_LEARNING.DISTANCE_MATRIX_PATH)['arr_0']
            distance_matrix = distance_matrix[np.ix_(self.relevant_indices, self.relevant_indices)]

            self.utility_func = lambda s: util.diversity(s, distance_matrix)
        print(f"Utility function set to: {utility_func_type}")

    def _resolve_cost_func(self):
        cost_func_type = self.cfg.COST.FN
        self.cost_func_type = cost_func_type
        assert cost_func_type in COST_FNS, f"Invalid cost function type: {cost_func_type}"

        self.cost_func = COST_FNS[cost_func_type]
        self.np_cost_func = NP_COST_FNS[cost_func_type]
        self.initial_set_cost = self.np_cost_func

        self.cost_domain = "unit"

        if cost_func_type == "pointwise_by_array":
            if self.cfg.COST.UNIT_COST_PATH is not None:
                with open(self.cfg.COST.UNIT_COST_PATH, "rb") as f:
                    self.cost_dict = dill.load(f)
                self.cost_array = np.array([self.cost_dict[u] for u in self.units])
            elif self.cfg.COST.ARRAY is not None:
                self.cost_array = np.array(self.cfg.COST.ARRAY)
                self.cost_array = self.cost_array[self.relevant_indices]
            else:
                raise(AssertionError)

            self.cost_func = lambda s: cost.pointwise_by_array(s, self.cost_array)
            self.np_cost_func = lambda s: np_cost.pointwise_by_array(s, self.cost_array)
            self.initial_set_cost = self.np_cost_func

        elif cost_func_type == "unit_aware_pointwise_cost":
            self.cost_domain = "point"
            labeled_indices = np.array([int(idx in set(self.lSet)) for idx in self.relevant_indices])

            self.units = np.arange(len(self.relevant_indices))

            labeled_units = set(self.unit_assignment[self.lSet])
            unit_labeled_array = [self.unit_assignment[i] in labeled_units for i in range(len(self.relevant_indices))]

            self.cost_func = lambda s: cost.unit_aware_pointwise_cost(s, labeled_indices, unit_labeled_array)
            self.np_cost_func = lambda s: np_cost.unit_aware_pointwise_cost(s, labeled_indices, unit_labeled_array)

    def solve_opt(self):
        assert self.utility_func_type != "Random", "Please do not use the optimization function for random selection"

        labeled_set = set(self.lSet)

        #make labeled inclusion vector of units
        unit_inclusion_vector = np.zeros(len(self.units), dtype=bool)

        if self.cost_domain == "unit":
            for i, u in enumerate(self.units):
                unit_point_indices = self.relevant_indices[self.unit_assignment == u]
                labeled_mask = np.array([idx in labeled_set for idx in unit_point_indices])
                
                if np.any(labeled_mask):
                    unit_inclusion_vector[i] = True
        else:
            for i in range(len(self.units)):
                unit_inclusion_vector[i] = True if self.units[i] in labeled_set else False

        n = len(self.units)
        s = cp.Variable(n, nonneg=True)

        assert s.shape == (n,), f"s should be shape {(n,)}, got {s.shape}"
        assert unit_inclusion_vector.shape == (n,), f"unit_inclusion_vector should be shape {(n,)}, got {unit_inclusion_vector.shape}"
        
        objective = self.utility_func(s)
        constraints = [
            0 <= s,
            s <= 1,
            self.cost_func(s) <= self.budget + self.np_cost_func(unit_inclusion_vector),
        ]

        if np.sum(unit_inclusion_vector) >= 1:
            constraints.append(s[unit_inclusion_vector] == 1)

        prob = cp.Problem(cp.Maximize(objective), constraints)
        prob.solve(solver = cp.MOSEK, verbose=False)

        assert prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE], \
            f"Optimization failed. Status: {prob.status}"

        if prob.status == cp.OPTIMAL_INACCURATE:
            print("Warning: Solution is OPTIMAL_INACCURATE. Results may be unreliable.")

        print("Optimal s is: ", s.value)

        return s.value

    def select_samples(self):
        # using only labeled+unlabeled indices, without validation set.
        assert hasattr(self, "cost_func") and self.cost_func is not None, "Need to specify cost function"

        labeled_set = set(self.lSet)
        np.random.seed(self.seed)

        #make labeled inclusion vector of units
        #clean this up, can prob make this variable for the class
        unit_index_map = {u: i for i, u in enumerate(self.units)}
        labeled_unit_inclusion_vector = np.zeros(len(self.units), dtype=bool)

        if self.cost_domain == "unit":
            for i, u in enumerate(self.units):
                unit_point_indices = self.relevant_indices[self.unit_assignment == u]
                labeled_mask = np.array([idx in labeled_set for idx in unit_point_indices])
                
                if np.any(labeled_mask):
                    labeled_unit_inclusion_vector[i] = True
        else:
            for i in range(len(self.units)):
                labeled_unit_inclusion_vector[i] = True if self.units[i] in labeled_set else False

        unit_inclusion_vector=labeled_unit_inclusion_vector.copy()

        unit_to_indices = {
            u: self.relevant_indices[self.unit_assignment[self.relevant_indices] == u]
            for u in self.units
        } if self.cost_domain == "unit" else {u: [self.relevant_indices[u]] for u in self.units}

        labeled_units = [
            u for i, u in enumerate(self.units)
            if np.any([idx in labeled_set for idx in self.relevant_indices[self.unit_assignment == u]])
        ] if self.cost_domain == "unit" else [
            u for u in self.units if u in self.lSet
        ]

        if self.utility_func_type == "random":
            non_labeled_units = np.setdiff1d(self.units, labeled_units)
            permuted_units = np.random.permutation(non_labeled_units)

            for u in permuted_units:
                unit_inclusion_vector[unit_index_map[u]] = 1
                if self.np_cost_func(unit_inclusion_vector) > self.budget + self.np_cost_func(labeled_unit_inclusion_vector):
                    unit_inclusion_vector[unit_index_map[u]] = 0
                    break

        elif self.utility_func_type in ["stratified", "match_population_proportion"]:
            from .representation import Representation
            from .Sampling import Sampling

            sampler = None
            if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "stratified":
                sampler = Representation(self.cfg, self.lSet, self.uSet, self.budget, strategy="balanced")
            elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "match_population_proportion":
                sampler = Representation(self.cfg, self.lSet, self.uSet, self.budget, strategy="match_population")

            assert sampler is not None, f"No sampler found for {self.cfg.ACTIVE_LEARNING.SAMPLING_FN}"

            unit_inclusion_vector[:] = 0
            labeled_cost = self.np_cost_func(labeled_unit_inclusion_vector)

            while self.np_cost_func(unit_inclusion_vector) + labeled_cost <= self.budget:
                sampled_units, _ = sampler.select_samples()
                if not sampled_units:
                    break

                for u in np.random.shuffle(sampled_units):
                    unit_inclusion_vector[unit_index_map[u]] = 1
                    if self.np_cost_func(unit_inclusion_vector) + labeled_cost > self.budget:
                        unit_inclusion_vector[unit_index_map[u]] = 0
                        break
        else:
            assert hasattr(self, "utility_func") and self.utility_func is not None, "Need to specify utility function"

            prob_path = os.path.join(self.cfg.EXP_DIR, "probabilities.pkl")

            if os.path.exists(prob_path):
                print("Loading probabilities from file...")
                with open(prob_path, "rb") as f:
                    probs = dill.load(f)["probs"] #already in the order of relevant indices based on how I saved these files
            else:
                probs = self.solve_opt()
                with open(prob_path, "wb") as f:
                    dill.dump({"ids": self.units, "probs": probs}, f)

            indices = np.arange(len(self.units))
            np.random.shuffle(indices)

            for i in indices:
                draw = np.random.binomial(1, probs[i])
                if draw == 1:
                    unit_inclusion_vector[i] = 1
                    total_cost = self.np_cost_func(unit_inclusion_vector)
                    if total_cost > self.budget + self.np_cost_func(labeled_unit_inclusion_vector):
                        unit_inclusion_vector[i] = 0
                        break

        
        selected_units = self.units[unit_inclusion_vector.astype(bool)]
        selected_units = np.setdiff1d(selected_units, labeled_units)

        activeSet = []
        for u in selected_units:
            available_idxs = unit_to_indices[u]
            unlabeled_idxs = [idx for idx in available_idxs if idx in self.uSet]

            if self.points_per_unit is None:
                selected_points = unlabeled_idxs
            elif len(unlabeled_idxs) <= self.points_per_unit:
                selected_points = unlabeled_idxs
            else:
                selected_points = np.random.choice(unlabeled_idxs, size=self.points_per_unit, replace=False)

            activeSet.extend(selected_points)

        # Ensure we include no overlap with already-labeled
        activeSet = np.array(sorted(set(activeSet)))
        remainSet = np.array(sorted(set(self.uSet) - set(activeSet)))
        total_sample_cost = self.np_cost_func(unit_inclusion_vector)

        print(f"Total Sample Cost: {total_sample_cost}")
        print(f"Finished the selection of {len(activeSet)} samples.")
        print(f"Active set is {activeSet}")


        if self.utility_func_type == "random":
            return activeSet, remainSet, self.np_cost_func(unit_inclusion_vector)
        else:
            return activeSet, remainSet, total_sample_cost, probs, self.relevant_indices