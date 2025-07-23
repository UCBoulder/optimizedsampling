import os
import dill
import json
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
    "region_aware_unit_cost": cost.region_aware_unit_cost
}

NP_COST_FNS = {
    "uniform": np_cost.uniform,
    "pointwise_by_array": np_cost.pointwise_by_array,
    "region_aware_unit_cost": np_cost.region_aware_unit_cost
}

class Opt:
    def __init__(self, cfg, lSet, uSet, budgetSize):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.lSet = lSet.astype(int)
        self.uSet = uSet.astype(int)

        self.budget = budgetSize

        # Set indices and assignments
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.unit_assignment = self._init_unit_assignment()
        self.units = np.unique(self.unit_assignment)
        self.unit_to_indices = self._compute_unit_to_indices()
        self.points_per_unit = self.cfg.UNITS.POINTS_PER_UNIT if self.cfg.UNITS.POINTS_PER_UNIT is not None else None

        self.labeled_unit_vector, self.labeled_unit_set = self._initialize_labeled_units()

        self._resolve_cost_func()

    def _initialize_labeled_units(self):
        labeled_unit_vector = np.zeros(len(self.units), dtype=bool)
        labeled_unit_list = []

        for i, u in enumerate(self.units):
            indices = self.unit_to_indices[u]
            if any(idx in self.lSet for idx in indices):
                labeled_unit_vector[i] = True
                labeled_unit_list.append(str(u))
        labeled_unit_set = set(labeled_unit_list)
        return labeled_unit_vector, labeled_unit_set

    def _init_unit_assignment(self):
        if self.cfg.UNITS.UNIT_ASSIGNMENT is not None:
            unit_assignment = np.array(self.cfg.UNITS.UNIT_ASSIGNMENT)
            unit_assignment = unit_assignment[self.relevant_indices]
        else:
            unit_assignment = self.relevant_indices.copy()
        return unit_assignment

    def _set_region_assignment(self):
        assert self.cfg.REGIONS.REGION_ASSIGNMENT is not None, "Need to specify region assignment in config"
        self.region_assignment = np.array(self.cfg.REGIONS.REGION_ASSIGNMENT)[self.relevant_indices]

        self.region_assignment_per_unit = self._get_region_per_unit()
        self.region_array_per_unit = np.array([self.region_assignment_per_unit[u] for u in self.units])

    def _get_region_per_unit(self):
        region_per_unit = {}

        for u in self.units:
            indices = np.where(self.unit_assignment == u)[0]
            regions = self.region_assignment[indices]
            unique_regions = np.unique(regions)

            if len(unique_regions) > 1:
                raise ValueError(f"Unit {u} has inconsistent region assignments: {unique_regions}")
            
            region_per_unit[u] = unique_regions[0]

        return region_per_unit
    
    def _compute_unit_to_indices(self):
        return {
            u: self.relevant_indices[self.unit_assignment == u]
            for u in self.units
        }
        #note, will need to adjust if the cost function is unit_aware_pointwise

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
            similarity_matrix = similarity_matrix[self.relevant_indices, :]

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
        
        if cost_func_type == "pointwise_by_array":
            if self.cfg.COST.UNIT_COST_PATH is not None:
                with open(self.cfg.COST.UNIT_COST_PATH, "rb") as f:
                    self.cost_dict = dill.load(f)
                self.cost_array = np.array([self.cost_dict[u] for u in self.units])
            elif self.cfg.COST.ARRAY is not None:
                self.cost_array = np.array(self.cfg.COST.ARRAY)
                self.cost_array = self.cost_array[self.units]
            else:
                raise(AssertionError)

            self.cost_func = lambda s: cost.pointwise_by_array(s, self.cost_array)
            self.np_cost_func = lambda s: np_cost.pointwise_by_array(s, self.cost_array)

        elif cost_func_type == "region_aware_unit_cost":
            in_labeled_set_unit_array = np.array([int(u in set(self.labeled_unit_set)) for u in self.units])
            self._set_region_assignment()

            labeled_regions = set(self.region_assignment[:len(self.lSet)])
            in_labeled_regions_unit_array = [self.region_assignment_per_unit[u] in labeled_regions for u in self.units]

            cost_kwargs = {}
            if self.cfg.REGIONS.IN_REGION_UNIT_COST is not None:
                cost_kwargs["c1"] = self.cfg.REGIONS.IN_REGION_UNIT_COST
            if self.cfg.REGIONS.OUT_OF_REGION_UNIT_COST is not None:
                cost_kwargs["c2"] = self.cfg.REGIONS.OUT_OF_REGION_UNIT_COST

            self.cost_func = lambda s: cost.region_aware_unit_cost(s, in_labeled_set_unit_array, in_labeled_regions_unit_array, **cost_kwargs)
            self.np_cost_func = lambda s: np_cost.region_aware_unit_cost(s, in_labeled_set_unit_array, in_labeled_regions_unit_array, **cost_kwargs)


    def solve_opt(self):
        assert self.utility_func_type != "Random", "Please do not use the optimization function for random selection"

        #make labeled inclusion vector of units
        unit_inclusion_vector = self.labeled_unit_vector.copy()

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

    def _load_probabilities(self):
        prob_path = os.path.join(self.cfg.SAMPLING_DIR, f"probabilities_seed_{self.seed}.pkl")
        if os.path.exists(prob_path):
            print("Loading probabilities from file...")
            with open(prob_path, "rb") as f:
                data = dill.load(f)
            saved_ids = data["ids"]
            saved_probs = data["probs"]

            # Map saved probabilities to current self.units order
            id_to_prob = dict(zip(saved_ids, saved_probs))
            aligned_probs = np.array([id_to_prob.get(u, 0.0) for u in self.units])
            return aligned_probs
        else:
            return None


    def _save_probabilities(self, probs):
        prob_path = os.path.join(self.cfg.SAMPLING_DIR, f"probabilities_seed_{self.seed}.pkl")
        with open(prob_path, "wb") as f:
            dill.dump({"ids": self.units, "probs": probs}, f)

    def _save_selection_metadata(self, total_sample_cost, num_selected_samples):
        save_path = os.path.join(self.cfg.EXP_DIR, "selection_metadata.json")
        metadata = {
            "total_sample_cost": float(total_sample_cost),  # convert np.float64 to native float
            "seed": self.seed,
            "num_selected_samples": num_selected_samples
        }
        with open(save_path, "w") as f:
            json.dump(metadata, f)

    def select_samples(self):
        assert hasattr(self, "cost_func") and self.cost_func is not None, "Need to specify cost function"
        assert hasattr(self, "utility_func") and self.utility_func is not None, "Need to specify utility function"

        np.random.seed(self.seed)
        unit_inclusion_vector = self.labeled_unit_vector.copy()

        probs = self._load_probabilities()
        if probs is None:
            probs = self.solve_opt()
            probs = np.clip(probs, 0.0, 1.0) # handle floating point issues
            self._save_probabilities(probs)

        shuffled_unit_indices = np.arange(len(self.units))
        np.random.shuffle(shuffled_unit_indices)

        probs_shuffled = probs[shuffled_unit_indices]

        for i in range(len(shuffled_unit_indices)):
            draw = np.random.binomial(1, probs_shuffled[i])
            unit_idx = shuffled_unit_indices[i]  # original index in dataset

            if draw == 1:
                unit_inclusion_vector[unit_idx] = 1
                total_cost = self.np_cost_func(unit_inclusion_vector)
                print(f"Total Cost: {total_cost}")

                if total_cost > self.budget + self.np_cost_func(self.labeled_unit_vector):
                    unit_inclusion_vector[unit_idx] = 0
                    break

        selected_units = self.units[unit_inclusion_vector.astype(bool)]
        labeled_units_array = np.array(list(self.labeled_unit_set), dtype=selected_units.dtype)
        selected_units = np.setdiff1d(selected_units, labeled_units_array)

        activeSet = []
        for u in selected_units:
            available_idxs = self.unit_to_indices[u]
            print(f"Available Indices: {available_idxs}")
            unlabeled_idxs = [idx for idx in available_idxs if idx in self.uSet]

            if self.points_per_unit is None:
                selected_points = unlabeled_idxs
            elif len(unlabeled_idxs) <= self.points_per_unit:
                selected_points = unlabeled_idxs
            else:
                selected_points = np.random.choice(unlabeled_idxs, size=self.points_per_unit, replace=False)

            print(f"Adding {selected_points} to activeSet")
            activeSet.extend(selected_points)

        activeSet = np.array(sorted(set(activeSet)))
        remainSet = np.array(sorted(set(self.uSet) - set(activeSet)))
        total_sample_cost = self.np_cost_func(unit_inclusion_vector)

        print(f"Total Sample Cost: {total_sample_cost}")
        print(f"Finished the selection of {len(activeSet)} samples.")
        print(f"Active set is {activeSet}")

        self._save_selection_metadata(total_sample_cost, len(activeSet))

        return activeSet, remainSet