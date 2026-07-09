import dill
import json
import os
import numpy as np

from . import cost

COST_FNS = {
    "uniform": cost.uniform,
    "pointwise_by_array": cost.pointwise_by_array,
    "region_aware_unit_cost": cost.region_aware_unit_cost
}

class Sampling:
    def __init__(self, cfg, lSet, uSet, budgetSize):
        self.cfg = cfg
        self.ds_name = cfg['DATASET']['NAME']
        self.seed = cfg['RNG_SEED']
        self.lSet = lSet.astype(int)
        self.uSet = uSet.astype(int)
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.budget = budgetSize
        self._set_unit_assignment()
        self.unit_index_map = {u: i for i, u in enumerate(self.units)}
        self.unit_to_indices = {u: self.relevant_indices[self.unit_assignment == u] for u in self.units}
        self.labeled_unit_vector, self.labeled_unit_set = self._initialize_labeled_units()
        self._resolve_cost_func()

    def _initialize_labeled_units(self):
        labeled_unit_vector = np.zeros(len(self.units), dtype=bool)
        labeled_unit_set = []
        for i, u in enumerate(self.units):
            if any(idx in self.lSet for idx in self.unit_to_indices[u]):
                labeled_unit_vector[i] = True
                labeled_unit_set.append(u)
        return labeled_unit_vector, set(labeled_unit_set)

    def _set_unit_assignment(self):
        self.unit_assignment = np.array(self.cfg.UNITS.UNIT_ASSIGNMENT) if self.cfg.UNITS.UNIT_ASSIGNMENT is not None else np.arange(len(self.relevant_indices))
        self.unit_assignment = self.unit_assignment[self.relevant_indices]
        self.units = np.unique(self.unit_assignment)
        self.points_per_unit = self.cfg.UNITS.POINTS_PER_UNIT if self.cfg.UNITS.POINTS_PER_UNIT is not None else None

    def _set_region_assignment(self):
        assert self.cfg.REGIONS.REGION_ASSIGNMENT is not None, "Need to specify region assignment in config"
        self.region_assignment = np.array(self.cfg.REGIONS.REGION_ASSIGNMENT)[self.relevant_indices]
        self.region_assignment_per_unit = self._get_region_per_unit()

    def _get_region_per_unit(self):
        region_per_unit = {}
        for u in self.units:
            regions = self.region_assignment[np.where(self.unit_assignment == u)[0]]
            unique_regions = np.unique(regions)
            if len(unique_regions) > 1:
                raise ValueError(f"Unit {u} has inconsistent region assignments: {unique_regions}")
            region_per_unit[u] = unique_regions[0]
        return region_per_unit

    def _resolve_cost_func(self):
        cost_func_type = self.cfg.COST.FN
        self.cost_func_type = cost_func_type
        assert cost_func_type in COST_FNS, f"Invalid cost function type: {cost_func_type}"
        self.cost_func = COST_FNS[cost_func_type]

        if cost_func_type == "pointwise_by_array":
            if self.cfg.COST.UNIT_COST_PATH is not None:
                with open(self.cfg.COST.UNIT_COST_PATH, "rb") as f:
                    self.cost_dict = dill.load(f)
                self.cost_array = np.array([self.cost_dict[u] for u in self.units])
            elif self.cfg.COST.ARRAY is not None:
                self.cost_array = np.array(self.cfg.COST.ARRAY)[self.relevant_indices]
            else:
                raise AssertionError
            self.cost_func = lambda s: cost.pointwise_by_array(s, self.cost_array)

        elif cost_func_type == "region_aware_unit_cost":
            in_labeled_set_unit_array = np.array([int(u in self.labeled_unit_set) for u in self.units])
            self._set_region_assignment()
            labeled_regions = set(self.region_assignment[:len(self.lSet)])
            in_labeled_regions_unit_array = [self.region_assignment_per_unit[u] in labeled_regions for u in self.units]

            cost_kwargs = {}
            if self.cfg.REGIONS.IN_REGION_UNIT_COST is not None:
                cost_kwargs["c1"] = self.cfg.REGIONS.IN_REGION_UNIT_COST
            if self.cfg.REGIONS.OUT_OF_REGION_UNIT_COST is not None:
                cost_kwargs["c2"] = self.cfg.REGIONS.OUT_OF_REGION_UNIT_COST

            self.cost_func = lambda s: cost.region_aware_unit_cost(s, in_labeled_set_unit_array, in_labeled_regions_unit_array, **cost_kwargs)

    def random(self, strategy='point'):
        if self.cost_func_type == 'uniform':
            return self.random_uniform()
        elif strategy == 'point':
            return self.random_point_cost_aware()
        else:
            return self.random_unit_cost_aware()

    def random_uniform(self):
        tempIdx = list(range(len(self.uSet)))
        np.random.shuffle(tempIdx)
        activeSet = self.uSet[tempIdx[:self.budget]]
        uSet = self.uSet[tempIdx[self.budget:]]
        return activeSet, uSet

    def random_point_cost_aware(self):
        np.random.seed(self.seed)
        lSet = set(self.lSet)
        uSet = set(self.uSet)
        unit_inclusion_vector = self.labeled_unit_vector.copy()

        all_unlabeled_points = list(uSet - lSet)
        np.random.shuffle(all_unlabeled_points)

        selected = []
        baseline_cost = self.cost_func(self.labeled_unit_vector)

        for idx in all_unlabeled_points:
            unit_idx = self.unit_index_map[self.unit_assignment[idx]]
            original = unit_inclusion_vector[unit_idx]
            unit_inclusion_vector[unit_idx] = 1

            if self.cost_func(unit_inclusion_vector) > self.budget + baseline_cost:
                unit_inclusion_vector[unit_idx] = original
                break

            selected.append(idx)
            lSet.add(idx)
            uSet.remove(idx)

        activeSet = np.array(selected)
        remainSet = np.array(sorted(uSet))
        total_sample_cost = self.cost_func(unit_inclusion_vector)
        self._save_selection_metadata(total_sample_cost, len(activeSet), baseline_cost)
        return activeSet, remainSet

    def random_unit_cost_aware(self):
        np.random.seed(self.seed)
        lSet = set(self.lSet)
        uSet = set(self.uSet)
        unit_inclusion_vector = self.labeled_unit_vector.copy()
        labeled_units = set(self.unit_assignment[self.lSet])

        selected = []
        baseline_cost = self.cost_func(self.labeled_unit_vector)
        non_labeled_units = np.setdiff1d(self.units, list(labeled_units))

        for u in np.random.permutation(non_labeled_units):
            unit_inclusion_vector[self.unit_index_map[u]] = 1
            added_cost = self.cost_func(unit_inclusion_vector) - baseline_cost
            if added_cost > self.budget:
                unit_inclusion_vector[self.unit_index_map[u]] = 0
                break

            unit_indices = self.unit_to_indices[u]
            if self.points_per_unit is None or len(unit_indices) <= self.points_per_unit:
                selected_points = unit_indices
            else:
                selected_points = np.random.choice(unit_indices, size=self.points_per_unit, replace=False)

            selected.extend(selected_points)
            lSet.update(selected_points)
            uSet.difference_update(selected_points)

        activeSet = np.array(selected)
        remainSet = np.array(sorted(uSet))
        total_sample_cost = self.cost_func(unit_inclusion_vector)
        self._save_selection_metadata(total_sample_cost, len(activeSet), baseline_cost)
        return activeSet, remainSet

    def _set_groups(self):
        assert self.cfg.GROUPS.GROUP_ASSIGNMENT is not None, "Group assignment must not be none for poprisk utility function"
        self.group_assignment = np.array(self.cfg.GROUPS.GROUP_ASSIGNMENT)[self.relevant_indices]

    def stratified(self):
        self._set_groups()
        return self._representation_sampling("balanced")

    def match_population_proportion(self):
        self._set_groups()
        return self._representation_sampling("match_population")

    def _representation_sampling(self, strategy):
        np.random.seed(self.seed)
        selected = []
        all_groups = np.unique(self.group_assignment)
        group_total = {g: np.sum(self.group_assignment == g) for g in all_groups}
        labeled_group_counts = dict(zip(*np.unique(self.group_assignment[list(self.lSet)], return_counts=True)))
        for g in all_groups:
            labeled_group_counts.setdefault(g, 0)

        lSet = set(self.lSet)
        uSet = set(self.uSet)
        unit_inclusion_vector = self.labeled_unit_vector.copy()
        baseline_cost = self.cost_func(self.labeled_unit_vector)

        while self.cost_func(unit_inclusion_vector) <= baseline_cost + self.budget:
            if strategy == "balanced":
                candidate_groups = sorted(labeled_group_counts, key=lambda g: labeled_group_counts[g])
            else:
                total_labeled = sum(labeled_group_counts.values()) or 1
                pop_prop = {g: group_total[g] / sum(group_total.values()) for g in all_groups}
                labeled_prop = {g: labeled_group_counts[g] / total_labeled for g in all_groups}
                deviation = {g: abs(labeled_prop[g] - pop_prop[g]) for g in all_groups}
                candidate_groups = sorted(deviation, key=deviation.get, reverse=True)
            np.random.shuffle(candidate_groups)

            chosen = None
            for g in candidate_groups:
                candidates = [idx for idx in uSet if self.group_assignment[idx] == g]
                if candidates:
                    chosen = np.random.choice(candidates)
                    break
            assert chosen is not None, "Selection failed"

            unit_idx = self.unit_index_map[self.unit_assignment[chosen]]
            unit_inclusion_vector[unit_idx] = 1
            if self.cost_func(unit_inclusion_vector) > baseline_cost + self.budget:
                unit_inclusion_vector[unit_idx] = 0
                break

            selected.append(chosen)
            uSet.remove(chosen)
            lSet.add(chosen)
            labeled_group_counts[self.group_assignment[chosen]] += 1

        activeSet = np.array(selected)
        remainSet = np.array(sorted(uSet))
        total_sample_cost = self.cost_func(unit_inclusion_vector)
        self._save_selection_metadata(total_sample_cost, len(activeSet), baseline_cost)
        return activeSet, remainSet

    def _save_selection_metadata(self, total_sample_cost, num_selected_samples, labeled_sample_cost):
        save_path = os.path.join(self.cfg.EXP_DIR, "selection_metadata.json")
        metadata = {
            "total_sample_cost": float(total_sample_cost),
            "labeled_sample_cost": float(labeled_sample_cost),
            "seed": self.seed,
            "num_selected_samples": num_selected_samples
        }
        with open(save_path, "w") as f:
            json.dump(metadata, f)

    def _save_costs_metadata(self):
        seed_dir = os.path.join(self.cfg.SAMPLING_DIR, f"seed_{self.seed}")
        original_path = os.path.join(seed_dir, "episode_0", "lSet.npy")
        new_path = os.path.join(seed_dir, "episode_1", "lSet.npy")
        if not os.path.exists(original_path) or not os.path.exists(new_path):
            raise FileNotFoundError("Missing lSet.npy file in expected location.")

        original_labeled_set = np.load(original_path, allow_pickle=True)
        new_labeled_set = np.load(new_path, allow_pickle=True)

        def get_labeled_unit_vector(labeled_indices):
            labeled_indices_set = set(labeled_indices)
            labeled_unit_vector = np.zeros(len(self.units), dtype=bool)
            labeled_unit_set = set()
            for i, u in enumerate(self.units):
                if any(idx in labeled_indices_set for idx in self.unit_to_indices[u]):
                    labeled_unit_vector[i] = True
                    labeled_unit_set.add(str(u))
            return labeled_unit_vector, labeled_unit_set

        original_vector, original_units = get_labeled_unit_vector(original_labeled_set)
        new_vector, new_units = get_labeled_unit_vector(new_labeled_set)
        original_cost = self.cost_func(original_vector)
        new_cost = self.cost_func(new_vector)

        save_path = os.path.join(self.cfg.EXP_DIR, "selection_metadata.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        metadata = {}
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                metadata = json.load(f)

        metadata["total_sample_cost"] = float(new_cost)
        metadata["labeled_sample_cost"] = float(original_cost)
        metadata["seed"] = self.seed
        metadata["num_selected_samples"] = len(new_units) - len(original_units)

        with open(save_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return original_cost
