import dill
import numpy as np

from . import cost

COST_FNS = {
    "uniform": cost.uniform,
    "pointwise_by_array": cost.pointwise_by_array,
    "unit_aware_pointwise_cost": cost.unit_aware_pointwise_cost
}

class Sampling:
    def __init__(self, cfg, lSet, uSet, budgetSize):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.lSet = lSet.astype(int)
        self.uSet = uSet.astype(int)
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.budget = budgetSize
        self._set_unit_assignment()
        self._resolve_cost_func()

        self.unit_index_map = {u: i for i, u in enumerate(self.units)}

        self.unit_to_indices = {
            u: self.relevant_indices[self.unit_assignment[self.relevant_indices] == u]
            for u in self.units
        }
        self.labeled_unit_vector = self._initialize_labeled_unit_vector()

    def _set_unit_assignment(self):
        self.unit_assignment = np.array(self.cfg.UNITS.UNIT_ASSIGNMENT) if self.cfg.UNITS.UNIT_ASSIGNMENT is not None else np.arange(len(self.relevant_indices))
        self.unit_assignment = self.unit_assignment[self.relevant_indices]

        self.units = np.unique(self.unit_assignment)
        self.points_per_unit = self.cfg.UNITS.POINTS_PER_UNIT if self.cfg.UNITS.POINTS_PER_UNIT is not None else None

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
                self.cost_array = np.array(self.cfg.COST.ARRAY)
                self.cost_array = self.cost_array[self.relevant_indices]
            else:
                raise(AssertionError)

            self.cost_func = lambda s: cost.pointwise_by_array(s, self.cost_array)


    def _compute_labeled_cost(self, labeled_unit_vector):
        return self.cost_func(labeled_unit_vector)

    def _initialize_labeled_unit_vector(self):
        labeled_unit_vector = np.zeros(len(self.units), dtype=bool)

        for i, u in enumerate(self.units):
            indices = self.unit_to_indices[u]
            if any(idx in self.lSet for idx in indices):
                labeled_unit_vector[i] = True
        return labeled_unit_vector

    def random(self, strategy=None):
        #self.cost_func_type = cfg.COST.FN
        if self.cost_func_type == 'uniform':
            return self.random_uniform()
        elif strategy == 'point':
            return self.random_point_cost_aware()
        else:
            return self.random_unit_cost_aware() #will sample points if units are points

    def random_uniform(self):
        tempIdx = [i for i in range(len(self.uSet))]
        np.random.shuffle(tempIdx)
        activeSet = self.uSet[tempIdx[0:self.budget]]
        uSet = self.uSet[tempIdx[self.budget:]]
        return activeSet, uSet
    
    def random_point_cost_aware(self):
        print("Running point-wise random selection...")
        np.random.seed(self.seed)

        lSet = set(self.lSet)
        uSet = set(self.uSet)

        unit_inclusion_vector = np.zeros(len(self.units), dtype=bool)
        labeled_units = set(self.unit_assignment[self.lSet])

        all_unlabeled_points = list(uSet - lSet)
        np.random.shuffle(all_unlabeled_points)

        selected = []

        for idx in all_unlabeled_points:
            unit = self.unit_assignment[idx]
            unit_idx = self.unit_index_map[unit]

            #temporarily include
            original = unit_inclusion_vector[unit_idx]
            unit_inclusion_vector[unit_idx] = 1

            #Check cost
            if self.cost_func(unit_inclusion_vector) > self.budget + self.cost_func(self.labeled_unit_vector):
                #revert inclusion if over budget
                unit_inclusion_vector[unit_idx] = original
                break

            selected.append(idx)
            lSet.add(idx)
            uSet.remove(idx)
            labeled_units.add(unit)

        activeSet = np.array(selected)
        remainSet = np.array(sorted(uSet))
        return activeSet, remainSet

    
    def random_unit_cost_aware(self):
        print("Running unit-wise random selection...")
        np.random.seed(self.seed)

        lSet = set(self.lSet)
        uSet = set(self.uSet)

        unit_inclusion_vector = np.zeros(len(self.units), dtype=bool)
        labeled_units = set(self.unit_assignment[self.lSet])

        selected = []
        while self.cost_func(unit_inclusion_vector) <= self.cost_func(self.labeled_unit_vector + self.budget):
            non_labeled_units = np.setdiff1d(self.units, list(labeled_units))
            permuted_units = np.random.permutation(non_labeled_units)

            for u in permuted_units:
                unit_inclusion_vector[self.unit_index_map[u]] = 1
                if self.cost_func(unit_inclusion_vector) > self.budget + self.cost_func(self.labeled_unit_vector):
                    unit_inclusion_vector[self.unit_index_map[u]] = 0
                    break

                # Get indices of points in the selected unit
                unit_indices = self.unit_to_indices[u]

                # Update sets
                selected.extend(unit_indices)
                lSet.update(unit_indices)
                uSet.difference_update(unit_indices)
                labeled_units.add(u)

        activeSet = np.array(selected)
        remainSet = np.array(sorted(uSet))
        return activeSet, remainSet

    def _set_groups(self):
        assert self.cfg.GROUPS.GROUP_ASSIGNMENT is not None, "Group assignment must not be none for poprisk utility function"

        self.group_assignment = np.array(self.cfg.GROUPS.GROUP_ASSIGNMENT)

        self.group_assignment = self.group_assignment[self.relevant_indices]


    def stratified(self):
        self._set_groups()
        return self._representation_sampling("balanced")

    def match_population_proportion(self):
        self._set_groups()
        return self._representation_sampling("match_population")

    def _representation_sampling(self, strategy):
        assert self.group_assignment is not None, "Group assignment must be provided for representation sampling"
        np.random.seed(self.seed)

        print(f"Starting representation sampling with strategy: {strategy}")
        selected = []
        all_groups = np.unique(self.group_assignment)
        group_total = {g: np.sum(self.group_assignment == g) for g in all_groups}

        print(f"All groups found: {all_groups}")
        print(f"Group population sizes: {group_total}")

        labeled_group_counts = dict(zip(*np.unique(self.group_assignment[list(self.lSet)], return_counts=True)))
        for g in all_groups:
            labeled_group_counts.setdefault(g, 0)

        print(f"Initial labeled group counts: {labeled_group_counts}")

        lSet = set(self.lSet)
        uSet = set(self.uSet)

        unit_inclusion_vector = np.zeros(len(self.units), dtype=bool)

        step = 0
        while self.cost_func(unit_inclusion_vector) <= self.cost_func(self.labeled_unit_vector) + self.budget:
            step += 1
            print(f"\nStep {step}")
            
            if strategy == "balanced":
                candidate_groups = sorted(labeled_group_counts, key=lambda g: labeled_group_counts[g])
                print(f"Candidate groups (least labeled first): {candidate_groups}")
            elif strategy == "match_population":
                total_labeled = sum(labeled_group_counts.values()) or 1
                pop_prop = {g: group_total[g] / sum(group_total.values()) for g in all_groups}
                labeled_prop = {g: labeled_group_counts[g] / total_labeled for g in all_groups}
                deviation = {g: abs(labeled_prop[g] - pop_prop[g]) for g in all_groups}
                candidate_groups = sorted(deviation, key=deviation.get, reverse=True)
                print(f"Candidate groups (by deviation from population prop): {candidate_groups}")

            np.random.shuffle(candidate_groups)
            print(f"Candidate groups after shuffle: {candidate_groups}")

            chosen = None
            for g in candidate_groups:
                candidates = [idx for idx in uSet if self.group_assignment[idx] == g]
                if candidates:
                    chosen = np.random.choice(candidates)
                    print(f"Chose index {chosen} from group {g}")
                    break

            assert chosen is not None, f"Selection failed"

            unit_of_chosen = self.unit_assignment[chosen]
            unit_idx = self.unit_index_map[unit_of_chosen]
            unit_inclusion_vector[unit_idx] = 1  # mark unit as selected

            if self.cost_func(unit_inclusion_vector) > self.cost_func(self.labeled_unit_vector) + self.budget:
                print(f"Adding unit {unit_of_chosen} (idx {unit_idx}) would exceed budget")
                unit_inclusion_vector[unit_idx] = 0
                break 
            
            selected.append(chosen)
            uSet.remove(chosen)
            lSet.add(chosen)
            labeled_group_counts[self.group_assignment[chosen]] += 1

            print(f"Updated labeled_group_counts: {labeled_group_counts}")
            print(f"Unit {unit_of_chosen} marked selected (index {unit_idx}).")

        activeSet = np.array(selected)
        remainSet = np.array(sorted(list(uSet)))
        print(f"\n✅ Sampling finished. Total selected: {len(activeSet)}")
        return activeSet, remainSet


    def _unit_vector_to_active_set(self, unit_inclusion_vector):
        selected_units = self.units[unit_inclusion_vector]
        active_set = []
        for u in selected_units:
            available = self.unit_to_indices[u]
            unlabeled = [idx for idx in available if idx in self.uSet]
            if self.points_per_unit is None:
                selected = unlabeled
            elif len(unlabeled) <= self.points_per_unit:
                selected = unlabeled
            else:
                selected = np.random.choice(unlabeled, size=self.points_per_unit, replace=False)
            active_set.extend(selected)

        active_set = np.array(sorted(set(active_set)))
        remain_set = np.array(sorted(self.uSet - set(active_set)))
        cost = self._compute_labeled_cost(unit_inclusion_vector)
        return active_set, remain_set, cost