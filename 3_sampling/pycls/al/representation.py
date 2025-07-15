import os
import dill
import numpy as np
import pandas as pd
import numpy as np

class Representation:
    def __init__(self, cfg, lSet, uSet, budgetSize, strategy="proportional"):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.lSet = lSet
        self.uSet = uSet
        self.budget = budgetSize
        self.strategy = strategy
        self._set_group_assignments()

    def _set_group_assignments(self):
        self.groups = np.array(self.cfg.GROUPS.GROUP_ASSIGNMENT)

    def select_samples(self):
        strategy = self.strategy

        np.random.seed(self.seed)

        lSet = set(self.lSet)
        uSet = set(self.uSet)

        selected = []
        # Compute group totals
        all_groups = np.unique(self.groups)
        group_total = {}
        for g in all_groups:
            group_total[g] = np.sum(self.groups == g)

        # Initialize labeled counts per group
        labeled_groups = self.groups[list(lSet)]
        labeled_group_labels, labeled_group_counts = np.unique(labeled_groups, return_counts=True)
        group_labeled_dict = dict(zip(labeled_group_labels, labeled_group_counts))
        for g in all_groups:
            if g not in group_labeled_dict:
                group_labeled_dict[g] = 0

        for i in range(self.budget):
            if strategy == "balanced":
                groups_sorted_by_count = sorted(group_labeled_dict.items(), key=lambda x: x[1])
                candidate_groups = [g for g, c in groups_sorted_by_count]

            elif strategy == "match_population":
                total_labeled = sum(group_labeled_dict.values()) or 1
                population_props = {g: group_total[g] / sum(group_total.values()) for g in all_groups}
                labeled_props = {g: group_labeled_dict[g] / total_labeled for g in all_groups}

                deviations = {g: abs(labeled_props[g] - population_props[g]) for g in all_groups}

                # Sort groups by deviation descending (largest mismatch first)
                groups_sorted_by_deviation = sorted(deviations.items(), key=lambda x: x[1], reverse=True)
                candidate_groups = [g for g, dev in groups_sorted_by_deviation]

            else:
                raise ValueError(f"Unknown strategy '{strategy}'")

            np.random.shuffle(candidate_groups)

            chosen = None
            for group in candidate_groups:
                candidates = [idx for idx in uSet if self.groups[idx] == group]
                if candidates:
                    chosen = np.random.choice(candidates)
                    break

            assert chosen is not None, f"Selection failed at step {i+1}"

            selected.append(chosen)
            uSet.remove(chosen)
            lSet.add(chosen)
            group_labeled_dict[self.groups[chosen]] += 1

            print(f"[{i+1}/{self.budget}] Selected index {chosen} from group {self.groups[chosen]} with labeled count {group_labeled_dict[self.groups[chosen]]}")

        selected = np.array(selected)
        assert len(selected) == self.budget
        assert len(np.intersect1d(selected, self.lSet)) == 0
        assert len(set(selected)) == len(selected)

        activeSet = selected
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        print(f"\nFinished the selection of {len(activeSet)} samples.")
        print(f"Active set: {activeSet}")
        print(f"Remaining unlabeled set size: {len(remainSet)}")
        return activeSet, remainSet