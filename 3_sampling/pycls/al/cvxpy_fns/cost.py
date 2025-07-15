import cvxpy as cp
import numpy as np

def uniform(s):
    return cp.sum(s)

def pointwise_by_array(s, cost_array):
    assert s.shape[0] == len(cost_array), "Cost array length mismatch"
    return s @ cost_array

def unit_aware_pointwise_cost(s, already_labeled_array, unit_labeled_array, c1=1, c2=5):
    unit_labeled_array = np.asarray(unit_labeled_array)
    already_labeled_array = np.asarray(already_labeled_array)
    assert s.shape == unit_labeled_array.shape == already_labeled_array.shape, \
        "Shape mismatch between s and labeled arrays"

    # Cost logic:
    # cost = c1 if not already_labeled and unit not labeled
    #        c2 otherwise
    condition = (~already_labeled_array.astype(bool)) & (~unit_labeled_array.astype(bool))
    cost_per_point = np.where(condition, c1, c2)

    return s @ cost_per_point