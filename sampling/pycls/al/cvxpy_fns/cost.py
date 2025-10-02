import cvxpy as cp
import numpy as np

def uniform(s):
    return cp.sum(s)

def pointwise_by_array(s, cost_array):
    assert s.shape[0] == len(cost_array), "Cost array length mismatch"
    return s @ cost_array

def region_aware_unit_cost(s, in_labeled_set_array, in_labeled_region_array, c1=1, c2=2): #unit_labeled_array is binary indicating whether that index point is in an already labeled unit
    in_labeled_set_array = np.asarray(in_labeled_set_array)
    in_labeled_region_array = np.asarray(in_labeled_region_array)
    assert s.shape == in_labeled_set_array.shape == in_labeled_region_array.shape, \
        "Shape mismatch between s and labeled arrays"

    # Cost logic:
    # cost = c1 if not already_labeled and region labeled
    #        c2 otherwise
    condition = (~in_labeled_set_array.astype(bool)) & (in_labeled_region_array.astype(bool))
    cost_per_point = np.where(condition, c1, c2)

    return s @ cost_per_point