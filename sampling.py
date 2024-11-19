import numpy as np
from mosaiks.code.mosaiks.utils import io
from oed import *
import random

'''
Determine indices of valid entries in a set (not NaN or inf)
'''
def valid(set, *args):
    valid = (~np.isnan(set) & ~np.isinf(set))[:,0]
    for arg in args:
        if len(arg.shape)==1:
            valid = valid & ~np.isnan(arg) & ~np.isinf(arg)
        else:
            valid= valid & (~np.isnan(arg) & ~np.isinf(arg))[:,0]
    return valid

'''
Determine number of data points that are not NaN
'''
def valid_num(c, label, X, latlons):
    c = io.get_filepaths(c, label)
    c_app = getattr(c, label)
    Y = io.get_Y(c, c_app["colname"])

    X = X.reindex(Y.index)
    latlons = latlons.reindex(Y.index)

    valid_rows = Y.notna() & (Y != -999)
    valid_rows = valid_rows & (X.notna().all(axis=1) & latlons.notna().all(axis=1))
    X = X[valid_rows]

    size_of_valid = X.shape[0]
    print(f"Size of valid set for {label}", size_of_valid)
    return size_of_valid

'''
Spatial-only baseline
Takes a random subset of training data
'''
def random_subset(X_train, Y_train, latlon, size):
    print("Generating subset using SRS...")
    random_subset = np.random.choice(len(X_train), size=size, replace=False)
    return X_train[random_subset], Y_train[random_subset], latlon[random_subset]

'''
Image-only baseline
Takes an OED subset of training data--Image-only baseline
Only works for V-optimality as of 11/4
'''
def image_subset(X_train, Y_train, latlon_train, rule, size):
    print("Generating subset using {rule}...".format(rule='V Optimal Design'))
    subset_idxs = sampling_with_prob(X_train, size, rule)
    return X_train[subset_idxs], Y_train[subset_idxs], latlon_train[subset_idxs]

'''
Image and Spatial based
Takes an OED subset of training data using SatCLIP embeddings
'''
def satclip_subset(X_train, Y_train, latlon_train, loc_emb_train, rule, size):
    print("Generating subset using satclip embeddings...")
    subset_idxs = sampling_with_prob(loc_emb_train, size, rule)
    return X_train[subset_idxs], Y_train[subset_idxs], latlon_train[subset_idxs]

'''
Takes a SRS subset of training data such that total cost does not exceed budget
'''
def constrained_random_subset(X_train, Y_train, latlon_train, budget):
    cost = 0
    idxs = [i for i in range(len(X_train))]
    subset_idxs = []
    while cost<=budget:
       sampled_idx = random.sample(idxs, 1)
       subset_idxs.append(sampled_idx)
       idxs = idxs[:sampled_idx] + idxs[sampled_idx + 1:]
       cost += cost[sampled_idx]
    return X_train[subset_idxs], Y_train[subset_idxs], latlon_train[subset_idxs]