""" 1. Decide train and tests splits of USAVars data (SRS, OED) with specified number of samples
    2. Parse data so it corresponds with features
    2. Train a ridge regression on train split
    3. Plot regression residuals for each variable of interest """

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from os.path import basename, dirname, join
from sklearn.metrics import r2_score
from oed import *
from pca import pca
# from satclip_files.satclip.satclip import get_embeddings

from mosaiks.code.mosaiks.utils import *
from mosaiks.code.mosaiks.utils import io
from mosaiks.code.mosaiks.solve import data_parser as parse
from mosaiks.code.mosaiks.solve import solve_functions as solve
from mosaiks.code.mosaiks.solve import interpret_results as ir


#To store results
results_dict = {}
results_dict_test = {}

#Path to store results
save_patt = join(
        "{save_dir}",
        "CONTUS_{{rule}}_{label}_{variable}_outcomes_{{reg_type}}_subset_{{subset_n}}.data"
    )

#Set solver to ridge regression
solver = solve.ridge_regression

def valid(set, *args):
    valid = (~np.isnan(set) & ~np.isinf(set))[:,0]
    for arg in args:
        if len(arg.shape)==1:
            valid = valid & ~np.isnan(arg) & ~np.isinf(arg)
        else:
            valid= valid & (~np.isnan(arg) & ~np.isinf(arg))[:,0]
    return valid

#Make sure X and latlons do not have NaN values
def valid_set(c, label, X, latlons):
    c = io.get_filepaths(c, label)
    c_app = getattr(c, label)
    Y = io.get_Y(c, c_app["colname"])

    X = X.reindex(Y.index)
    latlons = latlons.reindex(Y.index)

    valid_rows = Y.notna() & (Y != -999)
    valid_rows = valid_rows & (X.notna().all(axis=1) & latlons.notna().all(axis=1))
    X = X[valid_rows]
    latlons = latlons[valid_rows]
    size_of_valid = X.shape[0]
    print(f"Size of valid set for {label}", size_of_valid)
    return size_of_valid, X, latlons

#Taking a random subset of training data--Spatial-only baseline
def random_subset(X_train, Y_train, latlon, size):
    random_subset = np.random.choice(len(X_train), size=size, replace=False)
    return X_train[random_subset], Y_train[random_subset], latlon[random_subset]

#Taking a OED subset of training data--Image-only baseline
#Only works for V-optimality as of 11/4
def image_subset(X_train, Y_train, latlon, rule, size):
    subset_indices = sampling(X_train, size, rule)
    return X_train[subset_indices], Y_train[subset_indices], latlon[subset_indices]

#TODO: check
# def satclip_subset(X_train, Y_train, latlon, rule, size):
#     emb = get_embeddings(latlon)
#     subset_indices = sampling(emb, size, rule)
#     return X_train[subset_indices], Y_train[subset_indices], latlon[subset_indices]

#Run ridge regression on mosaiks features for label
def train_and_test(c, label, X, latlons, subset_n=None, rule=None):
    print("*** Running regressions for: {label} with {num} samples".format(label=label, num=subset_n))

    #Test all lambdas (specified in config file)
    this_lambdas = io.get_lambdas(c, label, best_lambda_fpath=None)

    #Access config file and specific label parameters from config file
    c = io.get_filepaths(c, label)
    c_app = getattr(c, label)
    sampling_type = c_app["sampling"]  # UAR 
    this_save_patt = save_patt.format(
        save_dir="data/output",
        label=label,
        variable=c_app["variable"]
    )

    #Bounds from config file
    if c_app["logged"]:
        bounds = np.array([c_app["us_bounds_log_pred"]])
    else:
        bounds = np.array([c_app["us_bounds_pred"]])

    ## Get save path
    if subset_n is not None:
        subset_str = f"_subset{subset_n}"
    else:
        subset_str = ""

    if rule is not None:
        rule_str = f"_{rule}"
    else:
        rule_str = ""
    save_path_validation = this_save_patt.format(reg_type="scatter", subset_n=subset_str, rule=rule_str)
    save_path_test = this_save_patt.format(reg_type="testset", subset_n=subset_str, rule=rule_str)

    (
        this_X,
        this_X_test,
        this_Y,
        this_Y_test,
        this_latlons,
        this_latlons_test,
    ) = parse.merge_dropna_transform_split_train_test(
        c, label, X, latlons
    )

    #Clean up; not sure why valid_set func does not return X, latlons with no NaNs
    valid_train=valid(this_X, this_Y, this_latlons)
    valid_test=valid(this_X_test, this_Y_test, this_latlons_test)

    this_X = this_X[valid_train]
    this_X_test = this_X_test[valid_test]
    this_Y = this_Y[valid_train]
    this_Y_test = this_Y_test[valid_test]
    this_latlons = this_latlons[valid_train]
    this_latlons_test = this_latlons_test[valid_test]

    #this_X, this_X_test = pca(this_X, this_X_test)

    # Take a random subset of size n
    if subset_n is not None:
            if rule is None:
                this_X, this_Y, this_latlons = random_subset(this_X, this_Y, this_latlons, subset_n)
            else:
                this_X, this_Y, this_latlons = image_subset(this_X, this_Y, this_latlons, rule, subset_n)
    else:
        while (this_X.shape[0]%5 != 0):
            this_X = this_X[:-1]
            this_latlons = this_latlons[:-1]
            this_Y = this_Y[:-1]

    from IPython import embed; embed()
    subset_n = this_X.shape[0]
    print("Training model...")
    import time

    st_train = time.time()
    kfold_results = solve.kfold_solve(
        this_X,
        this_Y,
        solve_function=solver,
        num_folds=c.ml_model["n_folds"],
        return_model=True,
        lambdas=this_lambdas,
        return_preds=True,
        svd_solve=False,
        clip_bounds=bounds,
    )
    print("")

    # get timing
    training_time = time.time() - st_train
    print("Training time:", training_time)

    ## Store the metrics and the predictions from the best performing model
    best_lambda_idx, best_metrics, best_preds = ir.interpret_kfold_results(
        kfold_results, "r2_score", hps=[("lambdas", c_app["lambdas"])]
    )
    best_lambda = this_lambdas[best_lambda_idx][0]

    ## combine out-of-sample predictions over folds
    preds = np.vstack([solve.y_to_matrix(i) for i in best_preds.squeeze()]).squeeze()
    truth = np.vstack(
        [solve.y_to_matrix(i) for i in kfold_results["y_true_test"].squeeze()]
    ).squeeze()

    # get latlons in same shuffled, cross-validated order
    ll = this_latlons[
        np.hstack([test for train, test in kfold_results["cv"].split(this_latlons)])
    ]

    data = {
        "truth": truth,
        "preds": preds,
        "lon": ll[:, 1],
        "lat": ll[:, 0],
        "best_lambda": best_lambda,
    }

    # Save validation set predictions
    print("Saving validation set results to {}".format(save_path_validation))
    with open(save_path_validation, "wb") as f:
        pickle.dump(data, f)
    results_dict[label + ";size" + str(subset_n)] = r2_score(truth, preds)

    # Get test set predictions
    st_test = time.time()
    holdout_results = solve.single_solve(
        this_X,
        this_X_test,
        this_Y,
        this_Y_test,
        lambdas=best_lambda,
        svd_solve=False,
        return_preds=True,
        return_model=False,
        clip_bounds=bounds,
    )

    #Get timing
    test_time = time.time() - st_test
    print("Test set training time:", test_time)

    #Save test set predictions
    ll = this_latlons_test
    data = {
        "truth": holdout_results["y_true_test"],
        "preds": holdout_results["y_pred_test"][0][0][0],
        "lon": ll[:, 1],
        "lat": ll[:, 0],
    }

    print("Saving test set results to {}".format(save_path_test))
    with open(save_path_test, "wb") as f:
        pickle.dump(data, f)

    print("R^2 score: ", holdout_results["metrics_test"][0][0][0]["r2_score"])
    ## Store the R2
    results_dict_test[label + ";size" + str(subset_n)] = holdout_results["metrics_test"][0][0][0]["r2_score"]
    print("Full reg time", time.time() - st_train)