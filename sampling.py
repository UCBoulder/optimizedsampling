""" 1. Decide train and tests splits of USAVars data (SRS, OED) with specified number of samples
    2. Parse data so it corresponds with features
    2. Train a ridge regression on train split
    3. Plot regression residuals for each variable of interest """

import os
import pickle
from pathlib import Path
import numpy as np
from os.path import basename, dirname, join

from sklearn.metrics import r2_score

from mosaiks.code.mosaiks.utils import *
from mosaiks.code.mosaiks.utils import io
from mosaiks.code.mosaiks import config as c
from mosaiks.code.mosaiks.solve import data_parser as parse
from mosaiks.code.mosaiks.solve import solve_functions as solve
from mosaiks.code.mosaiks.solve import interpret_results as ir

#Path for saving results
save_patt = join(
        "{save_dir}",
        "subset{subset_n}outcomes_{{reg_type}}_obsAndPred_{label}_{variable}_CONTUS_16_640_{sampling}_"
        f"{c.sampling['n_samples']}_{c.sampling['seed']}_random_features_{c.features['random']['patch_size']}_"
        f"{c.features['random']['seed']}{{subset}}.data",
    )

#Initialize X and latlons
results_dict = {}
results_dict_test = {}
X = {}
latlons = {}

#Set X (feature matrix) and corresponding lat lons
X["UAR"], latlons["UAR"] = io.get_X_latlon(c, "UAR")

#Set solver to ridge regression
solver = solve.ridge_regression

#Taking a random subset of training data
def random_subset(X_train, Y_train, latlon, size):
    random_subset = np.random.choice(X_train.shape[0], size=size, replace=False)
    return X_train[random_subset], Y_train[random_subset], latlon[random_subset]


#Run ridge regression for mosaiks features for all labels
def train_and_test(labels_to_run, subset_n):

    if labels_to_run == "all":
        labels_to_run = c.app_order

    ## get X, Y, latlon values of training data
    for label in labels_to_run:
        train_and_test(label, subset_n)


#Run ridge regression on mosaiks features for label
def train_and_test(label, subset_n=None):
    #Test all lambdas
    this_lambdas = io.get_lambdas(c, label, best_lambda_fpath=None)

    #Bounds from config file
    if c_app["logged"]:
        bounds = np.array([c_app["us_bounds_log_pred"]])
    else:
        bounds = np.array([c_app["us_bounds_pred"]])

    ## Get save path
    if (subset_n != slice(None)):
        subset_str = "_subset"
    else:
        subset_str = ""
    save_path_validation = this_save_patt.format(reg_type="scatter", subset=subset_str)
    save_path_test = this_save_patt.format(reg_type="testset", subset=subset_str)

    c = io.get_filepaths(c, label)
    c_app = getattr(c, label)
    sampling_type = c_app["sampling"]  # UAR or POP
    this_save_patt = save_patt.format(
        subset="",
        save_dir=c.fig_dir_prim,
        label=label,
        variable=c_app["variable"],
        sampling=c_app["sampling"],
        subset_n=subset_n
    )

    (
        this_X,
        this_X_test,
        this_Y,
        this_Y_test,
        this_latlons,
        this_latlons_test,
    ) = parse.merge_dropna_transform_split_train_test(
        c, label, X[sampling_type], latlons[sampling_type]
    )

    if subset_n is not None:
        this_X, this_Y, this_latlons = random_subset(this_X, this_latlons, subset_n)

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
    best_lambda = this_lambdas[best_lambda_idx]

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

    ## save validation set predictions
    print("Saving validation set results to {}".format(save_path_validation))
    with open(save_path_validation, "wb") as f:
        pickle.dump(data, f)
    results_dict = r2_score(truth, preds)

    ## Get test set predictions
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

    # get timing
    test_time = time.time() - st_test
    print("Test set training time:", test_time)

    ## Save test set predictions
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

    ## Store the R2
    results_dict_test[label] = holdout_results["metrics_test"][0][0][0]["r2_score"]
    print("Full reg time", time.time() - st_train)