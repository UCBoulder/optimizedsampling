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
from plot_coverage import plot_lat_lon
from pca import pca
from sampling import *

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

#Run ridge regression on mosaiks features for label
def train_and_test(c, label, X, latlons, subset_n=None, rule=None, loc_emb=None):

    if rule==v_optimal_design:
        rule_str = 'V Optimal Design'
    else: 
        rule_str = ''

    if loc_emb is not None:
        satclip_str = "satclip embeddings"
    else:
        satclip_str = "no satclip embeddings"
    print("*** Running regressions for: {label} with {num} samples using {rule} with {satclip_str}".format(label=label, num=subset_n, rule=rule_str, satclip_str=satclip_str))

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
        this_emb,
        this_emb_test
    ) = parse.merge_dropna_transform_split_train_test(
        c, label, X, latlons, loc_emb
    )
    
    # Perform PCA
    # this_X, this_X_test = pca(this_X, this_X_test)

    # Take a random subset of size n
    if subset_n is not None:
            if rule is None:
                this_X, this_Y, this_latlons = random_subset(this_X, this_Y, this_latlons, subset_n)
            elif loc_emb is None:
                this_X, this_Y, this_latlons = image_subset(this_X, this_Y, this_latlons, rule, subset_n)
            else:
                this_X, this_Y, this_latlons = satclip_subset(this_X, this_Y, this_latlons, this_emb, rule, subset_n)
    else:
        while (this_X.shape[0]%5 != 0):
            this_X = this_X[:-1]
            this_latlons = this_latlons[:-1]
            this_Y = this_Y[:-1]


    #Plot coverage
    # print("plotting coverage ...")
    # fig = plot_lat_lon(this_latlons[:,0], this_latlons[:,1], title="Coverage for {satclip_str} with {num} samples".format(satclip_str=satclip_str, num=subset_n), color="green", alpha=1)
    # fig.savefig("plots/Coverage for {satclip_str} chosen with {rule} with {num} samples.png".format(satclip_str=satclip_str, num=subset_n, rule=rule))

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