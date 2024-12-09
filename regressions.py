""" 1. Write function to resave features with splits (train, val, test)
    2. Use Scikit Learn Ridge over range of lambda
    2. Make Sampler function
    3. Make PCA function
    3. Plot regression residuals for each variable of interest """

import dill
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from oed import *
from sampling import *
from format_data import *
from feasibility import *
from plot_coverage import plot_lat_lon

results = {}
costs = {}

'''
Run regressions
Parameters:
    label: "population", "treecover", or "elevation
    rule: None (implies no subset is taken), "random", "image", or "satclip"
    subset_size: None (no subset is taken), int
'''
def run_regression(label, rule=None, subset_size=None):
    print("*** Running regressions for: {label} with {num} samples using {rule} rule".format(label=label, num=subset_size, rule=rule))

    (
        X_train,
        X_test,
        y_train,
        y_test,
        latlon_train,
        latlon_test,
        loc_emb_train,
        loc_emb_test,
        ids_train,
        ids_test
    ) = retrieve_splits(label)
    
    cost_path = "data/cost/costs_by_city_dist.pkl"
    cost_train = costs_of_train_data(cost_path, ids_train)

    dist_path = "data/cost/distance_to_closest_city.pkl"
    dist_train = dists_of_train_data(dist_path, ids_train)
    r = 500

    #Take subset according to rule
    if rule=="random":
        X_train, y_train, latlon_train, ids_train, total_cost = random_subset_and_cost(X_train, y_train, latlon_train, ids_train, cost_train, subset_size)
        costs[label + ";size" + str(subset_size)] = total_cost 

        #Record latlons used
        record_latlons_ids(label, latlon_train, ids_train, "random", subset_size)
    if rule=="image":
        X_train, y_train, latlon_train, ids_train, total_cost = image_subset(X_train, y_train, latlon_train, ids_train, cost_train, v_optimal_design, subset_size)
        costs[label + ";size" + str(subset_size)] = total_cost 

        #Record latlons used
        record_latlons_ids(label, latlon_train, ids_train, "image", subset_size)
    if rule=="satclip":
        X_train, y_train, latlon_train, ids_train, total_cost = satclip_subset(X_train, y_train, latlon_train, loc_emb_train, ids_train, cost_train, v_optimal_design, subset_size)
        costs[label + ";size" + str(subset_size)] = total_cost 

        #Record latlons used
        record_latlons_ids(label, latlon_train, ids_train, "satclip", subset_size)
    if rule=="lowcost":
        X_train, y_train, latlon_train, ids_train, total_cost = sampling_by_lin(X_train, y_train, latlon_train, ids_train, cost_train, subset_size)
        costs[label + ";size" + str(subset_size)] = total_cost

        #Record latlons used
        record_latlons_ids(label, latlon_train, ids_train, "lowcost", subset_size)
    if rule=="dist":
        X_train, y_train, latlon_train, ids_train = sample_by_lin_rad(X_train, y_train, latlon_train, ids_train, dist_train, r, subset_size)
        #Record latlons used
        record_latlons_ids(label, latlon_train, ids_train, "dist", subset_size)
    if rule=="rad":
        X_train, y_train, latlon_train, ids_train = sample_by_bin_rad(X_train, y_train, latlon_train, ids_train, dist_train, r, subset_size)
        #Record latlons used
        record_latlons_ids(label, latlon_train, ids_train, "rad", subset_size)
    
    if rule is None:
        costs[label + ";size" + str(subset_size)] = cost_train

    #Plot coverage
    # print("Plotting coverage ...")
    # fig = plot_lat_lon(latlon_train[:,0], latlon_train[:,1], title="Coverage for {rule} with {num} samples".format(rule=rule, num=subset_size), color="orange", alpha=1)
    # fig.savefig("plots/Coverage for {rule} with {num} samples.png".format(rule=rule, num=subset_size))

    #Range of alphas for ridge regression
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1, 10, 100]

    # Perform Ridge regression with cross-validation
    reg = RidgeCV(alphas=alphas, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=42))  # 5-fold cross-validation
    print("Fitting regression...")
    reg.fit(X_train, y_train)

    # Optimal alpha
    best_alpha = reg.alpha_
    print(f"Best alpha: {best_alpha}")

    # Make predictions on the test set
    yhat_test = reg.predict(X_test)

    # Calculate R2 score
    r2 = r2_score(y_test, yhat_test)
    print(f"R2 score on test set: {r2}")
    results[label + ";size" + str(subset_size)] = r2