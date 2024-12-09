""" 1. Write function to resave features with splits (train, val, test)
    2. Use Scikit Learn Ridge over range of lambda
    2. Make Sampler function
    3. Make PCA function
    3. Plot regression residuals for each variable of interest """

import dill
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from oed import *
from sampling import *
from format_data import *
from feasibility import *
from sampler import Sampler
from plot_coverage import plot_lat_lon

results = {}
budget = {}

'''
Run regressions
Parameters:
    label: "population", "treecover", or "elevation
    rule: None (implies no subset is taken), "random", "image", or "satclip"
    subset_size: None (no subset is taken), int
'''
def run_regression(label, cost_func, *params, budget=float('inf'), rule='random'):
    print("*** Running regressions for: {label} with ${budget} budget using {rule} rule".format(label=label, budget=budget, rule=rule))

    if cost_func == cost_lin:
        cost_str = "linear wrt distance with alpha={alpha}, beta={beta}".format(alpha=params[0], beta=params[1])
    elif cost_func == cost_lin_with_r:
        cost_str = "linear outside or radius {r} km with alpha={alpha}, beta = {beta}, c={c}".format(r=params[3], alpha=params[0], beta=params[1], c=params[2])

    print("Cost function is {cost_str}".format(cost_str=cost_str))

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
    
    dist_path = "data/cost/distance_to_closest_city.pkl"
    costs = cost_func(dist_path, ids_train, *params)

    if budget != float('inf'):
        sampler = Sampler(ids_train, X_train, y_train, latlon_train, rule=rule, loc_emb=loc_emb_train, costs=costs)
        X_train, y_train, latlon_train = sampler.sample_with_budget(budget)

    #Plot coverage
    # print("Plotting coverage ...")
    # fig = plot_lat_lon(latlon_train[:,0], latlon_train[:,1], title="Coverage for {rule} with {num} samples".format(rule=rule, num=subset_size), color="orange", alpha=1)
    # fig.savefig("plots/Coverage for {rule} with {num} samples.png".format(rule=rule, num=subset_size))

    n_samples = X_train.shape[0]
    n_folds = 5

    #Range of alphas for ridge regression
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1, 10, 100]
    
    if n_samples >= n_folds:
        # Perform Ridge regression with cross-validation
        reg = RidgeCV(alphas=alphas, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=42))  # 5-fold cross-validation
        print("Fitting regression...")
        reg.fit(X_train, y_train)

        # Optimal alpha
        best_alpha = reg.alpha_
        print(f"Best alpha: {best_alpha}")
    else:
        #Maybe change this alpha
        reg = Ridge(alpha=10)
        print("Fitting regression...")
        reg.fit(X_train, y_train)
    
    # Make predictions on the test set
    yhat_test = reg.predict(X_test)

    # Calculate R2 score
    r2 = r2_score(y_test, yhat_test)
    print(f"R2 score on test set: {r2}")
    results[label + ";budget" + str(budget)] = r2