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

avgr2 = {}
stdr2 = {}
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

    n_folds = 5
    seeds = [42, 123, 456, 789, 1011]
    r2_scores = []

    if budget != float('inf'):
        sampler = Sampler(ids_train, X_train, y_train, rule=rule, loc_emb=loc_emb_train, costs=costs)

        for seed in seeds:
            print(f"Using Seed {seed} to sample...")
            X_train_sampled, y_train_sampled = sampler.sample_with_budget(budget, seed)

            r2 = ridge_regression(X_train_sampled, y_train_sampled, X_test, y_test, n_folds=n_folds)
            if r2 is not None:
                r2_scores.append(r2)

                print(f"Seed {seed}: R2 score on test set: {r2}")
    else:
        r2 = ridge_regression(X_train, y_train, X_test, y_test, n_folds=n_folds)
        if r2 is not None:
            r2_scores.append(r2)

            print(f"R2 score on test set: {r2}")
    
    #Add to results
    if len(r2_scores) != 0:
        avg_r2 = np.nanmean(r2_scores)
        std_r2 = np.std(r2_scores)
        print(f"Average R2 score across seeds: {avg_r2}")

        avgr2[label + ";budget" + str(budget)] = avg_r2
        stdr2[label + ";budget" + str(budget)] = std_r2
    else:
        avgr2[label + ";budget" + str(budget)] = None
        stdr2[label + ";budget" + str(budget)] = None

'''
Run ridge regression and return R2 score
'''
def ridge_regression(X_train, y_train, X_test, y_test, n_folds=5, alphas=[1e-8, 1e-6, 1e-4, 1e-2, 1, 10, 100]):
    n_samples = X_train.shape[0]

    if n_samples < n_folds:
        print("Not enough samples for cross-validation.")
        return
     
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

    if abs(r2) > 1:
        print("Warning: Severe overfitting. Add more samples.")
    return r2
            