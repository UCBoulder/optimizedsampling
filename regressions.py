""" 1. Write function to resave features with splits (train, val, test)
    2. Use Scikit Learn Ridge over range of lambda
    2. Make Sampler function
    3. Make PCA function
    3. Plot regression residuals for each variable of interest """

import dill
import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from oed import *
import opt
from format_data import *
from cost import *
from sampler import Sampler
from plot_coverage import plot_lat_lon
import config as c

r2_dict = {}
avgr2 = {}
stdr2 = {}
num_samples_dict = {}

'''
Run regressions
Parameters:
    label: "population", "treecover", or "elevation
    rule: None (implies no subset is taken), "random", "image", or "satclip"
    subset_size: None (no subset is taken), int
'''
def run_regression(label, 
                   cost_func, 
                   rule='random', 
                   budget=float('inf'),
                   avg_results=False, 
                   **kwargs
                   ):
    print("*** Running regressions for: {label} with ${budget} budget using {rule} rule".format(label=label, budget=budget, rule=rule))

    (
        X_train,
        X_test,
        y_train,
        y_test,
        latlon_train,
        latlon_test,
        ids_train,
        ids_test
    ) = retrieve_splits(label)
    
    if cost_func == compute_state_cost:
        states = kwargs.get('states', 1)
        costs = cost_func(states, latlon_train)
    elif cost_func == compute_unif_cost:
        costs = cost_func(ids_train)
    elif cost_func == compute_cluster_cost:
        costs = cost_func(ids_train, 
                          cluster_type=kwargs.get('cluster_type', 'NLCD_percentages'))
    else:
        dist_path = "data/cost/distance_to_closest_city.pkl"
        costs = cost_func(dist_path, ids_train, **kwargs)

    n_folds = 5
    seeds = [42, 123, 456, 789, 1011]
    r2_scores = []
    sample_costs = []
    num_samples_arr = []

    sampler = Sampler(ids_train, 
                        X_train, 
                        y_train, 
                        latlon_train,
                        rule=rule,
                        costs=costs,
                        cluster_type=kwargs.get('cluster_type', 'NLCD_percentages')
                    )
    
    if rule == 'invsize':
        probs = sampler.compute_probs(budget, kwargs.get('l', 0.5))

        #Ensure probs are not slightly more or less than 1 or 0
        probs = np.clip(probs, 0, 1)

    for seed in seeds:
        print(f"Using Seed {seed} to sample...")

        if rule == 'invsize':
            X_train_sampled, y_train_sampled, latlon_train_sampled, sample_cost = sampler.sample_with_prob(probs, seed)
        else:
            X_train_sampled, y_train_sampled, latlon_train_sampled, sample_cost = sampler.sample_with_budget(budget, seed)

        num_samples = X_train_sampled.shape[0]
        print("Number of samples: ", num_samples)
        if num_samples == sampler.total_valid:
            print("Used all samples.")
            c.used_all_samples = True
        num_samples_arr.append(num_samples)

        #Plot Coverage
        # fig = plot_lat_lon(latlon_train_sampled[:,0], latlon_train_sampled[:,1], title=f"Coverage with Budget {budget}", color="orange", alpha=1)
        # fig.savefig(f"plots/c Coverage with Budget {budget}.png")

        r2 = ridge_regression(X_train_sampled, 
                                y_train_sampled, 
                                X_test, 
                                y_test, 
                                n_folds=n_folds)

        if avg_results:
            if r2 is not None:
                r2_scores.append(r2)
        else:
            if r2 is not None:
                key = label + ";cost" + str(sample_cost)
                if key not in r2_dict:
                    r2_dict[key] = []
                r2_dict[key].append(r2)

        # print(f"Seed {seed}: R2 score on test set: {r2}")

    if avg_results:
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
    avg_num_samples = np.nanmean(num_samples_arr)
    num_samples_dict[label + ";budget" + str(budget)] = avg_num_samples

'''
Run ridge regression and return R2 score
'''
def ridge_regression(X_train, 
                     y_train, 
                     X_test, 
                     y_test, 
                     n_folds=5, 
                     alphas=np.logspace(-5, 5, 10)):
    
    n_samples = X_train.shape[0]

    if n_samples < 2*n_folds:
        print("Not enough samples for cross-validation.")
        return
     
    print("Fitting regression...")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    #Pipeline that scales and then fits ridge regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),     # Step 1: Standardize features
        ('ridgecv', RidgeCV(alphas=alphas, scoring='r2', cv=kf))  # Step 2: RidgeCV with 5-fold CV
    ])

    #Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Optimal alpha
    best_alpha = pipeline.named_steps['ridgecv'].alpha_
    print(f"Best alpha: {best_alpha}")
            
    # Make predictions on the test set
    r2 = pipeline.score(X_test, y_test)

    if abs(r2) > 1:
        print("Warning: Severe overfitting. Add more samples.")
    return r2
            