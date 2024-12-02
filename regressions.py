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

from sampling import *
from mosaiks.code.mosaiks.solve import data_parser as parse

results = {}

'''
Run regressions
Parameters:
    data_path: path to access data and splits
    rule: None (implies no subset is taken), "random", "image", or "satclip"
    subset_size: None (no subset is taken), int
'''
def run_regression(label, rule=None, subset_size=None):
    print("*** Running regressions for: {label} with {num} samples using {rule} rule".format(label=label, num=subset_size, rule=rule))

    data_path = "data/int/feature_matrices/CONTUS_UAR_{label}_with_splits.pkl".format(label=label)
    with open(data_path, "rb") as f:
        arrs = dill.load(f)

    X_train = arrs["X_train"]
    X_test = arrs["X_test"]
    y_train = arrs["y_train"]
    y_test = arrs["y_test"]
    latlons_train = arrs["latlons_train"]
    latlons_test = arrs["latlons_test"]
    loc_emb_train = arrs["loc_emb_train"]
    loc_emb_test = arrs["loc_emb_test"]

    #Take subset according to rule
    if rule=="random":
        X_train, y_train, latlons_train = random_subset(X_train, y_train, latlons_train, subset_size)
    if rule=="image":
         X_train, y_train, latlons_train = image_subset(X_train, y_train, latlons_train, subset_size)
    if rule=="satclip":
         X_train, y_train, latlons_train = satclip_subset(X_train, y_train, latlons_train, subset_size)

    #Range of alphas for ridge regression
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1, 10, 100]

    # Perform Ridge regression with cross-validation
    reg = RidgeCV(alphas=alphas, scoring='r2', cv=5, normalize=True)  # 5-fold cross-validation
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