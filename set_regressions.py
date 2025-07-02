""" 1. Write function to resave features with splits (train, val, test)
    2. Use Scikit Learn Ridge over range of lambda
    2. Make Sampler function
    3. Make PCA function
    3. Plot regression residuals for each variable of interest """

import dill
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from format_data import retrieve_splits, subset_train_with_ids
from plot_coverage import plot_lat_lon
import config as c


'''
Run regressions
Parameters:
    label: "population", "treecover", or "elevation
    rule: None (implies no subset is taken), "random", "image", or "satclip"
    subset_size: None (no subset is taken), int
'''
def run_regression(label, 
                   ids
                   ):

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

    X_train_sampled, y_train_sampled, latlon_train_sampled, ids_train_sampled = subset_train_with_ids(ids, X_train, y_train, latlon_train, ids_train)

    n_folds = 5


    num_samples = X_train_sampled.shape[0]

    r2 = ridge_regression(X_train_sampled, 
                            y_train_sampled, 
                            X_test, 
                            y_test, 
                            n_folds=n_folds)

    print(f"R2 score on test set: {r2}")
    return r2

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
    print(f"NUM SAMPLES: {X_train.shape[0]}")
    pipeline.fit(X_train, y_train)

    # Optimal alpha
    best_alpha = pipeline.named_steps['ridgecv'].alpha_
    print(f"Best alpha: {best_alpha}")

    # Make predictions on the test set
    r2 = pipeline.score(X_test, y_test)

    if abs(r2) > 1:
        print("Warning: Severe overfitting. Add more samples.")
    return r2



if __name__ == "__main__":
    # import os
    # import dill

    # label = "income"
    # folder = "/home/libe2152/optimizedsampling/initial_sample/income/cluster_sampling"

    # print(f"LABEL: {label}")
    # print(f"FOLDER: {folder}")

    # for fname in sorted(os.listdir(folder)):
    #     if fname.endswith(".pkl"):
    #         id_path = os.path.join(folder, fname)

    #         try:
    #             with open(id_path, "rb") as f:
    #                 arrs = dill.load(f)
    #             ids = arrs
    #             if len(ids) > 1000:
    #                 r2 = run_regression(label, ids)

    #                 print(f"\nFILE: {fname}")
    #                 print(f"  R^2: {r2:.4f}")
    #                 print(f"  NUM SAMPLES: {len(ids)}")
    #         except Exception as e:
    #             print(f"\nFAILED TO PROCESS {fname}: {e}")

    label = "income"
    id_path = "/home/libe2152/optimizedsampling/data/int/feature_matrices/CONTUS_UAR_income_with_splits_torchgeo4096.pkl"


    try:
        with open(id_path, "rb") as f:
            arrs = dill.load(f)
        invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
        ids_train = arrs['ids_train']
        valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
        ids = ids_train[valid_idxs]
        if len(ids) > 1000:
            r2 = run_regression(label, ids)

            print(f"  R^2: {r2:.4f}")
            print(f"  NUM SAMPLES: {len(ids)}")
    except Exception as e:
        print(f"\nFAILED TO PROCESS : {e}")
            