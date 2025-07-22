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
    print(r2)

    if abs(r2) > 1:
        print("Warning: Severe overfitting. Add more samples.")
    return r2


if __name__ == "__main__":
    import dill

    with open("/home/libe2152/optimizedsampling/0_data/features/india_secc/India_SECC_with_splits_4000.pkl", "rb") as f:
        arrs = dill.load(f)
    full_ids = arrs['ids_train']
    X_train_full = arrs['X_train']
    y_train_full = arrs['y_train']
    X_test = arrs['X_test']
    y_test = arrs['y_test']

    id_to_index = {str(id_): i for i, id_ in enumerate(full_ids)}

    file_path = "/home/libe2152/optimizedsampling/0_data/initial_samples/india_secc/cluster_sampling/randomstrata/sample_5_state_district_desired_20ppc_3000_size_seed_1.pkl"

    with open(file_path, "rb") as f:
        sampled_ids = dill.load(f)['sampled_ids']
    sampled_ids = [str(x) for x in sampled_ids]
    sampled_indices = [id_to_index[i] for i in sampled_ids if i in id_to_index]

    X_subset = X_train_full[sampled_indices]
    y_subset = y_train_full[sampled_indices]

    r2 = ridge_regression(X_subset, y_subset, X_test, y_test)
    from IPython import embed; embed()
