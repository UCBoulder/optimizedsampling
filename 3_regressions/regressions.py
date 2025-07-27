""" 1. Write function to resave features with splits (train, val, test)
    2. Use Scikit Learn Ridge over range of lambda
    2. Make Sampler function
    3. Make PCA function
    3. Plot regression residuals for each variable of interest """

import dill
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
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

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    param_grid = {
        'ridge__alpha': alphas
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    ridge_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',        
        cv=cv,
        n_jobs=-1                   #parallelize across folds
    )


    def evaluate_r2(model, X_test, y_test):
        return model.score(X_test, y_test)
    
    ridge_search.fit(X_train, y_train)
    r2 = evaluate_r2(ridge_search, X_test, y_test)

    if abs(r2) > 1:
        print("Warning: Severe overfitting. Add more samples.")
    return r2


if __name__ == "__main__":
    import dill
    import pandas as pd

    results = []

    for label in ['ph_h2o', 'k', 'p', 'mo']:
        for typestr in ['P20', 'P40', 'P10', 'P30']:
            for year in [2017, 2018, 2019, 2020, 2021, 2022]:
                for month1 in ['Jan', 'Jul', 'jan']:
                    for month2 in ['Jun', 'Dec', 'jun']:
                        feature_identifier = f"{year}_{month1}_{month2}_{typestr}"
                        
                        try:

                            with open(f"/home/libe2152/optimizedsampling/0_data/features/togo/togo_fertility_data_all_{feature_identifier}.pkl", "rb") as f:
                                arrs = dill.load(f)
                            full_ids = arrs['ids_train']
                            X_train_full = arrs['X_train']
                            y_train_full = arrs[f'{label}_train']
                            X_test = arrs['X_test']
                            y_test = arrs[f'{label}_test']

                            train_mask = ~np.isnan(y_train_full)
                            full_ids = full_ids[train_mask]
                            X_train_full = X_train_full[train_mask]
                            y_train_full = y_train_full[train_mask]

                            test_mask = ~np.isnan(y_test)
                            X_test = X_test[test_mask]
                            y_test = y_test[test_mask]

                            r2 = ridge_regression(X_train_full, y_train_full, X_test, y_test)
                            results.append({'label': label, 'features': feature_identifier, 'r2': r2})
                            print(f"R2 for label {label}, feature identifier {feature_identifier}: {r2}")

                        except Exception as e:
                            continue

    # Write to CSV
    df = pd.DataFrame(results)
    df.to_csv("ridge_regression_results.csv", index=False)
