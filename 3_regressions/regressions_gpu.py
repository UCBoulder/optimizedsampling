from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import numpy as np
import cupy as cp
from cuml.linear_model import Ridge as cuRidge
from sklearn.base import BaseEstimator, RegressorMixin

class cuRidgeWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = None

    def fit(self, X, y):
        self.model = cuRidge(alpha=self.alpha)
        self.model.fit(cp.asarray(X), cp.asarray(y))
        return self

    def predict(self, X):
        return cp.asnumpy(self.model.predict(cp.asarray(X)))

def ridge_regression(X_train, 
                     y_train, 
                     X_test, 
                     y_test, 
                     n_folds=5, 
                     alphas=np.logspace(-5, 5, 10)):
    
    n_samples = X_train.shape[0]
    if n_samples < 2 * n_folds:
        print("not enough samples for cross-validation.")
        return

    print("Fitting ridge regression with GPU support...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {}
    for alpha in alphas:
        print(f"Testing alpha = {alpha:.1e}...")
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            pipeline = Pipeline([
                ('scaler', StandardScaler()), #in sklearn pipeline with cv
                ('ridge', cuRidgeWrapper(alpha=alpha))
            ])

            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            scores.append(r2_score(y_val, y_pred))

        results[alpha] = {
            "mean_r2": np.mean(scores),
            "std_r2": np.std(scores),
        }
        print(f"Alpha {alpha:.1e}: Mean R² = {results[alpha]['mean_r2']:.4f} ± {results[alpha]['std_r2']:.4f}")

    best_alpha = max(results, key=lambda a: results[a]['mean_r2'])
    print(f"\nBest alpha: {best_alpha:.1e} with Mean R² = {results[best_alpha]['mean_r2']:.4f}")

    pipeline.named_steps['ridge'].alpha = best_alpha
    pipeline.fit(X_train, y_train)

    y_test_pred = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"\nFinal R² on test set: {test_r2:.4f}")

    if abs(test_r2) > 1:
        print("Warning: Severe overfitting or data leakage.")

    return {
        "best_alpha": best_alpha,
        "cv_results": results,
        "test_r2": test_r2
    }



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

                        except Exception as e:
                            continue

    # Write to CSV
    df = pd.DataFrame(results)
    df.to_csv("ridge_regression_results.csv", index=False)