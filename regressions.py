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
    import os
    import dill
    import pandas as pd
    from tqdm import tqdm

    base_results_dir = "/home/libe2152/optimizedsampling/0_results"
    os.makedirs(base_results_dir, exist_ok=True)

    for label in ['population', 'treecover']:
        print(f"\n[Label: {label}] Starting processing...")
        results = []

        id_path = f"/home/libe2152/optimizedsampling/0_data/features/usavars/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl"
        cluster_dir = f"/home/libe2152/optimizedsampling/0_data/initial_samples/usavars/{label}/cluster_sampling"

        try:
            with open(id_path, "rb") as f:
                arrs = dill.load(f)
            full_ids = arrs['ids_train']
            X_train_full = arrs['X_train']
            y_train_full = arrs['y_train']
        except Exception as e:
            print(f"[ERROR] Failed to load feature file for {label}: {e}")
            continue

        id_to_index = {str(id_): i for i, id_ in enumerate(full_ids)}

        for fname in tqdm(sorted(os.listdir(cluster_dir)), desc=f"Processing {label}"):
            parts = fname.replace(".pkl", "").split("_")

            seed = parts[-1]
            if not fname.endswith(".pkl"):
                continue

            full_path = os.path.join(cluster_dir, fname)

            try:
                with open(full_path, "rb") as f:
                    sampled_ids = dill.load(f)
                sampled_ids = [str(x) for x in sampled_ids]
            except Exception as e:
                print(f"[WARNING] Failed to load {fname}: {e}")
                continue

            try:
                sampled_indices = [id_to_index[i] for i in sampled_ids if i in id_to_index]
            except Exception as e:
                print(f"[WARNING] Some IDs missing from training set for {fname}: {e}")
                continue

            if len(sampled_indices) < 10:
                print(f"[SKIP] Too few samples in {fname}: {len(sampled_indices)}")
                continue

            X_subset = X_train_full[sampled_indices]
            y_subset = y_train_full[sampled_indices]

            try:
                print(f'Running Ridge Regression for seed: {seed}')
                r2 = ridge_regression(X_subset, y_subset, arrs['X_test'], arrs['y_test'])
            except Exception as e:
                print(f"[ERROR] Failed regression on {fname}: {e}")
                continue

            metadata = {
                "filename": fname,
                "seed": seed,
                "sample_size": parts[-3],
                "points_per_cluster": parts[-6],
                "strata": parts[2],
                "cluster": parts[4],
                "r2": r2
            }
            results.append(metadata)

        # Save per-label CSV
        label_csv_path = os.path.join(base_results_dir, f"cluster_sampling_r2_scores_{label}.csv")
        df = pd.DataFrame(results)
        df.to_csv(label_csv_path, index=False)
        print(f"✅ Saved {len(results)} results to {label_csv_path}")

            