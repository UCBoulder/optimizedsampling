import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'regressions'))
from data_loading import load_data_from_pkl, create_id_mapping, load_sample_file

def ridge_regression_metrics(X_train, y_train, X_test, y_test, n_folds=5, alphas=np.logspace(-5, 5, 10)):
    if X_train.shape[0] < 2 * n_folds:
        return None

    model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge_search = GridSearchCV(estimator=model, param_grid={'ridge__alpha': alphas}, scoring='r2', cv=cv, n_jobs=-1)
    ridge_search.fit(X_train, y_train)

    y_pred = ridge_search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return {
        'r2': ridge_search.score(X_test, y_test),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(y_test, y_pred)
    }

def compute_r2_statistics(sampling_dir, id_to_index, X_train_full, y_train_full, X_test, y_test,
                         ridge_regression_fn, min_samples=0):
    results = {}
    sample_files = sorted(f for f in os.listdir(sampling_dir) if f.endswith(".pkl"))

    for fname in tqdm(sample_files, desc="Processing samples"):
        sampled_ids = load_sample_file(os.path.join(sampling_dir, fname), verbose=False)
        if sampled_ids is None:
            continue

        sampled_indices = [id_to_index[i] for i in sampled_ids if i in id_to_index]
        if len(sampled_indices) < min_samples:
            continue

        metrics = ridge_regression_fn(X_train_full[sampled_indices], y_train_full[sampled_indices], X_test, y_test)
        if metrics is None:
            continue

        base_name = fname.replace(".pkl", "")
        results[base_name] = {'base_name': base_name, 'sample_size': len(sampled_indices), **metrics}

    return results

def save_results_to_json(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for base_name, stats in results.items():
        with open(os.path.join(output_dir, f"{base_name}.json"), 'w') as f:
            json.dump(stats, f, indent=2)

def seeded_sampling_r2_analysis(features_path, sampling_dir, results_dir, ridge_regression_fn, min_samples=0, **kwargs):
    data = load_data_from_pkl(features_path, **kwargs)
    if data is None:
        return

    full_ids, X_train_full, y_train_full, X_test, y_test = data
    id_to_index = create_id_mapping(full_ids)

    results = compute_r2_statistics(sampling_dir, id_to_index, X_train_full, y_train_full, X_test, y_test,
                                     ridge_regression_fn, min_samples)
    save_results_to_json(results, results_dir)
    return results

def main():
    parser = argparse.ArgumentParser(description="Run seeded sampling R2 analysis.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--cluster_sampling_dir", type=str, default=None)
    parser.add_argument("--convenience_sampling_urban_dir", type=str, default=None)
    parser.add_argument("--convenience_sampling_region_dir", type=str, default=None)
    parser.add_argument("--random_sampling_dir", type=str, default=None)
    parser.add_argument("--min_samples", type=int, default=0)
    args = parser.parse_args()

    common_args = {
        'features_path': args.features_path,
        'results_dir': args.results_dir,
        'ridge_regression_fn': ridge_regression_metrics,
        'min_samples': args.min_samples,
        'dataset': args.dataset,
    }
    if args.label:
        common_args['label'] = args.label

    sampling_dirs = [
        args.cluster_sampling_dir,
        args.convenience_sampling_urban_dir,
        args.convenience_sampling_region_dir,
        args.random_sampling_dir,
    ]
    for sampling_dir in sampling_dirs:
        if sampling_dir:
            seeded_sampling_r2_analysis(sampling_dir=sampling_dir, **common_args)

if __name__ == "__main__":
    main()
