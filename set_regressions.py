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

def generate_latex_table(run_regression):
    import dill
    from collections import defaultdict

    # Store rows organized by label and type_str
    grouped = defaultdict(lambda: defaultdict(list))

    for label in ["population", "treecover"]:
        for type_str in ["clustered", "density"]:
            for num_counties in [25, 50, 75, 100, 125, 150, 175, 200]:
                id_path = f"county_sampling/{label}/{type_str}/IDs_{num_counties}_counties_10_radius_seed_42.pkl"
                try:
                    with open(id_path, "rb") as f:
                        arrs = dill.load(f)
                    ids = arrs
                    r2 = run_regression(label, ids)
                    num_samples = len(ids)
                    grouped[label][type_str].append((num_counties, num_samples, r2))
                except FileNotFoundError:
                    print(f"File not found: {id_path}")
                except Exception as e:
                    print(f"Error processing {id_path}: {e}")

    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\caption{Regression results by Label and Sampling Type}")
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Label & Type & \# Counties & \# Samples & Test $R^2$ \\\\")
    lines.append("\\midrule")

    for label in grouped:
        label_total_rows = sum(len(v) for v in grouped[label].values())
        label_printed = False
        for type_str in grouped[label]:
            type_rows = grouped[label][type_str]
            type_total_rows = len(type_rows)
            type_printed = False
            for i, (num_counties, num_samples, r2) in enumerate(type_rows):
                row = []
                if not label_printed:
                    row.append(f"\\multirow{{{label_total_rows}}}{{*}}{{{label}}}")
                    label_printed = True
                else:
                    row.append("")  # skip label

                if not type_printed:
                    row.append(f"\\multirow{{{type_total_rows}}}{{*}}{{{type_str}}}")
                    type_printed = True
                else:
                    row.append("")  # skip type

                row.extend([str(num_counties), str(num_samples), f"{r2:.3f} \\\\"])
                lines.append(" & ".join(row))

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    print("\n".join(lines))



if __name__ == "__main__":
    # label = "treecover"
    # type_str = "clustered"

    # for label in ["population", "treecover"]:
    #     for type_str in ["clustered", "density"]:
    #         for num_counties in [100, 125, 150, 175, 200]:

    #             print(f"LABEL: {label}")
    #             print(f"TYPE: {type_str}")
    #             print(f"NUM COUNTIES: {num_counties}")

    #             id_path = f"county_sampling/{label}/{type_str}/IDs_{num_counties}_counties_10_radius_seed_42.pkl"
    #             print(f"ID_PATH: {id_path}")

    #             with open(id_path, "rb") as f:
    #                 arrs = dill.load(f)
    #             ids = arrs
    #             r2 = run_regression(label, ids)
    generate_latex_table(run_regression)
            