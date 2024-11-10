from pathlib import Path
import pandas as pd

from sampling import valid_set, train_and_test, results_dict, results_dict_test
from oed import *
from pca import *

from mosaiks.code.mosaiks import config as cfg
from mosaiks.code.mosaiks.utils import io

#Run
def run(labels_to_run, rule=None):
    for label in labels_to_run:
        #Set X (feature matrix) and corresponding lat lons
        #UAR is the only option when working with data from torchgeo
        X_df, latlons_df= io.get_X_latlon(cfg, "UAR")
        #Remove NaN
        valid_num, X_df, latlons_df = valid_set(cfg, label, X_df, latlons_df)

        #PCA
        # X_df = pca(X_df)

        #List of sizes for subsetting the dataset
        size_of_subset = [0.05, 0.1, 0.20, 0.35, 0.5, 0.75]
        size_of_subset = [int(np.floor(valid_num*percent)) for percent in size_of_subset]
        size_of_subset = [n - (n%5) for n in size_of_subset]

        #uncomment to run all sizes
        # for size in size_of_subset:
        #      train_and_test(cfg, label, X_df, latlons_df, size, rule=rule)
        train_and_test(cfg, label, X_df, latlons_df, subset_n=None, rule=None)

    #Save results (R2 score) in csv
    results_df = pd.DataFrame(
        {"Cross-validation R2": results_dict, "Test R2": results_dict_test}
    )
    results_df.index.name = "label"
    if rule is None:
        results_df.to_csv(Path("results/TestSetPerformancePCA.csv"), index=True)
    elif rule==v_optimal_design:
        results_df.to_csv(Path("results/TestSetPerformancePCAVOptimality.csv"), index=True)

#Labels from torchgeo dataset, UAR samples
labels_to_run = ["population", "treecover", "elevation"]
# rule=v_optimal_design
run(labels_to_run)