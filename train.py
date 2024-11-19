from pathlib import Path
import pandas as pd
import dill

from regressions import valid_num, train_and_test, results_dict, results_dict_test
from oed import *
from pca import *
from satclip import get_satclip

from mosaiks.code.mosaiks import config as cfg
from mosaiks.code.mosaiks.utils import io

#Run
def run(labels_to_run, X_df, latlons_df, rule=None, loc_emb=None):
    for label in labels_to_run:
        #Set X (feature matrix) and corresponding lat lons
        #UAR is the only option when working with data from torchgeo

        #Remove NaN
        valid = valid_num(cfg, label, X_df, latlons_df)

        #PCA
        # X_df = pca(X_df)

        #List of sizes for subsetting the dataset
        size_of_subset = [0.005, 0.01, 0.05, 0.1, 0.20, 0.35, 0.5, 0.75]
        size_of_subset = [int(np.floor(valid*0.80*percent)) for percent in size_of_subset]
        size_of_subset = [n - (n%5) for n in size_of_subset]

        for size in size_of_subset:
            train_and_test(cfg, label, X_df, latlons_df, size, rule=rule, loc_emb=loc_emb)
        train_and_test(cfg, label, X_df, latlons_df, subset_n=None, rule=None, loc_emb=None)

    #Save results (R2 score) in csv
    results_df = pd.DataFrame(
        {"Cross-validation R2": results_dict, "Test R2": results_dict_test}
    )
    results_df.index.name = "label"
    if rule is None:
        rule_str = ''
        loc_str = ''
    elif rule==v_optimal_design:
        rule_str = 'VOptimality'

        if loc_emb is not None:
            loc_str = 'SatCLIP'
        else:
            loc_str = ''

    results_df.to_csv(Path("results/TestSetPerformance{rule_str}{loc_str}.csv".format(rule_str=rule_str, loc_str=loc_str)), index=True)

#MOSAIKS features
X_df, latlons_df= io.get_X_latlon(cfg, "UAR")

#TorchGeo RCF Features
# with open("data/int/feature_matrices/CONTUS_UAR_torchgeo.pkl", "rb") as f:
#         arrs = dill.load(f)
# X_df = pd.DataFrame(
#     arrs["X"].astype(np.float64),
#     index=arrs["ids_X"],
#     columns=["X_" + str(i) for i in range(arrs["X"].shape[1])],
# )

# # get latlons
# latlons_df = pd.DataFrame(arrs["latlon"], index=arrs["ids_X"], columns=["lat", "lon"])

# # sort both
# latlons_df = latlons_df.sort_values(["lat", "lon"], ascending=[False, True])
# X_df = X_df.reindex(latlons_df.index)

satclip_df = get_satclip(cfg, X_df)

labels_to_run = ["population", "treecover", "elevation"]
rule=v_optimal_design
loc_emb=satclip_df

#Run Random
run(labels_to_run, X_df, latlons_df, rule=None, loc_emb=None)

#Run with V Optimal Design
# run(labels_to_run, X_df, latlons_df, rule=rule, loc_emb=None)

#Run with SatCLIP embeddings
# run(labels_to_run, X_df, latlons_df, rule=rule, loc_emb=loc_emb)