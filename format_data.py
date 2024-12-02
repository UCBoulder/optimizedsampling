import dill
import pandas as pd
import numpy as np

from mosaiks.code.mosaiks.solve import data_parser as parse
from mosaiks.code.mosaiks import config as cfg

def save_with_splits(c, label, feature_path, loc_emb_path=None):
    with open(feature_path, "rb") as f:
        arrs = dill.load(f)

    # get features
    X = pd.DataFrame(
        arrs["X"].astype(np.float64),
        index=arrs["ids_X"],
        columns=["X_" + str(i) for i in range(arrs["X"].shape[1])],
    )

    # get latlons
    latlons = pd.DataFrame(arrs["latlon"], index=arrs["ids_X"], columns=["lat", "lon"])

    # sort both
    latlons = latlons.sort_values(["lat", "lon"], ascending=[False, True])
    X = X.reindex(latlons.index)

    if loc_emb_path is not None:
        with open(loc_emb_path, "rb") as f:
            arrs = dill.load(f)

        # get embeddings
        loc_emb = pd.DataFrame(arrs["emb"], index=arrs["ids"], columns=["L_" + str(i) for i in range(arrs["emb"].shape[1])])

        # reindex embeddings according to X
        loc_emb = loc_emb.reindex(X.index)
    else:
        loc_emb=None

    (
        X_train,
        X_test,
        y_train,
        y_test,
        latlons_train,
        latlons_test,
        loc_emb_train,
        loc_emb_test,
        ids_train,
        ids_test
    ) = parse.merge_dropna_transform_split_train_test(
        c, label, X, latlons, loc_emb
    )

    out_path = "data/int/feature_matrices/CONTUS_UAR_{label}_with_splits.pkl".format(label=label)

    if loc_emb is not None:
        with open(out_path, "wb") as f:
            dill.dump(
                {"X_train": X_train, 
                "latlons_train": latlons_train,
                "y_train": y_train,
                "loc_emb_train": loc_emb_train,
                "X_test": X_test,
                "latlons_test": latlons_test,
                "y_test": y_test,
                "loc_emb_test": loc_emb_test,
                "ids_train": ids_train,
                "ids_test": ids_test},
                f,
                protocol=4,
            )
    else:
        with open(out_path, "wb") as f:
            dill.dump(
                {"X_train": X_train, 
                "latlons_train": latlons_train,
                "y_train": y_train,
                "X_test": X_test,
                "latlons_test": latlons_test,
                "y_test": y_test},
                f,
                protocol=4,
            )

def retrieve_splits(label):
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
    ids_train = arrs["ids_train"]
    ids_test = arrs["ids_test"]

    return X_train, X_test, y_train, y_test, latlons_train, latlons_test, loc_emb_train, loc_emb_test, ids_train, ids_test

'''
    Returns numpy array of cost with same order as training data
'''
def costs_of_train_data(cost_path, ids_train):
    with open(cost_path, "rb") as f:
        arrs = dill.load(f)

    # get costs
    costs = pd.DataFrame(
        arrs["cost"].astype(np.float64),
        index=arrs["ids"],
        columns=["cost"],
    )
    
    cost_train = costs.loc[ids_train].to_numpy()[:,0]

    return cost_train
