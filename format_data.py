import dill
import pandas as pd
import numpy as np

from mosaiks.code.mosaiks.solve import data_parser as parse
from mosaiks.code.mosaiks import config as cfg

#IDs contain NaN in the metadata (transform, latlon,...) or latlon is outside US
invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])

def save_with_splits(c, label, out_fpath, feature_path, loc_emb_path=None):
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

    out_fpath = out_fpath.format(label=label)

    if loc_emb is not None:
        with open(out_fpath, "wb") as f:
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
        with open(out_fpath, "wb") as f:
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
    print("Retrieving TorchGeo Feature splits...")

    data_path = "data/int/feature_matrices/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl".format(label=label)
    with open(data_path, "rb") as f:
        arrs = dill.load(f)

    X_train = arrs["X_train"]
    X_test = arrs["X_test"]
    y_train = arrs["y_train"]
    y_test = arrs["y_test"]
    latlons_train = arrs["latlons_train"]
    latlons_test = arrs["latlons_test"]
    #loc_emb_train = arrs["loc_emb_train"]
    #loc_emb_test = arrs["loc_emb_test"]
    ids_train = arrs["ids_train"]
    ids_test = arrs["ids_test"]

    valid_train_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
    ids_train = ids_train[valid_train_idxs]
    X_train = X_train[valid_train_idxs]
    y_train = y_train[valid_train_idxs]
    latlons_train = latlons_train[valid_train_idxs]
    #loc_emb_train = loc_emb_train[valid_train_idxs]

    valid_test_idxs = np.where(~np.isin(ids_test, invalid_ids))[0]
    ids_test = ids_test[valid_test_idxs]
    X_test = X_test[valid_test_idxs]
    y_test = y_test[valid_test_idxs]
    latlons_test = latlons_test[valid_test_idxs]
    #loc_emb_test = loc_emb_test[valid_test_idxs]

    return X_train, X_test, y_train, y_test, latlons_train, latlons_test, ids_train, ids_test

def retrieve_train_X(label):
    data_path = "data/int/feature_matrices/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl".format(label=label)
    with open(data_path, "rb") as f:
        arrs = dill.load(f)

    X_train = arrs["X_train"]

    return X_train

def retrieve_train_loc_emb(label):
    data_path = "data/int/feature_matrices/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl".format(label=label)
    with open(data_path, "rb") as f:
        arrs = dill.load(f)

    loc_emb_train = arrs["loc_emb_train"]

    return loc_emb_train

def retrieve_train_ids(label):
    data_path = "data/int/feature_matrices/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl".format(label=label)
    with open(data_path, "rb") as f:
        arrs = dill.load(f)

    ids_train = arrs["ids_train"]

    return ids_train


'''
    Writes array of latlons to pkl file
'''
def record_latlons(label, latlons, rule, size):
    print("Recording points used...")
    latlon_path = "data/latlons/{label}_sample_{rule}_{size}.pkl".format(label=label, rule=rule, size=size)

    with open(latlon_path, "wb") as f:
        dill.dump(
            {"latlon": latlons},
            f,
            protocol=4,
        )

'''
    Writes array of latlons and isz to pkl file
'''
def record_latlons_ids(label, latlons, ids, rule, size):
    print("Recording points used...")
    latlon_path = "data/latlons_ids/{label}_sample_{rule}_{size}.pkl".format(label=label, rule=rule, size=size)

    with open(latlon_path, "wb") as f:
        dill.dump(
            {"latlon": latlons, "ids": ids},
            f,
            protocol=4,
        )