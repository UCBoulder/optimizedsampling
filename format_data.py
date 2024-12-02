import dill
import pandas as pd
import numpy as np

from mosaiks.code.mosaiks.solve import data_parser as parse

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
        loc_emb_test
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
                "loc_emb_test": loc_emb_test},
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
