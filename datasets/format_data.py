import argparse
import dill
import numpy as np
import pandas as pd

from mosaiks.code.mosaiks.solve import data_parser as parse
from mosaiks.code.mosaiks import config as cfg

# IDs containing NaN in metadata or outside US
invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])

def save_with_splits(c, label, out_fpath, feature_path):
    with open(feature_path, "rb") as f:
        arrs = dill.load(f)

    X = pd.DataFrame(
        arrs["X"].astype(np.float64),
        index=arrs["ids_X"],
        columns=["X_" + str(i) for i in range(arrs["X"].shape[1])],
    )

    latlons = pd.DataFrame(arrs["latlon"], index=arrs["ids_X"], columns=["lat", "lon"])

    latlons = latlons.sort_values(["lat", "lon"], ascending=[False, True])
    X = X.reindex(latlons.index)

    #call Mosaiks parser for merge, dropna, transform, split
    (
        X_train,
        X_test,
        y_train,
        y_test,
        latlons_train,
        latlons_test,
        ids_train,
        ids_test
    ) = parse.merge_dropna_transform_split_train_test(
        c, label, X, latlons
    )

    out_fpath = out_fpath.format(label=label)

    with open(out_fpath, "wb") as f:
        dill.dump(
            {
                "X_train": X_train,
                "latlons_train": latlons_train,
                "y_train": y_train,
                "X_test": X_test,
                "latlons_test": latlons_test,
                "y_test": y_test,
                "ids_train": ids_train,
                "ids_test": ids_test,
            },
            f,
            protocol=4,
        )


def retrieve_splits(label):
    print("Retrieving TorchGeo Feature splits...")

    data_path = f"data/int/feature_matrices/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl"
    with open(data_path, "rb") as f:
        arrs = dill.load(f)

    X_train = arrs["X_train"]
    X_test = arrs["X_test"]
    y_train = arrs["y_train"]
    y_test = arrs["y_test"]
    latlons_train = arrs["latlons_train"]
    latlons_test = arrs["latlons_test"]
    ids_train = arrs["ids_train"]
    ids_test = arrs["ids_test"]

    valid_train_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
    ids_train = ids_train[valid_train_idxs]
    X_train = X_train[valid_train_idxs]
    y_train = y_train[valid_train_idxs]
    latlons_train = latlons_train[valid_train_idxs]

    valid_test_idxs = np.where(~np.isin(ids_test, invalid_ids))[0]
    ids_test = ids_test[valid_test_idxs]
    X_test = X_test[valid_test_idxs]
    y_test = y_test[valid_test_idxs]
    latlons_test = latlons_test[valid_test_idxs]

    return X_train, X_test, y_train, y_test, latlons_train, latlons_test, ids_train, ids_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save or retrieve data splits for USAVars features")

    parser.add_argument(
        "--label", type=str, required=True, help="Target label name, e.g. population"
    )
    parser.add_argument(
        "--feature_path", type=str, default=None, help="Path to feature .pkl file (required for saving)"
    )
    parser.add_argument(
        "--out_fpath",
        type=str,
        default="CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl",
        help="Output path template for saving splits",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="If set, save splits; otherwise retrieve and print info",
    )

    args = parser.parse_args()

    if args.save:
        if args.feature_path is None:
            raise ValueError("feature_path is required when saving splits")

        from mosaiks.code.mosaiks import config as c
        save_with_splits(c, args.label, args.out_fpath, args.feature_path)
        print(f"Saved splits for label '{args.label}' to {args.out_fpath.format(label=args.label)}")

    else:
        splits = retrieve_splits(args.label)
        print(f"Loaded splits for label '{args.label}':")
        keys = ["X_train", "X_test", "y_train", "y_test", "latlons_train", "latlons_test", "ids_train", "ids_test"]
        for k, v in zip(keys, splits):
            if hasattr(v, "shape"):
                print(f"{k}: shape {v.shape}")
            else:
                print(f"{k}: length {len(v)}")
