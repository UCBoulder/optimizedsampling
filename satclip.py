from os.path import join

import dill
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def get_satclip(c, X):
    """Get random features matrices for the main (CONTUS) analysis.

    Parameters
    ----------
    X: df to match index of
    c: config

    Returns
    -------
    satclip: :class:`pandas.DataFrame`
        100000 x 256 array of location embeddings, indexed by i,j ID
    """

    # Load the feature matrix locally
    local_path = join(
        c.features_dir,
        "satclip_embeddings.pkl",
    )
    with open(local_path, "rb") as f:
        arrs = dill.load(f)

    # get embeddings
    emb = pd.DataFrame(arrs["emb"], index=arrs["ids"], columns=["L_" + str(i) for i in range(arrs["emb"].shape[1])])

    # reindex embeddings according to X
    emb = emb.reindex(X.index)

    return emb

def satclip_pca(emb):
    """Get satclip embeddings and perform pca.

    Parameters
    ----------
    emb: (num of samples)x256 array

    Returns
    -------
    emb_pca: (num of samples)x3 array
    """
    pca = PCA(n_components=3)
    emb_pca = pca.fit_transform(emb)

    min_vals = emb_pca.min(axis=0)
    max_vals = emb_pca.max(axis=0)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    scaled_emb_pca = (emb_pca - min_vals)/(range_vals)

    return scaled_emb_pca
