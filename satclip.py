from os.path import join

import dill
import pandas as pd
import numpy as np

def get_satclip(c, X):
    """Get random features matrices for the main (CONUS) analysis.

    Parameters
    ----------
    X: df to match index of

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