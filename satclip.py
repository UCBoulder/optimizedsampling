from os.path import join

import dill
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

def satclip_k_means(emb_path, max_k):
    best_score = -1
    best_k = 0
    best_labels = None

    with open(emb_path, 'rb') as f:
        arrs = dill.load(f)
        data = arrs['emb']
        ids = arrs['ids']

    for k in range(2, max_k):  # Test k from 2 to 50
        try:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
            score = silhouette_score(data, kmeans.labels_)

            print(f"For k={k}, silhouette score={score}")
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = kmeans.labels_

        except ValueError as e:
            print(f"Skipping k={k} due to error: {e}")

    print(f"Best k={best_k} with silhouette score={best_score}")

    return best_labels, ids

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


