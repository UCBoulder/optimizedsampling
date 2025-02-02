import dill
import pandas as pd
import numpy as np
from nlcd import nlcd_land_cover_class, nlcd_k_means
from satclip import satclip_k_means


def cluster_and_save(feat_type):
    cluster_path = f"data/clusters/{feat_type}_cluster_assignment.pkl"
    data_path = 'data/int/feature_matrices/satclip_embeddings.pkl'

    labels, ids = satclip_k_means(data_path) #Change depending on desired clustering

    with open(cluster_path, "wb") as f:
        dill.dump(
            {"ids": ids, "clusters": labels},
            f,
            protocol=4,
        )

def retrieve_clusters(ids, cluster_path):
    with open(cluster_path, "rb") as f:
        arrs = dill.load(f)

    clusters_df = pd.DataFrame(arrs["clusters"], index=arrs["ids"], columns=["cluster"])
    cluster_labels = np.empty((len(ids),), dtype=float)

    for i in range(len(ids)):
        cluster_labels[i] = clusters_df["cluster"].get(ids[i], np.nan)

    return cluster_labels

def retrieve_all_clusters(cluster_path):
    with open(cluster_path, "rb") as f:
        arrs = dill.load(f)

    return arrs['clusters']

if __name__ == '__main__':
    cluster_and_save("SatCLIP")