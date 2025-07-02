import dill
import pandas as pd
import numpy as np
from nlcd import nlcd_land_cover_class, nlcd_k_means


def cluster_and_save(feat_type, data_path, max_k=10):
    cluster_path = f"data/clusters/{feat_type}_cluster_assignment_{max_k}.pkl"

    labels, ids = nlcd_k_means(data_path, max_k) #Change depending on desired clustering

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
    data_path = "data/clusters/NLCD_percentages.pkl"
    cluster_and_save("NLCD_percentages", data_path, max_k=10)