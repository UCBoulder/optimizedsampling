import dill
import pandas as pd
import numpy as np
from nlcd import nlcd_land_cover_class, nlcd_k_means


def cluster_and_save(feat_type):
    cluster_path = f"data/clusters/{feat_type}_cluster_assignment.pkl"
    data_path = 'data/clusters/NLCD_percentages.pkl'

    labels, ids = nlcd_k_means(data_path) #Change depending on desired clustering

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
    cluster_labels = np.empty((len(ids),), dtype=int)

    for i in range(len(ids)):
        cluster_labels[i] = clusters_df.loc[ids[i], "cluster"]

    return cluster_labels

if __name__ == '__main__':
    cluster_and_save("NLCD_percentages")