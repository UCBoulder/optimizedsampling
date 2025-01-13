import dill
import pandas as pd
import numpy as np
from nlcd import nlcd_land_cover_class


def cluster_and_save(ids, latlons, feat_type):
    cluster_path = f"data/clusters/{feat_type}_cluster_assignment.pkl"

    labels = nlcd_land_cover_class(latlons) #Change depending on desired clustering

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

with open("data/int/feature_matrices/CONTUS_UAR_torchgeo4096.pkl", "rb") as f:
    data = dill.load(f)

data_ids = data["ids_X"]
data_latlons = data["latlon"]

cluster_and_save(data_ids, data_latlons, "NLCD")