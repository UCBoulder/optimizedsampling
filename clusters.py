import argparse
import dill
import pandas as pd
import numpy as np

import optuna
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from format_data import *

def optimize_k(data, min_k, max_k, n_trials=100):
    print("Optimizing for k...")
    #Define objective
    def objective(trial):
        # Suggest hyperparameters
        k = trial.suggest_int("k", min_k, max_k)  # Optimize k between min_val, max_val

        # Fit KMeans with the suggested k
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(data)

        # Compute the silhouette score
        score = silhouette_score(data, labels)

        # Return the silhouette score [make sure to maximize]
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params["k"]

'''
Use to determine clusters in featurized data
'''
def cluster_labels(data):
    best_k = optimize_k(data, 2, 100)
    
    print(f"Running K Means for best k:{best_k}")
    kmeans = KMeans(n_clusters=best_k, random_state=42).fit(data)
        
    best_labels = kmeans.labels_

    return best_labels

def cluster_and_save(ids_train, featurized_data, feat_type):
    cluster_path = f"data/clusters/KMeans_{feat_type}_cluster_assignment.pkl"

    labels = cluster_labels(featurized_data)

    with open(cluster_path, "wb") as f:
        dill.dump(
            {"ids": ids_train, "clusters": labels},
            f,
            protocol=4,
        )

def retrieve_clusters(ids, cluster_path):
    with open(cluster_path, "rb") as f:
        arrs = dill.load(f)

    clusters_df = pd.DataFrame(arrs["clusters"], index=arrs["ids"], columns=["clusters"])
    cluster_labels = np.empty((len(ids),), dtype=int)

    for i in range(len(ids)):
        cluster_labels[i] = clusters_df.loc[ids[i], "clusters"]

    return cluster_labels

ids_train = retrieve_train_ids('treecover')
X_train = retrieve_train_X("treecover")
loc_emb_train = retrieve_train_loc_emb("treecover")

parser = argparse.ArgumentParser()
parser.add_argument('-t', 
                    "--type", 
                    required=True)
args = parser.parse_args()
feat_type = args.type

if feat_type == "torchgeo":
    cluster_and_save(ids_train, X_train, feat_type)
elif feat_type == "satclip":
    cluster_and_save(ids_train, loc_emb_train, feat_type)
else:
    print("Not valid type.")