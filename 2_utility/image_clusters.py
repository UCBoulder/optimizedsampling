import dill
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score

DATASET_FEATURE_PATHS = {
    "usavars_population": "../0_data/features/usavars/CONTUS_UAR_population_with_splits_torchgeo4096.pkl",
    "usavars_treecover": "../0_data/features/usavars/CONTUS_UAR_treecover_with_splits_torchgeo4096.pkl",
    "india_secc": "../0_data/features/india_secc/India_SECC_with_splits_4000.pkl",
    "togo_ph_h20": "../0_data/features/togo/togo_fertility_data_all_2021_Jan_Jun_P20.pkl"
}

CLUSTER_SAVE_PATHS = {
    "usavars_population": "../0_data/groups/usavars/population/image_{num_clusters}_cluster_assignments.pkl",
    "usavars_treecover": "../0_data/groups/usavars/treecover/image_{num_clusters}_cluster_assignments.pkl",
    "india_secc": "../0_data/groups/india_secc/image_{num_clusters}_cluster_assignments.pkl",
    "togo_ph_h20": "../0_data/groups/togo/image_{num_clusters}_cluster_assignments.pkl"
}

def load_features(pkl_path):
    with open(pkl_path, "rb") as f:
        arrs = dill.load(f)

    feature_matrix = arrs['X_train']
    ids = arrs['ids_train']
    return feature_matrix, ids

def cluster(features, num_clusters):
    return KMeans(n_clusters=num_clusters).fit_predict(features)

def determine_best_cluster(features, num_clusters_max=50):
    best_score = -1
    best_clusters = None
    best_k = None

    for num_clusters in range(2, num_clusters_max + 1):
        print(f"Number of clusters: {num_clusters}")
        preds = cluster(features, num_clusters)
        score = silhouette_score(features, preds)
        print(f"Score: {score}")
        if score > best_score:
            best_score = score
            best_clusters = preds
            best_k = num_clusters

    return best_clusters, best_k


def save_clusters(clusters, ids, save_path):
    id_to_cluster = dict(zip(ids, clusters))
    with open(save_path, "wb") as f:
        dill.dump(id_to_cluster, f)

def determine_clusters_and_save(dataset):
    features, ids = load_features(DATASET_FEATURE_PATHS[dataset])
    clusters, num_clusters = determine_best_cluster(features)
    save_path = CLUSTER_SAVE_PATHS[dataset].format(num_clusters=num_clusters)
    save_clusters(clusters, ids, save_path)

if __name__ == "__main__":
    for dataset in DATASET_FEATURE_PATHS.keys():
        #determine_clusters_and_save(dataset)
        features, ids = load_features(DATASET_FEATURE_PATHS[dataset])
        preds = cluster(features, 8)
        save_path = CLUSTER_SAVE_PATHS[dataset].format(num_clusters=8)
        save_clusters(preds, ids, save_path)
