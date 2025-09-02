import dill
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
import os

DEFAULT_FEATURE_PATHS = {
    "usavars_population": "../0_data/features/usavars/CONTUS_UAR_population_with_splits_torchgeo4096.pkl",
    "usavars_treecover": "../0_data/features/usavars/CONTUS_UAR_treecover_with_splits_torchgeo4096.pkl",
    "india_secc": "../0_data/features/india_secc/India_SECC_with_splits_4000.pkl",
    "togo_ph_h20": "../0_data/features/togo/togo_fertility_data_all_2021_Jan_Jun_P20.pkl"
}

DEFAULT_SAVE_PATHS = {
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
    model = KMeans(n_clusters=num_clusters, random_state=42)
    return model.fit_predict(features)

def determine_best_cluster(features, max_clusters=50):
    best_score = -1
    best_clusters = None
    best_k = None
    for k in range(2, max_clusters + 1):
        preds = cluster(features, k)
        score = silhouette_score(features, preds)
        print(f"Clusters: {k}, Silhouette Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_clusters = preds
            best_k = k
    return best_clusters, best_k

def save_clusters(clusters, ids, save_path):
    id_to_cluster = dict(zip(ids, clusters))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        dill.dump(id_to_cluster, f)
    print(f"Saved clusters to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate clusters and save cluster assignments.")
    parser.add_argument("--dataset", type=str, choices=DEFAULT_FEATURE_PATHS.keys(), required=True,
                        help="Dataset to cluster.")
    parser.add_argument("--feature_path", type=str, default=None,
                        help="Path to feature .pkl file. Overrides default for dataset.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save cluster assignment .pkl file. Overrides default for dataset.")
    parser.add_argument("--num_clusters", type=int, default=None,
                        help="Number of clusters to use. If not set, silhouette will be used to select best.")
    parser.add_argument("--max_clusters", type=int, default=50,
                        help="Max clusters to try when determining best cluster count.")
    args = parser.parse_args()

    feature_path = args.feature_path if args.feature_path else DEFAULT_FEATURE_PATHS[args.dataset]
    output_path = args.output_path if args.output_path else DEFAULT_SAVE_PATHS[args.dataset]

    features, ids = load_features(feature_path)

    if args.num_clusters:
        clusters = cluster(features, args.num_clusters)
        num_clusters_used = args.num_clusters
    else:
        clusters, num_clusters_used = determine_best_cluster(features, max_clusters=args.max_clusters)

    save_path = output_path.format(num_clusters=num_clusters_used)
    save_clusters(clusters, ids, save_path)

if __name__ == "__main__":
    main()
