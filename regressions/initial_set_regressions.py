import os
import csv
import re
import argparse
from tqdm import tqdm
from data_loading import load_data_from_pkl, create_id_mapping, load_sample_file
from regressions import ridge_regression


def parse_cluster_metadata(fname, sampled_indices, r2):
    parts = os.path.basename(fname).replace(".pkl", "").split("_")
    return {
        "filename": fname,
        "seed": parts[-1],
        "sample_size": len(sampled_indices),
        "points_per_cluster": parts[-4].replace("ppc", ""),
        "strata": parts[3],
        "cluster": parts[4],
        "r2": r2,
    }


def parse_convenience_metadata(fname, sampled_indices, r2):
    parts = os.path.basename(fname).replace(".pkl", "").split("_")
    source = next((s for s in ("urban_based", "region_based", "cluster_based") if s in fname), "unknown")
    seed = parts[-1] if "seed" in parts else None
    method = parts[-2] if "seed" in parts else None

    num_urban = None
    for p in parts:
        m = re.match(r"top(\d+)", p)
        if m:
            num_urban = int(m.group(1))
            break

    cluster_type = points_per_cluster = num_clusters = None
    if "cluster" in parts:
        idx = parts.index("cluster")
        cluster_type = parts[idx + 1] if idx + 1 < len(parts) else None
    if "ppc" in parts:
        idx = parts.index("ppc")
        points_per_cluster = int(parts[idx - 1]) if idx > 0 else None
    if "clusters" in parts:
        idx = parts.index("clusters")
        num_clusters = int(parts[idx - 1]) if idx > 0 else None

    return {
        "filename": os.path.basename(fname),
        "sample_size": len(sampled_indices),
        "seed": seed,
        "source": source,
        "method": method,
        "num_urban": num_urban,
        "cluster_type": cluster_type,
        "points_per_cluster": points_per_cluster,
        "num_clusters": num_clusters,
        "r2": r2,
    }


def parse_random_metadata(fname, sampled_indices, r2):
    m = re.search(r"random_sample_(\d+)_points_seed_(\d+)\.pkl$", fname)
    return {
        "filename": fname,
        "sample_size": int(m.group(1)) if m else len(sampled_indices),
        "seed": int(m.group(2)) if m else None,
        "r2": r2,
    }


PARSERS = {
    "cluster": parse_cluster_metadata,
    "convenience": parse_convenience_metadata,
    "random": parse_random_metadata,
}


def run(features_path, sampling_dir, results_dir, sampling_type, dataset=None, label=None, min_samples=100):
    full_ids, X_train_full, y_train_full, X_test, y_test = load_data_from_pkl(features_path, dataset=dataset, label=label)
    id_to_index = create_id_mapping(full_ids)
    metadata_parser = PARSERS[sampling_type]

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{sampling_type}_sampling_r2_scores.csv")
    results = []

    sample_files = sorted(f for f in os.listdir(sampling_dir) if f.endswith(".pkl"))
    for fname in tqdm(sample_files, desc="Processing samples"):
        sampled_ids = load_sample_file(os.path.join(sampling_dir, fname))
        sampled_indices = [id_to_index[i] for i in sampled_ids if i in id_to_index]
        if len(sampled_indices) < min_samples:
            continue

        r2 = ridge_regression(X_train_full[sampled_indices], y_train_full[sampled_indices], X_test, y_test)
        if r2 is None:
            continue

        results.append(metadata_parser(fname, sampled_indices, r2))

    if not results:
        return

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", required=True)
    p.add_argument("--sampling_dir", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--sampling_type", required=True, choices=PARSERS.keys())
    p.add_argument("--dataset", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--min_samples", type=int, default=100)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
