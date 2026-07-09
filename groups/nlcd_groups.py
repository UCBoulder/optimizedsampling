import os
import argparse
import glob
import rasterio
import dill
import numpy as np
from pyproj import Transformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

EXPECTED_LABELS = np.array([11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95])
NLCD_CLASS_NAMES = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed High Intensity",
    31: "Barren Land (Rock/Sand/Clay)",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",
}


def all_pixels_latlons(img_path):
    with rasterio.open(img_path) as src:
        rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing="ij")
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(np.array(xs).ravel(), np.array(ys).ravel())
        return np.column_stack([lat, lon])


def nlcd_land_cover_class(latlons, nlcd_path, land_cover_classes=None):
    with rasterio.open(nlcd_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        lons, lats = latlons[:, 1], latlons[:, 0]
        x, y = transformer.transform(lons, lats)
        rows, cols = src.index(x, y)
        if land_cover_classes is None:
            land_cover_classes = src.read(1)
        return land_cover_classes[rows, cols]


def calculate_nlcd_percentages(img_path, nlcd_path, land_cover_classes=None):
    latlons = all_pixels_latlons(img_path)
    labels = nlcd_land_cover_class(latlons, nlcd_path, land_cover_classes)
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_count_dict = dict(zip(unique_labels, counts))

    count_array = np.zeros_like(EXPECTED_LABELS, dtype=int)
    for i, label in enumerate(EXPECTED_LABELS):
        if label in label_count_dict:
            count_array[i] = label_count_dict[label]

    total = np.sum(count_array)
    if total == 0:
        return np.zeros_like(EXPECTED_LABELS, dtype=np.float32)
    return (count_array / total).astype(np.float32)


def find_optimal_clusters(data, k_range=(2, 10)):
    best_score, best_k, best_labels = -1, 0, None
    for k in range(k_range[0], k_range[1] + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(data)
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels
        except ValueError:
            continue
    return best_labels, best_k, best_score


def process_naip_images(input_dir, nlcd_path, output_dir, file_pattern="*.tif"):
    os.makedirs(output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(input_dir, file_pattern))
    if not image_files:
        raise ValueError(f"No files found matching pattern {file_pattern} in {input_dir}")

    with rasterio.open(nlcd_path) as src:
        land_cover_classes = src.read(1)

    ids, nlcd_percentages = [], []
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            img_id = os.path.basename(img_path).replace("tile_", "").replace(".tif", "")
            percentages = calculate_nlcd_percentages(img_path, nlcd_path, land_cover_classes)
            if not np.any(np.isnan(percentages)) and np.sum(percentages) > 0:
                ids.append(img_id)
                nlcd_percentages.append(percentages)
        except Exception:
            continue

    if not ids:
        raise RuntimeError("No images were successfully processed")

    ids = np.array(ids)
    nlcd_percentages = np.array(nlcd_percentages)
    return nlcd_percentages, ids, len(ids)


def save_results(nlcd_percentages, ids, cluster_labels, output_dir, dataset_name="usavars"):
    nlcd_path = os.path.join(output_dir, f"{dataset_name}_NLCD_percentages.pkl")
    with open(nlcd_path, "wb") as f:
        dill.dump({
            "NLCD_percentages": nlcd_percentages,
            "ids": ids,
            "class_names": NLCD_CLASS_NAMES,
            "expected_labels": EXPECTED_LABELS,
        }, f, protocol=4)

    if cluster_labels is not None:
        cluster_dict = dict(zip(ids.astype(str), cluster_labels))
        cluster_path = os.path.join(output_dir, f"{dataset_name}_nlcd_cluster_assignments.pkl")
        with open(cluster_path, "wb") as f:
            dill.dump(cluster_dict, f, protocol=4)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--nlcd_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="usavars")
    parser.add_argument("--file_pattern", type=str, default="*.tif")
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=10)
    parser.add_argument("--skip_clustering", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not os.path.exists(args.nlcd_path):
        raise FileNotFoundError(f"NLCD file not found: {args.nlcd_path}")

    nlcd_percentages, ids, _ = process_naip_images(
        args.input_dir, args.nlcd_path, args.output_dir, args.file_pattern
    )

    cluster_labels = None
    if not args.skip_clustering:
        cluster_labels, _, _ = find_optimal_clusters(nlcd_percentages, k_range=(args.k_min, args.k_max))

    save_results(nlcd_percentages, ids, cluster_labels, args.output_dir, args.dataset_name)


if __name__ == "__main__":
    main()
