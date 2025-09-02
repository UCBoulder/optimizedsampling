#!/usr/bin/env python3
"""
NLCD Land Cover Classification and Clustering Script

This script processes NAIP satellite imagery to extract land cover classifications
using the National Land Cover Database (NLCD), and performs k-means clustering
on the land cover percentages.

Usage:
    python nlcd_groups.py --input_dir /path/to/naip/images --nlcd_path /path/to/nlcd.tif --output_dir /path/to/output
"""

import os
import argparse
import rasterio
from pyproj import Transformer
import dill
import numpy as np
from pyproj import Transformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import logging
import requests
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NLCD Download URLs
NLCD_URLS = {
    "2016": "https://cloud.sdp.earthdata.nasa.gov/mrlc/nlcd_2016_land_cover_l48_20210604.zip", 
}

# NLCD Land Cover Classes
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
    95: "Emergent Herbaceous Wetlands"
}

def all_pixels_latlons(img_path):
    with rasterio.open(img_path) as src:
        transform = src.transform #Affine transform
        crs = src.crs
        width, height = src.width, src.height

        #2D grid for each pixel in the raster
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        from IPython import embed; embed()

        #Convert pixel indices to spatial coordinates corresponding to each pixel
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        xs, ys = np.array(xs), np.array(ys)

        #Transform xs, ys (spatial coordinates) into lat lons
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(np.array(xs).ravel(), np.array(ys).ravel())

        latlons = np.column_stack([lat, lon])

        return latlons

def nlcd_land_cover_class(latlons, nlcd_path, land_cover_classes=None):
    """
    Extract NLCD land cover classes for given lat/lon coordinates.
    
    Args:
        latlons (np.array): Array of shape (n, 2) with [lat, lon] coordinates
        nlcd_path (str): Path to NLCD raster file
        land_cover_classes (np.array, optional): Pre-loaded land cover data
    
    Returns:
        np.array: Land cover class labels for each coordinate
    """
    with rasterio.open(nlcd_path) as src:
        crs = src.crs
        
        # Convert lat/lon to NLCD CRS
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        
        # Extract lats and lons as separate arrays
        lons, lats = latlons[:, 1], latlons[:, 0]
        
        # Batch transform lat/lon coordinates
        x, y = transformer.transform(lons, lats)
        
        # Convert x, y to row, col indices
        rows, cols = src.index(x, y)
        
        # Load land cover data if not provided
        if land_cover_classes is None:
            land_cover_classes = src.read(1)
            
        # Extract labels
        labels = land_cover_classes[rows, cols]
    
    return labels

def calculate_nlcd_percentages(img_path, nlcd_path, land_cover_classes=None):
    """
    Calculate NLCD land cover percentages for a NAIP image.
    
    Args:
        img_path (str): Path to NAIP image
        nlcd_path (str): Path to NLCD raster
        land_cover_classes (np.array, optional): Pre-loaded land cover data
    
    Returns:
        np.array: Percentage of each NLCD class in the image
    """
    # Get all pixel coordinates
    latlons = all_pixels_latlons(img_path)
    
    # Get NLCD labels for these coordinates
    labels = nlcd_land_cover_class(latlons, nlcd_path, land_cover_classes)
    
    # Count occurrences of each label
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_count_dict = dict(zip(unique_labels, counts))
    
    # Create count array for expected labels only
    count_array = np.zeros_like(EXPECTED_LABELS, dtype=int)
    for i, label in enumerate(EXPECTED_LABELS):
        if label in label_count_dict:
            count_array[i] = label_count_dict[label]
    
    # Convert to percentages
    total = np.sum(count_array)
    if total == 0:
        logger.warning(f"No valid NLCD labels found for {img_path}")
        return np.zeros_like(EXPECTED_LABELS, dtype=np.float32)
    
    percentage_array = (count_array / total).astype(np.float32)
    return percentage_array

def find_optimal_clusters(data, k_range=(2, 10)):
    """
    Find optimal number of clusters using silhouette analysis.
    
    Args:
        data (np.array): Data to cluster
        k_range (tuple): Range of k values to test
    
    Returns:
        tuple: (best_labels, best_k, best_score)
    """
    best_score = -1
    best_k = 0
    best_labels = None
    
    logger.info(f"Testing k-means clustering for k in range {k_range}")
    
    for k in range(k_range[0], k_range[1] + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            
            logger.info(f"k={k}, silhouette score={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                
        except ValueError as e:
            logger.warning(f"Skipping k={k} due to error: {e}")
    
    logger.info(f"Best clustering: k={best_k} with silhouette score={best_score:.4f}")
    return best_labels, best_k, best_score

def process_naip_images(input_dir, nlcd_path, output_dir, file_pattern="*.tif"):
    """
    Process all NAIP images in a directory to extract NLCD percentages.
    
    Args:
        input_dir (str): Directory containing NAIP images
        nlcd_path (str): Path to NLCD raster file
        output_dir (str): Directory to save results
        file_pattern (str): File pattern to match
    
    Returns:
        tuple: (nlcd_percentages, ids, successful_count)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    import glob
    image_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not image_files:
        raise ValueError(f"No files found matching pattern {file_pattern} in {input_dir}")
    
    logger.info(f"Found {len(image_files)} image files to process")
    
    # Pre-load NLCD data
    logger.info(f"Loading NLCD data from {nlcd_path}")
    with rasterio.open(nlcd_path) as src:
        land_cover_classes = src.read(1)
    
    # Initialize arrays
    ids = []
    nlcd_percentages = []
    successful_count = 0
    
    # Process images with progress bar
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Extract ID from filename
            filename = os.path.basename(img_path)
            img_id = filename.replace('tile_', '').replace('.tif', '')
            
            # Calculate NLCD percentages
            percentages = calculate_nlcd_percentages(img_path, nlcd_path, land_cover_classes)
            
            # Check for valid data
            if not np.any(np.isnan(percentages)) and np.sum(percentages) > 0:
                ids.append(img_id)
                nlcd_percentages.append(percentages)
                successful_count += 1
            else:
                logger.warning(f"Invalid data for {img_id}, skipping")
                
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue
    
    logger.info(f"Successfully processed {successful_count}/{len(image_files)} images")
    
    if successful_count == 0:
        raise RuntimeError("No images were successfully processed")
    
    # Convert to numpy arrays
    ids = np.array(ids)
    nlcd_percentages = np.array(nlcd_percentages)
    
    return nlcd_percentages, ids, successful_count

def save_results(nlcd_percentages, ids, cluster_labels, output_dir, dataset_name="usavars"):
    """
    Save NLCD percentages and cluster assignments.
    
    Args:
        nlcd_percentages (np.array): NLCD percentage data
        ids (np.array): Image IDs
        cluster_labels (np.array): Cluster assignments
        output_dir (str): Output directory
        dataset_name (str): Dataset name for filenames
    """
    # Save NLCD percentages
    nlcd_path = os.path.join(output_dir, f"{dataset_name}_NLCD_percentages.pkl")
    with open(nlcd_path, "wb") as f:
        dill.dump({
            "NLCD_percentages": nlcd_percentages,
            "ids": ids,
            "class_names": NLCD_CLASS_NAMES,
            "expected_labels": EXPECTED_LABELS
        }, f, protocol=4)
    
    logger.info(f"Saved NLCD percentages to {nlcd_path}")
    
    # Save cluster assignments
    if cluster_labels is not None:
        cluster_dict = dict(zip(ids.astype(str), cluster_labels))
        cluster_path = os.path.join(output_dir, f"{dataset_name}_nlcd_cluster_assignments.pkl")
        with open(cluster_path, "wb") as f:
            dill.dump(cluster_dict, f, protocol=4)
        
        logger.info(f"Saved cluster assignments to {cluster_path}")
        
        # Print cluster summary
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        logger.info("Cluster distribution:")
        for cluster, count in zip(unique_clusters, counts):
            logger.info(f"  Cluster {cluster}: {count} images ({count/len(cluster_labels)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description="Extract NLCD land cover percentages from NAIP images and perform clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing NAIP image files"
    )
    
    parser.add_argument(
        "--nlcd_path", 
        type=str,
        required=True,
        help="Path to existing NLCD .tif raster file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="usavars",
        help="Dataset name for output filenames"
    )
    
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.tif",
        help="File pattern to match in input directory"
    )
    
    parser.add_argument(
        "--k_min",
        type=int,
        default=2,
        help="Minimum number of clusters to test"
    )
    
    parser.add_argument(
        "--k_max",
        type=int,
        default=10,
        help="Maximum number of clusters to test"
    )
    
    parser.add_argument(
        "--skip_clustering",
        action="store_true",
        help="Skip clustering step, only extract NLCD percentages"
    )
    
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    if not os.path.exists(args.nlcd_path):
        raise FileNotFoundError(f"NLCD file not found: {args.nlcd_path}")
    
    logger.info("Starting NLCD processing pipeline")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"NLCD file: {args.nlcd_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Process NAIP images
        nlcd_percentages, ids, successful_count = process_naip_images(
            args.input_dir, 
            args.nlcd_path, 
            args.output_dir,
            args.file_pattern
        )
        
        # Perform clustering if requested
        cluster_labels = None
        if not args.skip_clustering:
            cluster_labels, best_k, best_score = find_optimal_clusters(
                nlcd_percentages, 
                k_range=(args.k_min, args.k_max)
            )
        
        # Save results
        save_results(nlcd_percentages, ids, cluster_labels, args.output_dir, args.dataset_name)
        
        logger.info("Processing completed successfully!")
        logger.info(f"Processed {successful_count} images")
        if cluster_labels is not None:
            logger.info(f"Created {len(np.unique(cluster_labels))} clusters")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()