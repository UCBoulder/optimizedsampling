import dill
import pandas as pd
import numpy as np
import rasterio
from pyproj import Transformer

def nlcd_land_cover_class(latlons):
    nlcd_path = "land_cover/Annual_NLCD_LndCov_2016_CU_C1V0.tif"

    # Open the NLCD TIFF file using rasterio
    with rasterio.open(nlcd_path) as src:
        # Get the CRS and transformation from the raster
        crs = src.crs

        # Convert lat/lon to NLCD CRS
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        # Extract lats and lons as separate arrays
        lons, lats = latlons[:, 1], latlons[:, 0]

        # Batch transform lat/lon coordinates
        x, y = transformer.transform(lons, lats)

        # Use the transform to convert x, y to row, col indices
        rows, cols = src.index(x, y)

        # Read the land cover classes for the batch of pixels in one go
        land_cover_classes = src.read(1)  # Read the first band
        labels = land_cover_classes[rows, cols]

    return labels

def cluster_and_save(ids, latlons, feat_type):
    cluster_path = f"data/clusters/{feat_type}_cluster_assignment.pkl"

    labels = nlcd_land_cover_class(latlons)

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