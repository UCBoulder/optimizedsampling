import os
import rasterio
import dill
from pyproj import Transformer

from naip import *

expected_labels = np.array([11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95])

def nlcd_land_cover_class(latlons, land_cover_classes=None):
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

        # Assign labels
        if land_cover_classes is None:
            land_cover_classes = src.read(1)  # Read the first band
        labels = land_cover_classes[rows, cols]

    return labels

def naip_img_labels(img_path):
    latlons = all_pixels_latlons(img_path)
    return nlcd_land_cover_class(latlons, land_cover_classes)

def naip_img_nlcd_label_counts(img_path, land_cover_classes):
    labels = naip_img_labels(img_path)
    unique_labels, counts = np.unique(labels, return_counts=True)

    label_count_dict = dict(zip(unique_labels, counts))

    count_array = np.zeros_like(expected_labels, dtype=int)

    for i, label in enumerate(expected_labels):
        if label in label_count_dict:
            count_array[i] = label_count_dict[label]

    total = np.sum(count_array)
    percentage_array = (count_array / total)

    return percentage_array

if __name__ == "__main__":
    from USAVars import USAVars

    nlcd_path = "land_cover/Annual_NLCD_LndCov_2016_CU_C1V0.tif"

    with rasterio.open(nlcd_path) as src:
        land_cover_classes = src.read(1)  # Read the first band

    root_dir = "/share/usavars/uar"
    file_count = len(os.listdir(root_dir))
 
    ids = np.empty((file_count,), dtype='U{}'.format(15))
    nlcd_percentages = np.empty((file_count, 16), dtype=np.float32)

    i = 0
    for file_name in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file_name)
        print(f"Processing Sample {i}")

        try:
            percentage_array = naip_img_nlcd_label_counts(file_path, land_cover_classes)
            nlcd_percentages[i] = percentage_array

            id = file_name.replace('tile_', '').replace('.tif', '')
            ids[i] = id
            i += 1
        
        except Exception as e:
            print(f"Skipping sample {i} due to error: {e}")
            i += 1

    from IPython import embed; embed()
    out_fpath = 'data/clusters/NLCD_percentages.pkl'
    with open(out_fpath, "wb") as f:
        dill.dump(
            {"NLCD_percentages": nlcd_percentages, "ids": ids},
            f,
            protocol=4,
        )
