import dill
import os
import numpy as np

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from naip import *

expected_labels = np.array([0,1])

def naip_img_labels(img_path, polygons_gdf):
    latlons = all_pixels_latlons(img_path)

    if np.any(np.isinf(latlons)):
            print("Warning: Infinity values detected in lat/lon coordinates.")
            # Optionally return None or handle this case in another way
            return None

    return urban_areas(latlons, polygons_gdf)

def naip_img_urban_areas_label_counts(img_path, polygons_gdf):
    labels = naip_img_labels(img_path, polygons_gdf)
    if labels is None:
        return None
    unique_labels, counts = np.unique(labels, return_counts=True)

    label_count_dict = dict(zip(unique_labels, counts))

    count_array = np.zeros_like(expected_labels, dtype=int)

    for i, label in enumerate(expected_labels):
        if label in label_count_dict:
            count_array[i] = label_count_dict[label]

    total = np.sum(count_array)
    percentage_array = (count_array / total)

    return percentage_array

def urban_areas(latlons, polygons_gdf):
    lats = latlons[:,0]
    lons = latlons[:,1]

    # Convert lat/lon NumPy array to a GeoDataFrame of Points
    points_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in latlons], crs="EPSG:4326")

    # Spatial join: Find which polygon each point belongs to
    joined = gpd.sjoin(points_gdf, polygons_gdf, how="left", predicate="within")

    # Convert to 0s and 1s: if 'index_right' is NaN, it's 0 (not in any polygon), else 1
    inside_polygon = np.where(joined["index_right"].notna(), 1, 0)

    return inside_polygon

if __name__ == '__main__':
    root_dir = "/share/usavars/uar"
    file_count = len(os.listdir(root_dir))

    #Read shape file with polygons
    polygons_gdf = gpd.read_file('country_boundaries/census/tl_2020_us_uac20.shp')
    polygons_gdf = polygons_gdf.to_crs("EPSG:4326")
 
    ids = np.empty((file_count,), dtype='U{}'.format(15))
    nlcd_percentages = np.empty((file_count, 2), dtype=np.float32)

    i = 0
    for file_name in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file_name)
        id = file_name.replace('tile_', '').replace('.tif', '')
        ids[i] = id
        print(f"Processing Sample {i}")

        try:
            percentage_array = naip_img_urban_areas_label_counts(file_path, polygons_gdf)
            nlcd_percentages[i] = percentage_array

            #Make sure no NaNs
            if ~np.any(np.isnan(percentage_array)):
                successful = True
            else:
                successful = False
        
        except Exception as e:
            print(f"Skipping sample {i} due to error: {e}")
            successful = False
    
        if successful:
            print(f"Adding sample {id}")
            i += 1
        else:
            print(f"Error in sample {id}")
            

    out_fpath = 'data/clusters/urban_areas_percentages.pkl'
    with open(out_fpath, "wb") as f:
        dill.dump(
            {"urban_area_percentages": urban_areas_percentages, "ids": ids},
            f,
            protocol=4,
        )
