import dill
import os
import numpy as np

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from naip import *

expected_labels = np.array([0,1])

def naip_img_labels(img_path, df_polygon, sindex):
    latlons = all_pixels_latlons(img_path)
    return urban_areas(latlons, df_polygon, sindex)

def naip_img_urban_areas_label_counts(img_path, df_polygon, sindex):
    labels = naip_img_labels(img_path, df_polygon, sindex)
    unique_labels, counts = np.unique(labels, return_counts=True)

    label_count_dict = dict(zip(unique_labels, counts))

    count_array = np.zeros_like(expected_labels, dtype=int)

    for i, label in enumerate(expected_labels):
        if label in label_count_dict:
            count_array[i] = label_count_dict[label]

    total = np.sum(count_array)
    percentage_array = (count_array / total)

    return percentage_array

# Function to check if a point is inside any polygon using spatial index
def is_inside_polygon(point, df_polygon, sindex):
    # Get potential polygon candidates using spatial index
    possible_matches_index = list(sindex.intersection(point.bounds))
    possible_matches = df_polygon.iloc[possible_matches_index]
    
    # Check for containment in those candidates
    return any(possible_matches.contains(point))

def urban_areas(latlons, df_polygon, sindex):
    lats = latlons[:,0]
    lons = latlons[:,1]

    # Convert to GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    gdf_samples = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326") 

    # Check if each point is inside any polygon
    inside_polygon = np.array([is_inside_polygon(point, df_polygon, sindex) for point in gdf_samples.geometry])
    inside_polygon = inside_polygon.astype(int)

    return inside_polygon

if __name__ == '__main__':
    root_dir = "/share/usavars/uar"
    file_count = len(os.listdir(root_dir))

    #Read shape file with polygons
    df_polygons = gpd.read_file('country_boundaries/census/tl_2020_us_uac20.shp')
    df_polygons = df_polygons.to_crs("EPSG:4326")
    sindex = df_polygons.sindex
 
    ids = np.empty((file_count,), dtype='U{}'.format(15))
    nlcd_percentages = np.empty((file_count, 2), dtype=np.float32)

    i = 0
    for file_name in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file_name)
        id = file_name.replace('tile_', '').replace('.tif', '')
        ids[i] = id
        print(f"Processing Sample {i}")

        try:
            percentage_array = naip_img_urban_areas_label_counts(file_path, df_polygons, sindex)
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
