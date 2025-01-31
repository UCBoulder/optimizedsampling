import dill
import numpy as np

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from naip import *

def naip_img_labels(img_path):
    latlons = all_pixels_latlons(img_path)
    return urban_areas(latlons)

def naip_img_urban_areas_label_counts(img_path):
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

def urban_areas(latlons):
    lats = latlons[:,0]
    lons = latlons[:,1]

    #Read shape file with polygons
    df_polygons = gpd.read_file('country_boundaries/census/tl_2020_us_uac20.shp')
    df_polygons = df_polygons.to_crs("EPSG:4326")

    # Convert to GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    gdf_samples = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326") 

    # Check if each point is inside any polygon
    inside_polygon = np.array([any(df_polygons.contains(point)) for point in gdf_samples.geometry])
    inside_polygon = inside_polygon.astype(int)
    from IPython import embed; embed()

    return inside_polygon

if __name__ == '__main__':
    data_path = "data/int/feature_matrices/CONTUS_UAR_torchgeo4096.pkl"
    cluster_path = "data/clusters/urban_areas_cluster_assignment.pkl"

    with open(data_path, "rb") as f:
        arrs = dill.load(f)

    ids = arrs['ids_X']

    latlons = arrs['latlon']
    lats = latlons[:, 0]
    lons = latlons[:, 1]

    urban_areas_groups = urban_areas_by_img(lats, lons)

    with open(cluster_path, "wb") as f:
        dill.dump(
            {"ids": ids, "clusters": urban_areas_groups},
            f,
            protocol=4,
        )