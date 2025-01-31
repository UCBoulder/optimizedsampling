#Utils for 4x256x256 NAIP imagery

import rasterio
import numpy as np
from pyproj import Transformer

def all_pixels_latlons(img_path):
    with rasterio.open(img_path) as src:
        transform = src.transform #Affine transform
        crs = src.crs
        width, height = src.width, src.height

        #2D grid for each pixel in the raster
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        #Convert pixel indices to spatial coordinates corresponding to each pixel
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        xs, ys = np.array(xs), np.array(ys)

        #Transform xs, ys (spatial coordinates) into lat lons
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(np.array(xs).ravel(), np.array(ys).ravel())

        latlons = np.column_stack([lat, lon])

        return latlons