#Utils for 4x256x256 NAIP imagery

import rasterio
import numpy as np
from pyproj import Transformer

def all_pixels_latlons(img_path):
    with rasterio.open(img_path) as src:
        transform = src.transform
        crs = src.crs
        width, height = src.width, src.height

        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        xs, ys = rasterio.transform.xy(transform, rows, cols)
        xs, ys = np.array(xs), np.array(ys)

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(np.array(xs).ravel(), np.array(ys).ravel())

        latlons = np.column_stack([lat, lon])

        return latlons