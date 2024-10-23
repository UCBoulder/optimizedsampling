import numpy as np
import random
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# from USAVars import USAVars

# train = USAVars(root="/share/usavars", split="train", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)

lat_min, lat_max = 24.396308, 49.384358  # Latitude range
lon_min, lon_max = -125.0, -66.93457     # Longitude range
world = gpd.read_file('country_boundaries/ne_110m_admin_1_states_provinces.shp')
exclude_states = ["Alaska", "Hawaii"]
contiguous_us = world[world['name'].isin(exclude_states) == False]
us_boundary = contiguous_us.union_all()
union_gdf = gpd.GeoDataFrame(geometry=[us_boundary], crs=contiguous_us.crs)

def generate_grid(n):
    points_per_side = int(np.sqrt(n))

    # Generate evenly spaced latitudes and longitudes
    lats = np.linspace(lat_min, lat_max, points_per_side)
    lons = np.linspace(lon_min, lon_max, points_per_side)

    points = []
    for lat in lats:
        for lon in lons:
            points.append(Point(lon,lat))
    
    return points

def points_within_us(points):
    us_boundary = contiguous_us.union_all()
    points_gdf = gpd.GeoDataFrame(geometry=points, crs=contiguous_us.crs)
    return points_gdf[points_gdf.geometry.within(us_boundary)]

def n_points_within_us(n, points):
    num = n
    us_boundary = contiguous_us.union_all()
    points_gdf = points_within_us(points)
    while (points_gdf.__len__() < n):
        num = num + 10
        points = generate_grid(num)
        points_gdf = points_within_us(points)
    return points_gdf[points_gdf.geometry.within(us_boundary)]

def plot_points_and_boundary(points):
    fig, ax = plt.subplots(figsize=(10,10))

    union_gdf.plot(ax=ax, color='lightblue', edgecolor='black')

    points.plot(ax=ax, color='red', markersize=5)

    fig.savefig("sampling.png")

points = generate_grid(100)
points_within_us = n_points_within_us(100, points)
print(points_within_us.__len__())
plot_points_and_boundary(points_within_us)