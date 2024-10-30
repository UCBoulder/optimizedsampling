#to plot lat lon on a map
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import torch
import matplotlib.pyplot as plt

from USAVars import USAVars

train = USAVars(root="/share/usavars", split="train", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)
test = USAVars(root="/share/usavars", split="test", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)
val = USAVars(root="/share/usavars", split="val", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)

def list_coords(dataset, coord):
    coords = []
    for sample in dataset:
        coords.append(torch.squeeze(sample[coord]))
    return coords

def plot_lat_lon(lats, lons):
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats))

    world = gpd.read_file("country_boundaries/ne_110m_admin_1_states_provinces.shp")
    exclude_states = ["Alaska", "Hawaii"]
    contiguous_us = world[world['name'].isin(exclude_states) == False]

    fig, ax = plt.subplots(figsize=(10,10))
    contiguous_us.boundary.plot(ax=ax, color='black') 

    gdf.plot(ax=ax, color='red', markersize=0.05)
    return fig

def plot_coverage(dataset, name):
    lats = list_coords(dataset, "centroid_lat")
    lons = list_coords(dataset, "centroid_lon")

    fig = plot_lat_lon(lats, lons)
    fig.suptitle(name + " Coverage", fontsize = 30)
    fig.savefig(name+ " Coverage")
    

plot_lat_lon(train, "Train")
# plot_coverage(test, "Test")
# plot_coverage(val, "Val")