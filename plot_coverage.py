#to plot lat lon on a map
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#from USAVars import USAVars

# train = USAVars(root="/share/usavars", split="train", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)
# test = USAVars(root="/share/usavars", split="test", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)
# val = USAVars(root="/share/usavars", split="val", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)

def list_coords(dataset, coord):
    coords = []
    for sample in dataset:
        coords.append(torch.squeeze(sample[coord]))
    return coords

def plot_lat_lon(lats, lons, title, color='orangered', markersize=1, alpha=0.5):
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats))

    world = gpd.read_file("country_boundaries/ne_110m_admin_1_states_provinces.shp", engine = "pyogrio")
    exclude_states = ["Alaska", "Hawaii"]
    contiguous_us = world[world['name'].isin(exclude_states) == False]
    contiguous_outline = contiguous_us.dissolve()

    fig, ax = plt.subplots(figsize=(10,10))
    contiguous_outline.boundary.plot(ax=ax, color='black', zorder=1, alpha=0.5) 

    gdf.plot(ax=ax, color=color, markersize=markersize, alpha=alpha, zorder=2)
    ax.set_title(title)
    ax.axis("off")
    return fig

def plot_coverage(dataset, name):
    lats = list_coords(dataset, "centroid_lat")
    lons = list_coords(dataset, "centroid_lon")

    fig = plot_lat_lon(lats, lons)
    fig.suptitle(name + " Coverage", fontsize = 30)
    fig.savefig(name+ " Coverage")

def plot_lat_lon_with_scores(lats, lons, scores, title):
    # Create a GeoDataFrame with leverage scores
    gdf = gpd.GeoDataFrame(
        {'leverage_score': scores},
        geometry=gpd.points_from_xy(lons, lats)
    )
    gdf['log_leverage_score'] = np.log10(gdf['leverage_score'] + 1e-10) 

    # Load and prepare the US boundaries
    world = gpd.read_file("country_boundaries/ne_110m_admin_1_states_provinces.shp", engine="pyogrio")
    exclude_states = ["Alaska", "Hawaii"]
    contiguous_us = world[~world['name'].isin(exclude_states)]
    contiguous_outline = contiguous_us.dissolve()

    # Plot with color map
    fig, ax = plt.subplots(figsize=(12, 12))
    contiguous_outline.boundary.plot(ax=ax, color='black')

    # Scatter plot of points with color representing leverage score
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    gdf.plot(
        ax=ax,
        column='log_leverage_score',
        cmap='plasma',  # Choose a color map suitable for continuous data
        markersize=5,
        alpha=0.6,
        legend=True,
        legend_kwds={'label': "Log-Transformed Leverage Score"},
        cax=cax
    )
    ax.set_title(title)
    ax.axis("off")
    return fig

def plot_lat_lon_with_rgb(lats, lons, loc_emb, title, markersize=1, alpha=0.5):
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats))
    colors = loc_emb

    world = gpd.read_file("country_boundaries/ne_110m_admin_1_states_provinces.shp", engine = "pyogrio")
    exclude_states = ["Alaska", "Hawaii"]
    contiguous_us = world[world['name'].isin(exclude_states) == False]
    contiguous_outline = contiguous_us.dissolve()

    fig, ax = plt.subplots(figsize=(10,10))
    contiguous_outline.boundary.plot(ax=ax, color='black', zorder=1, alpha=0.5) 

    for i, (geom, color) in enumerate(zip(gdf.geometry, colors)):
        ax.plot(geom.x, geom.y, 'o', color=color, markersize=markersize, alpha=alpha, zorder=2)
    ax.set_title(title)
    ax.axis("off")
    return fig

# plot_lat_lon(train, "Train")
# plot_coverage(test, "Test")
# plot_coverage(val, "Val")