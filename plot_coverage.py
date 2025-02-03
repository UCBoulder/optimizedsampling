import dill
import rasterio

#to plot lat lon on a map
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

def list_coords(dataset, coord):
    coords = []
    for sample in dataset:
        coords.append(torch.squeeze(sample[coord]))
    return coords

def plot_lat_lon(lats, lons, title, color='orangered', markersize=1, alpha=0.5):
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons, lats),
        crs = 'EPSG:26914'
    )

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
        geometry=gpd.points_from_xy(lons, lats),
        crs = 'EPSG:26914'
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
        cmap='Spectral',  # Choose a color map suitable for continuous data
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
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons, lats),
        crs = 'EPSG:26914'
    )
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

def plot_lat_lon_cluster(lats, lons, clusters, title, markersize=1, alpha=0.5):
        # Create a GeoDataFrame with leverage scores
    gdf = gpd.GeoDataFrame(
        {'cluster': clusters},
        geometry=gpd.points_from_xy(lons, lats),
        crs = 'EPSG:4326' #FIX MAYBE
    )

    # Load and prepare the US boundaries
    world = gpd.read_file("country_boundaries/ne_110m_admin_1_states_provinces.shp", engine="pyogrio")
    exclude_states = ["Alaska", "Hawaii"]
    contiguous_us = world[~world['name'].isin(exclude_states)]
    contiguous_outline = contiguous_us.dissolve()

    # Plot with color map
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.tight_layout(pad=3.0)
    contiguous_outline.boundary.plot(ax=ax, color='black')

    # Define colors for clusters using a colormap
    cmap = plt.get_cmap('tab10')
    unique_clusters = sorted(gdf['cluster'].unique())  # Ensure clusters are sorted
    cluster_colors = [cmap(int(i)) for i in unique_clusters]

    # Plot each cluster with its assigned color
    for cluster_id, color in zip(unique_clusters, cluster_colors):
        cluster_points = gdf[gdf['cluster'] == cluster_id]
        cluster_points.plot(ax=ax, color=color, markersize=markersize, alpha=alpha, label=f'Cluster {cluster_id}')

    # Create a sorted custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Cluster {int(i)}')
                    for i, color in zip(unique_clusters, cluster_colors)]

    ax.legend(handles=legend_elements, loc='lower right', fontsize=14)
    ax.axis("off")

    return fig

def plot_cluster_sample(cluster_num):
    with open('data/clusters/NLCD_percentages_cluster_assignment.pkl', 'rb') as f:
        arrs = dill.load(f)

    clusters = arrs['clusters']
    ids = arrs['ids']

    cluster_idxs = np.where(clusters == cluster_num)[0]
    random_idxs = np.random.choice(cluster_idxs, size=6, replace=False)

    chosen_ids = ids[random_idxs]
    file_paths = [f'/share/usavars/uar/tile_{id}.tif' for id in chosen_ids]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
    axs = axs.flatten()  # Flatten the 2D array of axes to make it easy to iterate

    for i, (file_path, ax) in enumerate(zip(file_paths, axs)):
        # Open the GeoTIFF file using rasterio
        with rasterio.open(file_path) as src:
            # Read the data from the first band
            data = src.read([1,2,3])
            rgb_image = data.transpose(1, 2, 0)

            # Plot the data without a colorbar
            ax.imshow(rgb_image)  # You can change the cmap as per your preference
            ax.set_title(f'GeoTIFF {i+1}')  # Title of the plot
            ax.axis('off')  # Hide axes for cleaner display

    fig.suptitle(f'GeoTIFFS from Cluster {cluster_num}', fontsize=16)
    plt.tight_layout()  # Adjust layout to avoid overlap
    return fig  # Return the figure object

if __name__ == '__main__':
    with open("data/int/feature_matrices/CONTUS_UAR_torchgeo4096.pkl", "rb") as f:
        arrs = dill.load(f)
    
    ids = arrs['ids_X']
    latlons = arrs['latlon']
    lats = latlons[:,0]
    lons = latlons[:,1]

    from clusters import retrieve_clusters

    clusters = retrieve_clusters(ids, "data/clusters/NLCD_percentages_cluster_assignment.pkl")
    valid_idxs = np.where(~np.isnan(clusters))[0]
    lats = lats[valid_idxs]
    lons = lons[valid_idxs]
    clusters = clusters[valid_idxs]

    fig = plot_lat_lon_cluster(lats, lons, clusters, "NLCD groups", markersize=1, alpha=0.5)
    fig.savefig("NLCD_groups.png", dpi=300, bbox_inches='tight')

