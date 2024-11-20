'''
Modeling travel cost/feasibility
'''
from os.path import join

import geopandas as gpd
import numpy as np
import dill
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from geopy.distance import geodesic #geodesic distance between two lat/lon

cities = {
    "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
    "latitude": [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
    "longitude": [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740]
}
cities = pd.DataFrame(cities)

'''
Determine distance to closest city
'''
def closest_city(latlon):
    closest_city_distance = float('inf')

    for _, city_row in cities.iterrows():
        city_lat = city_row['latitude']
        city_lon = city_row['longitude']
        distance = geodesic(latlon, (city_lat, city_lon)).km
        if distance <= closest_city_distance:
            closest_city_distance = distance
    
    return closest_city_distance

'''
Use closest city distance to determine cost
'''
def cost_by_closest_city_dist(latlon, alpha, beta):
    closest_city_distance = closest_city(latlon)

    return (alpha*closest_city_distance + beta)
'''
Creates array of costs
'''
def cost_array_by_city_dist(latlons, alpha, beta):
    n = latlons.shape[0]
    costs = np.empty((n,), dtype=np.float32)

    for i in range(n):
        costs[i] = cost_by_closest_city_dist(latlons[i], alpha, beta)

    return costs

def save_costs(latlon_path, out_fpath):
    #Retrieve data
    with open(latlon_path, "rb") as f:
        arrs = dill.load(f)
        
    # get latlons
    latlons = pd.DataFrame(arrs["latlon"], index=arrs["ids_X"], columns=["lat", "lon"])

    # sort
    latlons = latlons.sort_values(["lat", "lon"], ascending=[False, True])
    ids = latlons.index.to_numpy()

    # Convert to numpy array
    latlons = latlons.values

    # get cost array
    costs = cost_array_by_city_dist(latlons, 100, 100)

    # save
    with open(out_fpath, "wb") as f:
        dill.dump(
            {"cost": costs, "ids": ids, "latlon": latlons},
            f,
            protocol=4,
        )


def plot_lat_lon_with_cost(lats, lons, costs, title):
    # Create a GeoDataFrame with costs
    gdf = gpd.GeoDataFrame(
        {'cost': costs},
        geometry=gpd.points_from_xy(lons, lats)
    )

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
        column='cost',
        cmap='plasma',  # Choose a color map suitable for continuous data
        markersize=5,
        alpha=0.6,
        legend=True,
        legend_kwds={'label': "Cost"},
        cax=cax
    )
    ax.set_title(title)
    ax.axis("off")
    return fig

def get_costs(X):
    """Get costs for samples.

    Parameters
    ----------
    X: df to match index of
    c: config

    Returns
    -------
    costs: :class:`pandas.DataFrame`
        100000 x 1 array of costs, indexed by i,j ID
    """

    # Load the feature matrix locally
    local_path = "/share/usavars/data/cost/costs_by_city_dist.pkl"
    with open(local_path, "rb") as f:
        arrs = dill.load(f)

    # get embeddings
    costs = pd.DataFrame(arrs["cost"], index=arrs["ids"], columns=["cost"])

    # reindex embeddings according to X
    costs = costs.reindex(X.index)

    return costs

'''
Returns total cost of subset
'''
def cost_of_subset(costs, subset_idxs):
    print("Determining cost of subset...")
    subset_costs = costs[subset_idxs]
    return np.sum(subset_costs)
