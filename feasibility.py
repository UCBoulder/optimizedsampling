'''
Modeling travel cost/feasibility
'''
from os.path import join
import os

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
    print("Determining closest city to {latlon}...".format(latlon=latlon))
    closest_city_distance = float('inf')

    for _, city_row in cities.iterrows():
        city_lat = city_row['latitude']
        city_lon = city_row['longitude']
        distance = geodesic(latlon, (city_lat, city_lon)).km
        if distance <= closest_city_distance:
            closest_city_distance = distance
    
    return closest_city_distance

'''
Determine distance to closest city for multiple latlons
'''
def closest_city_array(latlons):
    closest_city_array = np.empty((len(latlons),), dtype=np.float32)
    for i in range(len(latlons)):
        closest_city_array[i] = closest_city(latlons[i])
    
    return closest_city_array

'''
Use closest city distance to determine cost
'''
def cost_by_closest_city_dist(latlon, alpha, beta):
    closest_city_distance = closest_city(latlon)
    print("Calculating cost...")
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

def retrieve_latlons(latlon_path):
    #Retrieve data
    with open(latlon_path, "rb") as f:
        arrs = dill.load(f)
        
    # get latlons
    return arrs["latlon"]

def retrieve_ids(latlon_path):
    #Retrieve data
    with open(latlon_path, "rb") as f:
        arrs = dill.load(f)
        
    # get latlons
    return  arrs["ids"]

def total_cost_from_latlon_file(label, rule, size, alpha, beta):
    latlon_path = "data/latlons/{label}_sample_{rule}_{size}.pkl".format(label=label, rule=rule, size=size)
    
    if os.path.exists(latlon_path):
        latlons = retrieve_latlons(latlon_path)
    else: 
        latlon_path = "data/int/feature_matrices/CONTUS_UAR_{label}_with_splits.pkl".format(label=label)
        with open(latlon_path, "rb") as f:
            arrs = dill.load(f)
        latlons = arrs["latlons_train"]

    total_cost = cost_array_by_city_dist(latlons, alpha, beta).sum()

    return total_cost

def total_cost_from_dist_file(label, rule, size, cost_func, *params):
    ids_path = "data/latlons_ids/{label}_sample_{rule}_{size}.pkl".format(label=label, rule=rule, size=size)

    #Note that for alpha*(dist to closest city) + beta all greedy lowcost functions will sample the same points
    #Suffices to use the samples from one of them
    #But, need to change if change cost function
    if os.path.exists(ids_path):
        ids = retrieve_ids(ids_path)
    else: 
        ids_path = "data/int/feature_matrices/CONTUS_UAR_{label}_with_splits.pkl".format(label=label)
        with open(ids_path, "rb") as f:
            arrs = dill.load(f)
        ids = arrs["ids_train"]

    dist_path = "data/cost/distance_to_closest_city.pkl"

    #Change depending on cost function
    total_cost = cost_func(dist_path, ids, *params)

    return total_cost

def write_cost_to_csv(old_csv, new_csv, rule, cost_func, *params):
    df = pd.read_csv(old_csv)
    df = df.set_index(['label', 'size_of_subset'])

    for (label, size_of_subset), row in df.iterrows():
        new_cost = total_cost_from_dist_file(label, rule, int(size_of_subset), cost_func, *params)
        df.at[(label, size_of_subset), 'Cost'] = new_cost

    df.index.name = "label"
    df.to_csv(new_csv, index=True)


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

def save_distances(latlon_path, out_fpath):
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
    distances = closest_city_array(latlons)

    # save
    with open(out_fpath, "wb") as f:
        dill.dump(
            {"distances": distances, "ids": ids, "latlon": latlons},
            f,
            protocol=4,
        )

def distance_of_subset(dist_path, ids):
    with open(dist_path, "rb") as f:
        arrs = dill.load(f)

    # get costs
    distances = pd.DataFrame(
        arrs["distances"].astype(np.float64),
        index=arrs["ids"],
        columns=["Distance"],
    )

    dist_of_subset = distances.loc[ids].to_numpy()[:,0]

    return dist_of_subset

def cost_lin(dist_path, ids, *params):
    dist_of_subset = distance_of_subset(dist_path, ids)
    alpha = params[0]
    beta = params[1]
    gamma = params[2]

    #Subject to change
    cost_of_subset = (alpha*(dist_of_subset)**gamma + beta).sum()

    return cost_of_subset

def cost_lin_with_r(dist_path, ids, *params):
    dist_of_subset = distance_of_subset(dist_path, ids)

    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    c = params[3]
    cost_of_subset = 0
    #Subject to change
    for i in range(len(dist_of_subset)):
        if (dist_of_subset[i] <= c):
            cost_of_subset += gamma
        if (dist_of_subset[i] > c):
            cost_of_subset += alpha*(dist_of_subset[i]) + beta

    return cost_of_subset

def cost_bin_r(dist_path, ids, *params):
    dist_of_subset = distance_of_subset(dist_path, ids)

    cost_of_subset = 0
    #Subject to change
    for i in range(len(dist_of_subset)):
        if (dist_of_subset[i] <= c):
            cost_of_subset += params[0]
        if (dist_of_subset[i] > params[2]):
            cost_of_subset += params[1]

    return cost_of_subset

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
    total_cost = np.sum(costs[subset_idxs])
    print("Subset of size {size} costs {cost}".format(size=len(subset_idxs), cost=total_cost))
    return total_cost

'''
    Returns numpy array of cost with same order as training data
'''
def costs_of_train_data(cost_path, ids_train):
    with open(cost_path, "rb") as f:
        arrs = dill.load(f)

    # get costs
    costs = pd.DataFrame(
        arrs["cost"].astype(np.float64),
        index=arrs["ids"],
        columns=["cost"],
    )
    
    cost_train = costs.loc[ids_train].to_numpy()[:,0]

    return cost_train

'''
    Returns numpy array of dist with same order as training data
'''
def dists_of_train_data(dist_path, ids_train):
    with open(dist_path, "rb") as f:
        arrs = dill.load(f)

    # get costs
    distances = pd.DataFrame(
        arrs["distances"].astype(np.float64),
        index=arrs["ids"],
        columns=["distances"],
    )
    
    dist_train = distances.loc[ids_train].to_numpy()[:,0]

    return dist_train

