import dill
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from utils import distance_of_subset

contus_states = [
    "Alabama", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", 
    "Florida", "Georgia", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", 
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", 
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", 
    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", 
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", 
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]

gdf_states = gpd.read_file("country_boundaries/ne_110m_admin_1_states_provinces.shp")

def compute_unif_cost(ids, **kwargs):
    gamma = kwargs.get('gamma', 1)

    return np.full(len(ids), gamma)

def compute_lin_cost(dist_path, ids, **kwargs):
    dist_of_subset = distance_of_subset(dist_path, ids)
    alpha = kwargs.get('alpha', 1)
    beta = kwargs.get('beta', 1)

    cost_str = f"linear wrt distance with alpha={alpha}, beta={beta}"
    print("Cost function is {cost_str}".format(cost_str=cost_str))

    costs = (alpha*(dist_of_subset) + beta)

    return costs

def compute_lin_w_r_cost(dist_path, ids, **kwargs):
    dist_of_subset = distance_of_subset(dist_path, ids)
    costs = np.empty(len(dist_of_subset), dtype=np.float32)

    alpha = kwargs.get('alpha', 1)
    beta = kwargs.get('beta', 1)
    gamma = kwargs.get('gamma', 1)
    r = kwargs.get('r', 0)

    cost_str = f"linear outside or radius {r} km with alpha={alpha}, beta = {beta}, gamma={gamma}"
    print("Cost function is {cost_str}".format(cost_str=cost_str))

    for i in range(len(dist_of_subset)):
        if (dist_of_subset[i] <= r):
            costs[i] = gamma
        if (dist_of_subset[i] > r):
            costs[i] = alpha*(dist_of_subset[i]) + beta

    return costs

def compute_state_cost(states, latlons):
    points = [Point(lon, lat) for lat, lon in latlons]
    gdf_points = gpd.GeoDataFrame(
        {'geometry': points},
        crs='EPSG:26914'
    )

    state_geom = gdf_states[gdf_states['name'].isin(states)].geometry.unary_union
    gdf_points['in_state'] = gdf_points.geometry.apply(lambda x: x.within(state_geom))

    costs = gdf_points['in_state'].apply(lambda x: 1 if x else np.inf).to_numpy()

    return costs