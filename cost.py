import dill
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from utils import distance_of_subset

from clusters import retrieve_clusters

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

def compute_state_cost(states, latlons=None, ids=None):
    if latlons is None:
        with open("data/int/feature_matrices/CONTUS_UAR_torchgeo4096.pkl", "rb") as f:
            arrs = dill.load(f)

        latlon_array = arrs['latlon']
        id_array = arrs['ids_X']

        id_to_latlon = {id_: latlon for id_, latlon in zip(id_array, latlon_array)}
        latlons = [id_to_latlon.get(id_, None) for id_ in ids]

    points = [Point(lon, lat) for lat, lon in latlons]
    gdf_points = gpd.GeoDataFrame(
        {'geometry': points},
        crs='EPSG:26914'
    )
    state_geom = gdf_states[gdf_states['name'].isin(states)].geometry.unary_union
    gdf_points['in_state'] = gdf_points.geometry.apply(lambda x: x.within(state_geom))

    costs = gdf_points['in_state'].apply(lambda x: 1 if x else np.inf).to_numpy() #change back to inf

    return costs

'''
Calculates cost based on clusters
    cluster_assignment: cluster for each sample
    cluster_cost: dict of cost for each cluster
'''
def compute_cluster_cost(ids, cluster_type):
    cluster_path = f"data/clusters/{cluster_type}_cluster_assignment.pkl"
    clusters = retrieve_clusters(ids, cluster_path)

    cluster_cost = None
    if cluster_type == 'NLCD':
        cluster_cost = {
            11: 100,
            12: 100,
            21: 1,
            22: 1,
            23: 1,
            24: 1,
            31: 100,
            41: 100,
            42: 100,
            43: 100,
            52: 100,
            71: 100,
            81: 1,
            82: 1,
            90: 100,
            95: 100,
            250: 1e9 #background or unmapped value, dont sample
        }
    if cluster_type == 'NLCD_percentages':
        cluster_cost = {
            0: 1,
            1: 10,
            2: 1,
            3: 10, 
            4: 10, 
            5: 1,
            6: 1,
            7: 10
        }
    if cluster_type =='urban_areas':
        cluster_cost = {
            0: 10,
            1: 1
        }
    assert cluster_cost is not None
    return np.array([cluster_cost[clusters[i]] for i in range(len(ids))])