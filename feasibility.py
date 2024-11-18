'''
Modeling travel cost/feasibility
'''
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from geopy.distance import geodesic #geodesic distance between two lat/lon

cities= gdf = gpd.read_file("/share/cost/ne_10m_populated_places_simple.shp")

'''
Determine lat/lon and rank of closest city
'''
def closest_city(lat, lon):
    closest_city_distance = float('inf')
    closest_city_rank = 0
    closest_city_lat = 0
    closest_city_lon = 0
    closest_city_name = ''

    for _, city_row in cities.iterrows():
        city_lat = city_row['latitude']
        city_lon = city_row['longitude']
        distance = geodesic((lat,lon), (city_lat, city_lon))
        if distance <= closest_city_distance:
            closest_city_distance = distance
            closest_city_rank = city_row['scalerank']
            closest_city_name = city_row['name']
            closest_city_lat = city_lat
            closest_city_lon = city_lon
    
    closest_city_latlon = [closest_city_lat, closest_city_lon]
    return closest_city_name, closest_city_latlon, closest_city_rank

'''
Use closest city distance to determine cost
'''
def cost_by_city_dist(lat, lon):
    closest_city_name, closest_city_latlon, closest_city_rank = closest_city(lat, lon)
    dist_to_city = geodesic((lat,lon), closest_city_latlon).km

    return (140*closest_city_rank+100 + 10*dist_to_city)
'''
Creates array of costs
'''
def costs_by_city_dist(lats, lons):
    n = len(lats)
    costs = np.empty((n,), dtype=np.float32)

    for i in range(n):
        costs[i] = cost_by_city_dist(lats[i], lons[i])

    return costs


def plot_lat_lon_with_cost(lats, lons, costs, title):
    # Create a GeoDataFrame with costs
    gdf = gpd.GeoDataFrame(
        {'costs': costs},
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