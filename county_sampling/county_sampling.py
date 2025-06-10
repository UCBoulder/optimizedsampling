import os
import ast
import re
import dill
import math
import numpy as np
import pandas as pd
from shapely.geometry import Point, box
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_latlon_str(s):
    # Remove 'np.float32' wrappers
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', s)
    # Evaluate and convert to tuple of np.float32
    return tuple(map(np.float32, ast.literal_eval(cleaned)))

def haversine_dist(lat1, lon1, lat2, lon2):
    """Compute haversine distance in kilometers."""
    R = 6371  # Earth radius in km
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def plot_counties(all_points, selected_points, total_counties_to_sample, points_per_county=np.nan, clustered=True, radius_km=10, seed=42, density=False, label="treecover"):
    fig, ax = plt.subplots(figsize=(12, 10))

    world = gpd.read_file("../country_boundaries/ne_110m_admin_1_states_provinces.shp", engine="pyogrio")
    exclude_states = ["Alaska", "Hawaii", "Puerto Rico"]
    contiguous_us = world[~world["name"].isin(exclude_states)]
    contiguous_outline = contiguous_us.dissolve()

    contiguous_outline.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=4, alpha=0.8)

    all_points.plot(ax=ax, color='#cccccc', markersize=5, label='All Points', zorder=1, alpha=0.6)

    legend_label = f'Sampled Points ({len(selected_points):,})'
    selected_points.plot(ax=ax, color='#d62728', markersize=5, label=legend_label, zorder=3, alpha=0.8)

    title_str = f'Geo-Spatial Clustering of Sampled Points\n({total_counties_to_sample} counties sampled, {radius_km} km radius clusters)' if clustered else f'Geo-Spatial Clustering of Sampled Points\n({total_counties_to_sample} counties sampled, {points_per_county} points per county)'
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    ax.legend(loc='lower left', fontsize=10, title='Legend', title_fontsize=11, frameon=True)

    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    type_str = 'density' if density else 'clustered'
    save_str = f'{label}/{type_str}/plots/{total_counties_to_sample}_counties_radius_{radius_km}.png' if np.isnan(points_per_county) else 'test.png'
    plt.savefig(save_str, dpi=300, bbox_inches='tight')
    plt.close()

def load_and_filter_counties(county_shapefile_path):
    gdf_counties = gpd.read_file(county_shapefile_path)

    if gdf_counties.crs != "EPSG:4326":
        gdf_counties = gdf_counties.to_crs("EPSG:4326")

    exclude_state_fps = ["02", "15", "60", "66", "69", "72", "78"]  # Non-CONTUS
    return gdf_counties[~gdf_counties['STATEFP'].isin(exclude_state_fps)].reset_index(drop=True)

def create_geodataframe_from_latlons(latlons):
    points = [Point(lon, lat) for lat, lon in latlons]
    return gpd.GeoDataFrame({'geometry': points}, crs='EPSG:4326')

def find_containing_county(point, counties_gdf):
    for idx, county_geom in enumerate(counties_gdf.geometry):
        if county_geom.contains(point):
            return idx
    return None

def assign_counties_to_points(gdf_points, gdf_counties, save_path, latlons):
    tqdm.pandas(desc="Processing points")
    gdf_points["county_idx"] = [
        find_containing_county(geom, gdf_counties) for geom in gdf_points.geometry
    ]
    gdf_points_with_county = gdf_points.merge(
        gdf_counties.reset_index(),
        left_on="county_idx",
        right_on="index",
        how="left",
        suffixes=("", "_county")
    )
    gdf_points_with_county['latlon'] = [tuple(latlon) for latlon in latlons]  # when first creating it
    try:
        gdf_points_with_county.to_file(save_path, driver="GeoJSON")
    except Exception as e:
        from IPython import embed; embed()
    return gdf_points_with_county

def load_or_generate_county_assignments(gdf_points, gdf_counties, save_path, latlons):
    if os.path.exists(save_path):
        print(f"Loading county GeoDataFrame from {save_path}")
        return gpd.read_file(save_path)
    else:
        print(f"Generating and saving county GeoDataFrame to {save_path}")
        return assign_counties_to_points(gdf_points, gdf_counties, save_path, latlons)

def sample_counties_by_state_proportion(gdf_counties, total_counties_to_sample, seed):
    print("Sampling counties according to number of counties in state...")
    county_counts = gdf_counties['STATEFP'].value_counts()
    county_props = county_counts / county_counts.sum()
    state_num_counties = (county_props * total_counties_to_sample).round().astype(int)

    sampled_counties = []
    for statefp, n_counties in state_num_counties.items():
        state_counties = gdf_counties[gdf_counties['STATEFP'] == str(statefp)]
        if not state_counties.empty:
            sampled = state_counties.sample(min(n_counties, len(state_counties)), random_state=seed)
            sampled_counties.append(sampled)

    return gpd.GeoDataFrame(pd.concat(sampled_counties, ignore_index=True), crs=gdf_counties.crs)

def sample_counties_by_density_proportion(gdf_counties, total_counties_to_sample, seed):
    print("Sampling counties according to population density...")
    df = gdf_counties.copy()
    df = df[(df['ALAND'] > 0) & (~df['2023'].isna())]

    df['pop_density'] = df['2023'] / (df['ALAND'] / 1e6)

    df = df[df['pop_density'].notna() & df['pop_density'].apply(np.isfinite)]

    probs = df['pop_density'] / df['pop_density'].sum()

    sampled = df.sample(n=total_counties_to_sample, weights=probs, random_state=seed)

    return gpd.GeoDataFrame(sampled, crs=gdf_counties.crs)

def sample_points_from_sampled_counties(sampled_counties, gdf_points_with_county, points_per_county, seed):
    selected_points = []

    for geoid in sampled_counties.GEOID:
        points_in_county = gdf_points_with_county[gdf_points_with_county['GEOID'] == geoid]

        if len(points_in_county) >= points_per_county:
            sampled_points = points_in_county.sample(points_per_county, random_state=seed)
        else:
            sampled_points = points_in_county

        selected_points.append(sampled_points)

    selected_points = gpd.GeoDataFrame(pd.concat(selected_points, ignore_index=True), crs=gdf_points_with_county.crs)
    selected_points['latlon'] = selected_points['latlon'].apply(parse_latlon_str) #convert latlon to tuple

    return selected_points

def sample_clustered_points_from_sampled_counties(sampled_counties, gdf_points_with_county, radius_km=10, seed=42):
    selected_points = []

    for geoid in sampled_counties.GEOID:
        points_in_county = gdf_points_with_county[gdf_points_with_county['GEOID'] == geoid]
        if points_in_county.empty:
            continue

        # Sample one center point
        center_point = points_in_county.sample(1, random_state=seed)
        center_lat = center_point.geometry.y.values[0]
        center_lon = center_point.geometry.x.values[0]

        # Compute haversine distance to all points
        lats = points_in_county.geometry.y.values
        lons = points_in_county.geometry.x.values
        dists = haversine_dist(center_lat, center_lon, lats, lons)

        within_radius = points_in_county[dists <= radius_km]
        selected_points.append(within_radius)

    selected_points = gpd.GeoDataFrame(pd.concat(selected_points, ignore_index=True), crs=gdf_points_with_county.crs)
    selected_points['latlon'] = selected_points['latlon'].apply(parse_latlon_str) #convert latlon to tuple

    return selected_points

def sample_points_from_counties(latlons, total_counties_to_sample, points_per_county=np.nan, radius_km=None, seed=42, sampling_method='radius', save_sampled_points=False, density=False, label="treecover"):
    np.random.seed(seed)

    gdf_counties = load_and_filter_counties("../country_boundaries/us_county/counties_with_population.shp")
    gdf_points = create_geodataframe_from_latlons(latlons)

    save_path = f'{label}/gdf_county.geojson'
    gdf_points_with_county = load_or_generate_county_assignments(gdf_points, gdf_counties, save_path, latlons)

    if density:
        print("Using density sampling...")
        sampled_counties = sample_counties_by_density_proportion(gdf_counties, total_counties_to_sample, seed)
    else:
        print("Using sampling by state proportions...")
        sampled_counties = sample_counties_by_state_proportion(gdf_counties, total_counties_to_sample, seed)

    # Handle the sampling based on the chosen method
    if sampling_method == 'radius' and radius_km is not None:
        sampled_points = sample_clustered_points_from_sampled_counties(sampled_counties, gdf_points_with_county, radius_km=radius_km, seed=seed)
    elif sampling_method == 'points_per_county' and not np.isnan(points_per_county):
        sampled_points = sample_points_from_sampled_counties(sampled_counties, gdf_points_with_county, points_per_county, seed)
    else:
        raise ValueError("Invalid combination of sampling parameters.")

    type_str = 'density' if density else 'clustered'
    if save_sampled_points:
        # Adjusting the output filename to include the sampling method and its parameters
        if sampling_method == 'radius':
            output_path = f'{label}/{type_str}/{total_counties_to_sample}_counties_{radius_km}km_radius_seed_{seed}.geojson'
        elif sampling_method == 'points_per_county':
            output_path = f'{label}/{type_str}/{total_counties_to_sample}_counties_{points_per_county}_points_seed_{seed}.geojson'
        
        print(f"Saving sampled points to {output_path}...")
        sampled_points.to_file(output_path, driver="GeoJSON")
    return gdf_points, sampled_points

def generate_and_save_ids(total_counties_to_sample, radius_km, points_per_county=np.nan, seed=42, plot=True, sampling_method='radius', density=False, label="treecover"):
    with open(f"../data/int/feature_matrices/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl", "rb") as f:
        arrs = dill.load(f)

    invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
    ids_train = arrs['ids_train']
    valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
    ids_train = ids_train[valid_idxs]

    latlon_train = arrs['latlons_train'][valid_idxs]

    all_points_gdf, sampled_points_gdf = sample_points_from_counties(latlon_train, total_counties_to_sample, radius_km=radius_km, points_per_county=points_per_county, sampling_method=sampling_method, seed=seed, save_sampled_points=True, density=density, label=label)
    
    if plot:
        plot_counties(all_points_gdf, sampled_points_gdf, total_counties_to_sample, radius_km=radius_km, seed=seed, density=density, label=label)

    sampled_latlons = sampled_points_gdf['latlon'].tolist()

    latlon_to_idx = {tuple(latlon_train[i]):i for i in range(len(latlon_train))}
    sampled_indices = [latlon_to_idx[latlon] for latlon in sampled_latlons]

    # Extract the corresponding sampled IDs
    sampled_ids = ids_train[sampled_indices]

    type_str = 'density' if density else 'clustered'

    with open(f'{label}/{type_str}/IDs_{total_counties_to_sample}_counties_{radius_km}_radius_seed_{seed}.pkl', 'wb') as f:
        dill.dump(sampled_ids, f)

if __name__ == '__main__':
    for label in ["population", "treecover"]:
        for num_counties in [300, 400, 500]:
            for radius_km in [10]:
                for density in [False, True]:
                    generate_and_save_ids(total_counties_to_sample=num_counties, radius_km=radius_km, plot=True, density=density, label=label)