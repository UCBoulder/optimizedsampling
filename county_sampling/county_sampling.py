import os
import dill
import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_counties(all_points, selected_points, total_counties_to_sample, points_per_county=6, seed=42):
    fig, ax = plt.subplots(figsize=(12, 10))

    world = gpd.read_file("../country_boundaries/ne_110m_admin_1_states_provinces.shp", engine="pyogrio")
    exclude_states = ["Alaska", "Hawaii", "Puerto Rico"]
    contiguous_us = world[~world["name"].isin(exclude_states)]
    contiguous_outline = contiguous_us.dissolve()

    contiguous_outline.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=4, alpha=0.8)

    all_points.plot(ax=ax, color='#cccccc', markersize=5, label='All Points', zorder=1, alpha=0.6)

    label = f'Sampled Points ({len(selected_points):,})'
    selected_points.plot(ax=ax, color='#d62728', markersize=5, label=label, zorder=3, alpha=0.8)

    ax.set_title(f'Geo-Spatial Clustering of Sampled Points\n({total_counties_to_sample} counties sampled, {points_per_county} points per county)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    ax.legend(loc='lower left', fontsize=10, title='Legend', title_fontsize=11, frameon=True)

    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig(f'county_sampling_plots/treecover_{total_counties_to_sample} counties sampled, {points_per_county} points per county.png', dpi=300, bbox_inches='tight')
    plt.close()

def sample_points_from_counties(latlons, total_counties_to_sample, points_per_county=6, seed=42, save_sampled_points=False):
    np.random.seed(seed)

    gdf_counties = gpd.read_file("../country_boundaries/us_county/tl_2024_us_county.shp")

    #Remove territories not in CONTUS
    exclude_state_fps = ["02", "15", "60", "66", "69", "72", "78"]
    gdf_counties = gdf_counties[~gdf_counties['STATEFP'].isin(exclude_state_fps)].reset_index(drop=True)

    points = [Point(lon, lat) for lat, lon in latlons]
    gdf_points = gpd.GeoDataFrame({'geometry': points}, crs='EPSG:4326')

    save_path = 'gdf_county.geojson'
    if os.path.exists(save_path):
        print(f"Loading county GeoDataFrame from {save_path}")
        gdf_points_with_county = gpd.read_file(save_path)
    else:
        def find_containing_county(point, counties_gdf):
            for idx, county_geom in enumerate(counties_gdf.geometry):
                if county_geom.contains(point):
                    return idx
            return None
        
        #Apply the function to each point
        gdf_points["county_idx"] = [
            find_containing_county(geom, gdf_counties) 
            for geom in tqdm(gdf_points.geometry, desc="Processing points", unit="point")
        ]

        #Merge back to include county info if needed
        gdf_points_with_county = gdf_points.merge(
            gdf_counties.reset_index(), 
            left_on="county_idx", 
            right_on="index", 
            how="left",
            suffixes=("", "_county")
        )

        print(f"Saving generated county GeoDataFrame to {save_path}...")
        gdf_points_with_county.to_file(save_path, driver="GeoJSON")

    #2. How many counties per state
    county_counts = gdf_counties['STATEFP'].value_counts()
    county_props = county_counts / county_counts.sum()

    #Decide how many counties to sample from each state
    state_num_counties = (county_props * total_counties_to_sample).round().astype(int)

    sampled_counties = []

    for statefp, n_counties in state_num_counties.items():
        state_counties = gdf_counties[gdf_counties['STATEFP'] == statefp]
        if len(state_counties) == 0:
            continue
        sampled = state_counties.sample(min(n_counties, len(state_counties)), random_state=seed)
        sampled_counties.append(sampled)

    sampled_counties = gpd.GeoDataFrame(pd.concat(sampled_counties, ignore_index=True), crs=gdf_counties.crs)

    #3. From each sampled county, sample points
    selected_points = []

    for idx, county_geom in enumerate(sampled_counties.geometry):
        #Points in this county
        points_in_county = gdf_points_with_county[gdf_points_with_county['county_idx'] == idx]

        if len(points_in_county) >= points_per_county:
            sampled_points = points_in_county.sample(points_per_county, random_state=seed)
            selected_points.append(sampled_points)
        else:
            #If not enough points, take all available
            selected_points.append(points_in_county)

    final_points = gpd.GeoDataFrame(pd.concat(selected_points, ignore_index=True), crs=gdf_points.crs)

    points_save_path = f'sampled_points_treecover/{total_counties_to_sample}_counties_{points_per_county}_points_seed_{seed}.geojson'
    if save_sampled_points:
        print(f"Saving sampled points to {points_save_path}...")
        final_points.to_file(points_save_path, driver="GeoJSON")

    return gdf_points, final_points

def generate_and_save_ids(latlons, total_counties_to_sample, points_per_county, seed=42, plot=True):
    all_points_gdf, sampled_points_gdf = sample_points_from_counties(latlons, total_counties_to_sample, points_per_county=points_per_county, seed=seed, save_sampled_points=True)

    if plot:
        plot_counties(all_points_gdf, sampled_points_gdf, total_counties_to_sample, points_per_county=points_per_county, seed=seed)

    sampled_latlons = np.array([(point.y, point.x) for point in sampled_points_gdf.geometry])

    with open("../data/int/feature_matrices/CONTUS_UAR_treecover_with_splits_torchgeo4096.pkl", "rb") as f:
        arrs = dill.load(f)

    invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
    ids_train = arrs['ids_train']
    valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
    ids_train = ids_train[valid_idxs]

    latlon_train = arrs['latlons_train'][valid_idxs]

    #TODO: make more efficient
    def is_within_tolerance(latlon1, latlon2, tolerance=1e-5):
        lat_diff = np.abs(latlon1[0] - latlon2[0])  # Latitude difference
        lon_diff = np.abs(latlon1[1] - latlon2[1])  # Longitude difference
        return lat_diff <= tolerance and lon_diff <= tolerance

    # Create latlon_to_idx dictionary with rounded values
    latlon_to_idx = {tuple(latlon): idx for idx, latlon in enumerate(latlon_train)}

    sampled_indices = []
    for latlon in sampled_latlons:
        latlon_tuple = tuple(latlon)  # Convert to tuple for comparison
        matching_indices = []
        
        # Find all latlons in latlon_to_idx that are within tolerance
        for stored_latlon in latlon_to_idx:
            if is_within_tolerance(latlon_tuple, stored_latlon):
                matching_indices.append(latlon_to_idx[stored_latlon])
        
        # Check if there are multiple matches
        if len(matching_indices) > 1:
            raise ValueError
        
        if matching_indices:
            sampled_indices.append(matching_indices[0])  # Pick the first match

    # Filter out None values (if any)
    sampled_indices = [idx for idx in sampled_indices if idx is not None]
    from IPython import embed; embed()

    # Extract the corresponding sampled IDs
    sampled_ids = ids_train[sampled_indices]

    with open(f'sampled_points_treecover/IDs_{total_counties_to_sample}_counties_{points_per_county}_points_seed_{seed}.pkl', 'wb') as f:
        dill.dump(sampled_ids, f) 

if __name__ == '__main__':
    with open("../data/int/feature_matrices/CONTUS_UAR_treecover_with_splits_torchgeo4096.pkl", "rb") as f:
        arrs = dill.load(f)

    invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])

    ids_train = arrs['ids_train']
    valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]

    latlon_train = arrs['latlons_train'][valid_idxs]

    # all_points, selected_points = sample_points_from_counties(latlon_train, 1000)
    # plot_counties(all_points, selected_points)
    generate_and_save_ids(latlon_train, 500, 4, plot=True)