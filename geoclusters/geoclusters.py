import dill
import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm

def sample_geo_clusters(latlons, num_centers, buffer_rad, seed=42, return_all_points=False, save_sampled_points=False):
    #Sample random centers
    idxs = np.arange(len(latlons))
    np.random.seed(seed)
    centroid_latlon_idxs = np.random.choice(idxs, size=num_centers)
    centroid_latlons = latlons[centroid_latlon_idxs]

    points = [Point(lon, lat) for lat, lon in latlons]
    centroid_points = [Point(lon, lat) for lat, lon in centroid_latlons]

    gdf_points = gpd.GeoDataFrame(
        {'geometry': points},
        crs='EPSG:4326'
    )
    gdf_centroids = gpd.GeoDataFrame(
        {'geometry': centroid_points},
        crs='EPSG:4326'
    )

    #1. Project centroid points to a distance-preserving CRS
    gdf_centroids_proj = gdf_centroids.to_crs('EPSG:5070')

    #2. Buffer in meters
    gdf_centroids_proj['buffer'] = gdf_centroids_proj.buffer(buffer_rad)

    #3. Convert buffer back to match gdf_points CRS (EPSG:4326)
    buffer_gdf = gdf_centroids_proj.set_geometry('buffer').to_crs(gdf_points.crs)

    #4. Combine all buffers into one
    combined_buffer = buffer_gdf.unary_union

    #5. Select points within combined buffer
    points_within = gdf_points[gdf_points.geometry.apply(lambda geom: combined_buffer.covers(geom))]

    print(f"Sampled points within a {buffer_rad}m radius of centroids...")

    if save_sampled_points:
        print("Saving sampled points...")
        points_within.to_file(f"sampled_points_population/{num_centers}_centers_{buffer_rad}m_radius_seed_{seed}.geojson", driver="GeoJSON")

    if return_all_points:
        return gdf_points, gdf_centroids, points_within

    return points_within

def plot_geo_clusters(latlons, num_centers=0, buffer_rad=0, **kwargs):
    all_points, center_points, selected_points = sample_geo_clusters(
        latlons, num_centers, buffer_rad, return_all_points=True, **kwargs
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    world = gpd.read_file("../country_boundaries/ne_110m_admin_1_states_provinces.shp", engine="pyogrio")
    exclude_states = ["Alaska", "Hawaii", "Puerto Rico"]
    contiguous_us = world[~world["name"].isin(exclude_states)]
    contiguous_outline = contiguous_us.dissolve()

    contiguous_outline.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=4, alpha=0.8)

    all_points.plot(ax=ax, color='#cccccc', markersize=5, label='All Points', zorder=1, alpha=0.6)
    from IPython import embed; embed()

    label = f'Sampled Points (within {buffer_rad}m) ({len(selected_points):,})'
    selected_points.plot(ax=ax, color='#d62728', markersize=5, label=label, zorder=3, alpha=0.8)

    label = f'Center Points ({len(center_points):,})'
    center_points.plot(ax=ax, color='#1f77b4', marker='x', markersize=20, label=label, zorder=2, alpha=0.9)

    if 'buffer' in center_points.columns:
        # Convert buffer GeoSeries to GeoDataFrame and reproject for plotting
        buffer_gdf = gpd.GeoDataFrame(geometry=center_points['buffer'], crs='EPSG:5070')
        buffer_gdf = buffer_gdf.to_crs('EPSG:4326')
        buffer_gdf.boundary.plot(ax=ax, edgecolor='black', alpha=0.4, linewidth=1, zorder=3, label='3km Buffer')


    ax.set_title(f'Geo-Spatial Clustering of Sampled Points\n({num_centers} centers with {buffer_rad}m Radius Buffers)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    ax.legend(loc='lower left', fontsize=10, title='Legend', title_fontsize=11, frameon=True)

    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig(f'geocluster_plots/geocluster_center_{num_centers}_rad_{buffer_rad}m.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_and_save_ids(latlons, num_centers, buffer_rad, seed=42):
    sampled_points_gdf = sample_geo_clusters(latlons, num_centers, buffer_rad, seed=seed, save_sampled_points=True)

    sampled_latlons = np.array([(point.y, point.x) for point in sampled_points_gdf.geometry])

    with open("../data/int/feature_matrices/CONTUS_UAR_population_with_splits_torchgeo4096.pkl", "rb") as f:
        arrs = dill.load(f)

    invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
    ids_train = arrs['ids_train']
    valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
    ids_train = ids_train[valid_idxs]

    latlon_train = arrs['latlons_train'][valid_idxs]

    latlon_to_idx = {tuple(latlon): idx for idx, latlon in enumerate(latlon_train)}

    sampled_indices = [latlon_to_idx[tuple(latlon)] for latlon in sampled_latlons]

    sampled_ids = ids_train[sampled_indices]

    with open(f'sampled_points_population/IDs_{num_centers}_centers_{buffer_rad}m_radius_seed_{seed}.pkl', 'wb') as f:
        dill.dump(sampled_ids, f) 

if __name__ == '__main__':
    with open("../data/int/feature_matrices/CONTUS_UAR_population_with_splits_torchgeo4096.pkl", "rb") as f:
        arrs = dill.load(f)

    invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])

    ids_train = arrs['ids_train']
    valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]

    latlon_train = arrs['latlons_train'][valid_idxs]
    for num_centers in [1500]:
        for buffer_rad in [10000, 15000, 20000]:
            plot_geo_clusters(latlon_train, num_centers, buffer_rad, save_sampled_points=True)
    # generate_and_save_ids(latlon_train, 500, 5000, 42)
    # generate_and_save_ids(latlon_train, 500, 20000, 42)
    # generate_and_save_ids(latlon_train, 750, 5000, 42)
    # generate_and_save_ids(latlon_train, 1500, 10000, 42)
