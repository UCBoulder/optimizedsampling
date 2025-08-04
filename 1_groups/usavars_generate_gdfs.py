import os
import dill
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from shapely.ops import nearest_points

COUNTY_SHP = "../../0_data/boundaries/us/us_county_2015"


def get_nearest_polygon_index(point, gdf, buffer_degrees=0.5):
    assert gdf.crs.to_string() == "EPSG:4326"
    assert isinstance(point, Point)
    
    lon, lat = point.x, point.y
    bbox = box(lon - buffer_degrees, lat - buffer_degrees,
               lon + buffer_degrees, lat + buffer_degrees)
    candidates = gdf[gdf.intersects(bbox)]

    if candidates.empty:
        candidates = gdf
        #raise ValueError("No polygons found within the bounding box.")

    distances = candidates.geometry.apply(
        lambda poly: geodesic(
            (point.y, point.x),
            (nearest_points(point, poly)[1].y, nearest_points(point, poly)[1].x)
        ).meters
    )

    return distances.idxmin()

def process_or_load_counties(gdf_points, label, year):
    counties_fp = f"../../0_data/admin_gdfs/usavars/{label}/gdf_counties_{year}.geojson"
    gdf_counties = load_counties_with_combined_id(COUNTY_SHP)

    if os.path.exists(counties_fp):
        print(f"Loading counties from {counties_fp}")
        gdf_points_with_counties = gpd.read_file(counties_fp)
    else:
        print("Generating counties...")
        gdf_points_with_counties = add_counties_to_points(gdf_points, gdf_counties)
        gdf_points_with_counties.to_file(counties_fp, driver="GeoJSON")

    return gdf_points_with_counties

def load_counties_with_combined_id(county_shp):
    gdf_counties = gpd.read_file(county_shp)

    gdf_counties['STATEFP'] = gdf_counties['STATEFP'].astype(str).str.zfill(2)
    gdf_counties['COUNTYFP'] = gdf_counties['COUNTYFP'].astype(str).str.zfill(3)
    gdf_counties['COUNTY_NAME'] = gdf_counties['NAME']

    if 'STATE_NAME' not in gdf_counties.columns:
        state_fips_to_name = {
            '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas', '06': 'California',
            '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware', '11': 'District of Columbia',
            '12': 'Florida', '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois',
            '18': 'Indiana', '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana',
            '23': 'Maine', '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
            '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska', '32': 'Nevada',
            '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico', '36': 'New York',
            '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio', '40': 'Oklahoma', '41': 'Oregon',
            '42': 'Pennsylvania', '44': 'Rhode Island', '45': 'South Carolina', '46': 'South Dakota',
            '47': 'Tennessee', '48': 'Texas', '49': 'Utah', '50': 'Vermont', '51': 'Virginia',
            '53': 'Washington', '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming',
        } 
        gdf_counties['STATE_NAME'] = gdf_counties['STATEFP'].map(state_fips_to_name)

    gdf_counties['combined_county_id'] = (
        gdf_counties['COUNTY_NAME'].astype(str) + "_" +
        gdf_counties['COUNTYFP'].astype(str) + "_" +
        gdf_counties['STATE_NAME'].astype(str) + "_" +
        gdf_counties['STATEFP'].astype(str)
    )

    return gdf_counties

def add_counties_to_points(gdf_points, gdf_counties):
    gdf_points = gdf_points.copy()
    gdf_points['combined_county_id'] = None

    print(f"Processing {len(gdf_points)} points and {len(gdf_counties)} counties...")

    gdf_points = gdf_points.to_crs("EPSG:4326")
    gdf_counties = gdf_counties.to_crs("EPSG:4326")

    joined = gpd.sjoin(
        gdf_points,
        gdf_counties[['geometry', 'combined_county_id']],
        how='left',
        predicate='within'
    )
    gdf_points.loc[joined.index, 'combined_county_id'] = joined['combined_county_id_right']

    num_missing = gdf_points['combined_county_id'].isna().sum()

    if num_missing > 0:
        print(f"{num_missing} points missing 'combined_county_id'. Filling using nearest polygon...")

        for idx, row in gdf_points[gdf_points['combined_county_id'].isna()].iterrows():
            nearest_idx = get_nearest_polygon_index(row.geometry, gdf_counties)
            gdf_points.at[idx, 'combined_county_id'] = gdf_counties.loc[nearest_idx, 'combined_county_id']

        print("Missing values filled using nearest county geometry.")
    else:
        print("All points have a 'combined_county_id'.")

    return gdf_points


def counts_per_division(gdf, division_col):
    counts = gdf[division_col].value_counts()
    avg_count = counts.mean()
    median_count = counts.median()
    min_count = counts.min()
    max_count = counts.max()
    
    print(f"\n=== {division_col} ===")
    print(f"Average: {avg_count:.2f}")
    print(f"Median:  {median_count}")
    print(f"Min:     {min_count}")
    print(f"Max:     {max_count}")
    
    return counts

def plot_points_distribution(label, counts, division_type, division_col, log_scale=False):
    plt.figure(figsize=(10,6))
    counts.plot(kind='hist', bins=100, alpha=0.7, color='skyblue')
    plt.title(f'Distribution of Number of Points per {division_col}', fontsize=14)
    plt.xlabel('Number of Points')
    plt.ylabel('Number of Divisions')
    plt.yscale('log')
    if log_scale:
        plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{label}/plots/{division_type}_hist.png", dpi=300)

if __name__ == "__main__":
    for label in ["population", "treecover"]:
        year = 2015

        with open(f"../../0_data/features/usavars/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl", "rb") as f:
            arrs = dill.load(f)
        
        invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
        ids_train = arrs['ids_train']
        valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
        ids = ids_train[valid_idxs]

        latlons = arrs['latlons_train'][valid_idxs]

        points = [Point(lon, lat) for lat, lon in latlons]

        df = pd.DataFrame({'id': ids})

        gdf = gpd.GeoDataFrame(df, geometry=points, crs="EPSG:4326")
        
        gdf_counties = process_or_load_counties(gdf, label, year)