import os
import argparse
import dill
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from shapely.ops import nearest_points

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

def process_or_load_counties(gdf_points, label, year, county_shape_file, output_dir):
    """
    Process or load GeoDataFrame of points with assigned counties.

    Args:
        gdf_points (GeoDataFrame): GeoDataFrame of points to assign counties.
        label (str): Label name (used for output filename).
        year (int): Year (used for output filename).
        county_shape_file (str or Path): Path to county shapefile.
        output_dir (str or Path): Directory where output GeoJSON will be saved or loaded from.

    Returns:
        GeoDataFrame: Points GeoDataFrame with counties assigned.
    """
    os.makedirs(output_dir, exist_ok=True)

    counties_fp = os.path.join(output_dir, f"gdf_counties_{label}_{year}.geojson")
    gdf_counties = load_counties_with_combined_id(county_shape_file)

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
    gdf_points['state_name'] = None  # Add state_name column
    gdf_points['state_fp'] = None    # Add state_fp column

    print(f"Processing {len(gdf_points)} points and {len(gdf_counties)} counties...")

    gdf_points = gdf_points.to_crs("EPSG:4326")
    gdf_counties = gdf_counties.to_crs("EPSG:4326")

    # Modified to include state information in the join
    joined = gpd.sjoin(
        gdf_points,
        gdf_counties[['geometry', 'combined_county_id', 'STATE_NAME', 'STATEFP']],
        how='left',
        predicate='within'
    )
    
    # Assign values from the join
    gdf_points.loc[joined.index, 'combined_county_id'] = joined['combined_county_id_right']
    gdf_points.loc[joined.index, 'state_name'] = joined['STATE_NAME']
    gdf_points.loc[joined.index, 'state_fp'] = joined['STATEFP']

    num_missing = gdf_points['combined_county_id'].isna().sum()

    if num_missing > 0:
        print(f"{num_missing} points missing 'combined_county_id'. Filling using nearest polygon...")

        for idx, row in gdf_points[gdf_points['combined_county_id'].isna()].iterrows():
            nearest_idx = get_nearest_polygon_index(row.geometry, gdf_counties)
            gdf_points.at[idx, 'combined_county_id'] = gdf_counties.loc[nearest_idx, 'combined_county_id']
            gdf_points.at[idx, 'state_name'] = gdf_counties.loc[nearest_idx, 'STATE_NAME']
            gdf_points.at[idx, 'state_fp'] = gdf_counties.loc[nearest_idx, 'STATEFP']

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
    parser = argparse.ArgumentParser(description="Process feature splits and assign counties.")
    parser.add_argument(
        "--labels",
        type=str,
        default="population,treecover",
        help="Comma-separated list of labels to process",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="..",
        help="Folder where CONTUS_UAR feature .pkl files are stored",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2015,
        help="Year for county processing",
    )
    parser.add_argument(
        "--county_shp",
        type=str,
        default="../../0_data/boundaries/us/us_county_2015",
        help="Path to counties shapefile (directory or .shp file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save/load processed counties GeoJSON files"
    )

    args = parser.parse_args()

    COUNTY_SHP = args.county_shp  # Define COUNTY_SHP here from args
    INVALID_IDS = ['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439']

    labels = [lbl.strip() for lbl in args.labels.split(",")]

    for label in labels:
        feature_path = f"{args.input_folder}/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl"

        print(f"Loading features for label: {label} from {feature_path}")
        with open(feature_path, "rb") as f:
            arrs = dill.load(f)

        ids_train = arrs["ids_train"]
        latlons_train = arrs["latlons_train"]

        valid_idxs = np.where(~np.isin(ids_train, INVALID_IDS))[0]
        valid_ids = ids_train[valid_idxs]
        valid_latlons = latlons_train[valid_idxs]

        points = [Point(lon, lat) for lat, lon in valid_latlons]

        df = pd.DataFrame({"id": valid_ids})
        gdf = gpd.GeoDataFrame(df, geometry=points, crs="EPSG:4326")

        gdf_counties = process_or_load_counties(gdf, label, args.year, COUNTY_SHP, output_dir=args.output_dir)

        print(f"Processed {len(gdf_counties)} points for label '{label}'")
        print(f"Columns in final GDF: {list(gdf_counties.columns)}")
        
        print("\nSample of processed data:")
        print(gdf_counties[['id', 'combined_county_id', 'state_name', 'state_fp']].head())