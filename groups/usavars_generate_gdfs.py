import os
import argparse
import dill
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from geo_utils import get_nearest_polygon_index, counts_per_division, plot_points_distribution

STATE_FIPS_TO_NAME = {
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

INVALID_IDS = ['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439']

def process_or_load_counties(gdf_points, label, year, county_shape_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    counties_fp = os.path.join(output_dir, f"gdf_counties_{label}_{year}.geojson")

    if os.path.exists(counties_fp):
        return gpd.read_file(counties_fp)

    gdf_counties = load_counties_with_combined_id(county_shape_file)
    gdf_points_with_counties = add_counties_to_points(gdf_points, gdf_counties)
    gdf_points_with_counties.to_file(counties_fp, driver="GeoJSON")
    return gdf_points_with_counties

def load_counties_with_combined_id(county_shp):
    gdf_counties = gpd.read_file(county_shp)

    gdf_counties['STATEFP'] = gdf_counties['STATEFP'].astype(str).str.zfill(2)
    gdf_counties['COUNTYFP'] = gdf_counties['COUNTYFP'].astype(str).str.zfill(3)
    gdf_counties['COUNTY_NAME'] = gdf_counties['NAME']

    if 'STATE_NAME' not in gdf_counties.columns:
        gdf_counties['STATE_NAME'] = gdf_counties['STATEFP'].map(STATE_FIPS_TO_NAME)

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
    gdf_points['state_name'] = None
    gdf_points['state_fp'] = None

    gdf_points = gdf_points.to_crs("EPSG:4326")
    gdf_counties = gdf_counties.to_crs("EPSG:4326")

    joined = gpd.sjoin(
        gdf_points,
        gdf_counties[['geometry', 'combined_county_id', 'STATE_NAME', 'STATEFP']],
        how='left',
        predicate='within'
    )

    gdf_points.loc[joined.index, 'combined_county_id'] = joined['combined_county_id_right']
    gdf_points.loc[joined.index, 'state_name'] = joined['STATE_NAME']
    gdf_points.loc[joined.index, 'state_fp'] = joined['STATEFP']

    for idx, row in gdf_points[gdf_points['combined_county_id'].isna()].iterrows():
        nearest_idx = get_nearest_polygon_index(row.geometry, gdf_counties)
        gdf_points.at[idx, 'combined_county_id'] = gdf_counties.loc[nearest_idx, 'combined_county_id']
        gdf_points.at[idx, 'state_name'] = gdf_counties.loc[nearest_idx, 'STATE_NAME']
        gdf_points.at[idx, 'state_fp'] = gdf_counties.loc[nearest_idx, 'STATEFP']

    return gdf_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="population,treecover")
    parser.add_argument("--input_folder", type=str, default="..")
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--county_shp", type=str, default="../../0_data/boundaries/us/us_county_2015")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    labels = [lbl.strip() for lbl in args.labels.split(",")]

    for label in labels:
        feature_path = f"{args.input_folder}/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl"
        with open(feature_path, "rb") as f:
            arrs = dill.load(f)

        ids_train = arrs["ids_train"]
        latlons_train = arrs["latlons_train"]

        valid_idxs = np.where(~np.isin(ids_train, INVALID_IDS))[0]
        valid_ids = ids_train[valid_idxs]
        valid_latlons = latlons_train[valid_idxs]

        points = [Point(lon, lat) for lat, lon in valid_latlons]
        gdf = gpd.GeoDataFrame(pd.DataFrame({"id": valid_ids}), geometry=points, crs="EPSG:4326")

        gdf_counties = process_or_load_counties(gdf, label, args.year, args.county_shp, output_dir=args.output_dir)
