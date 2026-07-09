import os
import argparse
import dill
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from unidecode import unidecode
from geo_utils import get_nearest_polygon_index, counts_per_division, plot_points_distribution

ADMIN2_CANONICAL = {
    'tandjouare': 'tandjoare',
    'binah': 'bimah',
    'centre': 'centrale',
    'tchaoudjo': 'tchaudjo',
    'mo': 'plaine du mo',
}

def normalize_admin2(val):
    val = unidecode(str(val)).lower().strip()
    return ADMIN2_CANONICAL.get(val, val)

def process_or_load_adm3(gdf_points, adm3_shp, adm3_out_path):
    if os.path.exists(adm3_out_path):
        return gpd.read_file(adm3_out_path)

    gdf_adm3 = load_adm3_admin_columns(adm3_shp)
    gdf_points_with_adm3 = add_adm3_to_points(gdf_points, gdf_adm3)
    gdf_points_with_adm3.to_file(adm3_out_path, driver="GeoJSON")
    return gdf_points_with_adm3

def load_adm3_admin_columns(adm3_shp):
    gdf_adm3 = gpd.read_file(adm3_shp)
    gdf_adm3['admin_3'] = gdf_adm3["ADM3_FR"]
    return gdf_adm3[["admin_3", "geometry"]].copy()

def add_adm3_to_points(gdf_points, gdf_adm3):
    gdf_points = gdf_points.copy()
    gdf_points['admin_3'] = None

    gdf_points = gdf_points.to_crs("EPSG:4326")
    gdf_adm3 = gdf_adm3.to_crs("EPSG:4326")

    joined = gpd.sjoin(gdf_points, gdf_adm3[['geometry', 'admin_3']], how='left', predicate='within')
    gdf_points.loc[joined.index, 'admin_3'] = joined['admin_3_right']
    gdf_points["admin_3"] = (
        gdf_points["admin_3"].astype(str).str.strip() + "__" +
        gdf_points["admin_2"].astype(str).str.strip()
    )

    for idx, row in gdf_points[gdf_points['admin_3'].isna()].iterrows():
        subset = gdf_adm3[(gdf_adm3["admin_1"] == row["admin_1"]) & (gdf_adm3["admin_2"] == row["admin_2"])]
        if subset.empty:
            subset = gdf_adm3[gdf_adm3["admin_1"] == row["admin_1"]]
            if subset.empty:
                subset = gdf_adm3
        nearest_idx = get_nearest_polygon_index(point=row.geometry, gdf=subset, buffer_degrees=0.5)
        gdf_points.at[idx, 'admin_3'] = gdf_adm3.loc[nearest_idx, 'admin_3']

    return gdf_points

def verify_point_admin_nesting(gdf_points):
    admin3_to_admin2_counts = gdf_points.groupby("admin_3")["admin_2"].nunique()
    admin2_to_admin1_counts = gdf_points.groupby("admin_2")["admin_1"].nunique()
    return admin3_to_admin2_counts[admin3_to_admin2_counts > 1], admin2_to_admin1_counts[admin2_to_admin1_counts > 1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_pkl", required=True)
    parser.add_argument("--soil_csv", required=True)
    parser.add_argument("--adm3_shp", required=True)
    parser.add_argument("--adm3_out_path", required=True)
    args = parser.parse_args()

    with open(args.features_pkl, "rb") as f:
        arrs = dill.load(f)

    ids = arrs['ids_train']
    latlons = list(zip(arrs['lats_train'], arrs['lons_train']))

    df_soil = pd.read_csv(args.soil_csv).set_index('unique_id')
    admin_1 = [df_soil.loc[uid, 'admin_1'] if uid in df_soil.index else None for uid in ids]
    admin_2 = [df_soil.loc[uid, 'admin_2'] if uid in df_soil.index else None for uid in ids]

    points = [Point(lon, lat) for lat, lon in latlons]
    geo_df = gpd.GeoDataFrame(pd.DataFrame({'id': ids, 'admin_1': admin_1, 'admin_2': admin_2}), geometry=points, crs="EPSG:4326")
    geo_df['admin_1'] = geo_df['admin_1'].apply(normalize_admin2)
    geo_df['admin_2'] = geo_df['admin_2'].apply(normalize_admin2)

    gdf_adm3 = process_or_load_adm3(geo_df, args.adm3_shp, args.adm3_out_path)
