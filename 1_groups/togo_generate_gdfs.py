import os
import dill
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from shapely.ops import nearest_points
from unidecode import unidecode

ADM3_SHP = "/share/togo/Shapefiles/tgo_admbnda_adm3_inseed_20210107.shp"

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

def get_nearest_polygon_index(point, gdf, buffer_degrees=0.5):
    assert gdf.crs.to_string() == "EPSG:4326"
    assert isinstance(point, Point)
    
    lon, lat = point.x, point.y
    bbox = box(lon - buffer_degrees, lat - buffer_degrees,
               lon + buffer_degrees, lat + buffer_degrees)
    candidates = gdf[gdf.intersects(bbox)]

    if candidates.empty:
        candidates = gdf

    distances = candidates.geometry.apply(
        lambda poly: geodesic(
            (point.y, point.x),
            (nearest_points(point, poly)[1].y, nearest_points(point, poly)[1].x)
        ).meters
    )

    return distances.idxmin()

def process_or_load_adm3(gdf_points):
    all_adm3_path = f"../../0_data/admin_gdfs/togo/gdf_adm3.geojson"
    gdf_adm3 = load_adm3_admin_columns(ADM3_SHP)

    if os.path.exists(all_adm3_path):
        gdf_points_with_adm3 = gpd.read_file(all_adm3_path)
    else:
        print("Generating adm3 assignments...")
        gdf_points_with_adm3 = add_adm3_to_points(gdf_points, gdf_adm3)
        gdf_points_with_adm3.to_file(all_adm3_path, driver="GeoJSON")

    return gdf_points_with_adm3

def load_adm3_admin_columns(adm3_shp):
    gdf_adm3 = gpd.read_file(adm3_shp)
    gdf_adm3['admin_3'] = gdf_adm3["ADM3_FR"]

    return gdf_adm3[["admin_3", "geometry"]].copy()


def add_adm3_to_points(gdf_points, gdf_adm3):
    gdf_points = gdf_points.copy()
    gdf_points['admin_3'] = None

    print(f"Processing {len(gdf_points)} points and {len(gdf_adm3)} admin_3 polygons...")

    # Ensure consistent CRS
    gdf_points = gdf_points.to_crs("EPSG:4326")
    gdf_adm3 = gdf_adm3.to_crs("EPSG:4326")

    # Initial spatial join
    joined = gpd.sjoin(
        gdf_points,
        gdf_adm3[['geometry', 'admin_3']],
        how='left',
        predicate='within'
    )

    gdf_points.loc[joined.index, 'admin_3'] = joined['admin_3_right']
    gdf_points["admin_3"] = (
        gdf_points["admin_3"].astype(str).str.strip() + "__" +
        gdf_points["admin_2"].astype(str).str.strip()
    )

    # Fill in unmatched points using nearest polygon
    num_missing = gdf_points['admin_3'].isna().sum()

    if num_missing > 0:
        print(f"{num_missing} points missing 'admin_3'. Attempting to fill using nearest polygon...")

        for idx, row in gdf_points[gdf_points['admin_3'].isna()].iterrows():
            # Filter polygons by admin_1 and admin_2
            subset = gdf_adm3[
                (gdf_adm3["admin_1"] == row["admin_1"]) &
                (gdf_adm3["admin_2"] == row["admin_2"])
            ]

            if subset.empty:
                from IPython import embed; embed()
                subset = gdf_adm3[gdf_adm3["admin_1"] == row["admin_1"]]
                print(f"Fallback to full admin_1 for point {row['id']}")
                if subset.empty:
                    subset = gdf_adm3  # fallback to full set
                    print(f"Fallback to full gdf_adm3 for point {row['id']}")

            nearest_idx = get_nearest_polygon_index(
                point=row.geometry,
                gdf=subset,
                buffer_degrees=0.5
            )

            gdf_points.at[idx, 'admin_3'] = gdf_adm3.loc[nearest_idx, 'admin_3']

        print("Missing admin_3 values filled using nearest geometry.")
    else:
        print("All points matched to admin_3 polygons successfully.")

    return gdf_points


def verify_point_admin_nesting(gdf_points):
    bad_3 = gdf_points.groupby("admin_3")["admin_2"].nunique()
    bad_2 = gdf_points.groupby("admin_2")["admin_1"].nunique()

    if (bad_3 > 1).any():
        print("Some admin_3 values map to multiple admin_2 in point assignments.")
        print(bad_3[bad_3 > 1])
    if (bad_2 > 1).any():
        print("Some admin_2 values map to multiple admin_1 in point assignments.")
        print(bad_2[bad_2 > 1])



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
    type_str = "2017_Jan_Jun_P20"
    with open(f"../../0_data/features/togo/togo_fertility_data_all_{type_str}.pkl", "rb") as f:
        arrs = dill.load(f)
    
    ids = arrs['ids_train']
    lats_train = arrs['lats_train']
    lons_train = arrs['lons_train']

    latlons = list(zip(lats_train, lons_train))

    df_soil = pd.read_csv("/share/togo/togo_soil_fertility_resampled.csv")


    df_soil = df_soil.set_index('unique_id')
    admin_1 = [df_soil.loc[uid, 'admin_1'] if uid in df_soil.index else None for uid in ids]
    admin_2 = [df_soil.loc[uid, 'admin_2'] if uid in df_soil.index else None for uid in ids]

    points = [Point(lon, lat) for lat, lon in latlons]

    geo_df = gpd.GeoDataFrame(pd.DataFrame({'id': ids, 'admin_1':admin_1, 'admin_2': admin_2}), geometry=points, crs="EPSG:4326")

    geo_df['admin_1'] = geo_df['admin_1'].apply(normalize_admin2)
    geo_df['admin_2'] = geo_df['admin_2'].apply(normalize_admin2)

    gdf_adm3 = process_or_load_adm3(geo_df)
    from IPython import embed; embed()
