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
    all_adm3_path = f"/home/libe2152/optimizedsampling/0_data/admin_gdfs/togo/gdf_adm3.geojson"
    gdf_adm3 = load_adm3_with_combined_id(ADM3_SHP)

    if os.path.exists(all_adm3_path):
        gdf_points_with_adm3 = gpd.read_file(all_adm3_path)
    else:
        print("Generating adm3 assignments...")
        gdf_points_with_adm3 = add_adm3_to_points(gdf_points, gdf_adm3)
        from IPython import embed; embed()
        gdf_points_with_adm3.to_file(all_adm3_path, driver="GeoJSON")

    return gdf_points_with_adm3

def load_adm3_with_combined_id(adm3_shp):
    gdf_adm3 = gpd.read_file(adm3_shp)

    gdf_adm3["ADM1_FR"] = gdf_adm3["ADM1_FR"].apply(normalize_admin2)
    gdf_adm3["ADM2_FR"] = gdf_adm3["ADM2_FR"].apply(normalize_admin2)

    # Create combined_adm_id column
    gdf_adm3['combined_adm_name'] = (
        gdf_adm3['ADM3_FR'].astype(str) + "_" +
        gdf_adm3['ADM2_FR'].astype(str) + "_" +
        gdf_adm3['ADM1_FR'].astype(str)
    )

    gdf_adm3['combined_adm_id'] = (
        gdf_adm3['ADM3_PCODE'].astype(str) + "_" +
        gdf_adm3['ADM2_PCODE'].astype(str) + "_" +
        gdf_adm3['ADM1_PCODE'].astype(str)
    )

    return gdf_adm3

def add_adm3_to_points(gdf_points, gdf_adm3):
    gdf_points = gdf_points.copy()
    gdf_points['combined_adm_id'] = None

    print(f"Processing {len(gdf_points)} points and {len(gdf_adm3)} adm3...")

    # Spatial join points to adm3 to assign combined_adm_id
    gdf_points = gdf_points.to_crs("EPSG:4326")
    gdf_adm3 = gdf_adm3.to_crs("EPSG:4326")

    # Now perform the spatial join
    joined = gpd.sjoin(
        gdf_points,
        gdf_adm3[['geometry', 'combined_adm_id']],
        how='left',
        predicate='within'
    )

    gdf_points.loc[joined.index, 'combined_adm_id'] = joined['combined_adm_id_right']

    num_missing = gdf_points['combined_adm_id'].isna().sum()

    if num_missing > 0:
        print(f"{num_missing} points missing 'combined_adm_id'. Filling using nearest polygon...")

        for idx, row in gdf_points[gdf_points['combined_adm_id'].isna()].iterrows():
            # Subset gdf_adm3 to matching admin_1 and admin_2 from the point
            gdf_adm2_subset = gdf_adm3[
                (gdf_adm3["ADM1_FR"] == row["admin_1"]) &
                (gdf_adm3["ADM2_FR"] == row["admin_2"])
            ]

            if gdf_adm2_subset.empty:
                gdf_adm2_subset = gdf_adm3[(gdf_adm3["ADM1_FR"] == row["admin_1"])]
                print(gdf_adm2_subset)
                print(row['admin_2'])
                print(gdf_adm2_subset['ADM2_FR'])
                from IPython import embed; embed()
                print(f"No polygons found for point ID {row['id']} in admin_1='{row['admin_1']}', admin_2='{row['admin_2']}'. Using full gdf_adm3 fallback.")
                gdf_adm2_subset = gdf_adm3

            nearest_idx = get_nearest_polygon_index(
                point=row.geometry,
                gdf=gdf_adm2_subset,  # narrowed down admin_2 region
                buffer_degrees=0.5
            )

            gdf_points.at[idx, 'combined_adm_id'] = gdf_adm3.loc[nearest_idx, 'combined_adm_id']

        print("Missing values filled using nearest adm3 geometry within admin_2.")
    else:
        print("All points matched with adm3 polygons correctly.")


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
    from pathlib import Path

    root = Path("/share/togo")

    df = pd.read_csv(root / "togo_soil_fertility_resampled.csv")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("%", "pct")
        .str.replace("/", "")
    )

    df = df.dropna(subset=["lat", "lon"])

    ids = df["unique_id"].astype(str).values
    admin_2 = df["admin_2"].astype(str).values
    admin_1 = df["admin_1"].astype(str).values
    latlons = df[["lat", "lon"]].values

    points = [Point(lon, lat) for lat, lon in latlons]

    geo_df = gpd.GeoDataFrame(pd.DataFrame({'id': ids, 'admin_1':admin_1, 'admin_2': admin_2}), geometry=points, crs="EPSG:4326")

    geo_df['admin_1'] = geo_df['admin_1'].apply(normalize_admin2)
    geo_df['admin_2'] = geo_df['admin_2'].apply(normalize_admin2)

    gdf_adm3 = process_or_load_adm3(geo_df)
