import os
import dill
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from shapely.ops import nearest_points

STATE_SHP = "../boundaries/us_states_provinces/ne_110m_admin_1_states_provinces.shp"
COUNTY_SHP = "../boundaries/us_county_2015/tl_2015_us_county.shp"
PUMA_SHP = "../boundaries/us_puma_2015/{STATEFP}/tl_2015_{STATEFP}_puma10.shp"
TRACT_SHP = "../boundaries/us_tract_2015/{STATEFP}/tl_2015_{STATEFP}_tract.shp"


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

def process_or_load(latlons, ids, label, year):
    states_fp = f"{label}/gdf_states_{year}.geojson"
    counties_fp = f"{label}/gdf_counties_{year}.geojson"
    pumas_fp = f"{label}/gdf_pumas_{year}.geojson"
    tracts_fp = f"{label}/gdf_tracts_{year}.geojson"

    if os.path.exists(states_fp):
        print(f"Loading states from {states_fp}")
        gdf_points_with_states = gpd.read_file(states_fp)
    else:
        print("Generating states...")
        gdf_points_with_states = add_states_to_points(latlons, ids)
        gdf_points_with_states.to_file(states_fp, driver="GeoJSON")

    if os.path.exists(counties_fp):
        print(f"Loading counties from {counties_fp}")
        gdf_points_with_counties = gpd.read_file(counties_fp)
    else:
        print("Generating counties...")
        gdf_points_with_counties = add_counties_to_points(gdf_points_with_states)
        gdf_points_with_counties.to_file(counties_fp, driver="GeoJSON")

    if os.path.exists(pumas_fp):
        print(f"Loading pumas from {pumas_fp}")
        gdf_points_with_pumas = gpd.read_file(pumas_fp)
    else:
        print("Generating pumas...")
        gdf_points_with_pumas = add_puma_to_points(gdf_points_with_states)
        gdf_points_with_pumas.to_file(pumas_fp, driver="GeoJSON")

    # if os.path.exists(tracts_fp):
    #     print(f"Loading tracts from {tracts_fp}")
    #     gdf_points_with_tracts = gpd.read_file(tracts_fp)
    # else:
    #     print("Generating tracts...")
    #     gdf_points_with_tracts = add_tracts_to_points(gdf_points_with_counties)
    #     gdf_points_with_tracts.to_file(tracts_fp, driver="GeoJSON")

    return gdf_points_with_states, gdf_points_with_counties, gdf_points_with_pumas #, gdf_points_with_tracts

def add_states_to_points(latlons, ids):
    print("Loading states shapefile...")
    gdf_states = gpd.read_file(STATE_SHP).to_crs("EPSG:4326")

    points = [Point(lon, lat) for lat, lon in latlons]
    gdf_points_with_states = gpd.GeoDataFrame(
        {'id': ids},
        geometry=points,
        crs="EPSG:4326"
    )

    print(f"Performing spatial join for {len(points)} points to find states...")
    gdf_points_with_states = gpd.sjoin(
        gdf_points_with_states,
        gdf_states[['geometry', 'fips', 'name']],
        how='left',
        predicate='within'
    )

    gdf_points_with_states['fips'] = gdf_points_with_states['fips'].str.replace('US', '', regex=False)
    gdf_points_with_states = gdf_points_with_states.rename(columns={'name': 'STATE_NAME', 'fips': 'STATEFP'})
    gdf_points_with_states = gdf_points_with_states.drop(columns=['index_right']).reset_index(drop=True)

    missing_mask = gdf_points_with_states['STATEFP'].isna()
    missing_points = gdf_points_with_states[missing_mask]

    if not missing_points.empty:
        print(f"Assigning nearest states to {len(missing_points)} unmatched points...")

        for idx, row in missing_points.iterrows():
            nearest_geom = get_nearest_polygon_index(row.geometry, gdf_states)
            nearest_state = gdf_states.loc[nearest_geom]
            gdf_points_with_states.at[idx, 'STATEFP'] = nearest_state['fips'].replace('US', '')
            gdf_points_with_states.at[idx, 'STATE_NAME'] = nearest_state['name']

    return gdf_points_with_states

def add_counties_to_points(gdf_points_with_states):
    print("Loading counties shapefile...")
    gdf_counties = gpd.read_file(COUNTY_SHP)
    exclude_statefps = ['02', '15', '60', '66', '69', '72', '78']

    gdf_counties = gdf_counties[~gdf_counties['STATEFP'].isin(exclude_statefps)]
    gdf_counties = gdf_counties.to_crs("EPSG:4326")

    gdf_points_with_counties = gdf_points_with_states.copy()

    gdf_points_with_counties['COUNTYFP'] = None
    gdf_points_with_counties['COUNTY_NAME'] = None

    unique_states = gdf_points_with_counties['STATEFP'].dropna().unique()
    print(f"Processing counties for {len(unique_states)} states...")

    for state_fp in unique_states:
        counties_sub = gdf_counties[gdf_counties['STATEFP'] == state_fp]
        points_sub = gdf_points_with_states[gdf_points_with_states['STATEFP'] == state_fp]

        print(f"State {state_fp}: {len(points_sub)} points, {len(counties_sub)} counties")

        if counties_sub.empty or points_sub.empty:
            print(f"Skipping state {state_fp} due to empty counties or points subset.")
            continue

        joined = gpd.sjoin(points_sub, counties_sub[['geometry', 'COUNTYFP', 'NAME']], how='left', predicate='within')

        gdf_points_with_counties.loc[joined.index, 'COUNTYFP'] = joined['COUNTYFP']
        gdf_points_with_counties.loc[joined.index, 'COUNTY_NAME'] = joined['NAME']

    missing = gdf_points_with_counties[gdf_points_with_counties['COUNTYFP'].isna()]
    for idx, row in missing.iterrows():
        nearest_geom = get_nearest_polygon_index(row.geometry, gdf_counties)

        gdf_points_with_counties.at[idx, 'COUNTYFP'] = gdf_counties.loc[nearest_geom, 'COUNTYFP']
        gdf_points_with_counties.at[idx, 'COUNTY_NAME'] = gdf_counties.loc[nearest_geom, 'NAME']

    return gdf_points_with_counties

def add_puma_to_points(gdf_points_with_states):
    print("Adding PUMA info...")
    gdf_points_with_puma = gdf_points_with_states.copy()
    gdf_points_with_puma['PUMACE10'] = None

    unique_states = gdf_points_with_puma['STATEFP'].dropna().unique()
    print(f"Processing PUMA for {len(unique_states)} states...")

    for state_fp in unique_states:
        state_fp_str = str(state_fp).zfill(2)
        puma_path = PUMA_SHP.format(STATEFP=state_fp_str)
        print(f"Loading PUMA shapefile for state {state_fp_str} from {puma_path}")

        try:
            gdf_pumas = gpd.read_file(puma_path).to_crs("EPSG:4326")
        except Exception as e:
            print(f"Warning: Could not load PUMA shapefile for state {state_fp_str}: {e}")
            continue

        points_sub = gdf_points_with_states[gdf_points_with_states['STATEFP'] == state_fp]

        print(f"State {state_fp}: {len(points_sub)} points, {len(gdf_pumas)} PUMA polygons")

        if points_sub.empty or gdf_pumas.empty:
            print(f"Skipping state {state_fp} due to empty points or PUMA data.")
            continue

        joined = gpd.sjoin(points_sub, gdf_pumas[['geometry', 'PUMACE10']], how='left', predicate='within')

        gdf_points_with_puma.loc[joined.index, 'PUMACE10'] = joined['PUMACE10']

        missing = gdf_points_with_puma[gdf_points_with_puma['PUMACE10'].isna() & (gdf_points_with_puma['STATEFP'] == state_fp)]
        for idx, row in missing.iterrows():
            nearest_geom = get_nearest_polygon_index(row.geometry, gdf_pumas)

            gdf_points_with_puma.at[idx, 'PUMACE10'] = gdf_pumas.loc[nearest_geom, 'PUMACE10']

    return gdf_points_with_puma

def add_tracts_to_points(gdf_points_with_counties):
    print("Adding tract info...")
    gdf_points_with_tract = gdf_points_with_counties.copy()

    gdf_points_with_tract['TRACTCE'] = None

    unique_states = gdf_points_with_tract['STATEFP'].dropna().unique()
    print(f"Processing tracts for {len(unique_states)} states...")

    for state_fp in unique_states:
        state_fp_str = str(state_fp).zfill(2)
        tract_path = TRACT_SHP.format(STATEFP=state_fp_str)
        print(f"Loading tract shapefile for state {state_fp_str} from {tract_path}")

        try:
            gdf_tracts = gpd.read_file(tract_path).to_crs("EPSG:4326")
        except Exception as e:
            print(f"Warning: Could not load tract shapefile for state {state_fp_str}: {e}")
            continue

        points_state = gdf_points_with_counties[gdf_points_with_counties['STATEFP'] == state_fp]
        unique_counties = points_state['COUNTYFP'].dropna().unique()
        print(f"State {state_fp}: {len(points_state)} points, {len(unique_counties)} counties")

        for county_fp in unique_counties:
            tracts_sub = gdf_tracts[gdf_tracts['COUNTYFP'] == county_fp]
            points_sub = points_state[points_state['COUNTYFP'] == county_fp]

            print(f"County {county_fp}: {len(points_sub)} points, {len(tracts_sub)} tracts")

            if tracts_sub.empty or points_sub.empty:
                print(f"Skipping county {county_fp} due to empty tracts or points subset.")
                continue

            joined = gpd.sjoin(points_sub, tracts_sub[['geometry', 'TRACTCE']], how='left', predicate='within')

            gdf_points_with_tract.loc[joined.index, 'TRACTCE'] = joined['TRACTCE']

            missing = gdf_points_with_tract[
                gdf_points_with_tract['TRACTCE'].isna() &
                (gdf_points_with_tract['STATEFP'] == state_fp) &
                (gdf_points_with_tract['COUNTYFP'] == county_fp)
            ]
            for idx, row in missing.iterrows():
                if tracts_sub.empty:
                    continue
                nearest_geom = get_nearest_polygon_index(row.geometry, tracts_sub)

                gdf_points_with_tract.at[idx, 'TRACTCE'] = tracts_sub.loc[nearest_geom, 'TRACTCE']

    return gdf_points_with_tract

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
    for label in ["income", "population", "treecover"]:
        year = 2015

        with open(f"../data/int/feature_matrices/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl", "rb") as f:
            arrs = dill.load(f)
        
        invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
        ids_train = arrs['ids_train']
        valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
        ids = ids_train[valid_idxs]

        latlons = arrs['latlons_train'][valid_idxs]
        
        gdf_states, gdf_counties, gdf_pumas = process_or_load(latlons, ids, label, year)

        counts_state = counts_per_division(gdf_states, 'STATEFP')
        plot_points_distribution(label, counts_state, "state", 'STATEFP')

        counts_county = counts_per_division(gdf_counties, 'COUNTYFP')
        plot_points_distribution(label, counts_county, "county", 'COUNTYFP', log_scale=False)

        counts_puma = counts_per_division(gdf_pumas, 'PUMACE10')
        plot_points_distribution(label, counts_puma, "puma", 'PUMACE10', log_scale=False)

        # counts_tract = counts_per_division(gdf_tracts, 'TRACTCE')
        # plot_points_distribution(label, counts_tract, "tract", 'TRACTCE', log_scale=False)