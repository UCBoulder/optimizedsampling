import dill
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box

states = gpd.read_file("/home/libe2152/optimizedsampling/country_boundaries/ne_110m_admin_1_states_provinces.shp")
us_states = states[states['adm0_a3'] == 'USA']

def create_geodataframe_from_latlons(latlons):
    points = [Point(lon, lat) for lat, lon in latlons]
    return gpd.GeoDataFrame({'geometry': points}, crs='EPSG:4326')

def state_assignments(label):
    with open(f"../data/int/feature_matrices/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl", "rb") as f:
        arrs = dill.load(f)

    invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
    ids_train = arrs['ids_train']
    valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
    ids_train = ids_train[valid_idxs]

    latlon_train = arrs['latlons_train'][valid_idxs]

    points_gdf = create_geodataframe_from_latlons(latlon_train)
    points_gdf = points_gdf.to_crs(us_states.crs)
    points_gdf['original_index'] = range(len(points_gdf))

    joined = gpd.sjoin(points_gdf, us_states, how='left', predicate='within')
    joined_sorted = joined.sort_values('original_index')

    assignments = joined_sorted['name'].tolist()

    type_str = 'state'

    with open(f'{label}/{type_str}.pkl', 'wb') as f:
        dill.dump(
            {"ids": ids_train, "assignments": assignments}, f)

if __name__ == "__main__":
    for label in ["population", "treecover"]:
        state_assignments(label)
