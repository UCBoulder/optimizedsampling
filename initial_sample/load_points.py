from pathlib import Path

import dill
import geopandas as gpd
import pandas as pd

DATASET_DEFAULTS = {
    "usavars": {
        "labels": ["population", "treecover"],
        "data_dir": "../../0_data",
        "id_col": "id",
        "strata_col": "state_name",
        "cluster_col": ["combined_county_id"],
        "country_shape_file": "../../0_data/boundaries/us/us_states_provinces/ne_110m_admin_1_states_provinces.shp",
        "country_name": None,
        "exclude_names": ["Alaska", "Hawaii", "Puerto Rico"],
    },
    "india_secc": {
        "labels": ["secc_cons_pc_combined"],
        "data_dir": "../data",
        "id_col": "condensed_shrug_id",
        "strata_col": "s_name",
        "cluster_col": ["district_id"],
        "country_shape_file": "../data/india_secc/shapefiles/state/state.shp",
        "country_name": None,
        "exclude_names": None,
    },
}


def load_gdf(dataset: str, data_dir: str, label: str) -> gpd.GeoDataFrame:
    if dataset == "usavars":
        return gpd.read_file(f"{data_dir}/admin_gdfs/usavars/{label}/gdf_counties_2015.geojson")

    with open(Path(data_dir) / "india_secc/splits/India_SECC_with_splits_4000.pkl", "rb") as f:
        data = dill.load(f)
    ids = list(data["ids_train"]) + list(data["ids_test"])
    geometries = list(data["geometries_train"]) + list(data["geometries_test"])
    gdf = gpd.GeoDataFrame({"condensed_shrug_id": ids, "geometry": geometries}, crs="EPSG:4326")

    shapefiles_dir = Path(data_dir) / "india_secc/shapefiles"
    shrids = pd.read_csv(
        Path(data_dir) / "india_secc/MOSAIKS/grouped.csv",
        usecols=["condensed_shrug_id", "shrid"],
        low_memory=False,
    )
    shrid_parts = shrids["shrid"].str.split("-", expand=True)
    shrids["pc11_s_id"] = shrid_parts[1]
    shrids["pc11_d_id"] = shrid_parts[2]
    shrids["district_id"] = shrids["pc11_s_id"] + "-" + shrids["pc11_d_id"]

    state_names = gpd.read_file(shapefiles_dir / "state/state.shp")[["pc11_s_id", "s_name"]]
    district_names = gpd.read_file(shapefiles_dir / "district/district.shp")[["pc11_s_id", "pc11_d_id", "d_name"]]
    shrids = shrids.merge(state_names, on="pc11_s_id", how="left")
    shrids = shrids.merge(district_names, on=["pc11_s_id", "pc11_d_id"], how="left")

    return gdf.merge(shrids, on="condensed_shrug_id", how="left")
