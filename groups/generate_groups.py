import pickle
import argparse
import geopandas as gpd
import dill
import os
import matplotlib.pyplot as plt
import numpy as np

def make_group_assignment(df, columns, id_col):
    group_ids = df[columns].astype(str).agg("_".join, axis=1)
    return dict(zip(df[id_col].astype(str), group_ids))

def save_dict_to_pkl(d, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(d, f)

def plot_group_assignments(
    gdf_points,
    group_dict,
    dataset,
    group_type,
    id_col='id',
    title=None,
    point_size=5,
    cmap='tab10',
    country_shape_file=None,
    country_name=None,
    exclude_names=None
):
    gdf = gdf_points.copy()
    gdf['group'] = gdf[id_col].astype(str).map({str(k): v for k, v in group_dict.items()})
    gdf = gdf.dropna(subset=['group'])

    fig, ax = plt.subplots(figsize=(12, 10))

    if country_shape_file:
        country = gpd.read_file(country_shape_file, engine="pyogrio")
        if country_name:
            country = country[country['NAME'] == country_name]
        if exclude_names:
            country = country[~country['name'].isin(exclude_names)]
        country = country.to_crs('EPSG:4326')
        country.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=1, alpha=0.8)

    unique_groups = sorted(gdf['group'].unique())
    cmap = plt.get_cmap(cmap, len(unique_groups)+5)

    for i, group in enumerate(unique_groups):
        group_gdf = gdf[gdf['group'] == group]
        group_gdf.plot(ax=ax, markersize=point_size, color=cmap(i*2), label=f"Group {group}", zorder=2)

    ax.legend(title="Groups", fontsize=10, title_fontsize=11)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{dataset}_{group_type}.png", dpi=300)
    plt.close()
    print(f"Saved plot to plots/{dataset}_{group_type}.png")

def main():
    parser = argparse.ArgumentParser(description="Plot group assignments for datasets.")
    parser.add_argument("--datasets", type=str, default="usavars_pop,usavars_tc,india_secc,togo",
                        help="Comma-separated list of datasets to process.")
    parser.add_argument("--gdf_paths", type=str, nargs="+",
                        default=[
                            "../0_data/admin_gdfs/usavars/population/gdf_counties_2015.geojson",
                            "../0_data/admin_gdfs/usavars/population/gdf_counties_2015.geojson",
                            "/share/india_secc/MOSAIKS/train_shrugs_with_admins.geojson",
                            "../0_data/admin_gdfs/togo/gdf_adm3.geojson",
                        ],
                        help="List of paths to GeoDataFrames, in same order as datasets.")
    parser.add_argument("--group_paths", type=str, nargs="+",
                        default=[
                            "../0_data/groups/usavars/population/image_8_cluster_assignments.pkl",
                            "../0_data/groups/usavars/treecover/image_8_cluster_assignments.pkl",
                            "../0_data/groups/india_secc/image_8_cluster_assignments.pkl",
                            "../0_data/groups/togo/image_8_cluster_assignments.pkl",
                        ],
                        help="List of paths to group assignment pkl files, same order as datasets.")
    parser.add_argument("--id_cols", type=str, nargs="+",
                        default=["id", "id", "condensed_shrug_id", "id"],
                        help="List of ID column names, same order as datasets.")
    parser.add_argument("--group_cols", type=str, nargs="+",
                        default=["state_name", "state_name", "admin_1", "admin_1"],
                        help="List of admin column names for grouping, same order as datasets.")
    parser.add_argument("--shape_files", type=str, nargs="+",
                        default=[
                            "../0_data/boundaries/us/us_states_provinces/ne_110m_admin_1_states_provinces.shp",
                            "../0_data/boundaries/us/us_states_provinces/ne_110m_admin_1_states_provinces.shp",
                            "../0_data/boundaries/world/ne_10m_admin_0_countries.shp",
                            "/share/togo/Shapefiles/tgo_admbnda_adm0_inseed_itos_20210107.shp",
                        ],
                        help="List of shapefile paths, same order as datasets.")
    parser.add_argument("--country_names", type=str, nargs="+",
                        default=[None, None, "India", None],
                        help="Country names or None, same order as datasets.")
    parser.add_argument("--exclude_names", type=str, nargs="+",
                        default=["Alaska,Hawaii,Puerto Rico", "Alaska,Hawaii,Puerto Rico", "", ""],
                        help="Comma separated names to exclude, empty string for none, same order as datasets.")
    parser.add_argument("--group_type", type=str, default="admin_groups",
                        help="Group type string to pass to plot_group_assignments.")

    args = parser.parse_args()

    datasets = args.datasets.split(",")
    exclude_lists = [ex.split(",") if ex else [] for ex in args.exclude_names]

    for i, dataset in enumerate(datasets):
        gdf_path = args.gdf_paths[i]
        group_path = args.group_paths[i]
        id_col = args.id_cols[i]
        group_col = args.group_cols[i]  # Get the group column for this dataset
        # shape_file = args.shape_files[i]
        # country_name = args.country_names[i]
        # exclude_names = exclude_lists[i]

        print(f"Processing dataset: {dataset}")
        print(f"Using group column: {group_col}")
        
        gdf_points = gpd.read_file(gdf_path)
        gdf_points = gdf_points.to_crs("EPSG:4326")

        if group_col not in gdf_points.columns:
            print(f"Warning: Column '{group_col}' not found in {dataset}. Available columns: {list(gdf_points.columns)}")
            continue

        # Load group assignments from file 
        # with open(group_path, "rb") as f:
        #     group_dict = dill.load(f)

        group_dict = make_group_assignment(gdf_points, columns=[group_col], id_col=id_col)
        
        output_dir = f"../0_data/groups/{dataset}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = f"{output_dir}/{args.group_type}_assignments.pkl"
        save_dict_to_pkl(group_dict, output_path)
        print(f"Saved group assignments to: {output_path}")
if __name__ == "__main__":
    main()