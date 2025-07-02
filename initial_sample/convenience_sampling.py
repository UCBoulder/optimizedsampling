import dill
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points
import matplotlib.pyplot as plt

def compute_or_load_distances_to_urban(gdf_points, gdf_urban_top):

    load_path = f"/home/libe2152/optimizedsampling/cost/{label}/convenience_sampling/distance_to_top{N_urban}_urban.pkl"

    if os.path.exists(load_path):
        print(f"Loading precomputed distances from {load_path}...")
        with open(load_path, "rb") as f:
            dist_dict = dill.load(f)
        return dist_dict['distances_to_urban_area']


    print("Computing distances to nearest urban areas using spatial index...")
    assert gdf_points.crs.to_string() == "EPSG:4326"
    assert gdf_urban_top.crs.to_string() == "EPSG:4326"

    #precompute R-tree index and geometry mapping
    urban_geoms = gdf_urban_top.geometry.values
    tree = STRtree(urban_geoms)
    geom_to_index = {id(geom): i for i, geom in enumerate(urban_geoms)}

    def arc_distance_to_urban(point):
        #query for nearest candidate urban polygon
        nearest_geom_idx = tree.nearest(point)
        nearest_geom = tree.geometries.take(nearest_geom_idx)
        nearest_pt = nearest_points(point, nearest_geom)[1]
        return geodesic((point.y, point.x), (nearest_pt.y, nearest_pt.x)).meters

    distances = gdf_points.geometry.apply(arc_distance_to_urban).to_numpy()
    print("Finished computing distances.")
    return distances

def convenience_sampling_prioritized_near_urban(gdf_points, urban_areas_gdf, n_samples, method='deterministic'):
    """
    Convenience sampling near urban areas.

    Parameters:
    - gdf_points: GeoDataFrame of candidate points (EPSG:4326)
    - urban_areas_gdf: GeoDataFrame of urban area polygons (EPSG:4326)
    - n_samples: number of points to sample
    - method: 'probabilistic' or 'deterministic'

    Returns:
    - GeoDataFrame of sampled points
    """
    assert method in ["deterministic", "probabilistic"], "Need to specify either deterministic or probabilistic method"

    distances = compute_or_load_distances_to_urban(gdf_points, urban_areas_gdf)
    gdf_points['dist'] = distances

    # Probabilistic method
    if method == 'probabilistic':
        inv_prob = 1 / (gdf_points['dist'] + 1e-6)
        probs = inv_prob / inv_prob.sum()

        sampled = gdf_points.sample(n=n_samples, weights=probs, random_state=42)
        return sampled.reset_index(drop=True)

    # Deterministic method
    elif method == 'deterministic':
        sampled = gdf_points.nsmallest(n_samples, 'dist')
        return sampled.reset_index(drop=True)

    else:
        raise ValueError("Method must be 'probabilistic' or 'deterministic'")

def plot(label, all_points, selected_points, desired_sample_size, n_urban):
    fig, ax = plt.subplots(figsize=(12, 10))

    world = gpd.read_file("../boundaries/us_states_provinces/ne_110m_admin_1_states_provinces.shp", engine="pyogrio")
    exclude_states = ["Alaska", "Hawaii", "Puerto Rico"]
    contiguous_us = world[~world["name"].isin(exclude_states)]
    contiguous_outline = contiguous_us.dissolve()

    contiguous_outline.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=4, alpha=0.8)

    all_points.plot(ax=ax, color='#cccccc', markersize=5, label='All Points', zorder=1, alpha=0.6)

    legend_label = f'Sampled Points ({len(selected_points):,})'
    selected_points.plot(ax=ax, color='#d62728', markersize=5, label=legend_label, zorder=3, alpha=0.8)

    title_str = f'Convenience Sampling\n({n_urban} urban areas; {desired_sample_size} desired sample size)'
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    ax.legend(loc='lower left', fontsize=10, title='Legend', title_fontsize=11, frameon=True)

    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    save_str = f'{label}/convenience_sampling/plots/top{N_urban}_urban_areas_{n_samples}_points.png' 
    plt.savefig(save_str, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    N_urban = 50 
    method = 'probabilistic'

    for n_samples in range(100, 1100, 100):
        for label in ["population", "income", "treecover"]:
            for method in ['deterministic', 'probabilistic']:
                with open(f"../data/int/feature_matrices/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl", "rb") as f:
                    arrs = dill.load(f)

                invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])
                ids_train = arrs['ids_train']
                valid_idxs = np.where(~np.isin(ids_train, invalid_ids))[0]
                ids = ids_train[valid_idxs]
                latlons = arrs['latlons_train'][valid_idxs]

                points = [Point(lon, lat) for lat, lon in latlons]
                gdf_points = gpd.GeoDataFrame({'id': ids}, geometry=points, crs="EPSG:4326")

                urban_areas_path = "/home/libe2152/optimizedsampling/boundaries/us_urban_area_census_2020/tl_2020_us_uac20_with_pop.shp"
                gdf_urban = gpd.read_file(urban_areas_path).to_crs("EPSG:4326")

                gdf_urban_top = gdf_urban.nlargest(N_urban, 'POP').copy()

                sampled_points = convenience_sampling_prioritized_near_urban(
                    gdf_points, gdf_urban_top, n_samples, method='probabilistic'
                )

                print(f"Sampled {len(sampled_points)} points using convenience sampling prioritized near top {N_urban} urban areas.")

                plot(
                    label, 
                    gdf_points, 
                    sampled_points, 
                    n_samples, 
                    N_urban
                )

                sampled_ids = sampled_points['id'].tolist()
                out_path = f"{label}/convenience_sampling/sampled_ids_top{N_urban}_urban_areas_{method}_{n_samples}_points.pkl"
                with open(out_path, "wb") as f:
                    dill.dump(sampled_ids, f)

                print(f"Saved sampled IDs to {out_path}")