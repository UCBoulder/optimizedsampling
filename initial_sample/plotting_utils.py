import os
import geopandas as gpd
import matplotlib.pyplot as plt

def plot_sampled_points_on_map(
    gdf_points, sampled_gdf, sample_size, country_shape_file, title, save_path,
    country_name=None, exclude_names=None, sampled_color='#d62728',
    legend_kwargs=None, equal_aspect=False, title_pad=20, use_tight_layout=True,
):
    fig, ax = plt.subplots(figsize=(12, 10))
    country = gpd.read_file(country_shape_file, engine="pyogrio")
    if country_name is not None:
        country = country[country['NAME'] == country_name]
    if exclude_names:
        country = country[~country['name'].isin(exclude_names)]
    country = country.to_crs('EPSG:4326')

    country.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=4, alpha=0.8)
    gdf_points.plot(ax=ax, color='#cccccc', markersize=5, label='All Points', zorder=1, alpha=0.6)
    sampled_gdf.plot(ax=ax, color=sampled_color, markersize=5, label=f'Sampled ({sample_size})', zorder=3, alpha=0.8)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=title_pad)
    ax.legend(**(legend_kwargs or {'loc': 'lower left', 'fontsize': 10}))
    if equal_aspect:
        ax.set_aspect('equal')
    ax.axis('off')
    if use_tight_layout:
        plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
