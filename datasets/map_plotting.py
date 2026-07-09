import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def plot_geometries_on_map(
    geometries,
    country_shape_file: str,
    country_name: str | None = None,
    exclude_names: list[str] | None = None,
    point_color: str = 'red',
    point_size: float = 5,
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """Plot the given point geometries on a country shapefile."""
    print("Plotting latlon subset...")
    country = gpd.read_file(country_shape_file)

    if country_name is not None and 'NAME' in country.columns:
        country = country[country['NAME'] == country_name]
    if exclude_names:
        country = country[~country['name'].isin(exclude_names)]

    points_gdf = gpd.GeoDataFrame(geometry=geometries, crs='EPSG:4326')

    fig, ax = plt.subplots(figsize=(12, 10))
    country.plot(ax=ax, edgecolor='black', facecolor='none')
    points_gdf.plot(ax=ax, color=point_color, markersize=point_size)

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig
