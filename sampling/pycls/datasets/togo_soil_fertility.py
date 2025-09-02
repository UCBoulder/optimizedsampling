from pathlib import Path
from typing import Sequence
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from torchgeo.datasets.geo import NonGeoDataset

from matplotlib.figure import Figure
from shapely.geometry import Point


class TogoSoilFertility(NonGeoDataset):
    """Togo Soil Fertility dataset."""

class TogoSoilFertility(NonGeoDataset):
    def __init__(
        self,
        root: Path | str,
        identifier: str,
        isTrain: bool = True,
        label_col: str = "p_mgkg",
    ) -> None:
        self.root = Path(root)
        self.split = "train" if isTrain else "test"
        self.label_col = label_col

        import dill
        pkl_path = self.root / f"togo_fertility_data_all_{identifier}.pkl"
        with open(pkl_path, "rb") as f:
            data = dill.load(f)

        suffix = "_train" if isTrain else "_test"
        X = data[f"X{suffix}"]
        ids = data[f"ids{suffix}"]
        y = data[f"{label_col}{suffix}"] if f"{label_col}{suffix}" in data else data[f"y{suffix}"]
        lats = data[f"lats{suffix}"]
        lons = data[f"lons{suffix}"]

        # Remove NaN labels
        mask = ~np.isnan(y)
        num_removed = np.sum(~mask)
        if num_removed > 0:
            warnings.warn(f"Removed {num_removed} samples with NaN labels in '{label_col}{suffix}'.")

        self.X = X[mask]
        self.ids = ids[mask]
        self.y = y[mask]
        self.lats = lats[mask]
        self.lons = lons[mask]
        self.latlons = np.column_stack((self.lats, self.lons))

    def __getitem__(self, index: int):
        if self.X is not None:
            return self.X[index], self.y[index]
        else:
            return self.ids[index], self.y[index]

    def __len__(self):
        return len(self.y)
    
    def plot_subset_on_map(
        self,
        indices: Sequence[int],
        country_shape_file: str = '/share/togo/Shapefiles/tgo_admbnda_adm0_inseed_itos_20210107.shp',
        country_name: str | None = None,
        exclude_names: list[str] | None = None,
        point_color: str = 'red',
        point_size: float = 1,
        title: str | None = None,
        save_path: str | None = None
    ) -> Figure:
        """
        Plot selected lat/lon points on a country shapefile.

        Args:
            indices: list of indices in self.latlons to plot.
            country_shape_file: path to a shapefile for plotting the country boundary.
            country_name: optional name to filter a specific country (must match shapefile's attribute).
            exclude_names: optional list of names to exclude (e.g., overseas territories).
            point_color: color of plotted points.
            point_size: size of plotted points.
            title: optional title for the plot.

        Returns:
            A matplotlib Figure showing the points on the map.
        """
        import matplotlib.pyplot as plt
        print("Plotting latlon subset...")
        # Load the country shapefile
        country = gpd.read_file(country_shape_file)

        if country_name is not None and 'NAME' in country.columns:
            country = country[country['NAME'] == country_name]

        if exclude_names:
            country = country[~country['name'].isin(exclude_names)]

        latlons = self.latlons[indices]
        points = [Point(lon, lat) for lat, lon in latlons]
        points_gdf = gpd.GeoDataFrame(geometry=points, crs='EPSG:4326')

        fig, ax = plt.subplots(figsize=(12, 10))
        country.plot(ax=ax, edgecolor='black', facecolor='none')
        points_gdf.plot(ax=ax, color=point_color, markersize=point_size)

        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=14)

        if save_path:
            fig.savefig(save_path, dpi=300)

        return fig