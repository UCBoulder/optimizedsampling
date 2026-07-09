# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""USAVars dataset."""

from typing import Sequence

import dill
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import Path

invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])


def load_from_pkl(label, split):
    data_path = f"../../0_data/features/usavars/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl"
    with open(data_path, "rb") as f:
        arrs = dill.load(f)

    X, y = arrs[f"X_{split}"], arrs[f"y_{split}"]
    latlons, ids = arrs[f"latlons_{split}"], arrs[f"ids_{split}"]

    valid_idxs = np.where(~np.isin(ids, invalid_ids))[0]
    return X[valid_idxs], y[valid_idxs], latlons[valid_idxs], ids[valid_idxs]


class USAVars(NonGeoDataset):
    """USAVars dataset: reproduction of "A generalizable and accessible approach to
    machine learning with global satellite imagery" (https://doi.org/10.1038/s41467-021-24638-z).

    Loads precomputed features/labels for tree cover, elevation, population density,
    and income, keyed by ~100k points sampled from the contiguous US.
    """

    def __init__(self, root: Path = 'data', is_train: bool = True, label: str = 'population') -> None:
        self.root = root
        self.split = 'train' if is_train else 'test'
        assert label in ('treecover', 'elevation', 'population', 'income'), "Label information does not exist."
        self.label = label
        self.X, self.y, self.latlons, self.ids = load_from_pkl(self.label, self.split)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, str]:
        return self.X[index], self.y[index], self.latlons[index], self.ids[index]

    def __len__(self) -> int:
        return self.X.shape[0]

    def plot_subset_on_map(
        self,
        indices: Sequence[int],
        country_shape_file: str = '../../0_data/boundaries/us/us_states_provinces/ne_110m_admin_1_states_provinces.shp',
        country_name: str | None = None,
        exclude_names: list[str] = ['Alaska', 'Hawaii', 'Puerto Rico'],
        point_color: str = 'red',
        point_size: float = 5,
        title: str | None = None,
        save_path: str | None = None,
    ) -> Figure:
        """Plot selected lat/lon points on a country shapefile."""
        country = gpd.read_file(country_shape_file)
        if country_name is not None and 'NAME' in country.columns:
            country = country[country['NAME'] == country_name]
        if exclude_names:
            country = country[~country['name'].isin(exclude_names)]

        latlons_subset = self.latlons[indices]
        points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(latlons_subset[:, 1], latlons_subset[:, 0]), crs='EPSG:4326')

        fig, ax = plt.subplots(figsize=(12, 10))
        country.plot(ax=ax, edgecolor='black', facecolor='none')
        points_gdf.plot(ax=ax, color=point_color, markersize=point_size)
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=14)
        if save_path:
            fig.savefig(save_path, dpi=300)
        return fig
