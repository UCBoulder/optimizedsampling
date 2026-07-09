"""India SECC dataset."""

from pathlib import Path
from typing import Sequence

import dill
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset


class IndiaSECC(NonGeoDataset):
    """India SECC dataset. Loads precomputed MOSAIKS features, not images.

    Adapted from https://www.ijcai.org/proceedings/2023/0653.pdf
    """

    def __init__(
        self,
        root: Path = 'data',
        is_train: bool = True,
        label: str = 'secc_cons_pc_combined',
    ) -> None:
        self.root = Path(root)
        self.split = 'train' if is_train else 'test'
        self.label = label

        if not self._try_load_from_pickle():
            self.X, self.y, self.ids, self.geometries = self._load_features_and_labels()

            splits_dir = self.root / "splits"
            splits_dir.mkdir(exist_ok=True, parents=True)
            train_split_file = splits_dir / "train_condensed_shrug_ids.csv"
            test_split_file = splits_dir / "test_condensed_shrug_ids.csv"
            if not (train_split_file.exists() and test_split_file.exists()):
                self._make_and_save_splits()

            self.split_mask = self._filter_split()
            self.X = self.X[self.split_mask]
            self.y = self.y[self.split_mask]
            self.ids = self.ids[self.split_mask]
            self.geometries = self.geometries[self.split_mask]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.X.shape[0]

    def _load_features_and_labels(self):
        features_path = self.root / "MOSAIKS/mosaiks_features_by_shrug_condensed_regions_25_max_tiles_100_india.csv"
        features_df = pd.read_csv(features_path)

        labels_path = self.root / "MOSAIKS/grouped.csv"
        labels_df = pd.read_csv(labels_path, usecols=["condensed_shrug_id", self.label], low_memory=False)

        geometry_path = self.root / 'MOSAIKS/villages_with_regions.shp'
        gdf = gpd.read_file(geometry_path, columns=['condensed_', 'geometry'])
        gdf.rename(columns={'condensed_': 'condensed_shrug_id'}, inplace=True)

        df = gdf.merge(features_df, on="condensed_shrug_id", how="inner")
        df = df.merge(labels_df, on="condensed_shrug_id", how="inner")
        df = df.dropna(subset=[self.label])

        X = df[[f"Feature{i}" for i in range(4000)]].values
        return X, df[self.label].values, df["condensed_shrug_id"].values, df['geometry'].values

    def _filter_split(self):
        split_ids = pd.read_csv(self.root / f"splits/{self.split}_condensed_shrug_ids.csv", header=None)[0].values
        return np.isin(self.ids, split_ids)

    def _make_and_save_splits(self):
        ids_train, ids_test = train_test_split(self.ids, test_size=0.2, random_state=42)
        pd.Series(ids_train).to_csv(self.root / "splits/train_condensed_shrug_ids.csv", index=False, header=False)
        pd.Series(ids_test).to_csv(self.root / "splits/test_condensed_shrug_ids.csv", index=False, header=False)
        return self._filter_split()

    def _try_load_from_pickle(self, pkl_name="India_SECC_with_splits_4000.pkl") -> bool:
        """Load X, y, ids, and geometries from a cached pickle if one exists."""
        pkl_path = self.root / "splits" / pkl_name
        if not pkl_path.exists():
            return False

        with open(pkl_path, "rb") as f:
            data = dill.load(f)

        suffix = "_train" if self.split == "train" else "_test"
        self.X = data[f"X{suffix}"]
        self.y = data[f"y{suffix}"]
        self.ids = data[f"ids{suffix}"]
        self.geometries = data[f"geometries{suffix}"]
        return True

    def plot_subset_on_map(
        self,
        indices: Sequence[int],
        country_shape_file: str = '../../0_data/boundaries/world/ne_10m_admin_0_countries.shp',
        country_name: str = 'India',
        exclude_names: list[str] | None = None,
        point_color: str = 'red',
        point_size: float = 5,
        title: str | None = None,
        save_path: str | None = None,
    ) -> Figure:
        """Plot selected points on a country shapefile."""
        country = gpd.read_file(country_shape_file)
        if country_name is not None and 'NAME' in country.columns:
            country = country[country['NAME'] == country_name]
        if exclude_names:
            country = country[~country['name'].isin(exclude_names)]

        points_gdf = gpd.GeoDataFrame(geometry=self.geometries[indices], crs='EPSG:4326')

        fig, ax = plt.subplots(figsize=(12, 10))
        country.plot(ax=ax, edgecolor='black', facecolor='none')
        points_gdf.plot(ax=ax, color=point_color, markersize=point_size)
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=14)
        if save_path:
            fig.savefig(save_path, dpi=300)
        return fig
