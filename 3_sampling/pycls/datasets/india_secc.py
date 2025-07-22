"""India SECC dataset."""

import glob
import os
from collections.abc import Callable, Sequence
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor
import dill
from pathlib import Path
from sklearn.model_selection import train_test_split

from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets import DatasetNotFoundError


class IndiaSECC(NonGeoDataset):
    """India SECC dataset.

    Adapted from https://www.ijcai.org/proceedings/2023/0653.pdf
    Note this dataset directly loads the MOSAIKS features, not the images
    """

    def __init__(
        self,
        root: Path = 'data',
        isTrain: bool = True,
        label: str = 'secc_cons_pc_combined'
    ) -> None:
        """Initialize a new IndiaSECC dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = Path(root)
        if isTrain:
            self.split = 'train'
        else:
            self.split = 'test'

        self.label = label

        self._check_for_dataset()

        if not self._try_load_from_pickle():
            self.X, self.y, self.ids, self.geometries = self._load_features_and_labels()

            #make sure splits directory exists
            splits_dir = self.root / "splits"
            splits_dir.mkdir(exist_ok=True, parents=True)

            #check if splits exist; if not, create them
            train_split_file = splits_dir / "train_condensed_shrug_ids.csv"
            test_split_file = splits_dir / "test_condensed_shrug_ids.csv"

            if not (train_split_file.exists() and test_split_file.exists()):
                self._make_and_save_splits()

            self.split_mask = self._filter_split()
            self.X = self.X[self.split_mask]
            self.y = self.y[self.split_mask]
            self.ids = self.ids[self.split_mask]
            self.geometries = self.geometries[self.split_mask]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        return self.X[index], self.y[index]


    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return self.X.shape[0]


    def _load_features_and_labels(self):
        """Load MOSAIKS features and SECC labels for India.

        Returns:
            X: np.ndarray of shape (n_samples, 4000) with MOSAIKS features.
            y: np.ndarray of shape (n_samples,) with label values.
            shrids: np.ndarray of SHRUG IDs corresponding to each row.
        """
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
        y = df[self.label].values
        condensed_shrug_ids = df["condensed_shrug_id"].values
        geometries = df['geometry'].values

        return X, y, condensed_shrug_ids, geometries

    def _filter_split(self):
        split_condensed_shrug_ids = pd.read_csv(self.root / f"splits/{self.split}_condensed_shrug_ids.csv", header=None)[0].values

        missing_ids = set(split_condensed_shrug_ids) - set(self.ids)
        if missing_ids:
            print(f"Warning: {len(missing_ids)} IDs from {self.split}_split not in dataset")

        mask = np.isin(self.ids, split_condensed_shrug_ids)
        return mask

    def _make_and_save_splits(self):
        ids_train, ids_test = train_test_split(
            self.ids, test_size=0.2, random_state=42
        )

        pd.Series(ids_train).to_csv(self.root / "splits/train_condensed_shrug_ids.csv", index=False, header=False)
        pd.Series(ids_test).to_csv(self.root  / "splits/test_condensed_shrug_ids.csv", index=False, header=False)

        print("Train/test splits saved!")

        return self._filter_split()


    def _check_for_dataset(self) -> None:
        required_files = [
            self.root / 'MOSAIKS/grouped.csv',
            self.root / 'MOSAIKS/villages_with_regions.shp',
            self.root / 'MOSAIKS/mosaiks_features_by_shrug_condensed_regions_25_max_tiles_100_india.csv',
        ]
        for file in required_files:
            if not file.exists():
                raise DatasetNotFoundError(f"Required file not found: {file}")

    def _try_load_from_pickle(self, pkl_name="India_SECC_with_splits_4000.pkl") -> bool:
        """
        Attempt to load X, y, ids, and geometries from a pickle file.

        Args:
            pkl_name (str): Name of the pickle file to load from.

        Returns:
            bool: True if data was successfully loaded from pickle, False otherwise.
        """
        import dill 

        pkl_path = self.root / "splits" / pkl_name
        if not pkl_path.exists():
            return False

        print(f"Loading saved data from {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = dill.load(f)

        if self.split == "train":
            self.X = data["X_train"]
            self.y = data["y_train"]
            self.ids = data["ids_train"]
            self.geometries = data["geometries_train"]
        else:
            self.X = data["X_test"]
            self.y = data["y_test"]
            self.ids = data["ids_test"]
            self.geometries = data["geometries_test"]

        return True

    def plot_subset_on_map(
        self,
        indices: Sequence[int],
        country_shape_file: str = '/home/libe2152/optimizedsampling/0_data/boundaries/world/ne_10m_admin_0_countries.shp',
        country_name: str = 'India',
        exclude_names: list[str] | None = None,
        point_color: str = 'red',
        point_size: float = 5,
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
        print("Plotting latlon subset...")
        # Load the country shapefile
        country = gpd.read_file(country_shape_file)

        if country_name is not None and 'NAME' in country.columns:
            country = country[country['NAME'] == country_name]

        if exclude_names:
            country = country[~country['name'].isin(exclude_names)]

        geometries_subset = self.geometries[indices]
        points_gdf = gpd.GeoDataFrame(geometry=geometries_subset, crs='EPSG:4326')

        fig, ax = plt.subplots(figsize=(12, 10))
        country.plot(ax=ax, edgecolor='black', facecolor='none')
        points_gdf.plot(ax=ax, color=point_color, markersize=point_size)

        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=14)

        if save_path:
            fig.savefig(save_path, dpi=300)

        return fig