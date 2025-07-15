# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""USAVars dataset."""

import glob
import os
from collections.abc import Callable, Sequence
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor
import dill

from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets import DatasetNotFoundError
from torchgeo.datasets.utils import Path, download_url, extract_archive

invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])

def load_from_pkl(label, split):
    data_path = f"/home/libe2152/optimizedsampling/0_data/features/usavars/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl"

    with open(data_path, "rb") as f:
        arrs = dill.load(f)

    X = arrs[f"X_{split}"]
    y = arrs[f"y_{split}"]
    latlons = arrs[f"latlons_{split}"]
    ids = arrs[f"ids_{split}"]

    valid_idxs = np.where(~np.isin(ids, invalid_ids))[0]
    ids = ids[valid_idxs]
    X = X[valid_idxs]
    y = y[valid_idxs]
    latlons = latlons[valid_idxs]

    return X, y, latlons, ids


class USAVars(NonGeoDataset):
    """USAVars dataset.

    The USAVars dataset is reproduction of the dataset used in the paper "`A
    generalizable and accessible approach to machine learning with global satellite
    imagery <https://doi.org/10.1038/s41467-021-24638-z>`_". Specifically, this dataset
    includes 1 sq km. crops of NAIP imagery resampled to 4m/px cenetered on ~100k points
    that are sampled randomly from the contiguous states in the USA. Each point contains
    three continuous valued labels (taken from the dataset released in the paper): tree
    cover percentage, elevation, and population density.

    Dataset format:

    * images are 4-channel GeoTIFFs
    * labels are singular float values

    Dataset labels:

    * tree cover
    * elevation
    * population density

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1038/s41467-021-24638-z

    .. versionadded:: 0.3
    """

    data_url = 'https://hf.co/datasets/torchgeo/usavars/resolve/01377abfaf50c0cc8548aaafb79533666bbf288f/{}'
    dirname = 'uar'

    md5 = '677e89fd20e5dd0fe4d29b61827c2456'

    label_urls: ClassVar[dict[str, str]] = {
        'housing': data_url.format('housing.csv'),
        'income': data_url.format('income.csv'),
        'roads': data_url.format('roads.csv'),
        'nightlights': data_url.format('nightlights.csv'),
        'population': data_url.format('population.csv'),
        'elevation': data_url.format('elevation.csv'),
        'treecover': data_url.format('treecover.csv'),
    }

    data_download_url = data_url.format('uar.zip')

    split_metadata: ClassVar[dict[str, dict[str, str]]] = {
        'train': {
            'url': data_url.format('train_split.txt'),
            'filename': 'train_split.txt',
            'md5': '3f58fffbf5fe177611112550297200e7',
        },
        'val': {
            'url': data_url.format('val_split.txt'),
            'filename': 'val_split.txt',
            'md5': 'bca7183b132b919dec0fc24fb11662a0',
        },
        'test': {
            'url': data_url.format('test_split.txt'),
            'filename': 'test_split.txt',
            'md5': '97bb36bc003ae0bf556a8d6e8f77141a',
        },
    }

    def __init__(
        self,
        root: Path = 'data',
        isTrain: bool = True,
        label: str = 'population',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new USAVars dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            labels: list of labels to include
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if invalid labels are provided
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        if isTrain:
            self.split = 'train'
        else:
            self.split = 'test'

        assert label in ('treecover', 'elevation', 'population', 'income'), "Label information does not exist."

        self.label = label
        self.label_dfs = {
            self.label: pd.read_csv(os.path.join(self.root, self.label + ".csv"), index_col="ID")
        }

        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.files = self._load_files()

        self.X, self.y, self.latlons, self.ids = load_from_pkl(self.label, self.split)


    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        return self.X[index], self.y[index], self.latlons[index], self.ids[index]


    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return self.X.shape[0]


    def _load_files(self) -> list[str]:
        """Loads file names."""
        with open(os.path.join(self.root, f'{self.split}_split.txt')) as f:
            files = f.read().splitlines()
        return files

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with rasterio.open(path) as f:
            array: np.typing.NDArray[np.int_] = f.read()
            tensor = torch.from_numpy(array).float()
            return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, 'uar')
        csv_pathname = os.path.join(self.root, '*.csv')
        split_pathname = os.path.join(self.root, '*_split.txt')

        csv_split_count = (len(glob.glob(csv_pathname)), len(glob.glob(split_pathname)))
        if glob.glob(pathname) and csv_split_count == (8, 3):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.dirname + '.zip')
        if glob.glob(pathname) and csv_split_count == (8, 3):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            print("Raising DatasetNotFoundError")
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for f_name in self.label_urls:
            download_url(self.label_urls[f_name], self.root, filename=f_name + '.csv')

        download_url(self.data_download_url, self.root, md5=self.md5 if self.checksum else None)

        for metadata in self.split_metadata.values():
            download_url(
                metadata['url'],
                self.root,
                md5=metadata['md5'] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.dirname + '.zip'))

    def plot(
        self,
        sample: dict[str, Tensor],
        show_labels: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_labels: flag indicating whether to show labels above panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample['image'][:3].numpy()  # get RGB inds
        image = np.moveaxis(image, 0, 2)
        image = image/256.0

        fig, axs = plt.subplots(figsize=(10, 10))
        axs.imshow(image)
        axs.axis('off')

        if show_labels:
            labels = [(lab, val) for lab, val in sample.items() if lab != 'image']
            label_string = ''
            for lab, val in labels:
                label_string += f'{lab}={round(val[0].item(), 2)} \n'
            axs.set_title(label_string)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig