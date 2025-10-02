from pathlib import Path
from typing import Sequence, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from torchgeo.datasets.geo import NonGeoDataset

from matplotlib.figure import Figure
from shapely.geometry import Point


class TogoSoilFertility(NonGeoDataset):
    """Togo Soil Fertility dataset."""

    def __init__(
        self,
        root: Path = Path("data"),
        isTrain: bool = True,
        label_col: str = "p_mgkg",
        unique_id_col: str = "unique_id",
        is_feather: bool = False,
        csv_name: str = 'togo_soil_fertility_resampled.csv',
        feature_filepath: str | None = None
    ) -> None:
        """
        Args:
            root: Path to the data directory
            isTrain: Whether to load the training split
            label_col: Column name for the label
            unique_id_col: Column name for unique identifiers
        """
        self.root = Path(root) if isinstance(root, str) else root
        self.label_col = label_col
        self.unique_id_col = unique_id_col
        self.split = "train" if isTrain else "test"
        self.csv_name = csv_name

        self.df = self._load_csv()

        self.outcome_cols = [
            col for col in self.df.columns
            if col != unique_id_col and pd.api.types.is_numeric_dtype(self.df[col])
        ]

        assert label_col in self.outcome_cols, f"{label_col} must be in outcome_cols"

        self._ensure_splits_exist()
        self.df = self._apply_split()

        #code from TDL
        RENAME_COLS = dict(zip([f'X_{i}' for i in range(4000)], [f'planet_{i}' for i in range(4000)]))
        if feature_filepath is None:
            feature_filepath = root / "features.feather"
        if is_feather:
            df_mosaiks = pd.read_feather(feature_filepath).rename(columns=RENAME_COLS)
        else:
            df_mosaiks = pd.read_parquet(feature_filepath)
        
        self.df = (df_mosaiks
            .merge(self.df, on='unique_id', how='inner')
            .rename(columns={'admin_1': 'region'})
        )

        self.X = self.df[[f'planet_{i}' for i in range(4000)]].values
        self.ids = self.df[unique_id_col].astype(str).values
        self.y = self.df[label_col].values.astype(np.float32)
        self.lats = self.df['lat'].values.astype(np.float32)
        self.lons = self.df['lon'].values.astype(np.float32)

        self.latlons = np.array(list(zip(self.lats, self.lons)))

    def __getitem__(self, index: int):
        if self.X is not None:
            return self.X[index], self.y[index]
        else:
            return self.ids[index], self.y[index]

    def __len__(self):
        return len(self.y)

    def _load_csv(self) -> pd.DataFrame:
        path = self.root / self.csv_name
        df = pd.read_csv(path)

        # Sanitize column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("%", "pct")
            .str.replace("/", "")
            .str.replace("__", "_")
        )

        # Rename label_col if needed
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in CSV columns: {df.columns.tolist()}")

        df = df.dropna(subset=[self.label_col])
        return df

    def _ensure_splits_exist(self, random_state=42):
        split_dir = self.root / "splits"
        split_dir.mkdir(exist_ok=True)

        train_path = split_dir / "train_ids.csv"
        test_path = split_dir / "test_ids.csv"

        if not train_path.exists() or not test_path.exists():
            ids = self.df[self.unique_id_col].values
            ids_train, ids_test = train_test_split(ids, test_size=0.2, random_state=random_state)
            pd.Series(ids_train).to_csv(train_path, index=False, header=False)
            pd.Series(ids_test).to_csv(test_path, index=False, header=False)

    def _apply_split(self) -> pd.DataFrame:
        split_path = self.root / "splits" / f"{self.split}_ids.csv"
        split_ids = pd.read_csv(split_path, header=None)[0].values
        return self.df[self.df[self.unique_id_col].isin(split_ids)].copy()
    
    def plot_subset_on_map(
        self,
        indices: Sequence[int],
        country_shape_file: str = 'tgo_admbnda_adm0_inseed_itos_20210107.shp',
        country_name: str | None = None,
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