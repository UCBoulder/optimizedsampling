from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchgeo.datasets.geo import NonGeoDataset


class TogoSoilFertility(NonGeoDataset):
    """Togo Soil Fertility dataset with required external feature file."""

    def __init__(
        self,
        root: Path = Path("data"),
        isTrain: bool = True,
        label_col: str = "p_mgkg",
        unique_id_col: str = "unique_id",
        feature_file: str = "togo_soil_features.feather",
    ) -> None:
        """
        Args:
            root: Path to the data directory
            isTrain: Whether to load the training split
            label_col: Name of the label column in the CSV
            unique_id_col: Column name for unique IDs
            feature_file: Feather file containing features (must include unique_id_col)
        """
        self.root = root
        self.label_col = label_col
        self.unique_id_col = unique_id_col
        self.feature_file = feature_file
        self.split = "train" if isTrain else "test"

        # Load label CSV
        self.df = self._load_csv()

        # Load and merge required feature file
        feature_path = self.root / self.feature_file
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        
        feature_df = pd.read_feather(feature_path)
        if self.unique_id_col not in feature_df.columns:
            raise ValueError(f"{self.unique_id_col} must be in the feature file.")
        
        self.df = self.df.merge(feature_df, on=self.unique_id_col, how="inner")
        self.feature_cols = [
            col for col in feature_df.columns if col != self.unique_id_col
        ]

        self._ensure_splits_exist()
        self.df = self._apply_split()

        self.ids = self.df[self.unique_id_col].values
        self.y = self.df[self.label_col].values.astype(np.float32)
        self.X = self.df[self.feature_cols].values.astype(np.float32)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)

    def _load_csv(self) -> pd.DataFrame:
        path = self.root / "togo_soil_fertility_resampled.csv"
        df = pd.read_csv(path)

        # Clean column names for safety
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("%", "pct")
            .str.replace("/", "")
        )

        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in CSV.")

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