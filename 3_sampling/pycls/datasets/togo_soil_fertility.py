from pathlib import Path
from typing import Sequence, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchgeo.datasets.geo import NonGeoDataset


class TogoSoilFertility(NonGeoDataset):
    """Togo Soil Fertility dataset."""

    def __init__(
        self,
        root: Path = Path("data"),
        isTrain: bool = True,
        label_col: str = "p_mgkg",
        unique_id_col: str = "unique_id",
        outcome_cols: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Args:
            root: Path to the data directory
            isTrain: Whether to load the training split
            label_col: Column name for the label
            unique_id_col: Column name for unique identifiers
            outcome_cols: List of all outcome columns (including the label); 
                          if None, all numeric columns except unique_id_col are used
        """
        self.root = Path(root) if isinstance(root, str) else root
        self.label_col = label_col
        self.unique_id_col = unique_id_col
        self.split = "train" if isTrain else "test"

        self.df = self._load_csv()

        if outcome_cols is None:
            self.outcome_cols = [
                col for col in self.df.columns
                if col != unique_id_col and pd.api.types.is_numeric_dtype(self.df[col])
            ]
        else:
            self.outcome_cols = list(outcome_cols)

        assert label_col in self.outcome_cols, f"{label_col} must be in outcome_cols"

        self._ensure_splits_exist()
        self.df = self._apply_split()

        self.ids = self.df[unique_id_col].astype(str).values
        self.y = self.df[label_col].values.astype(np.float32)

        # Features = all other outcome columns (optional)
        self.feature_cols = [col for col in self.outcome_cols if col != label_col]
        if self.feature_cols:
            self.X = self.df[self.feature_cols].values.astype(np.float32)
        else:
            self.X = None

    def __getitem__(self, index: int):
        if self.X is not None:
            return self.X[index], self.y[index]
        else:
            return self.ids[index], self.y[index]

    def __len__(self):
        return len(self.y)

    def _load_csv(self) -> pd.DataFrame:
        path = self.root / "togo_soil_fertility_resampled.csv"
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