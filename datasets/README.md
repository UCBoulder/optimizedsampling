# Datasets

Three datasets are supported: **USAVars**, **India SECC**, and **Togo soil fertility** (not yet publicly released — will be released by the Togolese Ministry of Agriculture).

Each dataset goes through the same stages: download raw data → featurize → split into train/test → load precomputed features for sampling experiments.

- `raw/` — download and featurize raw imagery/tabular data.
- `featurized/` — dataset classes that load precomputed features/labels for the sampling pipeline (used via `featurized/data.py`'s `Data` class).

## USAVars

**Download** with `torchgeo` — see `raw/usavars.py` (adapted from [TorchGeo's USAVars dataset](https://torchgeo.readthedocs.io/en/v0.3.1/_modules/torchgeo/datasets/usavars.html)). Labels: `treecover`, `elevation`, `population`, `income`.

**Featurize** (RCF or a pretrained SwinV2-B backbone):

```bash
cd datasets/raw
python featurization.py \
  --dataset_name USAVars \
  --data_root /path/to/data/root \
  --labels treecover,population \
  --num_features 4096
```

**Split** into train/test (80/20), saved as a `.pkl` with `{X, y, latlon}_{train, test}`:

```bash
python format_data.py --save --label population --feature_path /path/to/CONTUS_UAR_torchgeo4096.pkl
```

## India SECC

Process/download from [satellite-fairness-replication](https://github.com/emilylaiken/satellite-fairness-replication), then save features as a `.pkl` (dict with keys `X`, `ids_X`, `latlon`). `featurized/india_secc.py` expects these files under the dataset root:

| File | Contents |
|---|---|
| `MOSAIKS/mosaiks_features_by_shrug_condensed_regions_25_max_tiles_100_india.csv` | Precomputed MOSAIKS features (4000-dim), keyed by `condensed_shrug_id` |
| `MOSAIKS/grouped.csv` | Labels, keyed by `condensed_shrug_id`; target column `secc_cons_pc_combined` |
| `MOSAIKS/villages_with_regions.shp` | Village polygons; `condensed_` column is renamed to `condensed_shrug_id` |

Train/test splits are generated and cached automatically on first load.

## Togo Soil Fertility

Not yet available. `featurized/togo_soil_fertility.py` expects a `togo_fertility_data_all_{identifier}.pkl` with `X`, `ids`, lat/lon, and label columns, split into `_train`/`_test`.
