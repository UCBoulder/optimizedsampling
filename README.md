# Optimized Sampling Pipeline

### Attribution This repository makes use of and builds on external code from the following sources: - **`datasets/mosaiks`**: Adapted from the [Global Policy Lab's mosaiks-paper repository](https://github.com/Global-Policy-Lab/mosaiks-paper), which provides code for feature extraction and dataset handling using MOSAIKS features. - **`sampling/`**: Adapted from [TypiClust's deep-al module](https://github.com/avihu111/TypiClust/tree/main/deep-al), which includes implementations for active learning and sampling methods. We thank the authors of these repositories for making their code available.

This repository contains a complete data processing pipeline for **optimized sampling analysis** across three datasets:

- **USAVars**
- **India SECC**
- **Togo soil fertility** (Note: not yet publicly released)

---

## Table of Contents

1. [Data Download](#data-download)  
2. [Featurization](#featurization)  
3. [Train-Test Split](#train-test-split)  
4. [Generate GeoDataFrames](#generate-geodataframes)  
5. [Group Creation](#group-creation)  
6. [Initial Sampling](#initial-sampling)  
7. [Running Sampling](#running-sampling)  
8. [Bash Scripts](#bash-scripts)  
9. [Results and Analysis](#results-and-analysis)  
10. [Contributing](#contributing)  

---

## Data Download

### USAVars Data

Download using `torchgeo`.  
- Docs: [TorchGeo USAVars Dataset](https://torchgeo.readthedocs.io/en/v0.3.1/_modules/torchgeo/datasets/usavars.html)

### India SECC Data

Process/download from this repository: https://github.com/emilylaiken/satellite-fairness-replication

#### India SECC Dataset Files:
- `mosaiks_features_by_shrug_condensed_regions_25_max_tiles_100_india.csv`  
 - **Description**: Precomputed MOSAIKS features (4000-dim)  
 - **Columns**:  
   - `condensed_shrug_id`: Unique ID per unit  
   - `Feature0` to `Feature3999`: Satellite features  

- `grouped.csv`  
 - **Description**: Contains labels  
 - **Columns**:  
   - `condensed_shrug_id` (matching above)  
   - `secc_cons_pc_combined`: Target variable  

- `villages_with_regions.shp`  
 - **Description**: Shapefile with spatial polygons  
 - **Columns**:  
   - `condensed_`: Will be renamed to `condensed_shrug_id`  
   - `geometry`: Polygon geometries  

### Togo Soil Fertility Data
*Not yet available.* Will be released by the Togolese Ministry of Agriculture.

---

## Featurization

### USAVars

Run:

```bash
python featurization.py \
 --dataset_name USAVars \
 --data_root /path/to/your/data/root \
 --labels treecover,population \
 --num_features 4096
```

### India SECC

Follow instructions at: [satellite-fairness-replication](https://github.com/emilylaiken/satellite-fairness-replication)  
- Save features as a `.pkl` file (dict format) with keys: `'X'`, `'ids_X'`, and `'latlon'`.

---

## Train-Test Split

Run `format_data.py`:

```bash
python format_data.py \
 --save \
 --label population \  # or other label
 --feature_path CONTUS_UAR_torchgeo4096.pkl  # or India features
```

Creates an 80/20 split, saved as a `.pkl` file.

---

## Generate GeoDataFrames

GeoDataFrames are used for clusters and region-based sampling strategies.

### USAVars

First, download US county shape files from [census.gov](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) (Here we use 2015 shape files):

```bash
python usavars_generate_gdfs.py \
 --labels population,treecover \
 --input_folder ../0_data/features/usavars \
 --year 2015 \
 --invalid_ids "615,2801 1242,645 539,3037 666,2792 1248,659 216,2439" \
 --county_shp ../0_data/boundaries/us/us_county_2015 \
 --output_dir ./admin_gdfs/usavars
```

### India SECC

Admin levels are pre-included in the shapefile. No processing needed.

### Togo

Use:

```bash
python togo_generate_gdfs.py
```

---

## Group Creation

### Available Group Types

- **Admin Groups**: States, regions  
- **Image Groups**: Feature-based KMeans clustering  
- **NLCD Groups**: Land cover classes (U.S. only)  

### Image Clustering

Run:

```bash
python image_clusters.py
```

### Group Assignment Example

#### Admin Groups

```bash
python generate_groups.py \
 --datasets "usavars_pop,usavars_tc,india_secc,togo" \
 --gdf_paths \
   "../0_data/admin_gdfs/usavars/gdf_counties_population_2015.geojson" \
   "../0_data/admin_gdfs/usavars/gdf_counties_treecover_2015.geojson" \
   "/share/india_secc/train_shrugs_with_admins.geojson" \
   "../../togo/gdf_adm3.geojson" \
 --id_cols "id" "id" "condensed_shrug_id" "id" \
 --shape_files "../0_data/boundaries/us/us_county_2015" \
 --country_names "None" "None" "India" "None" \
 --exclude_names "Alaska,Hawaii,Puerto Rico" \
 --group_type "counties" \
 --group_cols "combined_county_id"
```

and

```bash
python generate_groups.py \
 --datasets "usavars_pop,usavars_tc,india_secc,togo" \
 --gdf_paths \
   "../0_data/admin_gdfs/usavars/gdf_counties_population_2015.geojson" \
   "../0_data/admin_gdfs/usavars/gdf_counties_treecover_2015.geojson" \
   "/share/india_secc/train_shrugs_with_admins.geojson" \
   "../../togo/gdf_adm3.geojson" \
 --id_cols "id" "id" "condensed_shrug_id" "id" \
 --shape_files "../0_data/boundaries/us/us_county_2015" \
 --country_names "None" "None" "India" "None" \
 --exclude_names "Alaska,Hawaii,Puerto Rico" \
 --group_type "states" \
 --group_cols "state_name"
```

#### Image Groups

```bash
python image_clusters.py
```

#### NLCD Groups (U.S. only)

- Download 2016 NLCD TIFF from: https://earthexplorer.usgs.gov/  
- Run `nlcd_groups.py` with the following required arguments:
 - `--input_dir`: Directory containing NAIP image files
 - `--nlcd_path`: Path to existing NLCD .tif raster file  
 - `--output_dir`: Directory to save output files
 - `--dataset_name`: Dataset name for output filenames (default: "usavars")
 - `--file_pattern`: File pattern to match in input directory (default: "*.tif")
 - `--k_min`: Minimum number of clusters to test (default: 2)
 - `--k_max`: Maximum number of clusters to test (default: 10)

**Specifically, make sure k = 8**

---

## Initial Sampling

Use Jupyter Notebooks:

- `usavars_initial_sample.ipynb`  
- `india_secc_initial_sample.ipynb`  
- `togo_initial_sample.ipynb`

Save the outputs as `.pkl` to:  
`0_data/initial_samples/{dataset}/...`

---

## Running Sampling

**Attention**: To solve optimization problem, cvxpy is run with MOSEK solver. You need a license to use MOSEK, which can be requested from https://www.mosek.com/products/academic-licenses/

### Step 1: Configure

Edit config files in:

```text
sampling/configs/{dataset}/*.yaml
```

Set correct paths, especially `DATASET.ROOT_DIR`.

### Step 2: Train

Example usage:

```bash
python train.py \
 --cfg ../configs/usavars/RIDGE_POP.yaml \
 --exp-name experiment_1 \
 --sampling_fn greedycost \
 --budget 1000 \
 --initial_set_str test \
 --seed 42 \
 --unit_assignment_path ../../0_data/groups/usavars_pop/counties_assignments.pkl \
 --id_path '../../0_data/initial_samples/usavars/population/cluster_sampling/fixedstrata_Idaho_16-Louisiana_22-Mississippi_28-New Mexico_35-Pennsylvania_42/sample_state_combined_county_id_5ppc_150_size_seed_1.pkl' \
 --cost_func uniform \
 --cost_name uniform \
 --group_assignment_path ../../0_data/groups/usavars_pop/states_assignments.pkl \
 --group_type states
```

### Additional Parameters

| Argument | Description |
|----------|-------------|
| `--cost_array_path` | Path to a NumPy cost array |
| `--unit_assignment_path` | Path to unit assignment file |
| `--region_assignment_path` | Path to region assignment file |
| `--util_lambda` | Utility lambda for optimization |
| `--alpha` | Alpha parameter for cost-based sampling |
| `--points_per_unit` | For unit-based sampling |

---

## Bash Scripts

### Base Experiments

- `run_india.sh`  
- `run_togo.sh`  
- `run_usavars_population.sh`  
- `run_usavars_treecover.sh`

### Representative Sampling – Admin Regions

- `run_india_rep_states.sh`  
- `run_togo_rep_regions.sh`  
- `run_usavars_population_rep_states.sh`  
- `run_usavars_treecover_rep_states.sh`

### Representative Sampling – Image Clusters

- `run_india_rep_image_8.sh`  
- `run_togo_cluster_rep_image_8.sh`  
- `run_usavars_population_rep_image_8.sh`  
- `run_usavars_treecover_rep_image_8.sh`

### Representative Sampling – NLCD

(U.S. only)

- `run_usavars_population_rep_nlcd.sh`  
- `run_usavars_treecover_rep_nlcd.sh`

### Multiple Initial Sets (Figure 4)

- `run_india_cluster_multiple.sh`  
- `run_togo_cluster_multiple_initial_set.sh`

### Cost Sensitivity (Figure 5)

- `run_togo_cost_diff.sh`

---

## Results and Analysis

Go to the `summarize/` directory.

### Generate CSVs from Logs

```bash
python parse_out_log.py --multiple True  # or False
```

### Generate Table 2

```bash
python generate_latex_table.py
```

### Generate Figure 4

```bash
python plot_multiple_initial_set.py
```

### Generate Figure 5

```bash
python plot_alpha.py
```

---

## Other

- Make sure **all data paths** in config files are set correctly.
- Check that required `.pkl` files (features, splits, groups) exist.

---