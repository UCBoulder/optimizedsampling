# Groups

Group assignments are `dict`s (id:group), which are used by group-aware and cost-aware sampling methods.

**Admin groups** (state, county, district, region, etc.):

```bash
cd groups
python generate_groups.py \
  --datasets usavars_pop \
  --gdf_paths ../0_data/admin_gdfs/usavars/gdf_counties_population_2015.geojson \
  --id_cols id \
  --group_cols combined_county_id \
  --group_type counties
```

Admin geodataframes are built with `usavars_generate_gdfs.py` (USAVars, requires [Census county shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)) or `togo_generate_gdfs.py` (Togo). India SECC ships its admin levels in the shapefile already, so no processing is needed.

**Image clusters** (KMeans over feature vectors):

```bash
python image_clusters.py --dataset usavars_population --feature_path /path/to/features.pkl --num_clusters 8
```

`--dataset` must be one of: `usavars_population`, `usavars_treecover`, `india_secc`, `togo_ph_h20`.

**NLCD land-cover clusters** (US only, requires a [2016 NLCD raster](https://earthexplorer.usgs.gov/)):

```bash
python nlcd_groups.py --input_dir /path/to/naip/tifs --nlcd_path /path/to/nlcd.tif --output_dir ./nlcd_out
```
