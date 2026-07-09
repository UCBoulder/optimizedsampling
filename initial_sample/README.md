# Initial Sample

Each script draws a sample of point IDs and saves them as a `.pkl`, for every combination of `--sample_sizes` x `--seeds`. Run from `initial_sample/`.

## Random

```bash
python random_sampling.py --labels population treecover --sample_sizes 1100 1200 --seeds 1 42
```

## Cluster (stratified by admin region, e.g. state, sampling clusters e.g. counties)

```bash
python cluster_sampling.py \
  --labels population \
  --strata_col state_name --cluster_col combined_county_id \
  --sample_sizes 1000 --points_per_cluster 5 \
  --fixed_strata Idaho Louisiana Mississippi "New Mexico" Pennsylvania \
  --seeds 1 42
```

Use `--n_strata 5` instead of `--fixed_strata ...` to pick strata at random. Saves to `0_data/initial_samples/usavars/{label}/cluster_sampling/`.

## Convenience (biased toward urban areas)

```bash
python infrastructure_convenience_sampling.py \
  --labels population \
  --urban_shp /path/to/urban_areas.shp --pop_col population --n_urban 5 \
  --cluster_col combined_county_id --points_per_cluster 5 \
  --sample_sizes 1000 --method probabilistic --seeds 1 42
```

Omit `--cluster_col` to sample individual points instead of whole clusters. Saves to `0_data/initial_samples/usavars/{label}/convenience_sampling/`.

Run any script with `--help` for the full argument list.
