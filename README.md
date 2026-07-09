# Optimized Sampling Pipeline

A framework for studying sampling strategies (random, cluster, convenience, cost-aware, active learning) for remote-sensing regression tasks across three datasets: **USAVars**, **India SECC**, and **Togo soil fertility**.

Data download and featurization instructions are in [datasets/README.md](datasets/README.md).

### Attribution

- `datasets/raw/mosaiks` is adapted from [Global Policy Lab/mosaiks-paper](https://github.com/Global-Policy-Lab/mosaiks-paper) (feature extraction and dataset handling).
- `sampling/` is adapted from [TypiClust's deep-al module](https://github.com/avihu111/TypiClust/tree/main/deep-al) (active learning and sampling methods) — see [sampling/README.md](sampling/README.md).

## Pipeline

1. **[Data](datasets/README.md)** - download and featurize each dataset.
2. **[Groups](groups/README.md)** - build geodataframes and group assignments (admin regions, image clusters, NLCD land cover) used by group-aware sampling strategies.
3. **[Initial sample](initial_sample/README.md)** - construct a starting labeled set (random, cluster, or convenience sampling).
4. **[Sampling](sampling/README.md)** - train a model while adding to the labeled set with a chosen sampling method.
5. **[Summarize](summarize/README.md)** - parse logs and generate tables/figures.
