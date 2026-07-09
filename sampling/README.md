# Sampling

Trains a model while growing the labeled set with a chosen sampling method. Adapted from [TypiClust's deep-al module](https://github.com/avihu111/TypiClust/tree/main/deep-al). Requires a [MOSEK license](https://www.mosek.com/products/academic-licenses/) (used by `cvxpy` for cost-aware optimization).

1. Edit a config in `configs/{dataset}/*.yaml`, especially `DATASET.ROOT_DIR`.
2. Run from `scripts/`:

```bash
python train_ridge.py \
  --cfg ../configs/usavars/population.yaml \
  --exp-name experiment_1 \
  --sampling_fn greedycost \
  --budget 1000 \
  --seed 42 \
  --cost_func uniform --cost_name uniform \
  --unit_assignment_path ../../0_data/groups/usavars_pop/counties_assignments.pkl \
  --group_assignment_path ../../0_data/groups/usavars_pop/states_assignments.pkl \
  --group_type states
```

Key optional arguments: `--cost_array_path`, `--unit_assignment_path`, `--region_assignment_path`, `--util_lambda`, `--alpha`, `--points_per_unit`. See `train.py --help` / `train_ridge.py --help` for the full list.

## Sweep scripts

`scripts/shell/` has sweeps configured via environment variables (see the header comment in each file for the full variable list):

| Script | Sweeps |
|---|---|
| `base.sh` | method x budget x seed, no initial set |
| `rep_sampling.sh` | representative sampling (states / image clusters / NLCD), grouped by `GROUP_TYPE` |
| `multiple_initial_sets.sh` | initial-set size x cost function |
| `cost_sensitivity.sh` | alpha x group-assignment |

```bash
DATASET=togo CFG=../configs/togo/RIDGE.yaml METHODS="poprisk" BUDGETS="100 500" \
  ./scripts/shell/base.sh
```
