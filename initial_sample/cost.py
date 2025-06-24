import numpy as np
import dill

def load_radius_cost(total_counties, density=False, radius_small=10, radius_large=20, save=True, label="treecover"):
    type_str = "density" if density else "clustered"

    with open(f"../data/int/feature_matrices/CONTUS_UAR_{label}_with_splits_torchgeo4096.pkl", "rb") as f:
        arrs = dill.load(f)

    invalid_ids = np.array(['615,2801', '1242,645', '539,3037', '666,2792', '1248,659', '216,2439'])

    all_ids = arrs['ids_train']
    valid_idxs = np.where(~np.isin(all_ids, invalid_ids))[0]
    all_ids = all_ids[valid_idxs]

    with open(f"{type_str}_sampled_points_{label}/IDs_{total_counties}_counties_{radius_small}_radius_seed_42.pkl", "rb") as f:
        in_cluster = dill.load(f)

    with open(f"{type_str}_sampled_points_{label}/IDs_{total_counties}_counties_{radius_large}_radius_seed_42.pkl", "rb") as f:
        in_radius= dill.load(f)

    set_in_cluster = set(in_cluster)
    set_in_radius = set(in_radius)

    cost_dict = {}
    for id_ in all_ids:
        if id_ in set_in_cluster:
            cost_dict[id_] = np.nan
        elif id_ in set_in_radius:
            cost_dict[id_] = 1
        else:
            cost_dict[id_] = 5

    with open(f"{label}/{type_str}/cost/{total_counties}_counties_r1_{radius_small}_r2_{radius_large}_seed_42_cost_1_vs_5.pkl", "wb") as f:
        dill.dump(cost_dict, f)

    return cost_dict

if __name__ == '__main__':
    label="population"
    for density in [True, False]:
        load_radius_cost(100, density=density, label=label)