import pandas as pd
import os

base_method_order = ['greedycost', 'random_unit', 'poprisk_admin', 'poprisk_img', 'poprisk_nlcd']
METRICS = ['r2', 'mse', 'rmse', 'mae']

def load_df(dataset, init_set, cost_type, csv_path="results/_metrics_{dataset}_{init_set}_{cost_type}.csv"):
    return pd.read_csv(csv_path.format(dataset=dataset, init_set=init_set, cost_type=cost_type))

def generate_latex_table(df, dataset, init_set, cost_type, method_map, metric='r2', delta=False):
    method_labels = [method_map.get(m, m) for m in base_method_order]
    std_suffix = "se" if delta else "std"
    prefix = "delta_" if delta else ""

    lines = ["\\hline%"]
    for _, row in df.iterrows():
        budget = int(row["budget"])
        cells = []
        for method in method_labels:
            mean = row.get(f"{method}_{prefix}{metric}_mean" if delta else f"{method}_updated_{metric}_mean")
            std = row.get(f"{method}_{prefix}{metric}_{std_suffix}" if delta else f"{method}_updated_{metric}_{std_suffix}")
            cells.append(f"{mean:.2f} ± {std:.2f}" if pd.notnull(mean) and pd.notnull(std) else "--")
        lines.append(f"{budget} & " + " & ".join(cells) + "\\\\%")
    lines.append("\\hline%")

    os.makedirs(f"latex_table_{metric}", exist_ok=True)
    tex_path = f"latex_table_{metric}/latex_table_{prefix}{metric}_{dataset}_{init_set}_{cost_type}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

def generate_table_from_csv(dataset, init_set, cost_type, method_map, delta=False,
                            csv_path="results/aggregated_metrics_{dataset}_{init_set}_{cost_type}.csv"):
    df = load_df(dataset, init_set, cost_type, csv_path=csv_path)
    for metric in METRICS:
        generate_latex_table(df, dataset, init_set, cost_type, method_map, metric=metric, delta=delta)

if __name__ == "__main__":
    import argparse

    DATASET_NAMES = {"INDIA_SECC", "USAVARS_POP", "USAVARS_TC", "TOGO_PH_H2O"}
    initial_set_strs = {
        'USAVARS_POP': '{num_strata}_fixedstrata_{ppc}ppc_{size}_size',
        'USAVARS_TC': '{num_strata}_fixedstrata_{ppc}ppc_{size}_size',
        'INDIA_SECC': '{num_strata}_state_district_desired_{ppc}ppc_{size}_size',
        'TOGO_PH_H2O': '{num_strata}_strata_desired_{ppc}ppc_{size}_size'
    }
    admin_types = {'USAVARS_POP': 'states', 'USAVARS_TC': 'states', 'INDIA_SECC': 'states', 'TOGO_PH_H2O': 'regions'}
    cluster_nums = {'USAVARS_POP': '8', 'USAVARS_TC': '8', 'INDIA_SECC': '8', 'TOGO_PH_H2O': '3'}

    parser = argparse.ArgumentParser(description="Run sampling evaluation script")
    parser.add_argument("--delta", type=lambda x: x.lower() == 'true', default=False)
    delta = parser.parse_args().delta

    for dataset in DATASET_NAMES:
        method_map = {
            'poprisk_admin': f'poprisk_{admin_types[dataset]}_0.5',
            'poprisk_nlcd': 'poprisk_nlcd_0.5',
            'poprisk_img': f'poprisk_image_clusters_{cluster_nums[dataset]}_0.5',
        }

        for ppc in [10, 20, 25]:
            for size in [100, 200, 300, 500, 2000]:
                for num_strata in [2, 5, 10]:
                    initial_set = initial_set_strs[dataset].format(num_strata=num_strata, ppc=ppc, size=size)
                    for alpha in [1, 5, 10, 15, 20, 25, 30]:
                        cost_type = f"cluster_based_c1_{ppc}_c2_{ppc + alpha}"
                        try:
                            generate_table_from_csv(dataset, initial_set, cost_type, method_map, delta=delta)
                        except Exception:
                            pass
