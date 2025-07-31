import pandas as pd

#base_method_order = ['random', 'random_unit', 'greedycost', 'poprisk_avg', 'poprisk', 'poprisk_img', 'poprisk_img_8', 'similarity', 'diversity']
base_method_order = ['random', 'random_unit', 'greedycost', 'poprisk_avg', 'poprisk', 'poprisk_img']
# base_method_order = ['poprisk', 'poprisk2']

#base_method_order = ['poprisk_avg', 'poprisk', 'poprisk_img', 'poprisk_img_8', 'poprisk_avg_img', 'poprisk_avg_img_8']

def load_df(dataset, init_set, cost_type, csv_path="results/aggregated_r2_{dataset}_{init_set}_{cost_type}.csv"):
    csv_path = csv_path.format(dataset=dataset, init_set=init_set, cost_type=cost_type)
    df = pd.read_csv(csv_path)
    return df

def generate_latex_table(df, dataset, init_set, cost_type, method_map):
    lines = []
    lines.append("\\hline%")

    method_labels = [method_map.get(method, method) for method in base_method_order]

    for _, row in df.iterrows():
        budget = int(row["budget"])
        cells = []
        for method in method_labels:
            print(method)
            mean = row.get(f"{method}_updated_r2_mean")
            std = row.get(f"{method}_updated_r2_std")
            if pd.notnull(mean) and pd.notnull(std):
                cells.append(f"{mean:.2f} ± {std:.2f}")
            else:
                cells.append("--")
        print(len(cells))
        lines.append(f"{budget} & " + " & ".join(cells) + "\\\\%")
    lines.append("\\hline%")

    tex_path = f"latex_table_r2/latex_table_r2_{dataset}_{init_set}_{cost_type}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f'Wrote latex table to {tex_path}')

def generate_delta_latex_table(df, dataset, init_set, cost_type, method_map):
    lines = []
    lines.append("\\hline%")

    method_labels = [method_map.get(method, method) for method in base_method_order]

    for _, row in df.iterrows():
        budget = int(row["budget"])
        cells = []
        for method in method_labels:
            print(method)
            mean = row.get(f"{method}_delta_r2_mean")
            std = row.get(f"{method}_delta_r2_se")
            if pd.notnull(mean) and pd.notnull(std):
                cells.append(f"{mean:.2f} ± {std:.2f}")
            else:
                cells.append("--")
        print(len(cells))
        lines.append(f"{budget} & " + " & ".join(cells) + "\\\\%")
    lines.append("\\hline%")

    tex_path = f"latex_table_r2/latex_table_delta_r2_{dataset}_{init_set}_{cost_type}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f'Wrote latex table to {tex_path}')

def generate_table_from_csv(dataset, 
                            init_set, 
                            cost_type, 
                            method_map,
                            delta=False,
                            csv_path="results/aggregated_r2_{dataset}_{init_set}_{cost_type}.csv"):
    df = load_df(dataset, init_set, cost_type, csv_path=csv_path)
    if delta:
        generate_delta_latex_table(df, dataset, init_set, cost_type, method_map)
    else:
        generate_latex_table(df, dataset, init_set, cost_type, method_map)

if __name__ == "__main__":

    DATASET_NAMES = {
        "INDIA_SECC", "USAVARS_POP", "USAVARS_TC", "TOGO_PH_H2O"
    }

    initial_set_strs = {
        'USAVARS_POP': '{num_strata}_fixedstrata_{ppc}ppc_{size}_size',
        'USAVARS_TC': '{num_strata}_fixedstrata_{ppc}ppc_{size}_size',
        'INDIA_SECC': '{num_strata}_state_district_desired_{ppc}ppc_{size}_size',
        'TOGO_PH_H2O': '{num_strata}_strata_desired_{ppc}ppc_{size}_size'
    }

    group_types = {
        'USAVARS_POP': 'nlcd',
        'USAVARS_TC': 'nlcd',
        'INDIA_SECC': 'urban_rural',
        'TOGO_PH_H2O': 'regions'
    }

    cluster_nums = {
        'USAVARS_POP': '8',
        'USAVARS_TC': '8',
        'INDIA_SECC': '8',
        'TOGO_PH_H2O': '8'
    }

    import argparse
    parser = argparse.ArgumentParser(description="Run sampling evaluation script")
    parser.add_argument("--delta", type=lambda x: x.lower() == 'true', default=False,
                        help="Whether to include delta R² (True/False)")
    args = parser.parse_args()
    delta = args.delta
    print(delta)

    for dataset in DATASET_NAMES:
        method_map = {
            'poprisk': f'poprisk_{group_types[dataset]}_0.5',
            'poprisk2': f'poprisk_{group_types[dataset]}_1.0',
            'poprisk_avg': f'poprisk_avg_{group_types[dataset]}_0.5',
            'poprisk_img': f'poprisk_image_clusters_{cluster_nums[dataset]}_0.5',
            'poprisk_avg_img': f'poprisk_avg_image_clusters_{cluster_nums[dataset]}_0.5',
        }

        for ppc in [10, 20, 25]:
            for size in [100, 200, 300, 500, 2000]:
                for num_strata in [2, 5, 10]:
                    initial_set = initial_set_strs[dataset].format(num_strata=num_strata, ppc=ppc, size=size)
                    sampling_type = 'cluster_sampling'
                    for alpha in [1, 5, 10, 15, 20, 25, 30]:
                        in_region_cost = ppc
                        out_of_region_cost = in_region_cost + alpha
                        cost_type = f"cluster_based_c1_{in_region_cost}_c2_{out_of_region_cost}"
                        try:
                            generate_table_from_csv(dataset, initial_set, cost_type, method_map, delta=delta)
                        except Exception as e:
                            print(e)
