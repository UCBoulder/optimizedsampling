import pandas as pd
import numpy as np

from utils import *
from format_data import *
import ast

'''
    After reading from csv file, formats dataframe with multi-index label and subset
'''
def format_dataframe_with_budget(df):
    # Split the index into 'label' and 'size_with_prefix' parts
    split = pd.DataFrame(df.index.str.split(';').tolist(), index=df.index, columns=['label', 'budget_with_prefix'])
    df['label'] = split['label']
    df['budget_with_prefix'] = split['budget_with_prefix']

    # Handle 'sizeNone' by assigning it a placeholder value (e.g., float('inf') for whole dataset)
    #change hardcoding
    df['Budget'] = df.apply(
        lambda row: row['budget_with_prefix'].replace('budget', ''),
        axis=1
    )

    # Set 'label' and 'size_of_subset' as a MultiIndex and drop the old index columns
    df.set_index(['label', 'Budget'], inplace=True)

    df.drop(columns='budget_with_prefix', inplace=True)

    df = df.reset_index()
    df = df.set_index(['label', 'Budget'])
    return df

'''
    After reading from csv file, formats dataframe with multi-index label and subset
'''
def format_dataframe_with_cost(df):
    # Split the index into 'label' and 'size_with_prefix' parts
    split = pd.DataFrame(df.index.str.split(';').tolist(), index=df.index, columns=['label', 'cost_with_prefix'])
    df['label'] = split['label']
    df['cost_with_prefix'] = split['cost_with_prefix']

    df['Cost'] = df.apply(
        lambda row: row['cost_with_prefix'].replace('cost', ''),
        axis=1
    )

    # Set 'label' and 'size_of_subset' as a MultiIndex and drop the old index columns
    df.set_index(['label', 'Cost'], inplace=True)

    df.drop(columns='cost_with_prefix', inplace=True)

    df['Test R2'] = df['Test R2'].apply(ast.literal_eval)
    df = df.explode('Test R2', ignore_index=False)

    df = df.reset_index()
    df = df.set_index(['label', 'Cost'])
    return df

'''
    Formats cost of dataframe
    (Use when array of cost is recorded to get total cost)
'''
def format_cost(df):
    format_dataframe(df)

    cost_path = "data/cost/costs_by_city_dist.pkl"

    pop_cost = costs_of_train_data(cost_path, retrieve_train_ids("population")).sum()
    tree_cost = costs_of_train_data(cost_path, retrieve_train_ids("treecover")).sum()
    el_cost = costs_of_train_data(cost_path, retrieve_train_ids("elevation")).sum()

    df.loc[("population", 54340.0), "Cost"] = pop_cost
    df.loc[("treecover", 97875.0), "Cost"] = tree_cost
    df.loc[("elevation", 97875.0), "Cost"] = el_cost

    return df

def compute_avg_std(csv_file, costs, tolerance=10, output_file="output.csv"):
    # Load the CSV file
    df = pd.read_csv(csv_file, header=None, names=["label", "Cost", "Test R2"])
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    df = df.dropna(subset=["Cost"])

    df["Test R2"] = pd.to_numeric(df["Test R2"], errors="coerce")
    df = df.dropna(subset=["Test R2"])
    
    results = []
    for label in df['label'].unique():
        label_df = df[df["label"] == label]  # Filter by label
        for target_cost in costs:
            # Filter rows where Cost is within the target range (±tolerance)
            filtered_df = label_df[(label_df["Cost"] >= target_cost - tolerance) & (label_df["Cost"] <= target_cost + tolerance)]
            
            # Compute mean and standard deviation of Test R2 values
            mean_r2 = filtered_df["Test R2"].mean()
            std_r2 = filtered_df["Test R2"].std()

            results.append([label, target_cost, mean_r2, std_r2])
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results, columns=["label", "Cost", "Avg R2", "Std R2"])
    results_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    # for method in ['clusters', 'greedycost']:
    #     df = pd.read_csv(f"results/Torchgeo4096_{method}_cost_cluster_urban_areas.csv", index_col=0)
    #     df = format_dataframe_with_cost(df)
    #     df.to_csv(f"results/Torchgeo4096_{method}_cost_cluster_urban_areas_formatted.csv", index=True)

    for method in ['random', 'invsize', 'greedycost', 'clusters']:
        lambda_str = ''
        if method == 'invsize':
            lambda_str = 'lambda_0.5_'

        csv_file = f"results/Torchgeo4096_{method}_cost_cluster_NLCD_percentages_1347_{lambda_str}formatted.csv"
        output_file = f"results/Torchgeo4096_{method}_cost_cluster_NLCD_percentages_1347_{lambda_str}averaged.csv"
        costs = list(range(500, 5001, 500))
        compute_avg_std(csv_file, costs, output_file=output_file)