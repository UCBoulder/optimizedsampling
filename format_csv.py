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

if __name__ == '__main__':
    for method in ['clusters', 'greedycost']:
        df = pd.read_csv(f"results/Torchgeo4096_{method}_cost_cluster_urban_areas.csv", index_col=0)
        df = format_dataframe_with_cost(df)
        df.to_csv(f"results/Torchgeo4096_{method}_cost_cluster_urban_areas_formatted.csv", index=True)

    # for region in ['Northeast', 'West', 'Midwest', 'South']:
    #     for method in ['invsize']:
    #         df = pd.read_csv(f"results/Torchgeo4096_{method}_State_{region}_urban_clusters.csv", index_col=0)
    #         df = format_dataframe_with_cost(df)
    #         df.to_csv(f"results/Torchgeo4096_{method}_State_{region}_urban_clusters_formatted.csv", index=True)