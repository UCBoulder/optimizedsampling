import pandas as pd
import numpy as np

from utils import *
from format_data import *
import ast


'''
    After reading from csv file, formats dataframe with multi-index label and subset
'''
def format_dataframe(df):
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
    #Jointobj
    df = pd.read_csv(f"results/Torchgeo4096_jointobj_State_['California', 'Colorado']_1.csv", index_col=0)
    df = format_dataframe(df)
    df.to_csv(f"results/Torchgeo4096_jointobj_State_['California', 'Colorado']_formatted.csv", index=True)

    df = pd.read_csv(f"results/Torchgeo4096_jointobj_State_['California', 'Oregon', 'Washington', 'Idaho', 'Montana', 'Wyoming', 'Utah', 'Colorado', 'Arizona', 'New Mexico']_1.csv", index_col=0)
    df = format_dataframe(df)
    df.to_csv(f"results/Torchgeo4096_jointobj_State_['California', 'Oregon', 'Washington', 'Idaho', 'Montana', 'Wyoming', 'Utah', 'Colorado', 'Arizona', 'New Mexico']_formatted.csv", index=True)

    df = pd.read_csv(f"results/Torchgeo4096_jointobj_State_['Colorado']_1.csv", index_col=0)
    df = format_dataframe(df)
    df.to_csv(f"results/Torchgeo4096_jointobj_State_['Colorado']_formatted.csv", index=True)

    #Random
    df = pd.read_csv(f"results/Torchgeo4096_random_State_['California', 'Oregon', 'Washington', 'Idaho', 'Montana', 'Wyoming', 'Utah', 'Colorado', 'Arizona', 'New Mexico']_1.csv", index_col=0)
    df = format_dataframe(df)
    df.to_csv(f"results/Torchgeo4096_random_State_['California', 'Oregon', 'Washington', 'Idaho', 'Montana', 'Wyoming', 'Utah', 'Colorado', 'Arizona', 'New Mexico']_formatted.csv", index=True)