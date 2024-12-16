from pathlib import Path
import pandas as pd
import dill
import argparse

from regressions import run_regression, avgr2, stdr2
from cost import *

budgets = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]

#Run
def run(labels_to_run, cost_func, rule='random', **kwargs):
    for label in labels_to_run:
        for budget in budgets:
            run_regression(label, cost_func, rule=rule, budget=budget, **kwargs)

    #Save results (R2 score) in csv
    results_df = pd.DataFrame(
        {"Test Avg R2": avgr2,
         "Test Std R2": stdr2
        }
    )
    results_df.index.name = "label"

    results_df.to_csv(Path("results/TestSetPerformance{rule}withBudget_torchgeo4096.csv".format(rule=rule)), index=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--labels', 
    nargs='+', 
    default = ['population', 'elevation', 'treecover'],
    help='Labels to run')
parser.add_argument(
    '--method',
    required=True,
    help='Sampling method: random, image, satclip, greedycost'
)
parser.add_argument(
    '--cost',
    default = 'unif',
    help='Cost function: unif, lin, lin+rad'
)
parser.add_argument(
    '-a', '--alpha',
    default=1,
    type=int,
    help='Value of alpha in alpha*dist+beta'
)
parser.add_argument(
    '-b', '--beta',
    default=1,
    type=int,
    help='Value of beta in alpha*dist+beta'
)
parser.add_argument(
    '--gamma',
    default=1,
    type=int,
    help='Value of constant valued in rad or unif'
)
parser.add_argument(
    '--radius',
    default=500,
    type=int,
    help='Radius around city'
)

args = parser.parse_args()

cost_func = args.cost
if cost_func == "unif":
    cost_func = compute_unif_cost
elif cost_func == "lin":
    cost_func = compute_lin_cost
elif cost_func == "lin+rad":
    cost_func = compute_lin_w_r_cost

run(
    args.labels, 
    cost_func, 
    rule=args.method, 
    alpha=args.alpha, 
    beta=args.beta, 
    gamma=args.gamma, 
    r=args.radius
    )