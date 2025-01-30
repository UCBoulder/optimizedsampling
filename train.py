from pathlib import Path
import pandas as pd
import dill
import argparse

from regressions import run_regression, r2_dict
import config as c
from cost import *

#Run
def run(labels_to_run, 
        cost_func, 
        budgets=[10, 100, 1e3, 1e4, 1e5, 1e6], 
        rule='random', 
        **kwargs):

    for label in labels_to_run:
        c.used_all_samples = False
        for budget in budgets:
            run_regression(label, cost_func, rule=rule, budget=budget, **kwargs)
            if c.used_all_samples:
                break

    #Save results (R2 score) in csv
    results_df = pd.DataFrame(
        {"Test R2": r2_dict
        }
    )
    results_df.index.name = "label"

    if cost_func == compute_unif_cost:
        cost_str = 'Unif'
    elif cost_func == compute_lin_cost:
        alpha = kwargs.get('alpha', 1)
        beta = kwargs.get('beta', 1)
        cost_str = f'Lin_alpha{alpha}_beta{beta}'
    elif cost_func == compute_lin_w_r_cost:
        alpha = kwargs.get('alpha', 1)
        beta = kwargs.get('beta', 1)
        gamma = kwargs.get('gamma', 1)
        r = kwargs.get('r', 0)
        cost_str = f'LinRad_alpha{alpha}_beta{beta}_gamma{gamma}_rad{r}'
    elif cost_func == compute_state_cost:
        state = kwargs.get('states', 0)
        gamma = kwargs.get('gamma', 1)
        cost_str = f'State_{state}_{gamma}'

    results_df.to_csv(Path(f"results/Torchgeo4096_{rule}_{cost_str}.csv"), index=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--labels', 
    nargs='+', 
    default = ['population', 'elevation', 'treecover'],
    help='Labels to run')
parser.add_argument(
    '--method',
    required=True,
    help='Sampling method: random, image, satclip, greedycost, clusters'
)
parser.add_argument(
    '--cost',
    default = 'unif',
    help='Cost function: unif, lin, lin+rad'
)
parser.add_argument(
    '--budgets',
    default=[10, 100, 1e3, 1e4, 1e5, 1e6],
    nargs='+',
    type=float,
    help='Budget'
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
parser.add_argument(
    '--states',
    default=None,
    nargs='+',
    type=str,
    help='States to sample from'
)

args = parser.parse_args()

cost_func = args.cost
if cost_func == "unif":
    cost_func = compute_unif_cost
elif cost_func == "lin":
    cost_func = compute_lin_cost
elif cost_func == "lin+rad":
    cost_func = compute_lin_w_r_cost
elif cost_func == "state":
    cost_func = compute_state_cost

run(
    args.labels, 
    cost_func, 
    budgets=args.budgets,
    rule=args.method, 
    alpha=args.alpha, 
    beta=args.beta, 
    gamma=args.gamma, 
    r=args.radius,
    states=args.states
    )