from pathlib import Path
import pandas as pd
import dill
import argparse

from regressions import run_regression, r2_dict, avgr2, stdr2, num_samples_dict
import config as c
from cost import *
from format_csv import *

#Run
def run(labels_to_run, 
        cost_func, 
        budgets=[10, 100, 1e3, 1e4, 1e5, 1e6], 
        rule='random', 
        avg_results = False,
        **kwargs):

    for label in labels_to_run:
        c.used_all_samples = False
        for budget in budgets:
            run_regression(label, cost_func, rule=rule, budget=budget, avg_results=avg_results, **kwargs)
            if c.used_all_samples:
                break

    #Save results (R2 score) in csv
    if avg_results:
        results_df = pd.DataFrame(
            {"Test Avg R2": avgr2,
            "Test Std R2": stdr2
            }
        )

    else:
        results_df = pd.DataFrame(
            {"Test R2": r2_dict
            }
        )
    results_df.index.name = "label"
    num_samples_df = pd.DataFrame(
            {"Sample Size": num_samples_dict
            }
        )

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
        state_set = kwargs.get('states', 0)
        for region, states in states_dict.items():
            if state_set == states:
                state_str = region
        gamma = kwargs.get('gamma', 1)
        cost_str = f'State_{state_str}_{gamma}'
    elif cost_func == compute_cluster_cost:
        cluster_type = kwargs.get('cluster_type', 'NLCD_percentages')
        cost_str = f'cost_cluster_{cluster_type}'

    lambda_str = ''
    if rule == 'invsize':
        l = kwargs.get('l', 0.5)
        lambda_str = f"_lambda_{l}"
        cluster_type = kwargs.get('cluster_type', 'NLCD_percentages')
        rule = f'{rule}_{cluster_type}'

    test_str = ''
    test_split = kwargs.get('test_split', None)
    if test_split is not None:
        for region, states in states_dict.items():
            if test_split == states:
                test_str = f'_test_{region}'

    #num_samples_df.to_csv(f"NUMSAMPLES_{rule}_{cost_str}{lambda_str}.csv", index=True)
    results_df.to_csv(Path(f"results/plus_final_{rule}_{cost_str}{lambda_str}{test_str}.csv"), index=True)
    results_df = pd.read_csv(f"results/plus_final_{rule}_{cost_str}{lambda_str}{test_str}.csv", index_col=0)
    results_df = format_dataframe_with_cost(results_df)
    results_df.to_csv(f"results/plus_final_{rule}_{cost_str}{lambda_str}{test_str}.csv", index=True)

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
    default=1.0,
    type=float,
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
parser.add_argument(
    '--avg',
    default=False,
    type=str,
    help='Average results'
)
parser.add_argument(
    '--l',
    default=0.5,
    type=float,
    help='Hyperparameter lambda in optimization problem'
)
parser.add_argument(
    '--cluster_type',
    default='NLCD_percentages',
    type=str,
    help='Cluster type'
)
parser.add_argument(
    '--test_split',
    default=None,
    type=str,
    nargs='+',
    help='Test region'
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
elif cost_func == "cluster":
    cost_func = compute_cluster_cost

states_dict = {
    "West": {
        'Montana', 'Idaho', 'Wyoming', 'Colorado', 'New Mexico',  # Mountain West
        'Washington', 'Oregon', 'California', 'Nevada',  # Pacific
        'Utah', 'Arizona'  # Southwest
    },
    "East": {
        'Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 'Connecticut',  # New England
        'New York', 'New Jersey', 'Pennsylvania',  # Mid-Atlantic
        'Delaware', 'Maryland', 'Virginia', 'West Virginia',  # Mid-Atlantic/Southeast
        'Kentucky', 'Tennessee', 'North Carolina', 'South Carolina', 'Georgia', 'Florida',  # Southeast
        'Alabama', 'Mississippi',  # Deep South
        'Ohio', 'Michigan', 'Indiana', 'Illinois', 'Wisconsin',  # Great Lakes
        'Missouri', 'Iowa', 'Minnesota', 'North Dakota', 'South Dakota', 'Nebraska', 'Kansas'  # Central Plains
    },
    "North": {
        'Washington', 'Oregon', 'Idaho', 'Montana', 'Wyoming',  # Northwest/Rockies
        'North Dakota', 'South Dakota', 'Nebraska',  # Northern Plains
        'Minnesota', 'Wisconsin', 'Iowa',  # Upper Midwest
        'Michigan', 'Illinois', 'Indiana', 'Ohio',  # Great Lakes
        'Pennsylvania', 'New York', 'New Jersey',  # Mid-Atlantic
        'Connecticut', 'Rhode Island', 'Massachusetts', 'Vermont', 'New Hampshire', 'Maine'  # New England
    },
    "South": {
        'California', 'Nevada', 'Utah', 'Colorado',  # Southwest/Rockies
        'Kansas', 'Missouri', 'Kentucky', 'West Virginia',  # Borderline states near the cutoff
        'Virginia', 'Maryland', 'Delaware',  # Mid-Atlantic/Southeast
        'North Carolina', 'South Carolina', 'Georgia', 'Florida',  # Southeast
        'Tennessee', 'Arkansas', 'Alabama', 'Mississippi',  # Deep South
        'Louisiana', 'Texas', 'Oklahoma', 'New Mexico', 'Arizona'  # Southern Plains & Southwest
    }
}

states=args.states
test_split=args.test_split
if states is not None:
    if states[0] in states_dict:
        states = states_dict.get(states[0], set())

if test_split is not None:
    if test_split[0] in states_dict:
        test_split = states_dict.get(test_split[0], set())

run(
    args.labels, 
    cost_func, 
    budgets=args.budgets,
    rule=args.method,
    avg_results=args.avg, 
    alpha=args.alpha, 
    beta=args.beta, 
    gamma=args.gamma, 
    r=args.radius,
    states=states,
    l=args.l,
    cluster_type=args.cluster_type,
    test_split=test_split
    )