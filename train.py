from pathlib import Path
import pandas as pd
import dill

from regressions import run_regression, results
from oed import *
from pca import *
from satclip import get_satclip
from feasibility import *

from mosaiks.code.mosaiks import config as cfg
from mosaiks.code.mosaiks.utils import io

#Run
def run(labels_to_run, cost_func, *params, rule='random'):
    for label in labels_to_run:
        budgets = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

        for budget in budgets:
            run_regression(label, cost_func, *params, budget=budget, rule=rule)

    #Save results (R2 score) in csv
    results_df = pd.DataFrame(
        {"Test R2": results
        }
    )
    results_df.index.name = "label"

    results_df.to_csv(Path("results/TestSetPerformance{rule}withBudget.csv".format(rule=rule)), index=True)

labels_to_run = ["population", "treecover", "elevation"]
alpha = 1
beta = 100
c1 = 1
r = 500

cost_func = cost_lin_with_r

#Run Random
#run(labels_to_run, cost_func, alpha, beta, c1, r, rule="random")

#Run with V Optimal Design
#run(labels_to_run, rule="image")

#Run with SatCLIP embeddings
#run(labels_to_run, rule="satclip")

#Run with linear cost function greedy algorithm
#run(labels_to_run, cost_lin, alpha, beta, rule='greedycost')

#Run with linear outside of radius cost function greedy algorithm
run(labels_to_run, cost_lin_with_r, alpha, beta, c1, r, rule='greedycost')

#Run with binary wrt radius cost function greedy algorithm
#run(labels_to_run, rule='binradcost')