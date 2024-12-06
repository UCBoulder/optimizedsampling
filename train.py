from pathlib import Path
import pandas as pd
import dill

from regressions import run_regression, results, costs
from oed import *
from pca import *
from satclip import get_satclip
from feasibility import *

from mosaiks.code.mosaiks import config as cfg
from mosaiks.code.mosaiks.utils import io

#Run
def run(labels_to_run, rule):
    for label in labels_to_run:
        valid = 0

        if label=="population":
            valid = 54343
        else:
            valid = 97876

        size_of_subset = [0.005, 0.01, 0.05, 0.1, 0.20, 0.35, 0.5, 0.75]
        size_of_subset = [int(np.floor(valid*percent)) for percent in size_of_subset]
        size_of_subset = [n - (n%5) for n in size_of_subset]

        for size in size_of_subset:
            run_regression(label, rule=rule, subset_size=size)
        run_regression(label, rule=None, subset_size=None)

    #Save results (R2 score) in csv
    results_df = pd.DataFrame(
        {"Test R2": results,
         "Cost": costs
        }
    )
    results_df.index.name = "label"

    results_df.to_csv(Path("results/TestSetPerformance{rule}withCost.csv".format(rule=rule)), index=True)

labels_to_run = ["population", "treecover", "elevation"]

#Run Random
#run(labels_to_run, rule="random")

#Run with V Optimal Design
#run(labels_to_run, rule="image")

#Run with SatCLIP embeddings
#run(labels_to_run, rule="satclip")

#Run with greedy cost algorithm
#run(labels_to_run, rule='lowcost')

#Run with greedy dist algorithm
#run(labels_to_run, rule='dist')

#Run with binary rad algorithm
run(labels_to_run, rule='rad')