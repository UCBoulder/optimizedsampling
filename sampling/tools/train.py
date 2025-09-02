import os
import sys
import argparse
import numpy as np
import dill
import json
from datetime import datetime
from utils import get_free_cpus

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import torch

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

# Local imports
from pycls.al.ActiveLearning import ActiveLearning
from pycls.core.config import cfg, dump_cfg
from pycls.datasets.data import Data
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def argparser():
    parser = argparse.ArgumentParser(description='Subset Selection - Ridge Regression')
    parser.add_argument('--cfg', dest='cfg_file', required=True, type=str)
    parser.add_argument('--exp-name', required=True, type=str)
    parser.add_argument('--sampling_fn', required=True, type=str)
    #parser.add_argument('--random_strategy', default=None, type=str)
    parser.add_argument('--budget', required=True, type=int)
    parser.add_argument('--initial_size', default=0, type=int)
    parser.add_argument('--id_path', default=None, type=str)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--initial_set_str', default="empty_initial_set", type=str)

    parser.add_argument('--cost_func', default=None, type=str)
    parser.add_argument('--cost_name', default=None, type=str)
    parser.add_argument('--cost_array_path', default=None, type=str)

    parser.add_argument('--group_type', default=None, type=str)
    parser.add_argument('--group_assignment_path', default=None, type=str)
    parser.add_argument('--ignore_groups', default=None, type=list)

    parser.add_argument('--unit_type', default=None, type=str)
    parser.add_argument('--points_per_unit', default=None, type=int)
    parser.add_argument('--unit_assignment_path', default=None, type=str)
    parser.add_argument('--unit_cost_path', default=None, type=str)

    parser.add_argument('--region_assignment_path', default=None, type=str)
    parser.add_argument('--in_region_unit_cost', default=None, type=int)
    parser.add_argument('--out_of_region_unit_cost', default=None, type=int)
    parser.add_argument('--alpha', default=None, type=float)

    parser.add_argument('--util_lambda', default=0.5, type=float)

    parser.add_argument('--similarity_matrix_path', default=None, type=str)
    parser.add_argument('--train_similarity_matrix_path', default=None, type=str)
    parser.add_argument('--similarity_per_unit_path', default=None, type=str)
    return parser

def main(cfg, train_data):
    cfg.OUT_DIR = os.path.abspath(cfg.OUT_DIR)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    dataset_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME)
    os.makedirs(dataset_dir, exist_ok=True)


    if cfg.EXP_NAME == 'auto':
        now = datetime.now()
        exp_dir = now.strftime('%Y%m%d_%H%M%S')
    else:
        seed_str = f"seed_{cfg.RNG_SEED}"

        if cfg.ACTIVE_LEARNING.SAMPLING_FN in ["match_population_proportion", "poprisk", "poprisk_avg"]:
            sampling_str = f"{cfg.ACTIVE_LEARNING.SAMPLING_FN}/{cfg.GROUPS.GROUP_TYPE}"
        else:
            sampling_str = f"{cfg.ACTIVE_LEARNING.SAMPLING_FN}"

        base_dir = (f"{cfg.INITIAL_SET.STR}/{cfg.COST.NAME}/opt/{sampling_str}/budget_{cfg.ACTIVE_LEARNING.BUDGET_SIZE}"
                    if cfg.ACTIVE_LEARNING.OPT
                    else f"{cfg.INITIAL_SET.STR}/{cfg.COST.NAME}/{sampling_str}/budget_{cfg.ACTIVE_LEARNING.BUDGET_SIZE}")
        print(base_dir)

        # Add random_strategy subfolder if cost_name is cluster_based and sampling_fn is random
        # if cfg.COST.NAME == "cluster_based" and cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":
        #     random_strategy = getattr(cfg.ACTIVE_LEARNING, "RANDOM_STRATEGY", None)
        #     if random_strategy:
        #         base_dir = f"{base_dir}/{random_strategy}"

        # Append util_lambda if sampling_fn is poprisk
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("poprisk"):
            util_lambda = getattr(cfg.ACTIVE_LEARNING, "UTIL_LAMBDA", None)
            if util_lambda is not None:
                base_dir = f"{base_dir}/util_lambda_{util_lambda}"

        # Check if INITIAL_SET.STR already ends with seed_{seed}
        if not cfg.INITIAL_SET.STR.endswith(seed_str):
            exp_dir = f"{base_dir}/{seed_str}"
        else:
            exp_dir = base_dir

    exp_dir = os.path.join(dataset_dir, exp_dir)
    # if os.path.exists(exp_dir) and cfg.ACTIVE_LEARNING.SAMPLING_FN not in ['similarity', 'diversity']:
    #     print("Experiment already done!")
    #     return
    
    os.makedirs(exp_dir, exist_ok=True)
    cfg.EXP_DIR = exp_dir
    cfg.INITIAL_SET_DIR = os.path.join(dataset_dir, cfg.INITIAL_SET.STR)
    cfg.SAMPLING_DIR = os.path.join(dataset_dir, base_dir)
    dump_cfg(cfg)

    lu.setup_logging(cfg)
    logger.info(f"Experiment directory: {exp_dir}")

    cfg.DATASET.ROOT_DIR = os.path.abspath(cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    #train_data, _ = data_obj.getDataset(isTrain=True)
    test_data, _ = data_obj.getDataset(isTrain=False)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    param_grid = {
        'ridge__alpha': np.logspace(-5, 5, 10)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    ridge_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',        
        cv=cv,
        n_jobs=-1                   #parallelize across folds
    )

    if cfg.LSET_IDS:
        lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets_from_ids(cfg.LSET_IDS, data=train_data, save_dir=cfg.EXP_DIR)
        lSet, uSet, _= data_obj.loadPartitions(lSetPath=lSet_path, \
            uSetPath=uSet_path, valSetPath = valSet_path )
    else:
        lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets_from_ids([], data=train_data, save_dir=cfg.EXP_DIR)
        lSet, uSet, _= data_obj.loadPartitions(lSetPath=lSet_path, \
            uSetPath=uSet_path, valSetPath = valSet_path )


    def evaluate_r2(model, X_test, y_test):
        return model.score(X_test, y_test)
    
    X_test, y_test = test_data[:][0], test_data[:][1]

    run_regression = True
    if len(lSet) > 0:
        initial_set_r2_path = f"{cfg.INITIAL_SET_DIR}/initial_set_r2_seed_{cfg.RNG_SEED}.json"
        if os.path.exists(initial_set_r2_path):

            print("Loading initial set r2 performance...")
            try:
                with open(initial_set_r2_path, "r") as f:
                    r2 = json.load(f)
                if isinstance(r2, dict):
                    r2 = r2['initial_set_r2']
                logger.info(f"Initial R² score: {r2:.4f}")
                run_regression = False
            except Exception as e:
                run_regression = True
        if run_regression:
            n_splits = 5
            print("Training ridge regression on initial set...")
            logger.info("Training ridge regression on initial set...")
            lSet = lSet.astype(int)
            uSet = uSet.astype(int)
            X_train, y_train = train_data[lSet][0], train_data[lSet][1]

            if X_train.shape[0] < 5:
                print("Not enough samples...")
            else:
                ridge_search.fit(X_train, y_train)
                r2 = evaluate_r2(ridge_search, X_test, y_test)
                logger.info(f"Initial R² score: {r2:.4f}")

                os.makedirs(cfg.INITIAL_SET_DIR, exist_ok=True)
                with open(initial_set_r2_path, "w") as f:
                    json.dump(r2, f)

            train_data.plot_subset_on_map(lSet, save_path=f"{cfg.INITIAL_SET_DIR}/lSet_plot.png")
    else:
        logger.info("Initial labeled set is empty; skipping to subset selection.")

    logger.info("Starting subset selection...")
    al_obj = ActiveLearning(data_obj, cfg)
    activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data, X_train=train_data.X)
    print(f"Sampled {len(activeSet)} points!")
    if len(activeSet) > 0:
        train_data.plot_subset_on_map(activeSet, save_path=f"{exp_dir}/activeSet_plot.png")

    logger.info(f"Selected {len(activeSet)} new samples.")
    data_obj.saveSets(lSet, uSet, activeSet, os.path.join(cfg.EXP_DIR, 'episode_0'))

    lSet_updated = np.concatenate((lSet, activeSet))
    lSet_updated = lSet_updated.astype(int)
    train_data.plot_subset_on_map(lSet_updated, save_path=f"{exp_dir}/lSet_updated_plot.png")

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    param_grid = {
        'ridge__alpha': np.logspace(-5, 5, 10)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    ridge_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',        
        cv=cv,
        n_jobs=get_free_cpus()                  #parallelize across folds
    )

    # Train again on the updated labeled set
    if lSet_updated.size > 0:
        n_splits = 5
        logger.info("Training ridge regression on updated labeled set...")
        X_train_updated, y_train_updated = train_data[lSet_updated][0], train_data[lSet_updated][1]

        if n_splits > X_train_updated.shape[0]:
            print("Not enough samples...")
        else:
            ridge_search.fit(X_train_updated, y_train_updated)
            r2_updated = evaluate_r2(ridge_search, X_test, y_test)
            logger.info(f"Updated R² score: {r2_updated:.4f}")

    data_obj.saveSets(lSet_updated, new_uSet, activeSet, os.path.join(cfg.EXP_DIR, 'episode_1'))

def main_wrapper():
    args = argparser().parse_args()

    cfg.merge_from_file(args.cfg_file)
    cfg.EXP_NAME = args.exp_name
    cfg.ACTIVE_LEARNING.SAMPLING_FN = args.sampling_fn
    cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget
    cfg.INITIAL_SET.STR = args.initial_set_str
    cfg.RNG_SEED = args.seed

    cfg.DATASET.ROOT_DIR = os.path.abspath(cfg.DATASET.ROOT_DIR)

    data_obj = Data(cfg)
    train_data, _ = data_obj.getDataset(isTrain=True)

    if args.id_path:
        with open(args.id_path, "rb") as f:
            arrs = dill.load(f)
        if isinstance(arrs, dict) and "sampled_ids" in arrs:
            ids = arrs["sampled_ids"]
        else:
            ids = arrs  #assume list or array
        cfg.LSET_IDS = ids if isinstance(ids, list) else ids.tolist()
    else:
        cfg.LSET_IDS = []

    if args.cost_func:
        cfg.COST.FN = args.cost_func
        cfg.COST.NAME = args.cost_name or args.cost_func
    else:
        cfg.COST.FN = 'uniform'
        cfg.COST.NAME = 'uniform'

    cfg.ACTIVE_LEARNING.OPT = args.sampling_fn in [
        'greedycost', 'poprisk', 'similarity', 'diversity', 'poprisk_avg'
    ]

    if args.cost_array_path:
        with open(args.cost_array_path, "rb") as f:
            loaded = dill.load(f)

        if 'ids' in loaded:
            idx_to_cost = dict(zip(loaded['ids'], loaded['costs']))
            cost_array = [idx_to_cost[idx] for idx in train_data.ids]
        else:
            cost_array = loaded['assignments']

        cfg.COST.ARRAY = [float(x) for x in cost_array]

    if args.group_assignment_path:
        cfg.GROUPS.GROUP_TYPE = args.group_type

        with open(args.group_assignment_path, "rb") as f:
            loaded = dill.load(f)

        if isinstance(loaded, dict):
            idx_to_assignment = loaded
        elif 'ids' in loaded:
            idx_to_assignment = dict(zip(loaded['ids'], loaded['assignments']))
        else:
            idx_to_assignment = dict(zip(train_data.ids, loaded['assignments']))

        assignments_ordered = [idx_to_assignment[idx] for idx in train_data.ids]
        cfg.GROUPS.GROUP_ASSIGNMENT = [str(x) for x in assignments_ordered]

        if args.ignore_groups:
            print("see arg ignored groups")
            cfg.GROUPS.IGNORED_GROUPS = args.ignore_groups

    if args.unit_assignment_path:
        cfg.UNITS.UNIT_TYPE = args.unit_type

        with open(args.unit_assignment_path, "rb") as f:
            loaded = dill.load(f)

        if isinstance(loaded, dict):
            idx_to_assignment = loaded
        elif 'ids' in loaded:
            idx_to_assignment = dict(zip(loaded['ids'], loaded['assignments']))
        else:
            idx_to_assignment = dict(zip(train_data.ids, loaded['assignments']))

        try:
            assignments_ordered = [idx_to_assignment[idx] for idx in train_data.ids]
        except Exception as e:
            from IPython import embed; embed()
        cfg.UNITS.UNIT_ASSIGNMENT = [str(x) for x in assignments_ordered]
        cfg.UNITS.POINTS_PER_UNIT = args.points_per_unit

        if args.unit_cost_path:
            cfg.COST.UNIT_COST_PATH = args.unit_cost_path

        if args.alpha:
            cfg.COST.ALPHA = args.alpha

    if args.region_assignment_path:
        with open(args.region_assignment_path, "rb") as f:
            loaded = dill.load(f)

        if isinstance(loaded, dict):
            idx_to_assignment = loaded
        elif 'ids' in loaded:
            idx_to_assignment = dict(zip(loaded['ids'], loaded['assignments']))
        else:
            idx_to_assignment = dict(zip(train_data.ids, loaded['assignments']))

        assignments_ordered = [idx_to_assignment[idx] for idx in train_data.ids]
        cfg.REGIONS.REGION_ASSIGNMENT = [str(x) for x in assignments_ordered]

        if args.in_region_unit_cost:
            cfg.REGIONS.IN_REGION_UNIT_COST = args.in_region_unit_cost

        if args.out_of_region_unit_cost:
            cfg.REGIONS.OUT_OF_REGION_UNIT_COST = args.out_of_region_unit_cost

    if args.util_lambda:
        cfg.ACTIVE_LEARNING.UTIL_LAMBDA = args.util_lambda

    cfg.ACTIVE_LEARNING.SIMILARITY_MATRIX_PATH = args.similarity_matrix_path
    cfg.ACTIVE_LEARNING.TRAIN_SIMILARITY_MATRIX_PATH = args.train_similarity_matrix_path
    cfg.ACTIVE_LEARNING.SIMILARITY_PER_UNIT_PATH = args.similarity_per_unit_path

    main(cfg, train_data)

if __name__ == "__main__":
    main_wrapper()