import os
import sys
import argparse
import numpy as np
import dill
import json
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

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
    parser.add_argument('--sampling-fn', required=True, type=str)
    parser.add_argument('--budget', required=True, type=int)
    parser.add_argument('--initial_size', default=0, type=int)
    parser.add_argument('--id-path', default=None, type=str)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--initial_set_str', default="empty_initial_set", type=str)

    parser.add_argument('--cost_func', default=None, type=str)
    parser.add_argument('--cost_name', default=None, type=str)
    parser.add_argument('--cost_array_path', default=None, type=str)

    parser.add_argument('--group_type', default=None, type=str)
    parser.add_argument('--group_assignment_path', default=None, type=str)

    parser.add_argument('--unit_type', default=None, type=str)
    parser.add_argument('--points_per_unit', default=None, type=int)
    parser.add_argument('--unit_assignment_path', default=None, type=str)
    parser.add_argument('--unit_cost_path', default=None, type=str)

    parser.add_argument('--util_lambda', default=0.5, type=float)

    parser.add_argument('--similarity_matrix_path', default=None, type=str)
    parser.add_argument('--distance_matrix_path', default=None, type=str)
    return parser

def main(cfg):
    cfg.OUT_DIR = os.path.abspath(cfg.OUT_DIR)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    dataset_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME)
    os.makedirs(dataset_dir, exist_ok=True)

    if cfg.EXP_NAME == 'auto':
        now = datetime.now()
        exp_dir = now.strftime('%Y%m%d_%H%M%S')
    else:
        seed_str = f"seed_{cfg.RNG_SEED}"
        exp_dir = f"{cfg.INITIAL_SET.STR}/cost_aware/{cfg.ACTIVE_LEARNING.SAMPLING_FN}/budget_{cfg.ACTIVE_LEARNING.BUDGET_SIZE}" if cfg.ACTIVE_LEARNING.COST_AWARE else f"{cfg.INITIAL_SET.STR}/{cfg.ACTIVE_LEARNING.SAMPLING_FN}/budget_{cfg.ACTIVE_LEARNING.BUDGET_SIZE}"

        # Check if INITIAL_SET.STR already ends with seed_{seed}
        if not cfg.INITIAL_SET.STR.endswith(seed_str):
            exp_dir = f"{exp_dir}/{seed_str}"


    exp_dir = os.path.join(dataset_dir, exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    cfg.EXP_DIR = exp_dir
    cfg.INITIAL_SET_DIR = os.path.join(dataset_dir, cfg.INITIAL_SET.STR)
    dump_cfg(cfg)

    lu.setup_logging(cfg)
    logger.info(f"Experiment directory: {exp_dir}")

    cfg.DATASET.ROOT_DIR = os.path.abspath(cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, _ = data_obj.getDataset(isTrain=True)
    test_data, _ = data_obj.getDataset(isTrain=False)


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

    if len(lSet) > 0:
        logger.info("Training ridge regression on initial set...")
        lSet = lSet.astype(int)
        uSet = uSet.astype(int)
        X_train, y_train = train_data[lSet][0], train_data[lSet][1]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridgecv', RidgeCV(alphas=np.logspace(-5, 5, 10), scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=42)))
        ])
        pipeline.fit(X_train, y_train)
        r2 = evaluate_r2(pipeline, X_test, y_test)
        logger.info(f"Initial R² score: {r2:.4f}")

        summary_path = os.path.join(cfg.INITIAL_SET_DIR, "episode_0/summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({'test_r2': r2}, f)
    else:
        logger.info("Initial labeled set is empty; skipping to subset selection.")

    logger.info("Starting subset selection...")
    al_obj = ActiveLearning(data_obj, cfg)
    model = pipeline if lSet.size > 0 else None
    activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data)

    logger.info(f"Selected {len(activeSet)} new samples.")
    data_obj.saveSets(lSet, uSet, activeSet, os.path.join(cfg.EXP_DIR, 'episode_0'))

    lSet_updated = np.concatenate((lSet, activeSet))
    lSet_updated = lSet_updated.astype(int)


    # Train again on the updated labeled set
    if lSet_updated.size > 0:
        logger.info("Training ridge regression on updated labeled set...")
        X_train_updated, y_train_updated = train_data[lSet_updated][0], train_data[lSet_updated][1]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridgecv', RidgeCV(alphas=np.logspace(-5, 5, 10), scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=42)))
        ])
        pipeline.fit(X_train_updated, y_train_updated)
        r2_updated = evaluate_r2(pipeline, X_test, y_test)
        logger.info(f"Updated R² score: {r2_updated:.4f}")

    data_obj.saveSets(lSet_updated, new_uSet, activeSet, os.path.join(cfg.EXP_DIR, 'episode_1'))


if __name__ == "__main__":
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
    test_data, _ = data_obj.getDataset(isTrain=False)

    if args.id_path:
        with open(args.id_path, "rb") as f:
            ids = dill.load(f)
        cfg.LSET_IDS = ids if isinstance(ids, list) else ids.tolist()
    else:
        cfg.LSET_IDS = []

    if args.cost_func:
        cfg.COST.FN = args.cost_func
        cfg.COST.NAME = args.cost_name or args.cost_func
        cfg.ACTIVE_LEARNING.COST_AWARE = True
    else:
        cfg.COST.FN = 'uniform'
        cfg.COST.NAME = 'uniform'

    if args.sampling_fn in ['poprisk', 'similarity', 'diversity']:
        cfg.ACTIVE_LEARNING.COST_AWARE = True

    if args.cost_array_path:
        with open(args.cost_array_path, "rb") as f:
            loaded = dill.load(f)
        if 'ids' in loaded:
            idx_to_cost = dict(zip(loaded['ids'], loaded['costs']))
            # train_data has indices
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
            assignments_ordered = [idx_to_assignment[idx] for idx in train_data.ids]
        elif 'ids' in loaded:
            idx_to_assignment = dict(zip(loaded['ids'], loaded['assignments']))
            # train_data has indices
            assignments_ordered = [idx_to_assignment[idx] for idx in train_data.ids]
        else:
            assignments_ordered = loaded['assignments']

        cfg.GROUPS.GROUP_ASSIGNMENT = [str(x) for x in assignments_ordered]


    if args.unit_assignment_path:
        cfg.UNITS.UNIT_TYPE = args.unit_type
        with open(args.unit_assignment_path, "rb") as f:
            loaded = dill.load(f)
        if isinstance(loaded, dict):
            idx_to_assignment = loaded
            assignments_ordered = [idx_to_assignment[idx] for idx in train_data.ids]
        elif 'ids' in loaded:
            idx_to_assignment = dict(zip(loaded['ids'], loaded['assignments']))
            assignments_ordered = [idx_to_assignment[idx] for idx in train_data.ids]
        else:
            assignments_ordered = loaded['assignments']

        cfg.UNITS.UNIT_ASSIGNMENT = [str(x) for x in assignments_ordered]

        cfg.UNITS.POINTS_PER_UNIT = args.points_per_unit

        if args.unit_cost_path:
            cfg.COST.UNIT_COST_PATH = args.unit_cost_path

    if args.util_lambda:
        cfg.ACTIVE_LEARNING.UTIL_LAMBDA = args.util_lambda

    cfg.ACTIVE_LEARNING.SIMILARITY_MATRIX_PATH = args.similarity_matrix_path
    cfg.ACTIVE_LEARNING.DISTANCE_MATRIX_PATH = args.distance_matrix_path

    main(cfg)
