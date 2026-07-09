import os
import sys
import argparse
import numpy as np
import dill
import json
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

from active_learning.ActiveLearning import ActiveLearning
from core.config import cfg, dump_cfg
from datasets.data import Data
import utils.logging as lu

logger = lu.get_logger(__name__)

def argparser():
    parser = argparse.ArgumentParser(description='Subset Selection - Ridge Regression')
    parser.add_argument('--cfg', dest='cfg_file', required=True, type=str)
    parser.add_argument('--exp-name', required=True, type=str)
    parser.add_argument('--sampling_fn', required=True, type=str)
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

def evaluate_all_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
    }

def make_ridge_search():
    model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])
    return GridSearchCV(
        estimator=model,
        param_grid={'ridge__alpha': np.logspace(-5, 5, 10)},
        scoring='r2',
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )

def log_metrics(prefix, metrics):
    logger.info(f"{prefix} R² score: {metrics['r2']:.4f}")
    logger.info(f"{prefix} MSE: {metrics['mse']:.4f}")
    logger.info(f"{prefix} RMSE: {metrics['rmse']:.4f}")
    logger.info(f"{prefix} MAE: {metrics['mae']:.4f}")

def try_plot(train_data, ids, save_path):
    try:
        train_data.plot_subset_on_map(ids, save_path=save_path)
    except Exception as e:
        logger.warning(f"Plotting {save_path} failed: {e}")

def build_exp_dir(cfg):
    if cfg.EXP_NAME == 'auto':
        return datetime.now().strftime('%Y%m%d_%H%M%S'), None

    seed_str = f"seed_{cfg.RNG_SEED}"
    if cfg.ACTIVE_LEARNING.SAMPLING_FN in ["match_population_proportion", "poprisk", "poprisk_avg"]:
        sampling_str = f"{cfg.ACTIVE_LEARNING.SAMPLING_FN}/{cfg.GROUPS.GROUP_TYPE}"
    else:
        sampling_str = cfg.ACTIVE_LEARNING.SAMPLING_FN

    opt_str = "opt/" if cfg.ACTIVE_LEARNING.OPT else ""
    base_dir = f"{cfg.INITIAL_SET.STR}/{cfg.COST.NAME}/{opt_str}{sampling_str}/budget_{cfg.ACTIVE_LEARNING.BUDGET_SIZE}"

    if cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("poprisk"):
        util_lambda = getattr(cfg.ACTIVE_LEARNING, "UTIL_LAMBDA", None)
        if util_lambda is not None:
            base_dir = f"{base_dir}/util_lambda_{util_lambda}"

    exp_dir = base_dir if cfg.INITIAL_SET.STR.endswith(seed_str) else f"{base_dir}/{seed_str}"
    return exp_dir, base_dir

def main(cfg, train_data):
    cfg.OUT_DIR = os.path.abspath(cfg.OUT_DIR)
    dataset_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME)
    os.makedirs(dataset_dir, exist_ok=True)

    exp_dir, base_dir = build_exp_dir(cfg)
    dataset_and_model_dir = os.path.join(dataset_dir, cfg.MODEL.TYPE)
    exp_dir = os.path.join(dataset_and_model_dir, exp_dir)

    if os.path.exists(exp_dir) and cfg.ACTIVE_LEARNING.SAMPLING_FN not in ['similarity', 'diversity']:
        logger.info(f"Experiment already done: {exp_dir}")
        return

    os.makedirs(exp_dir, exist_ok=True)
    cfg.EXP_DIR = exp_dir
    cfg.INITIAL_SET_DIR = os.path.join(dataset_and_model_dir, cfg.INITIAL_SET.STR)
    cfg.SAMPLING_DIR = os.path.join(dataset_and_model_dir, base_dir) if base_dir else exp_dir
    dump_cfg(cfg)

    lu.setup_logging(cfg)
    logger.info(f"Experiment directory: {exp_dir}")

    cfg.DATASET.ROOT_DIR = os.path.abspath(cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    test_data, _ = data_obj.getDataset(is_train=False)
    X_test, y_test = test_data[:][0], test_data[:][1]

    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets_from_ids(cfg.LSET_IDS, data=train_data, save_dir=cfg.EXP_DIR)
    lSet, uSet, _ = data_obj.loadPartitions(lSetPath=lSet_path, uSetPath=uSet_path, valSetPath=valSet_path)

    X_train = None
    if len(lSet) > 0:
        initial_set_metrics_path = f"{cfg.INITIAL_SET_DIR}/initial_set_metrics_seed_{cfg.RNG_SEED}.json"
        metrics = None
        if os.path.exists(initial_set_metrics_path):
            try:
                with open(initial_set_metrics_path, "r") as f:
                    metrics = json.load(f)
            except Exception:
                metrics = None

        if metrics is not None:
            log_metrics("Initial", metrics)
        else:
            lSet = lSet.astype(int)
            uSet = uSet.astype(int)
            X_train, y_train = train_data[lSet][0], train_data[lSet][1]

            if X_train.shape[0] < 5:
                logger.info("Not enough samples to train initial-set model.")
            else:
                ridge_search = make_ridge_search()
                ridge_search.fit(X_train, y_train)
                metrics = evaluate_all_metrics(ridge_search, X_test, y_test)
                log_metrics("Initial", metrics)

                os.makedirs(cfg.INITIAL_SET_DIR, exist_ok=True)
                with open(initial_set_metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)

            try_plot(train_data, lSet, f"{cfg.INITIAL_SET_DIR}/lSet_plot.png")
    else:
        logger.info("Initial labeled set is empty; skipping to subset selection.")

    logger.info("Starting subset selection...")
    al_obj = ActiveLearning(data_obj, cfg)
    activeSet, new_uSet = al_obj.sample_from_uSet(make_ridge_search(), lSet, uSet, train_data, X_train=train_data.X)
    logger.info(f"Sampled {len(activeSet)} points.")
    if len(activeSet) > 0:
        try_plot(train_data, activeSet, f"{exp_dir}/activeSet_plot.png")

    data_obj.saveSets(lSet, uSet, activeSet, os.path.join(cfg.EXP_DIR, 'episode_0'))

    lSet_updated = np.concatenate((lSet, activeSet)).astype(int)
    try_plot(train_data, lSet_updated, f"{exp_dir}/lSet_updated_plot.png")

    if lSet_updated.size >= 5:
        logger.info("Training ridge regression on updated labeled set...")
        X_train_updated, y_train_updated = train_data[lSet_updated][0], train_data[lSet_updated][1]

        ridge_search = make_ridge_search()
        ridge_search.fit(X_train_updated, y_train_updated)
        metrics_updated = evaluate_all_metrics(ridge_search, X_test, y_test)
        log_metrics("Updated", metrics_updated)

        with open(f"{exp_dir}/updated_set_metrics_seed_{cfg.RNG_SEED}.json", "w") as f:
            json.dump(metrics_updated, f, indent=4)

    data_obj.saveSets(lSet_updated, new_uSet, activeSet, os.path.join(cfg.EXP_DIR, 'episode_1'))

def load_assignments(path, train_data):
    with open(path, "rb") as f:
        loaded = dill.load(f)

    if isinstance(loaded, dict):
        idx_to_assignment = loaded
    elif 'ids' in loaded:
        idx_to_assignment = dict(zip(loaded['ids'], loaded['assignments']))
    else:
        idx_to_assignment = dict(zip(train_data.ids, loaded['assignments']))

    return [str(idx_to_assignment[idx]) for idx in train_data.ids]

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
    train_data, _ = data_obj.getDataset(is_train=True)

    if args.id_path:
        with open(args.id_path, "rb") as f:
            arrs = dill.load(f)
        ids = arrs["sampled_ids"] if isinstance(arrs, dict) and "sampled_ids" in arrs else arrs
        cfg.LSET_IDS = ids if isinstance(ids, list) else ids.tolist()
    else:
        cfg.LSET_IDS = []

    if args.cost_func:
        cfg.COST.FN = args.cost_func
        cfg.COST.NAME = args.cost_name or args.cost_func
    else:
        cfg.COST.FN = 'uniform'
        cfg.COST.NAME = 'uniform'

    cfg.ACTIVE_LEARNING.OPT = args.sampling_fn in ['greedycost', 'poprisk', 'similarity', 'diversity', 'poprisk_avg']

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
        cfg.GROUPS.GROUP_ASSIGNMENT = load_assignments(args.group_assignment_path, train_data)
        if args.ignore_groups:
            cfg.GROUPS.IGNORED_GROUPS = args.ignore_groups

    if args.unit_assignment_path:
        cfg.UNITS.UNIT_TYPE = args.unit_type
        cfg.UNITS.UNIT_ASSIGNMENT = load_assignments(args.unit_assignment_path, train_data)
        cfg.UNITS.POINTS_PER_UNIT = args.points_per_unit

        if args.unit_cost_path:
            cfg.COST.UNIT_COST_PATH = args.unit_cost_path
        if args.alpha:
            cfg.COST.ALPHA = args.alpha

    if args.region_assignment_path:
        cfg.REGIONS.REGION_ASSIGNMENT = load_assignments(args.region_assignment_path, train_data)
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
