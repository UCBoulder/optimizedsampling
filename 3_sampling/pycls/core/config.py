# This file is modified from code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564

# ----------------------------------------------------------
# This file is modified from official pycls repository to adapt in AL settings.

"""Configuration file (powered by YACS)."""

import os

from yacs.config import CfgNode as CN


# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
# Output directory (will be created at the projec root)
_C.OUT_DIR = '../../0_output'
# Experiment directory
_C.EXP_DIR = ''
# Higher Level Experiment directory without seed
_C.EXP_ROOT = ''
# Initial Set Directory
_C.INITIAL_SET_DIR = ''
_C.SAMPLING_DIR = ''
# Episode directory
_C.EPISODE_DIR = ''
# Config destination (in OUT_DIR)
_C.CFG_DEST = 'config.yaml'
# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = None
# Folder name where best model logs etc are saved. "auto" creates a timestamp based folder 
_C.EXP_NAME = 'auto' 
# Which GPU to run on
# Log destination ('stdout' or 'file')
_C.LOG_DEST = 'file'

#if lset is initialized with ids (usavars)
_C.ID_PATH = None
_C.LSET_IDS = []

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIM = CN()

# Ridge regression
_C.OPTIM.CV = True
_C.OPTIM.ALPHAS = []

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
# Dataset and split
_C.TRAIN.DATASET = ''
_C.TRAIN.SPLIT = 'train'

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Dataset and split
_C.TEST.DATASET = ''


# #-------------------------------------------------------------------------------#
# #  ACTIVE LEARNING options
# #-------------------------------------------------------------------------------#
_C.ACTIVE_LEARNING = CN()
_C.ACTIVE_LEARNING.OPT = False
_C.ACTIVE_LEARNING.SAMPLING_FN = 'random'
_C.ACTIVE_LEARNING.RANDOM_STRATEGY = 'point'
_C.ACTIVE_LEARNING.LSET_PATH = ''
_C.ACTIVE_LEARNING.USET_PATH = ''

_C.ACTIVE_LEARNING.UTIL_LAMBDA = None
_C.ACTIVE_LEARNING.SIMILARITY_MATRIX_PATH = None
_C.ACTIVE_LEARNING.TRAIN_SIMILARITY_MATRIX_PATH = None
_C.ACTIVE_LEARNING.SIMILARITY_PER_UNIT_PATH= None

# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #
_C.DATASET = CN()
_C.DATASET.LABEL = None
_C.DATASET.FEATURE_IDENTIFIER = None
_C.DATASET.NAME = None
# For Tiny ImageNet dataset, ROOT_DIR must be set to the dataset folder ("data/tiny-imagenet-200/"). For others, the outder "data" folder where all datasets can be stored is expected.
_C.DATASET.ROOT_DIR = None
# Accepted Datasets
_C.DATASET.ACCEPTED = ['USAVARS_POP', 'USAVARS_TC', 'USAVARS_EL', 'USAVARS_INC', 'INDIA_SECC', 'TOGO_P', 'TOGO_PH_H2O']

# #-------------------------------------------------------------------------------#
# #  INITIAL SET options
# #-------------------------------------------------------------------------------#
_C.INITIAL_SET = CN()
_C.INITIAL_SET.STR = None

# #-------------------------------------------------------------------------------#
# #  COST options
# #-------------------------------------------------------------------------------#
_C.COST = CN()
_C.COST.FN = None
_C.COST.NAME = None
_C.COST.ARRAY = None
_C.COST.UNIT_COST_PATH = None

# #-------------------------------------------------------------------------------#
# #  GROUP options
# #-------------------------------------------------------------------------------#
_C.GROUPS = CN()
_C.GROUPS.GROUP_TYPE = None
_C.GROUPS.GROUP_ASSIGNMENT = None
_C.GROUPS.IGNORED_GROUPS = None

# #-------------------------------------------------------------------------------#
# #  BLOCK options
# #-------------------------------------------------------------------------------#
_C.UNITS = CN()
_C.UNITS.TYPE = None
_C.UNITS.UNIT_ASSIGNMENT = None
_C.UNITS.POINTS_PER_UNIT = None

# #-------------------------------------------------------------------------------#
# #  REGION options
# #-------------------------------------------------------------------------------#
_C.REGIONS = CN()
_C.REGIONS.TYPE = None
_C.REGIONS.REGION_ASSIGNMENT = None
_C.REGIONS.IN_REGION_UNIT_COST = None
_C.REGIONS.OUT_OF_REGION_UNIT_COST = None

def assert_cfg():
    """Checks config values invariants."""
    assert _C.TRAIN.SPLIT in ['train', 'val', 'test'], \
        'Train split \'{}\' not supported'.format(_C.TRAIN.SPLIT)
    assert _C.TEST.SPLIT in ['train', 'val', 'test'], \
        'Test split \'{}\' not supported'.format(_C.TEST.SPLIT)

def custom_dump_cfg(temp_cfg):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(temp_cfg.EXP_DIR, temp_cfg.CFG_DEST)
    with open(cfg_file, 'w') as f:
        _C.dump(stream=f)


def dump_cfg(cfg):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(cfg.EXP_DIR, cfg.CFG_DEST)
    with open(cfg_file, 'w') as f:
        cfg.dump(stream=f)


def load_cfg(out_dir, cfg_dest='config.yaml'):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)