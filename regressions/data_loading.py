import dill
import numpy as np


def load_data_from_pkl(features_path, dataset=None, label=None):
    with open(features_path, "rb") as f:
        arrs = dill.load(f)

    if dataset != "togo":
        return arrs['ids_train'], arrs['X_train'], arrs['y_train'], arrs['X_test'], arrs['y_test']

    y_test = arrs[f'{label}_test']
    valid_idxs = np.where(~np.isnan(y_test))[0]
    return arrs['ids_train'], arrs['X_train'], arrs[f'{label}_train'], arrs['X_test'][valid_idxs], y_test[valid_idxs]


def create_id_mapping(full_ids):
    return {str(id_): i for i, id_ in enumerate(full_ids)}


def load_sample_file(filepath):
    with open(filepath, "rb") as f:
        loaded = dill.load(f)
    sampled_ids = loaded['sampled_ids'] if isinstance(loaded, dict) and 'sampled_ids' in loaded else loaded
    return [x if isinstance(x, str) else str(x) for x in sampled_ids]
