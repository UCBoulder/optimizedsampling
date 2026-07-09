import os
import numpy as np

from datasets.usavars import USAVars
from datasets.india_secc import IndiaSECC
from datasets.togo_soil_fertility import TogoSoilFertility


class Data:
    """Loads a dataset by name and manages labeled/unlabeled/validation index sets."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = cfg.DATASET.NAME
        self.datasets_accepted = cfg.DATASET.ACCEPTED

    def getDataset(self, is_train=True):
        if self.dataset == 'USAVARS_POP':
            data = USAVars(root=self.cfg.DATASET.ROOT_DIR, is_train=is_train, label='population')
        elif self.dataset == 'USAVARS_TC':
            data = USAVars(root=self.cfg.DATASET.ROOT_DIR, is_train=is_train, label='treecover')
        elif self.dataset == 'USAVARS_EL':
            data = USAVars(root=self.cfg.DATASET.ROOT_DIR, is_train=is_train, label='elevation')
        elif self.dataset == 'USAVARS_INC':
            data = USAVars(root=self.cfg.DATASET.ROOT_DIR, is_train=is_train, label='income')
        elif self.dataset == "INDIA_SECC":
            data = IndiaSECC(root=self.cfg.DATASET.ROOT_DIR, is_train=is_train)
        elif self.dataset.startswith("TOGO"):
            data = TogoSoilFertility(root=self.cfg.DATASET.ROOT_DIR, is_train=is_train,
                                      label_col=self.cfg.DATASET.LABEL, identifier=self.cfg.DATASET.FEATURE_IDENTIFIER)
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not supported")
        return data, len(data)

    def makeLUVSets_from_ids(self, ids, data, save_dir):
        """Build lSet/uSet/valSet from a list of labeled sample ids and save them to save_dir."""
        if self.dataset.startswith("USAVARS"):
            ids_to_idxs = {data[i][3]: i for i in range(len(data))}
        elif self.dataset == "INDIA_SECC" or self.dataset.startswith("TOGO"):
            ids_to_idxs = {data.ids[i]: i for i in range(len(data))}
        else:
            raise ValueError("method not yet implemented")
        return self._save_luv_sets_from_idxs(ids, ids_to_idxs, save_dir)

    def _save_luv_sets_from_idxs(self, ids, ids_to_idxs, save_dir):
        assert self.dataset in self.datasets_accepted, \
            f"Sorry the dataset {self.dataset} is not supported. Currently we support {self.datasets_accepted}"

        lSet = np.array([ids_to_idxs[id_] for id_ in ids], dtype=np.ndarray)
        uSet = np.array([i for id_, i in ids_to_idxs.items() if id_ not in ids], dtype=np.ndarray)
        valSet = np.array([], dtype=np.ndarray)

        np.save(f'{save_dir}/lSet.npy', lSet)
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/valSet.npy', valSet)

        return f'{save_dir}/lSet.npy', f'{save_dir}/uSet.npy', f'{save_dir}/valSet.npy'

    def loadPartitions(self, lSetPath, uSetPath, valSetPath):
        lSet = np.load(lSetPath, allow_pickle=True)
        uSet = np.load(uSetPath, allow_pickle=True)
        valSet = np.load(valSetPath, allow_pickle=True)

        # Partitions must be disjoint.
        assert len(set(valSet) & set(uSet)) == 0, "Intersection is not allowed between validation set and uSet"
        assert len(set(valSet) & set(lSet)) == 0, "Intersection is not allowed between validation set and lSet"
        assert len(set(uSet) & set(lSet)) == 0, "Intersection is not allowed between uSet and lSet"

        return lSet, uSet, valSet

    def saveSets(self, lSet, uSet, activeSet, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        np.save(f'{save_dir}/lSet.npy', np.array(lSet, dtype=np.ndarray))
        np.save(f'{save_dir}/uSet.npy', np.array(uSet, dtype=np.ndarray))
        np.save(f'{save_dir}/activeSet.npy', activeSet)
