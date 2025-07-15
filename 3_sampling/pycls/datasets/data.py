# This file is modified from a code implementation shared with me by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------
import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import pycls.utils.logging as lu
from pycls.datasets.usavars import USAVars
from pycls.datasets.india_secc import IndiaSECC

logger = lu.get_logger(__name__)

class Data:
    """
    Contains all data related functions. For working with new dataset 
    make changes to following functions:
    0. Create labeled.txt and unlabaled.txt for Active Learning
    1. getDataset
    2. getAugmentations
    3. getDataLoaders

    """
    def __init__(self, cfg):
        """
        Initializes dataset attribute of (Data class) object with specified "dataset" argument.
        INPUT:
        cfg: yacs.config, config object
        """
        self.cfg = cfg
        self.dataset = cfg.DATASET.NAME
        self.data_dir = cfg.DATASET.ROOT_DIR
        self.datasets_accepted = cfg.DATASET.ACCEPTED
        # self.target_dir = {"test": cfg.DATASET.TEST_DIR, "train": cfg.DATASET.TRAIN_DIR, "val": cfg.DATASET.VAL_DIR}
        self.eval_mode = False

    def about(self):
        """
        Show all properties of this class.
        """
        print(self.__dict__)


    def make_data_lists(self, exp_dir):
        """
        Creates train.txt, test.txt and valid.txt. Text format is chosen to allow readability. 
        Keyword arguments:
            exp_dir -- Full path to the experiment directory where index lists will be saved
        """
        train = os.path.join(exp_dir, 'train.txt')
        test = os.path.join(exp_dir, 'test.txt')
        
        if os.path.exists(train) or os.path.exists(test):
            out = f'train.txt or test.text already exist at {exp_dir}'
            return None
        
        train_list = glob.glob(os.path.join(path, 'train/**/*.png'), recursive=True)
        test_list = glob.glob(os.path.join(path, 'test/**/*.png'), recursive=True)

        with open(train, 'w') as filehandle:
            filehandle.writelines("%s\n" % index for index in train_list)
        
        with open(test, 'w') as filehandle:
            filehandle.writelines("%s\n" % index for index in test_list)


    def getPreprocessOps(self):
        """
        This function specifies the steps to be accounted for preprocessing.
        
        INPUT:
        None
        
        OUTPUT:
        Returns a list of preprocessing steps. Note the order of operations matters in the list.
        """
        if self.dataset in self.datasets_accepted:
            ops = []
            norm_mean = []
            norm_std = []

            if self.dataset in ["CIFAR10", "CIFAR100", 'IMBALANCED_CIFAR10', 'IMBALANCED_CIFAR100']:
                ops = [transforms.RandomCrop(32, padding=4)]
                norm_mean = [0.4914, 0.4822, 0.4465]
                norm_std = [0.247 , 0.2435, 0.2616]
            elif self.dataset == "MNIST":
                ops = [transforms.Resize(32)] 
                norm_mean = [0.1307,]
                norm_std = [0.3081,]
            elif self.dataset == "TINYIMAGENET":
                # ops = [transforms.RandomResizedCrop(64)]
                ops = [transforms.RandomResizedCrop(64, scale=(0.5, 1.))]

                # Using ImageNet values 
                norm_mean = [0.485, 0.456, 0.406]
                norm_std = [0.229, 0.224, 0.225]
            elif self.dataset in ["IMAGENET", 'IMAGENET50', 'IMAGENET100', 'IMAGENET200']:
                ops = [transforms.RandomResizedCrop(224, scale=(0.5, 1.))]
                # Using ImageNet values
                norm_mean = [0.485, 0.456, 0.406]
                norm_std = [0.229, 0.224, 0.225]
            elif self.dataset in ["SVHN"]:
                ops = [transforms.RandomCrop(32, padding=4)]
                norm_mean = [0.4376, 0.4437, 0.4728]
                norm_std = [0.1980, 0.2010, 0.1970]
            else:
            #     raise NotImplementedError
                return

            ops.append(transforms.ToTensor())
            ops.append(transforms.Normalize(norm_mean, norm_std))

            if self.eval_mode:
                ops = [ops[0], transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
            else:
                print("Preprocess Operations Selected ==> ", ops)
                # logger.info("Preprocess Operations Selected ==> ", ops)
            return ops
        else:
            print("Either the specified {} dataset is not added or there is no if condition in getDataset function of Data class".format(self.dataset))
            logger.info("Either the specified {} dataset is not added or there is no if condition in getDataset function of Data class".format(self.dataset))
            #raise NotImplementedError


    def getDataset(self, isTrain=True, isDownload=False):
        """
        This function returns the dataset instance and number of data points in it.
        
        INPUT:
        save_dir: String, It specifies the path where dataset will be saved if downloaded.
        
        preprocess_steps(optional): List, Contains the ordered operations used for preprocessing the data.
        
        isTrain (optional): Bool, If true then Train partition is downloaded else Test partition.
        
        isDownload (optional): Bool, If true then dataset is saved at path specified by "save_dir".
        
        OUTPUT:
        (On Success) Returns the tuple of dataset instance and length of dataset.
        (On Failure) Returns Message as <dataset> not specified.
        """
        self.eval_mode = True
        test_preops_list = self.getPreprocessOps()
        self.eval_mode = False
        
        if isTrain:
            preprocess_steps = self.getPreprocessOps()
        else:
            preprocess_steps = test_preops_list
        preprocess_steps = transforms.Compose(preprocess_steps)
        
        if self.dataset == 'USAVARS_POP':
            usavars_pop = USAVars(root='/share/usavars', isTrain=isTrain, label='population')
            return usavars_pop, len(usavars_pop)
        
        elif self.dataset == 'USAVARS_TC':
            usavars_tc = USAVars(root='/share/usavars', isTrain=isTrain, label='treecover')
            return usavars_tc, len(usavars_tc)
        
        elif self.dataset == 'USAVARS_EL':
            usavars_el = USAVars(root='/share/usavars', isTrain=isTrain, label='elevation')
            return usavars_el, len(usavars_el)

        elif self.dataset == 'USAVARS_INC':
            usavars_inc = USAVars(root='/share/usavars', isTrain=isTrain, label='income')
            return usavars_inc, len(usavars_inc)

        elif self.dataset == "INDIA_SECC":
            india_secc = IndiaSECC(root='/share/india_secc', isTrain=isTrain)
            return india_secc, len(india_secc)

        else:
            print("Either the specified {} dataset is not added or there is no if condition in getDataset function of Data class".format(self.dataset))
            logger.info("Either the specified {} dataset is not added or there is no if condition in getDataset function of Data class".format(self.dataset))
            raise NotImplementedError


    def makeLUVSets(self, train_split_num, val_split_ratio, data, seed_id, save_dir):
        """
        Initialize the labelled and unlabelled set by splitting the data into train
        and validation according to split_ratios arguments.

        Visually it does the following:

        |<------------- Train -------------><--- Validation --->

        |<--- Labelled --><---Unlabelled --><--- Validation --->

        INPUT:
        train_split_ratio: Float, Specifies the proportion of data in train set.
        For example: 0.8 means beginning 80% of data is training data.

        val_split_ratio: Float, Specifies the proportion of data in validation set.
        For example: 0.1 means ending 10% of data is validation data.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.
        
        OUTPUT:
        (On Success) Sets the labelled, unlabelled set along with validation set
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        #assert isinstance(train_split_ratio, float),"Train split ratio is of {} datatype instead of float".format(type(train_split_ratio))
        assert isinstance(val_split_ratio, float),"Val split ratio is of {} datatype instead of float".format(type(val_split_ratio))
        assert self.dataset in self.datasets_accepted, "Sorry the dataset {} is not supported. Currently we support {}".format(self.dataset, self.datasets_accepted)

        lSet = []
        uSet = []
        valSet = []
        
        n_dataPoints = len(data)
        all_idx = [i for i in range(n_dataPoints)]
        np.random.shuffle(all_idx)
        train_splitIdx = train_split_num
        #To get the validation index from end we multiply n_datapoints with 1-val_ratio 
        val_splitIdx = int((1-val_split_ratio)*n_dataPoints)
        #Check there should be no overlap with train and val data
        # assert train_split_ratio + val_split_ratio < 1.0, "Validation data over laps with train data as last train index is {} and last val index is {}. \
        #     The program expects val index > train index. Please satisfy the constraint: train_split_ratio + val_split_ratio < 1.0; currently it is {} + {} is not < 1.0 => {} is not < 1.0"\
        #         .format(train_splitIdx, val_splitIdx, train_split_ratio, val_split_ratio, train_split_ratio + val_split_ratio)
        
        lSet = all_idx[:train_splitIdx]
        uSet = all_idx[train_splitIdx:val_splitIdx]
        valSet = all_idx[val_splitIdx:]

        # print("=============================")
        # print("lSet len: {}, uSet len: {} and valSet len: {}".format(len(lSet),len(uSet),len(valSet)))
        # print("=============================")
        
        lSet = np.array(lSet, dtype=np.ndarray)
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        
        np.save(f'{save_dir}/lSet.npy', lSet)
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/valSet.npy', valSet)
        
        return f'{save_dir}/lSet.npy', f'{save_dir}/uSet.npy', f'{save_dir}/valSet.npy'

    def makeLUVSets_from_ids(self, ids, data, save_dir):
        if self.dataset.startswith("USAVARS"):
            return self.makeLUVSets_from_ids_usavars(ids, data, save_dir)
        elif self.dataset == "INDIA_SECC":
            return self.makeLUVSets_from_ids_india_secc(ids, data, save_dir)
        else:
            raise ValueError("method not yet implemented")
        return
    
    def makeLUVSets_from_ids_usavars(self, ids, data, save_dir):
        """
            For USAVars, make labeled set from ids.
        """
        print("Sampling initial pool from IDS")
        assert self.dataset in self.datasets_accepted, "Sorry the dataset {} is not supported. Currently we support {}".format(self.dataset, self.datasets_accepted)

        lSet = []
        uSet = []
        valSet = []

        ids_to_idxs = {data[i][3]: i for i in range(len(data))}
        labeled_idxs = [ids_to_idxs[id_] for id_ in ids]
        unlabeled_idxs = [i for id_, i in ids_to_idxs.items() if id_ not in ids]
        
        #Check there should be no overlap with train and val data
        # assert train_split_ratio + val_split_ratio < 1.0, "Validation data over laps with train data as last train index is {} and last val index is {}. \
        #     The program expects val index > train index. Please satisfy the constraint: train_split_ratio + val_split_ratio < 1.0; currently it is {} + {} is not < 1.0 => {} is not < 1.0"\
        #         .format(train_splitIdx, val_splitIdx, train_split_ratio, val_split_ratio, train_split_ratio + val_split_ratio)
        
        n_dataPoints = len(data)
        all_idx = [i for i in range(n_dataPoints)]

        lSet = [all_idx[i] for i in labeled_idxs]
        uSet = [all_idx[i] for i in unlabeled_idxs]
        valSet = [all_idx[i] for i in []]

        # print("=============================")
        # print("lSet len: {}, uSet len: {} and valSet len: {}".format(len(lSet),len(uSet),len(valSet)))
        # print("=============================")
        
        lSet = np.array(lSet, dtype=np.ndarray)
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        
        np.save(f'{save_dir}/lSet.npy', lSet)
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/valSet.npy', valSet)
        
        return f'{save_dir}/lSet.npy', f'{save_dir}/uSet.npy', f'{save_dir}/valSet.npy'

    def makeLUVSets_from_ids_india_secc(self, ids, data, save_dir):
        """
            For India_SECC, make labeled set from ids.
        """
        print("Sampling initial pool from IDS")
        assert self.dataset in self.datasets_accepted, "Sorry the dataset {} is not supported. Currently we support {}".format(self.dataset, self.datasets_accepted)

        lSet = []
        uSet = []
        valSet = []

        ids_to_idxs = {data.ids[i]:i for i in range(len(data))}
        labeled_idxs = [ids_to_idxs[id_] for id_ in ids]
        unlabeled_idxs = [i for id_, i in ids_to_idxs.items() if id_ not in ids]
        
        #Check there should be no overlap with train and val data
        # assert train_split_ratio + val_split_ratio < 1.0, "Validation data over laps with train data as last train index is {} and last val index is {}. \
        #     The program expects val index > train index. Please satisfy the constraint: train_split_ratio + val_split_ratio < 1.0; currently it is {} + {} is not < 1.0 => {} is not < 1.0"\
        #         .format(train_splitIdx, val_splitIdx, train_split_ratio, val_split_ratio, train_split_ratio + val_split_ratio)
        
        n_dataPoints = len(data)
        all_idx = [i for i in range(n_dataPoints)]

        lSet = [all_idx[i] for i in labeled_idxs]
        uSet = [all_idx[i] for i in unlabeled_idxs]
        valSet = [all_idx[i] for i in []]

        # print("=============================")
        # print("lSet len: {}, uSet len: {} and valSet len: {}".format(len(lSet),len(uSet),len(valSet)))
        # print("=============================")
        
        lSet = np.array(lSet, dtype=np.ndarray)
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        
        np.save(f'{save_dir}/lSet.npy', lSet)
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/valSet.npy', valSet)
        
        return f'{save_dir}/lSet.npy', f'{save_dir}/uSet.npy', f'{save_dir}/valSet.npy'

    def makeTVSets(self, val_split_ratio, data, seed_id, save_dir):
        """
        Initialize the train and validation sets by splitting the train data according to split_ratios arguments.

        Visually it does the following:

        |<------------- Train -------------><--- Validation --->

        INPUT:
        val_split_ratio: Float, Specifies the proportion of data in validation set.
        For example: 0.1 means ending 10% of data is validation data.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.
        
        OUTPUT:
        (On Success) Sets the train set and the validation set
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        assert isinstance(val_split_ratio, float),"Val split ratio is of {} datatype instead of float".format(type(val_split_ratio))
        assert self.dataset in self.datasets_accepted, "Sorry the dataset {} is not supported. Currently we support {}".format(self.dataset, self.datasets_accepted)

        trainSet = []
        valSet = []
        
        n_dataPoints = len(data)
        all_idx = [i for i in range(n_dataPoints)]
        np.random.shuffle(all_idx)

        # To get the validation index from end we multiply n_datapoints with 1-val_ratio 
        val_splitIdx = int((1-val_split_ratio)*n_dataPoints)
        
        trainSet = all_idx[:val_splitIdx]
        valSet = all_idx[val_splitIdx:]

        # print("=============================")
        # print("lSet len: {}, uSet len: {} and valSet len: {}".format(len(lSet),len(uSet),len(valSet)))
        # print("=============================")
        
        trainSet = np.array(trainSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        
        np.save(f'{save_dir}/trainSet.npy', trainSet)
        np.save(f'{save_dir}/valSet.npy', valSet)
        
        return f'{save_dir}/trainSet.npy', f'{save_dir}/valSet.npy'

    def makeUVSets(self, val_split_ratio, data, seed_id, save_dir): 
        """
        Initial labeled pool should already be sampled. We use this function to initialize the train and validation sets by splitting the train data according to split_ratios arguments.

        Visually it does the following:

        |<------------- Unlabeled -------------><--- Validation --->

        INPUT:
        val_split_ratio: Float, Specifies the proportion of data in validation set.
        For example: 0.1 means ending 10% of data is validation data.

        data: reference to uSet instance post initial pool sampling. This can be obtained by calling getDataset function of Data class.
        
        OUTPUT:
        (On Success) Sets the unlabeled set and the validation set
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        assert isinstance(val_split_ratio, float),"Val split ratio is of {} datatype instead of float".format(type(val_split_ratio))
        assert self.dataset in self.datasets_accepted, "Sorry the dataset {} is not supported. Currently we support {}".format(self.dataset, self.datasets_accepted)
        uSet = []
        valSet = []
        
        n_dataPoints = len(data)
        # all_idx = [i for i in range(n_dataPoints)]
        np.random.shuffle(data)

        # To get the validation index from end we multiply n_datapoints with 1-val_ratio 
        val_splitIdx = int((1-val_split_ratio)*n_dataPoints)
        
        uSet = data[:val_splitIdx]
        valSet = data[val_splitIdx:]

        # print("=============================")
        # print("lSet len: {}, uSet len: {} and valSet len: {}".format(len(lSet),len(uSet),len(valSet)))
        # print("=============================")
        
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/valSet.npy', valSet)
        
        return f'{save_dir}/uSet.npy', f'{save_dir}/valSet.npy'

    def loadPartitions(self, lSetPath, uSetPath, valSetPath):

        assert isinstance(lSetPath, str), "Expected lSetPath to be a string."
        assert isinstance(uSetPath, str), "Expected uSetPath to be a string."
        assert isinstance(valSetPath, str), "Expected valSetPath to be a string."

        lSet = np.load(lSetPath, allow_pickle=True)
        uSet = np.load(uSetPath, allow_pickle=True)
        valSet = np.load(valSetPath, allow_pickle=True)

        #Checking no overlap
        assert len(set(valSet) & set(uSet)) == 0,"Intersection is not allowed between validationset and uset"
        assert len(set(valSet) & set(lSet)) == 0,"Intersection is not allowed between validationset and lSet"
        assert len(set(uSet) & set(lSet)) == 0,"Intersection is not allowed between uSet and lSet"

        return lSet, uSet, valSet

    def loadTVPartitions(self, trainSetPath, valSetPath):

        assert isinstance(trainSetPath, str), "Expected trainSetPath to be a string."
        assert isinstance(valSetPath, str), "Expected valSetPath to be a string."

        trainSet = np.load(trainSetPath, allow_pickle=True)
        valSet = np.load(valSetPath, allow_pickle=True)

        #Checking no overlap
        assert len(set(valSet) & set(trainSet)) == 0,"Intersection is not allowed between validationset and trainSet"

        return trainSet, valSet


    def loadPartition(self, setPath):

        assert isinstance(setPath, str), "Expected setPath to be a string."

        setArray = np.load(setPath, allow_pickle=True)
        return setArray


    def saveSets(self, lSet, uSet, activeSet, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        lSet = np.array(lSet, dtype=np.ndarray)
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(activeSet, dtype=np.ndarray)

        np.save(f'{save_dir}/lSet.npy', lSet)
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/activeSet.npy', activeSet)

        # return f'{save_dir}/lSet.npy', f'{save_dir}/uSet.npy', f'{save_dir}/activeSet.npy'


    def saveSet(self, setArray, setName, save_dir):

        setArray = np.array(setArray, dtype=np.ndarray)
        np.save(f'{save_dir}/{setName}.npy', setArray)
        return f'{save_dir}/{setName}.npy'