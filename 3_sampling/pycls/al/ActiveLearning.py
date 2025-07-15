# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

from .Sampling import Sampling
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

class ActiveLearning:
    """
    Implements standard active learning methods.
    """

    def __init__(self, dataObj, cfg):
        self.dataObj = dataObj
        self.sampler = Sampling(dataObj=dataObj,cfg=cfg)
        self.cfg = cfg
        
    def sample_from_uSet(self, clf_model, lSet, uSet, supportingModels=None, **kwargs):
        """
        Sample from uSet using cfg.ACTIVE_LEARNING.SAMPLING_FN.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]

        supportingModels: List of models which are used for sampling process.

        OUTPUT
        -------
        Returns activeSet, uSet
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
        .format(len(uSet), self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)

        if self.cfg.ACTIVE_LEARNING.COST_AWARE == True:
            from .opt import Opt
            opt = Opt(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
            opt.set_utility_func(self.cfg.ACTIVE_LEARNING.SAMPLING_FN)
                
            if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":
                activeSet, uSet, total_cost = opt.select_samples()
                return activeSet, uSet, total_cost
            else:
                activeSet, uSet, total_cost, probs, relevant_indices = opt.select_samples()
                return activeSet, uSet, total_cost, probs, relevant_indices 

        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":

            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("typiclust"):
            from .typiclust import TypiClust
            is_scan = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.endswith('dc')
            tpc = TypiClust(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, is_scan=is_scan)
            activeSet, uSet = tpc.select_samples()
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["inverse_typiclust", 'inversetypiclust']:
            from .typiclust import TypiClust
            is_scan = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.endswith('dc')
            tpc = TypiClust(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, inverse=True, is_scan=is_scan)
            activeSet, uSet = tpc.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN in ["stratified"]:
            from .representation import Representation
            rep = Representation(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, strategy="balanced")
            activeSet, uSet = rep.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN in ["match_population_proportion"]:
            from .representation import Representation
            rep = Representation(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, strategy="match_population")
            activeSet, uSet = rep.select_samples()

        else:
            print(f"{self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError

        return activeSet, uSet
        
