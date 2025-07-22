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

        if self.cfg.ACTIVE_LEARNING.OPT == True:
            from .opt import Opt
            opt = Opt(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
            opt.set_utility_func(self.cfg.ACTIVE_LEARNING.SAMPLING_FN)
                
            activeSet, uSet = opt.select_samples()
            return activeSet, uSet

        self.sampler = Sampling(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":
            #strategy = self.cfg.ACTIVE_LEARNING.RANDOM_STRATEGY
            activeSet, uSet = self.sampler.random()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN in ["stratified"]:
            activeSet, uSet = self.sampler.stratified()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN in ["match_population_proportion"]:
            activeSet, uSet = self.sampler.match_population_proportion()

        else:
            print(f"{self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError

        return activeSet, uSet
        
