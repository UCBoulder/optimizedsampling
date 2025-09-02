# This file is modified from official pycls repository

"""Model and loss construction functions."""
from pycls.models.ridge import *

import torch
from torch import nn
from torch.nn import functional as F
# Supported models
_models = {
    #Ridge regression
    'ridge': ridge_regression
}


class FeaturesNet(nn.Module):
    def __init__(self, in_layers, out_layers, use_mlp=False, penultimate_active=False):
        super().__init__()
        self.use_mlp = use_mlp
        self.penultimate_active = penultimate_active
        self.lin1 = nn.Linear(in_layers, in_layers)
        self.lin2 = nn.Linear(in_layers, in_layers)
        self.final = nn.Linear(in_layers, out_layers)

    def forward(self, x):
        feats = x
        if self.use_mlp:
            x = F.relu(self.lin1(x))
            x = F.relu((self.lin2(x)))
        out = self.final(x)
        if self.penultimate_active:
            return feats, out
        return out


def get_model(cfg):
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]


def build_model(cfg):
    """Builds the model."""
    if cfg.MODEL.LINEAR_FROM_FEATURES:
        num_features = 384 if cfg.DATASET.NAME in ['IMAGENET50', 'IMAGENET100', 'IMAGENET200'] else 512
        return FeaturesNet(num_features, cfg.MODEL.NUM_CLASSES)
    
    if cfg.MODEL.TYPE == 'ridge':
        return get_model(cfg)

    model = get_model(cfg)(num_classes=cfg.MODEL.NUM_CLASSES, use_dropout=True)
    if cfg.DATASET.NAME == 'MNIST':
        model.conv1 =  torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    return model 


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor
