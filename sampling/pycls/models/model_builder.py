from .ridge import RidgeModel
from .swin import SwinV2Model
from .resnet import ResNet18Model

def build_model(cfg):
    """Factory function to build models based on config."""
    model_type = cfg.MODEL.TYPE.lower()
    
    if model_type == 'ridge':
        return RidgeModel(cfg)
    elif model_type == 'swin':
        return SwinV2Model(cfg)
    elif model_type == 'resnet':
        return ResNet18Model(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")