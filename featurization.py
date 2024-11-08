from pathlib import Path

from torchgeo.models import RCF
from torchgeo.datsets import USAVars
from mosaiks.code.mosaiks import config as c
from mosaiks.code.mosaiks.featurization import featurize_and_save

image_folder = Path("/share/usavars/uar")
#image_folder = Path("test_image")

def mosaiks_featurization():
    out_fpath = Path(c.features_dir) / f"{image_folder.name}_mosaiks.pkl"

    featurize_and_save(image_folder, out_fpath, c)

def torchgeo_featurization():
    out_fpath = Path(c.features_dir) / f"{image_folder.name}_torchgeo.pkl"

    #Torchgeo Random Convolutional Feature Implementation
    rcf = RCF(
        in_channels=4, 
        features=256, #maybe change
        kernel_size=4, #maybe change
        bias=-1.0, 
        seed=None, 
        mode='empirical', 
        dataset=USAVars)
    
    