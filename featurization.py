from pathlib import Path
import torch
import dill
import numpy as np

from torchgeo.models import RCF
# from mosaiks.code.mosaiks import config as c
# from mosaiks.code.mosaiks.featurization import featurize_and_save
from USAVars import USAVars

train = USAVars(root="/share/usavars", split="train", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)
test = USAVars(root="/share/usavars", split="test", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)
val = USAVars(root="/share/usavars", split="val", labels=('treecover', 'elevation', 'population'), transforms=None, download=False, checksum=False)

image_folder = Path("/share/usavars/uar")
#image_folder = Path("test_image")

# def mosaiks_featurization():
#     out_fpath = Path(c.features_dir) / f"{image_folder.name}_mosaiks.pkl"

#     featurize_and_save(image_folder, out_fpath, c)

def torchgeo_featurization():
    out_fpath = "data/int/feature_matrices/CONTUS_UAR_torchgeo.pkl"

    imgs, ids, latlons = format_data(train, val, test)

    from IPython import embed; embed()

    #Torchgeo Random Convolutional Feature Implementation
    rcf = RCF(
        in_channels=4, 
        features=256, #if 256, not enough storage
        kernel_size=4, #maybe change; patch in mosaiks is 3
        bias=-1.0, 
        seed=42, 
        mode='empirical',
        dataset=train)
    
    from IPython import embed; embed()

    #Forward pass of TorchGeo RCF
    print("Featurizing images...")
    featurized_imgs = []
    for i in range(len(imgs)):
        print("Featurizing image ", i)
        img = torch.tensor(imgs[i])
        featurized_imgs.append(rcf.forward(img).numpy())
    featurized_imgs = np.array(featurized_imgs)

    with open(out_fpath, "wb") as f:
        dill.dump(
            {"X": featurized_imgs, "ids_X": ids, "latlon": latlons},
            f,
            protocol=4,
        )

'''
    Retrieve images, ids, and latlons from dataset
'''
def retrieve_data(dataset):
    print("Retrieving images from torchgeo...")
    imgs = np.array([dataset[i]['image'] for i in range(len(dataset))])

    print("Retrieving ids from torchgeo...")
    ids = np.array([dataset[i]['name'].replace('tile_', '').replace('.tif', '') for i in range(len(dataset))])

    print("Retrieving latlon from torchgeo...")
    latlons = np.array([[dataset[i]['centroid_lat'].item(), dataset[i]['centroid_lon'].item()] for i in range(len(dataset))])

    print("Done retrieving")
    return imgs, ids, latlons

'''
    Append train, val, and test data
'''
def format_data(*args):
    print("Formatting data...")
    combined_imgs = []
    combined_ids = []
    combined_latlons = []

    for arg in args:
        imgs, ids, latlons = retrieve_data(arg)

        from IPython import embed; embed()
        for i in range(len(imgs)):
            combined_imgs.append(imgs[i])
            combined_ids.append(ids[i])
            combined_latlons.append(latlons)

    combined_imgs = np.array(combined_imgs)
    combined_ids = np.array(combined_ids)
    combined_latlons = np.array(combined_latlons)

    return combined_imgs, combined_ids, combined_latlons

torchgeo_featurization()